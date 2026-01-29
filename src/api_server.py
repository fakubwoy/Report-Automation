from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import asyncio
import json
import os
import logging
from datetime import datetime, timedelta, date
from typing import List, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import the new ML Engine
# Note: We import inside the function or ensuring path is correct, 
# assuming src package is available
from src.ml_engine import perform_ml_analysis

logging.basicConfig(level=logging.INFO)

# Database setup
def get_db_engine():
    user = os.getenv('DB_USER', 'mfg_user')
    password = os.getenv('DB_PASSWORD', 'mfg_pass123')
    host = os.getenv('DB_HOST', 'db')
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME', 'manufacturing')
    return create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}")

engine = get_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logging.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Helper function to serialize data for JSON
def serialize_for_json(obj):
    """Convert date/datetime objects to ISO strings for JSON serialization"""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

# Background task for streaming live data
async def stream_live_data():
    """Background task that continuously fetches and broadcasts live data"""
    from src.ingestion import get_all_opcua_data
    
    logging.info("Starting live data stream background task...")
    
    while True:
        try:
            if len(manager.active_connections) > 0:
                live_data = await get_all_opcua_data(max_retries=2, retry_delay=1)
                if live_data:
                    # Serialize the data to ensure JSON compatibility
                    serialized_data = serialize_for_json(live_data)
                    
                    # Broadcast to all connected WebSocket clients
                    await manager.broadcast({
                        "type": "live_update",
                        "timestamp": datetime.now().isoformat(),
                        "data": serialized_data
                    })
                    logging.debug(f"Broadcast live data to {len(manager.active_connections)} clients")
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logging.error(f"Error in live data stream: {e}", exc_info=True)
            await asyncio.sleep(5)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting FastAPI application...")
    task = asyncio.create_task(stream_live_data())
    yield
    # Shutdown
    logging.info("Shutting down FastAPI application...")
    task.cancel()

app = FastAPI(title="Manufacturing Analytics API", version="2.0.0", lifespan=lifespan)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ProductionRecord(BaseModel):
    production_date: str
    machine_id: str
    units_produced: float
    defective_units: float
    downtime_min: float
    shift: str

class KPIResponse(BaseModel):
    total_units: int
    total_defects: int
    yield_percentage: float
    avg_downtime: float
    report_date: str

class MachineStats(BaseModel):
    machine_id: str
    units_produced: float
    defective_units: float
    downtime_minutes: float

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# REST API Endpoints

@app.get("/")
async def root():
    return {
        "service": "Manufacturing Analytics API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "production_data": "/api/production",
            "kpis": "/api/kpis",
            "machines": "/api/machines",
            "reports": "/api/reports/latest",
            "websocket": "/ws",
            "websocket_status": "/api/websocket/status",
            "ml_forecast": "/api/ml/forecast"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")

@app.get("/api/websocket/status")
async def websocket_status():
    """Check WebSocket connection status"""
    return {
        "websocket_endpoint": "/ws",
        "active_connections": len(manager.active_connections),
        "status": "operational"
    }

@app.get("/api/production", response_model=List[ProductionRecord])
async def get_production_data(
    days: int = 7,
    machine_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get production data with optional filtering"""
    try:
        query = """
            SELECT production_date, machine_id, units_produced, 
                   defective_units, downtime_min, shift
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '%s days'
        """ % days
        
        if machine_id:
            query += f" AND machine_id = '{machine_id}'"
        
        query += " ORDER BY production_date DESC, machine_id"
        
        df = pd.read_sql_query(query, engine)
        # Convert date to string for JSON serialization
        if 'production_date' in df.columns:
            df['production_date'] = df['production_date'].astype(str)
        
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kpis", response_model=KPIResponse)
async def get_kpis(days: int = 7, db: Session = Depends(get_db)):
    """Get calculated KPIs"""
    try:
        query = f"""
            SELECT SUM(units_produced) as total_units,
                   SUM(defective_units) as total_defects,
                   AVG(downtime_min) as avg_downtime
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        
        df = pd.read_sql_query(query, engine)
        
        total_units = int(df['total_units'].iloc[0]) if df['total_units'].iloc[0] else 0
        total_defects = int(df['total_defects'].iloc[0]) if df['total_defects'].iloc[0] else 0
        avg_downtime = float(df['avg_downtime'].iloc[0]) if df['avg_downtime'].iloc[0] else 0
        
        yield_pct = ((total_units - total_defects) / total_units * 100) if total_units > 0 else 0
        
        return {
            "total_units": total_units,
            "total_defects": total_defects,
            "yield_percentage": round(yield_pct, 2),
            "avg_downtime": round(avg_downtime, 2),
            "report_date": datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/machines", response_model=List[MachineStats])
async def get_machine_stats(days: int = 7, db: Session = Depends(get_db)):
    """Get aggregated statistics per machine"""
    try:
        query = f"""
            SELECT machine_id,
                   SUM(units_produced) as units_produced,
                   SUM(defective_units) as defective_units,
                   SUM(downtime_min) as downtime_minutes
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY machine_id
            ORDER BY machine_id
        """
        
        df = pd.read_sql_query(query, engine)
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- NEW ML ENDPOINT ----
@app.get("/api/ml/forecast")
async def get_ml_forecast(days: int = 30):
    """
    Trains a lightweight model on recent data and predicts 
    downtime for the next shift.
    """
    try:
        # Fetch data for ML
        query = f"""
            SELECT units_produced as "Units Produced", 
                   defective_units as "Defective Units", 
                   downtime_min as "Downtime (minutes)"
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        df = pd.read_sql_query(query, engine)
        
        predicted_downtime, score, _ = perform_ml_analysis(df)
        
        risk_level = "Low"
        if predicted_downtime > 20: risk_level = "Medium"
        if predicted_downtime > 40: risk_level = "High"

        return {
            "predicted_downtime_next_shift": predicted_downtime,
            "model_confidence_score": score,
            "risk_assessment": risk_level,
            "message": "Prediction based on linear regression of units vs defects."
        }
    except Exception as e:
        # Fallback if DB is empty or error occurs
        logging.error(f"ML Forecast Error: {e}")
        return {
            "predicted_downtime_next_shift": 0,
            "model_confidence_score": 0,
            "risk_assessment": "Unknown",
            "message": "Insufficient data."
        }
# -------------------------

@app.get("/api/reports/latest")
async def get_latest_report():
    """Download the latest PDF report"""
    report_path = "reports/pdf/Report.pdf"
    if os.path.exists(report_path):
        return FileResponse(
            report_path,
            media_type='application/pdf',
            filename=f'Manufacturing_Report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    else:
        raise HTTPException(status_code=404, detail="No report available")

@app.post("/api/production")
async def create_production_record(record: ProductionRecord, db: Session = Depends(get_db)):
    """Create a new production record (for third-party integrations)"""
    try:
        query = text("""
            INSERT INTO live_production 
            (production_date, machine_id, units_produced, defective_units, downtime_min, shift)
            VALUES (:prod_date, :machine_id, :units, :defects, :downtime, :shift)
        """)
        
        with engine.connect() as conn:
            conn.execute(query, {
                "prod_date": record.production_date,
                "machine_id": record.machine_id,
                "units": record.units_produced,
                "defects": record.defective_units,
                "downtime": record.downtime_min,
                "shift": record.shift
            })
            conn.commit()
        
        return {"status": "success", "message": "Record created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/machines/live")
async def get_live_machine_data():
    """Get current live data from OPC-UA"""
    try:
        from src.ingestion import get_all_opcua_data
        live_data = await get_all_opcua_data(max_retries=2, retry_delay=1)
        
        # Serialize data for JSON
        serialized_data = serialize_for_json(live_data)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "machines": serialized_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial connection confirmation
        await websocket.send_json({"type": "connected", "message": "WebSocket connected successfully"})
        
        while True:
            # Keep connection alive by receiving messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back for heartbeat
                await websocket.send_json({"type": "heartbeat", "status": "connected"})
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logging.info("Client disconnected from WebSocket")
    except Exception as e:
        logging.error(f"WebSocket error: {e}", exc_info=True)
        try:
            manager.disconnect(websocket)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
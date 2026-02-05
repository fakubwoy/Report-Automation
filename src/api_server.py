from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
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

# Import the ML Engine (both versions for backward compatibility)
try:
    from src.ml_engine_advanced import perform_advanced_ml_analysis, perform_ml_analysis
    from src.ai_insights import get_comprehensive_ai_insights
    ADVANCED_ML_AVAILABLE = True
    logging.info("Advanced ML and AI engines loaded successfully")
except ImportError:
    try:
        from src.ml_engine import perform_ml_analysis
        ADVANCED_ML_AVAILABLE = False
        logging.warning("Advanced ML engine not available, using basic version")
    except ImportError:
        logging.error("No ML engine available")
        ADVANCED_ML_AVAILABLE = False

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
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

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

# Function to save live data to DB safely
def save_live_data_to_db(live_data):
    try:
        if not live_data:
            return
            
        # Create a DataFrame
        df = pd.DataFrame(live_data)
        
        # We need to ensure the columns match the DB schema
        # The ingestion typically returns: machine_id, units_produced, defective_units, downtime_min, production_date, shift
        
        # Use a separate engine connection to avoid threading issues with the global session
        with engine.connect() as conn:
            df.to_sql('live_production', conn, if_exists='append', index=False)
            conn.commit()
            
        logging.debug(f"Persisted {len(df)} live records to database.")
    except Exception as e:
        logging.error(f"Failed to persist live data to DB: {e}")

# Background task for streaming live data
async def stream_live_data():
    """Background task that fetches, broadcasts, AND persists live data"""
    from src.ingestion import get_all_opcua_data
    
    logging.info("Starting live data stream background task...")
    
    while True:
        try:
            # 1. Fetch Data
            # We fetch even if no clients are connected so we can record history to DB
            live_data = await get_all_opcua_data(max_retries=2, retry_delay=1)
            
            if live_data:
                # 2. Persist to DB (CRITICAL FIX: This ensures charts update without restart)
                # Run in a separate thread to not block the async event loop
                await asyncio.to_thread(save_live_data_to_db, live_data)

                # 3. Broadcast to WebSocket Clients
                if len(manager.active_connections) > 0:
                    serialized_data = serialize_for_json(live_data)
                    await manager.broadcast({
                        "type": "live_update",
                        "timestamp": datetime.now().isoformat(),
                        "data": serialized_data
                    })
                    
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logging.error(f"Error in live data stream: {e}", exc_info=True)
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting FastAPI application...")
    task = asyncio.create_task(stream_live_data())
    yield
    logging.info("Shutting down FastAPI application...")
    task.cancel()

app = FastAPI(title="Manufacturing Analytics API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ROUTES ---

@app.get("/")
async def root():
    return {
        "service": "Manufacturing Analytics API",
        "status": "operational",
        "endpoints": ["/api/production", "/api/kpis", "/api/machines", "/ws"]
    }

@app.get("/api/health")
async def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")

@app.get("/api/production", response_model=List[ProductionRecord])
async def get_production_data(days: int = 7, machine_id: Optional[str] = None):
    try:
        query = f"SELECT * FROM live_production WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'"
        if machine_id:
            query += f" AND machine_id = '{machine_id}'"
        query += " ORDER BY production_date DESC, machine_id"
        
        df = pd.read_sql_query(query, engine)
        if 'production_date' in df.columns:
            df['production_date'] = df['production_date'].astype(str)
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kpis", response_model=KPIResponse)
async def get_kpis(days: int = 7):
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
async def get_machine_stats(days: int = 7):
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

@app.get("/api/ml/forecast")
async def get_ml_forecast(days: int = 30):
    """Advanced ML forecast with ensemble predictions"""
    try:
        query = f"""
            SELECT units_produced as "Units Produced", 
                   defective_units as "Defective Units", 
                   downtime_min as "Downtime (minutes)"
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        df = pd.read_sql_query(query, engine)
        
        if ADVANCED_ML_AVAILABLE:
            # Use advanced ML engine
            prediction, confidence, insights, _ = perform_advanced_ml_analysis(df)
            
            return {
                "predicted_downtime_next_shift": prediction,
                "model_confidence_score": confidence,
                "risk_assessment": insights.get('risk_level', 'Unknown'),
                "anomalies_detected": insights.get('anomalies_detected', 0),
                "feature_importance": insights.get('feature_importance', {}),
                "recommendations": insights.get('recommendations', []),
                "model_type": "Advanced Ensemble (RF + XGBoost)",
                "message": "Prediction based on ensemble machine learning models."
            }
        else:
            # Fallback to basic ML
            predicted_downtime, score, _ = perform_ml_analysis(df)
            
            risk_level = "Low"
            if predicted_downtime > 20: risk_level = "Medium"
            if predicted_downtime > 40: risk_level = "High"

            return {
                "predicted_downtime_next_shift": predicted_downtime,
                "model_confidence_score": score,
                "risk_assessment": risk_level,
                "model_type": "Basic Linear Regression",
                "message": "Prediction based on linear regression."
            }
    except Exception as e:
        logging.error(f"ML Forecast Error: {e}", exc_info=True)
        return {
            "predicted_downtime_next_shift": 0, 
            "model_confidence_score": 0, 
            "risk_assessment": "Unknown",
            "error": str(e)
        }

@app.get("/api/ai/insights")
async def get_ai_insights_endpoint(days: int = 7):
    """
    NEW: Get comprehensive AI-powered strategic insights
    """
    try:
        if not ADVANCED_ML_AVAILABLE:
            return {
                "error": "Advanced AI features not available",
                "message": "Install required packages: aiohttp, xgboost"
            }
        
        # Get production data
        query = f"""
            SELECT units_produced as "Units Produced", 
                   defective_units as "Defective Units", 
                   downtime_min as "Downtime (minutes)",
                   machine_id, production_date, shift
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        df = pd.read_sql_query(query, engine)
        
        # Get KPIs
        kpi_query = f"""
            SELECT SUM(units_produced) as total_units,
                   SUM(defective_units) as total_defects,
                   AVG(downtime_min) as avg_downtime
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        kpi_df = pd.read_sql_query(kpi_query, engine)
        
        total_units = int(kpi_df['total_units'].iloc[0]) if kpi_df['total_units'].iloc[0] else 0
        total_defects = int(kpi_df['total_defects'].iloc[0]) if kpi_df['total_defects'].iloc[0] else 0
        yield_pct = ((total_units - total_defects) / total_units * 100) if total_units > 0 else 0
        
        kpi_summary = {
            'total_units': total_units,
            'total_defects': total_defects,
            'yield_percentage': round(yield_pct, 2),
            'avg_downtime': float(kpi_df['avg_downtime'].iloc[0]) if kpi_df['avg_downtime'].iloc[0] else 0
        }
        
        # Get machine stats
        machine_query = f"""
            SELECT machine_id,
                   SUM(units_produced) as units_produced,
                   SUM(defective_units) as defective_units,
                   SUM(downtime_min) as downtime_minutes
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY machine_id
        """
        machine_df = pd.read_sql_query(machine_query, engine)
        machine_stats = machine_df.to_dict('records')
        
        # Perform ML analysis
        prediction, confidence, ml_insights, _ = perform_advanced_ml_analysis(df)
        
        # Get AI insights
        ai_insights = await get_comprehensive_ai_insights(
            ml_insights, 
            df.to_dict('records')[:10],  # Send sample data
            kpi_summary,
            machine_stats
        )
        
        return {
            "ml_analysis": {
                "prediction": prediction,
                "confidence": confidence,
                "insights": ml_insights
            },
            "ai_insights": ai_insights,
            "kpis": kpi_summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"AI Insights Error: {e}", exc_info=True)
        return {
            "error": str(e),
            "message": "Failed to generate AI insights"
        }

@app.get("/api/ai/maintenance-plan")
async def get_maintenance_plan(days: int = 7):
    """
    NEW: Get AI-generated maintenance schedule
    """
    try:
        if not ADVANCED_ML_AVAILABLE:
            return {"error": "AI features not available"}
        
        from src.ai_insights import AIInsightsEngine
        
        # Get ML insights
        query = f"""
            SELECT units_produced as "Units Produced", 
                   defective_units as "Defective Units", 
                   downtime_min as "Downtime (minutes)"
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        df = pd.read_sql_query(query, engine)
        
        prediction, confidence, ml_insights, _ = perform_advanced_ml_analysis(df)
        
        # Get machine stats
        machine_query = f"""
            SELECT machine_id,
                   SUM(units_produced) as units_produced,
                   SUM(defective_units) as defective_units,
                   SUM(downtime_min) as downtime_minutes
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY machine_id
        """
        machine_df = pd.read_sql_query(machine_query, engine)
        machine_stats = machine_df.to_dict('records')
        
        # Generate maintenance plan
        ai_engine = AIInsightsEngine()
        maintenance_plan = await ai_engine.generate_maintenance_plan(
            ml_insights, machine_stats
        )
        
        return {
            "maintenance_plan": maintenance_plan,
            "ml_prediction": prediction,
            "risk_level": ml_insights.get('risk_level', 'Unknown'),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Maintenance Plan Error: {e}", exc_info=True)
        return {"error": str(e)}

# --- REPORTING ENDPOINTS ---

@app.post("/api/reports/generate")
async def generate_report(background_tasks: BackgroundTasks):
    """Trigger the report generation pipeline manually"""
    try:
        # Deferred import to avoid circular dependency
        from main import run_pipeline
        # Run in background to not block the response
        background_tasks.add_task(run_pipeline)
        return {"status": "success", "message": "Report generation started"}
    except ImportError:
        raise HTTPException(status_code=500, detail="Reporting module not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/latest")
async def get_latest_report():
    """Download the latest PDF report, generating it if missing"""
    report_path = "reports/pdf/Report.pdf"
    
    # If report doesn't exist, try to generate it synchronously
    if not os.path.exists(report_path):
        try:
            from main import run_pipeline
            logging.info("Report missing, generating new one...")
            run_pipeline()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

    if os.path.exists(report_path):
        return FileResponse(
            report_path,
            media_type='application/pdf',
            filename=f'Manufacturing_Report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    else:
        raise HTTPException(status_code=404, detail="No report available")

@app.get("/api/machines/live")
async def get_live_machine_data():
    try:
        from src.ingestion import get_all_opcua_data
        live_data = await get_all_opcua_data(max_retries=2, retry_delay=1)
        serialized_data = serialize_for_json(live_data)
        return {"timestamp": datetime.now().isoformat(), "machines": serialized_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "WebSocket connected successfully"})
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                await websocket.send_json({"type": "heartbeat", "status": "connected"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        try:
            manager.disconnect(websocket)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
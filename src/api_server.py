from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks, Header
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

# Import tenant manager
try:
    from src.tenant_manager import get_tenant_manager
    TENANT_SUPPORT = True
    logging.info("Tenant management available")
except ImportError:
    TENANT_SUPPORT = False
    logging.warning("Tenant management not available")

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

app = FastAPI(title="Manufacturing Analytics API", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

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

# Multi-tenant Pydantic models
class TenantCreate(BaseModel):
    tenant_id: str
    tenant_name: str
    plant_location: Optional[str] = None
    timezone: str = "UTC"
    config: Optional[dict] = {}

class MachineRegister(BaseModel):
    machine_id: str
    machine_name: str
    machine_type: Optional[str] = None
    capacity: Optional[float] = None

class UserAccess(BaseModel):
    user_id: str
    user_name: str
    user_email: str
    role: str = "viewer"

class ProductionDataSubmit(BaseModel):
    production_date: str
    machine_id: str
    units_produced: float
    defective_units: float
    downtime_min: float
    shift: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to extract tenant from header
def get_tenant_id(x_tenant_id: Optional[str] = Header(None)) -> str:
    """Extract tenant ID from request header"""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="Missing X-Tenant-ID header")
    return x_tenant_id

# ============================================================================
# STANDARD ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "Manufacturing Analytics API",
        "status": "operational",
        "version": "2.2.0",
        "tenant_support": TENANT_SUPPORT,
        "endpoints": ["/api/production", "/api/kpis", "/api/machines", "/api/tenants", "/ws"]
    }

@app.get("/api/health")
async def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected", "tenant_support": TENANT_SUPPORT}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")

@app.get("/api/diagnostics")
async def diagnostics():
    """Diagnostic endpoint to check system status"""
    diag = {
        "tenant_support": TENANT_SUPPORT,
        "advanced_ml": ADVANCED_ML_AVAILABLE,
        "database": "unknown",
        "tenant_tables": {}
    }
    
    try:
        with engine.connect() as conn:
            # Check database connection
            conn.execute(text("SELECT 1"))
            diag["database"] = "connected"
            
            # Check if tenant tables exist
            tables_to_check = ['tenants', 'production_data_tenant', 'machines_tenant', 'users_tenant']
            for table in tables_to_check:
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    )
                """))
                diag["tenant_tables"][table] = result.scalar()
    except Exception as e:
        diag["database_error"] = str(e)
    
    return diag

@app.get("/api/additional/metrics")
async def get_additional_metrics(days: int = 7):
    """Get additional metrics like energy consumption and cycle time"""
    try:
        # Since this data might not be in your database yet, 
        # you can calculate it or provide default values
        
        # First, let's get some actual data to calculate from
        query = f"""
            SELECT SUM(units_produced) as total_units,
                   SUM(downtime_min) as total_downtime
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        df = pd.read_sql_query(query, engine)
        
        total_units = int(df['total_units'].iloc[0]) if df['total_units'].iloc[0] else 0
        total_downtime = float(df['total_downtime'].iloc[0]) if df['total_downtime'].iloc[0] else 0
        
        # Calculate or estimate other metrics
        # Energy consumption: estimate 0.5 kWh per unit produced
        energy_consumption = round(total_units * 0.5, 2)
        
        # Average cycle time: estimate based on units and downtime
        # Assume 8 hour shifts, 7 days
        available_time = (days * 24 * 60) - total_downtime  # minutes
        avg_cycle_time = round(available_time / total_units, 2) if total_units > 0 else 0
        
        return {
            "energy_consumption_kwh": energy_consumption,
            "avg_cycle_time_min": avg_cycle_time,
            "total_units": total_units,
            "days_analyzed": days
        }
    except Exception as e:
        logging.error(f"Additional metrics error: {e}")
        return {
            "energy_consumption_kwh": 0,
            "avg_cycle_time_min": 0,
            "error": str(e)
        }

@app.get("/api/production")
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

@app.post("/api/production")
async def add_production_record(record: ProductionRecord):
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO live_production 
                (production_date, machine_id, units_produced, defective_units, downtime_min, shift)
                VALUES (:date, :machine, :units, :defects, :downtime, :shift)
            """), {
                'date': record.production_date,
                'machine': record.machine_id,
                'units': record.units_produced,
                'defects': record.defective_units,
                'downtime': record.downtime_min,
                'shift': record.shift
            })
            conn.commit()
        return {"status": "success", "message": "Record added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kpis")
async def get_kpis(days: int = 7):
    try:
        query = f"""
            SELECT SUM(units_produced) as total_units,
                   SUM(defective_units) as total_defects,
                   AVG(downtime_min) as avg_downtime,
                   SUM(downtime_min) as total_downtime
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        df = pd.read_sql_query(query, engine)
        
        total_units = int(df['total_units'].iloc[0]) if df['total_units'].iloc[0] else 0
        total_defects = int(df['total_defects'].iloc[0]) if df['total_defects'].iloc[0] else 0
        avg_downtime = float(df['avg_downtime'].iloc[0]) if df['avg_downtime'].iloc[0] else 0
        total_downtime = float(df['total_downtime'].iloc[0]) if df['total_downtime'].iloc[0] else 0
        
        yield_pct = ((total_units - total_defects) / total_units * 100) if total_units > 0 else 0
        defect_rate = (total_defects / total_units * 100) if total_units > 0 else 0
        
        # Calculate efficiency (simplified: uptime percentage)
        # Assuming 480 minutes per day (8 hours) per machine, 3 machines
        theoretical_max_time = days * 480 * 3
        actual_uptime = theoretical_max_time - total_downtime
        efficiency = (actual_uptime / theoretical_max_time * 100) if theoretical_max_time > 0 else 0
        
        return {
            "total_units": total_units,
            "total_defects": total_defects,
            "yield_percentage": round(yield_pct, 2),
            "defect_rate": round(defect_rate, 2),
            "avg_downtime": round(avg_downtime, 2),
            "total_downtime": round(total_downtime, 2),
            "efficiency": round(efficiency, 2),
            "report_date": datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/machines")
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

@app.get("/api/ml/prediction")
async def get_ml_prediction(days: int = 30):
    """ML prediction endpoint - compatible with dashboard"""
    return await get_ml_forecast(days)

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
        
        if len(df) == 0:
            # Return default values if no data
            return {
                "predicted_downtime_next_shift": 0,
                "model_confidence_score": 0,
                "risk_assessment": "Unknown",
                "anomalies_detected": 0,
                "feature_importance": {},
                "recommendations": [],
                "model_type": "No Data",
                "message": "Insufficient data for prediction"
            }
        
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
async def get_ai_insights(days: int = 7):
    """Get AI-powered strategic insights"""
    try:
        if not ADVANCED_ML_AVAILABLE:
            return {"error": "AI features not available"}
        
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
        
        if len(df) < 5:
            return {"error": "Insufficient data for AI analysis"}
        
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
        avg_downtime = float(kpi_df['avg_downtime'].iloc[0]) if kpi_df['avg_downtime'].iloc[0] else 0
        
        kpi_summary = {
            'total_units': total_units,
            'total_defects': total_defects,
            'yield_percentage': yield_pct,
            'avg_downtime': avg_downtime
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


@app.get("/api/forecasting/advanced")
async def get_advanced_forecast(days: int = 30):
    """Get advanced time-series forecast data"""
    try:
        query = f"""
            SELECT units_produced as "Units Produced", 
                   defective_units as "Defective Units", 
                   downtime_min as "Downtime (minutes)"
            FROM live_production
            WHERE production_date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY production_date
        """
        df = pd.read_sql_query(query, engine)
        
        if len(df) < 10:
            return {
                "error": "Insufficient data for forecasting",
                "message": "Need at least 10 days of data"
            }
        
        # Import and run the forecasting engine
        from src.forecasting_engine import perform_advanced_forecasting
        
        lstm_forecast, prophet_forecast, ensemble_forecast, chart_path = perform_advanced_forecasting(
            df, 
            target_col='Downtime (minutes)', 
            forecast_periods=7
        )
        
        # Extract forecast values
        lstm_values = lstm_forecast[0].tolist() if lstm_forecast and lstm_forecast[0] is not None else None
        prophet_values = prophet_forecast.tail(7)['yhat'].tolist() if prophet_forecast is not None else None
        ensemble_values = ensemble_forecast.tolist() if ensemble_forecast is not None else None
        
        # Historical data for chart
        historical_downtime = df['Downtime (minutes)'].tail(20).tolist()
        
        return {
            "historical_data": historical_downtime,
            "lstm_forecast": lstm_values,
            "prophet_forecast": prophet_values,
            "ensemble_forecast": ensemble_values,
            "forecast_periods": 7,
            "chart_path": chart_path,
            "lstm_prediction": lstm_values[0] if lstm_values else None,
            "prophet_prediction": prophet_values[0] if prophet_values else None,
            "ensemble_prediction": ensemble_values[0] if ensemble_values else None,
            "model_confidence_score": 0.85  # Example confidence score
        }
        
    except Exception as e:
        logging.error(f"Advanced forecasting failed: {e}")
        return {"error": str(e)}
    
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

# ============================================================================
# MULTI-TENANT ENDPOINTS
# ============================================================================

@app.post("/api/tenants/create")
async def create_tenant(tenant: TenantCreate):
    """
    Create a new tenant (plant/factory)
    
    Example:
    ```
    POST /api/tenants/create
    {
        "tenant_id": "plant_chicago",
        "tenant_name": "Chicago Manufacturing Plant",
        "plant_location": "Chicago, IL, USA",
        "timezone": "America/Chicago"
    }
    ```
    """
    if not TENANT_SUPPORT:
        return {
            "status": "error",
            "message": "Tenant management not enabled. Please ensure tenant_manager.py is in the src/ directory.",
            "tenant_support": False
        }
    
    try:
        tm = get_tenant_manager()
        
        # Validate tenant_id
        if not tenant.tenant_id or len(tenant.tenant_id) < 3:
            raise HTTPException(status_code=400, detail="tenant_id must be at least 3 characters")
        
        success = tm.create_tenant(
            tenant.tenant_id,
            tenant.tenant_name,
            tenant.plant_location,
            tenant.timezone,
            tenant.config
        )
        
        if success:
            return {"status": "success", "tenant_id": tenant.tenant_id, "message": "Tenant created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Tenant creation failed - check database connection")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Tenant creation error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to create tenant: {str(e)}",
            "error_type": type(e).__name__
        }


@app.get("/api/tenants/list")
async def list_tenants(active_only: bool = True):
    """
    Get list of all tenants
    
    Example:
    ```
    GET /api/tenants/list?active_only=true
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        tenants = tm.get_tenant_list(active_only)
        
        # Serialize datetime objects
        for tenant in tenants:
            for key, value in tenant.items():
                if isinstance(value, (date, datetime)):
                    tenant[key] = value.isoformat()
        
        return {"tenants": tenants, "count": len(tenants)}
        
    except Exception as e:
        logging.error(f"List tenants error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/machines/register")
async def register_machine(tenant_id: str, machine: MachineRegister):
    """
    Register a machine for a tenant
    
    Example:
    ```
    POST /api/tenants/plant_chicago/machines/register
    {
        "machine_id": "M001",
        "machine_name": "Assembly Line 1",
        "machine_type": "Assembly",
        "capacity": 500
    }
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        success = tm.register_machine(
            tenant_id,
            machine.machine_id,
            machine.machine_name,
            machine.machine_type,
            machine.capacity
        )
        
        if success:
            return {"status": "success", "machine_id": machine.machine_id}
        else:
            raise HTTPException(status_code=500, detail="Machine registration failed")
            
    except Exception as e:
        logging.error(f"Machine registration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tenants/{tenant_id}/machines")
async def get_tenant_machines(tenant_id: str, active_only: bool = True):
    """
    Get all machines for a tenant
    
    Example:
    ```
    GET /api/tenants/plant_chicago/machines?active_only=true
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        machines = tm.get_tenant_machines(tenant_id, active_only)
        
        # Serialize datetime objects
        for machine in machines:
            for key, value in machine.items():
                if isinstance(value, (date, datetime)):
                    machine[key] = value.isoformat()
        
        return {"tenant_id": tenant_id, "machines": machines, "count": len(machines)}
        
    except Exception as e:
        logging.error(f"Get tenant machines error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/production/submit")
async def submit_production_data(tenant_id: str, data: List[ProductionDataSubmit]):
    """
    Submit production data for a tenant
    
    Example:
    ```
    POST /api/tenants/plant_chicago/production/submit
    [
        {
            "production_date": "2026-02-07",
            "machine_id": "M001",
            "units_produced": 450,
            "defective_units": 5,
            "downtime_min": 15,
            "shift": "Day"
        }
    ]
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        
        # Convert to list of dicts
        records = [record.dict() for record in data]
        
        success = tm.add_production_data(tenant_id, records)
        
        if success:
            return {"status": "success", "records_added": len(records)}
        else:
            raise HTTPException(status_code=500, detail="Data submission failed")
            
    except Exception as e:
        logging.error(f"Submit production data error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tenants/{tenant_id}/production")
async def get_tenant_production(tenant_id: str, days: int = 7, machine_id: Optional[str] = None):
    """
    Get production data for a tenant (isolated)
    
    Example:
    ```
    GET /api/tenants/plant_chicago/production?days=7&machine_id=M001
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        df = tm.get_production_data(tenant_id, days, machine_id)
        
        # Convert to JSON-serializable format
        data = df.to_dict('records')
        
        # Handle date serialization
        for record in data:
            if 'production_date' in record and isinstance(record['production_date'], date):
                record['production_date'] = record['production_date'].isoformat()
        
        return {
            "tenant_id": tenant_id,
            "data": data,
            "count": len(data)
        }
        
    except Exception as e:
        logging.error(f"Get tenant production error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tenants/{tenant_id}/kpis")
async def get_tenant_kpis(tenant_id: str, days: int = 7):
    """
    Get KPIs for a specific tenant
    
    Example:
    ```
    GET /api/tenants/plant_chicago/kpis?days=7
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        df = tm.get_production_data(tenant_id, days)
        
        if df.empty:
            return {
                "tenant_id": tenant_id,
                "message": "No data available",
                "kpis": {}
            }
        
        # Rename columns to match KPI engine expectations
        df_renamed = df.rename(columns={
            'units_produced': 'Units Produced',
            'defective_units': 'Defective Units',
            'downtime_min': 'Downtime (minutes)',
            'machine_id': 'Machine ID'
        })
        
        from src.kpi_engine import calculate_kpis
        summary, machine_stats = calculate_kpis(df_renamed)
        
        # Convert machine_stats to dict if it's a DataFrame
        if hasattr(machine_stats, 'to_dict'):
            machine_stats_data = machine_stats.to_dict('records')
        else:
            machine_stats_data = machine_stats
        
        return {
            "tenant_id": tenant_id,
            "kpis": summary,
            "machine_stats": machine_stats_data
        }
        
    except Exception as e:
        logging.error(f"Get tenant KPIs error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/corporate/summary")
async def get_corporate_summary(days: int = 7):
    """
    Get aggregated summary across all tenants (corporate overview)
    
    Example:
    ```
    GET /api/corporate/summary?days=7
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        df = tm.get_cross_tenant_summary(days)
        
        summary_data = df.to_dict('records')
        
        return {
            "summary": summary_data,
            "total_tenants": len(summary_data),
            "period_days": days
        }
        
    except Exception as e:
        logging.error(f"Corporate summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/users/grant-access")
async def grant_user_access(tenant_id: str, user: UserAccess):
    """
    Grant user access to a tenant
    
    Example:
    ```
    POST /api/tenants/plant_chicago/users/grant-access
    {
        "user_id": "john_doe",
        "user_name": "John Doe",
        "user_email": "john@company.com",
        "role": "manager"
    }
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        success = tm.grant_user_access(
            user.user_id,
            tenant_id,
            user.user_name,
            user.user_email,
            user.role
        )
        
        if success:
            return {"status": "success", "user_id": user.user_id, "tenant_id": tenant_id}
        else:
            raise HTTPException(status_code=500, detail="User access grant failed")
            
    except Exception as e:
        logging.error(f"Grant user access error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/tenants")
async def get_user_tenants(user_id: str):
    """
    Get all tenants a user has access to
    
    Example:
    ```
    GET /api/users/john_doe/tenants
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        tenants = tm.get_user_tenants(user_id)
        
        # Serialize datetime objects
        for tenant in tenants:
            for key, value in tenant.items():
                if isinstance(value, (date, datetime)):
                    tenant[key] = value.isoformat()
        
        return {
            "user_id": user_id,
            "tenants": tenants,
            "count": len(tenants)
        }
        
    except Exception as e:
        logging.error(f"Get user tenants error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/migrate-legacy-data")
async def migrate_legacy_data(tenant_id: str):
    """
    Migrate legacy data from live_production table to tenant-isolated table
    
    Example:
    ```
    POST /api/tenants/plant_chicago/migrate-legacy-data
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        success = tm.migrate_legacy_data(tenant_id)
        
        if success:
            return {"status": "success", "message": f"Legacy data migrated to tenant {tenant_id}"}
        else:
            raise HTTPException(status_code=500, detail="Migration failed")
            
    except Exception as e:
        logging.error(f"Migrate legacy data error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
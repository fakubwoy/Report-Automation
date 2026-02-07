# Automated Manufacturing Intelligence System (AMIS)

## 1. Project Overview
An enterprise-grade manufacturing analytics platform featuring real-time data collection, AI-powered predictive maintenance, advanced time-series forecasting, multi-tenant support, and automated reporting with strategic insights.

**Key Features:**
- **Real-time Data Collection**: Live OPC-UA industrial PLC data streaming
- **Advanced ML Engine**: Ensemble models (Random Forest + XGBoost) for predictive maintenance
- **Time-Series Forecasting**: LSTM and Prophet models with ensemble forecasting
- **Multi-Tenant Architecture**: Isolated data management for multiple plants/factories
- **AI Strategic Insights**: Groq-powered analysis for actionable recommendations
- **Live Dashboard**: WebSocket-powered analytics with interactive visualizations
- **RESTful API**: Comprehensive REST API with production data, KPIs, ML, AI, and tenant endpoints
- **Database Persistence**: PostgreSQL with tenant isolation and JSONB support
- **Automated Reporting**: Scheduled PDF generation with ML insights and AI analysis
- **Containerized Deployment**: Docker Compose orchestration with health checks

## 2. System Architecture

### 2.1 Core Services
* **PLC Simulator** (`plc-simulator`): Simulates 3 manufacturing machines via OPC-UA protocol
* **PostgreSQL Database** (`db`): Persistent storage for production history
* **API Server** (`api-server`): FastAPI with REST + WebSocket streaming + ML + AI endpoints
* **Dashboard** (`dashboard`): Nginx-served interactive web dashboard with live updates
* **Reporting Pipeline** (`reporting-app`): Automated data collection, ML/AI analysis, and report generation

### 2.2 Application Modules
* **Data Ingestion** (`src/ingestion.py`): OPC-UA client with retry logic
* **API Server** (`src/api_server.py`): FastAPI with REST, WebSocket, ML, AI, and multi-tenant endpoints
* **Tenant Manager** (`src/tenant_manager.py`): Multi-tenant data isolation and management
* **Advanced ML Engine** (`src/ml_engine_advanced.py`): Ensemble models with anomaly detection
* **Forecasting Engine** (`src/forecasting_engine.py`): LSTM and Prophet time-series forecasting
* **AI Insights** (`src/ai_insights.py`): Groq-powered strategic analysis
* **Validation** (`src/validation.py`): Data integrity checks and anomaly logging
* **KPI Engine** (`src/kpi_engine.py`): Calculates Yield, Defect Rates, Downtime metrics
* **Report Generator** (`src/report_generator.py`): PDF reports with ML visualizations
* **Scheduler** (`src/scheduler.py`): Daily automated execution at 08:00 AM

## 3. Directory Structure
```text
report_automation/
├── dashboard/
│   └── dashboard.html          # Real-time web dashboard
├── data/
│   ├── raw/                    # Input fallback files
│   └── processed/              # Historical archives
├── reports/
│   ├── pdf/                    # Generated PDF reports
│   ├── chart_prod.png          # Production charts
│   ├── chart_down_trend.png    # Downtime trends
│   ├── chart_down_var.png      # Machine comparisons
│   ├── chart_ml_advanced.png   # Advanced ML visualizations
│   └── ai_insights.txt         # AI strategic analysis
├── src/
│   ├── opcua_simulator.py      # PLC simulator
│   ├── ingestion.py            # OPC-UA + PostgreSQL
│   ├── api_server.py           # FastAPI REST + WebSocket + ML + AI
│   ├── tenant_manager.py       # Multi-tenant data isolation
│   ├── ml_engine_advanced.py   # Advanced ML engine
│   ├── forecasting_engine.py   # LSTM/Prophet forecasting
│   ├── ai_insights.py          # AI insights generator
│   ├── validation.py           # Data quality checks
│   ├── kpi_engine.py           # KPI calculations
│   ├── report_generator.py     # PDF and charts
│   └── scheduler.py            # Automation scheduler
├── .env                        # Environment variables
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Container build
├── main.py                     # Pipeline orchestrator
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## 4. Technology Stack
* **Core**: Python 3.9, FastAPI, Uvicorn
* **Real-time**: WebSockets, asyncio
* **Data**: Pandas, NumPy
* **ML**: scikit-learn (Random Forest, Isolation Forest), XGBoost
* **Forecasting**: TensorFlow/Keras (LSTM), Prophet
* **AI**: Groq API (Llama 3.3 70B) for strategic insights
* **Visualizations**: Chart.js (web), Matplotlib, Seaborn (PDF)
* **Reporting**: FPDF, XlsxWriter
* **Industrial Protocol**: asyncua (OPC-UA)
* **Database**: PostgreSQL 15 with SQLAlchemy and JSONB support
* **Deployment**: Docker Compose

## 5. Setup and Deployment

### 5.1 Prerequisites
* Docker Engine 20.10+ and Docker Compose V2
* 2GB RAM, 5GB disk space
* Modern web browser
* **(Optional)** Groq API key for AI insights - Get at https://console.groq.com

### 5.2 Environment Configuration
Create `.env` file:

```bash
# Database
DB_HOST=db
DB_PORT=5432
DB_NAME=manufacturing
DB_USER=mfg_user
DB_PASSWORD=mfg_pass123

# OPC-UA
OPCUA_SERVER_URL=opc.tcp://plc-simulator:4840/freeopcua/server/

# AI Features (Optional - Free)
GROQ_API_KEY=your_groq_api_key_here

# Email (Optional)
EMAIL_SENDER=your-email@gmail.com
EMAIL_RECEIVER=manager@factory.com
EMAIL_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### 5.3 Quick Start

```bash
# Build and start all services
docker compose up -d --build

# Watch initialization
docker logs -f mfg_api_server

# Verify services
docker ps
```

### 5.4 Access Points

**Live Dashboard**: http://localhost:8080
- Real-time KPI cards
- Production trend charts
- Machine performance comparison
- Live status updates (every 5 seconds)

**API Endpoints**: http://localhost:8000
```bash
# Health check
curl http://localhost:8000/api/health

# Get KPIs
curl http://localhost:8000/api/kpis?days=7

# Get production data
curl http://localhost:8000/api/production?days=7

# Advanced ML forecast
curl http://localhost:8000/api/ml/forecast?days=30

# Advanced time-series forecast (LSTM/Prophet)
curl http://localhost:8000/api/ml/forecast-advanced?days=7

# AI strategic insights
curl http://localhost:8000/api/ai/insights?days=7

# AI maintenance plan
curl http://localhost:8000/api/ai/maintenance-plan?days=7

# Get live machine data
curl http://localhost:8000/api/machines/live

# Multi-tenant endpoints
curl http://localhost:8000/api/tenants/list
curl -X POST http://localhost:8000/api/tenants/create \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"plant_chicago","tenant_name":"Chicago Plant"}'
curl http://localhost:8000/api/tenants/plant_chicago/kpis?days=7
curl http://localhost:8000/api/corporate/summary?days=7

# Download latest PDF
curl http://localhost:8000/api/reports/latest --output report.pdf
```

**WebSocket**: ws://localhost:8000/ws
- Real-time data streaming every 5 seconds

## 6. Advanced ML & AI Features 

### 6.1 Ensemble Machine Learning
**Models**: Random Forest + XGBoost with weighted averaging
- **Features**: 8 engineered features including defect rates, quality scores, production efficiency, and rolling statistics
- **Anomaly Detection**: Isolation Forest for identifying unusual patterns
- **Confidence Scoring**: Model agreement-based confidence metrics
- **Risk Assessment**: 4-level risk classification (Low/Medium/High/Critical)

**ML Endpoint**: `/api/ml/forecast`
```bash
curl http://localhost:8000/api/ml/forecast?days=30
```

**Response Example:**
```json
{
  "predicted_downtime_next_shift": 28.5,
  "model_confidence_score": 0.89,
  "risk_assessment": "Medium",
  "feature_importance": {...},
  "anomalies_detected": 3,
  "recommendations": [...]
}
```

### 6.2 Advanced Time-Series Forecasting
**Models**: LSTM Neural Network + Prophet with ensemble forecasting

**Forecasting Capabilities**:
- **LSTM**: Deep learning model for complex temporal patterns
- **Prophet**: Facebook's time-series forecasting for trend and seasonality
- **Ensemble**: Weighted combination of both models for robust predictions
- **Multi-period**: Forecasts up to 30 days ahead with confidence intervals

**Forecast Endpoint**: `/api/ml/forecast-advanced`
```bash
curl http://localhost:8000/api/ml/forecast-advanced?days=7
```

**Response Structure**:
```json
{
  "lstm_forecast": [23.5, 24.1, 22.8, ...],
  "prophet_forecast": [24.2, 23.9, 23.1, ...],
  "ensemble_forecast": [23.9, 24.0, 22.9, ...],
  "forecast_periods": 7,
  "model_info": {
    "lstm_trained": true,
    "prophet_trained": true
  }
}
```

### 6.3 AI-Powered Strategic Insights 
**Powered by**: Groq API (Llama 3.3 70B) - Free tier available

**AI Capabilities**:
- **Strategic Analysis**: Executive summaries and root cause analysis
- **Maintenance Planning**: 7-day predictive maintenance schedules
- **Quality Insights**: Defect analysis and improvement recommendations
- **Risk Assessment**: Financial impact and timeline for action
- **Actionable Recommendations**: Priority-ordered action items

**AI Endpoints**:
```bash
# Comprehensive strategic insights
curl http://localhost:8000/api/ai/insights?days=7

# AI-generated maintenance plan
curl http://localhost:8000/api/ai/maintenance-plan?days=7
```

**AI Response Structure**:
```json
{
  "strategic_insights": {
    "full_analysis": "...",
    "sections": {
      "executive_summary": "...",
      "root_cause_analysis": "...",
      "recommendations": "..."
    }
  },
  "maintenance_plan": "...",
  "quality_insights": "...",
  "ml_analysis": {...}
}
```

### 6.4 Advanced Visualizations
The ML and forecasting engines generate comprehensive visualizations:
- **Feature Importance**: Top factors affecting downtime
- **Model Accuracy**: Actual vs predicted scatter plot
- **Anomaly Detection**: Timeline with flagged anomalies
- **Downtime Forecast**: Multi-period prediction with confidence bands
- **LSTM/Prophet Comparison**: Side-by-side model performance

All visualizations are automatically included in PDF reports.

## 7. Multi-Tenant Architecture

### 7.1 Overview
Support for multiple isolated plants/factories with data segregation and cross-tenant analytics.

**Key Features**:
- **Data Isolation**: Separate production data per tenant with PostgreSQL schemas
- **Machine Registry**: Track machines per plant with capacity and metadata
- **User Access Control**: Grant users access to specific tenants with roles
- **Corporate Overview**: Aggregated cross-tenant analytics and reporting
- **Legacy Migration**: Tools to migrate existing data to tenant structure

### 7.2 Tenant Management Endpoints

**Create Tenant**:
```bash
curl -X POST http://localhost:8000/api/tenants/create \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "plant_chicago",
    "tenant_name": "Chicago Manufacturing Plant",
    "plant_location": "Chicago, IL, USA",
    "timezone": "America/Chicago"
  }'
```

**List Tenants**:
```bash
curl http://localhost:8000/api/tenants/list?active_only=true
```

**Register Machine**:
```bash
curl -X POST http://localhost:8000/api/tenants/plant_chicago/machines/register \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M001",
    "machine_name": "Assembly Line 1",
    "machine_type": "Assembly",
    "capacity": 500
  }'
```

**Get Tenant KPIs**:
```bash
curl http://localhost:8000/api/tenants/plant_chicago/kpis?days=7
```

**Corporate Summary** (all tenants):
```bash
curl http://localhost:8000/api/corporate/summary?days=7
```

### 7.3 Tenant Data Submission
```bash
curl -X POST http://localhost:8000/api/tenants/plant_chicago/production/submit \
  -H "Content-Type: application/json" \
  -d '[{
    "production_date": "2026-02-07",
    "machine_id": "M001",
    "units_produced": 450,
    "defective_units": 5,
    "downtime_min": 15,
    "shift": "Day"
  }]'
```

### 7.4 User Access Management
```bash
# Grant user access
curl -X POST http://localhost:8000/api/tenants/plant_chicago/users/grant-access \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "john_doe",
    "user_name": "John Doe",
    "user_email": "john@company.com",
    "role": "manager"
  }'

# Get user's tenants
curl http://localhost:8000/api/users/john_doe/tenants
```

## 8. API Reference

### 8.1 REST Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/api/health` | GET | System health check |
| `/api/production` | GET | Production data with filters |
| `/api/kpis` | GET | Calculated KPIs |
| `/api/machines` | GET | Machine statistics |
| `/api/machines/live` | GET | Real-time OPC-UA data |
| `/api/ml/forecast` | GET | Ensemble ML downtime prediction |
| `/api/ml/forecast-advanced` | GET | LSTM/Prophet time-series forecasting |
| `/api/ai/insights` | GET | AI strategic analysis |
| `/api/ai/maintenance-plan` | GET | AI maintenance schedule |
| `/api/tenants/list` | GET | List all tenants |
| `/api/tenants/create` | POST | Create new tenant |
| `/api/tenants/{id}/kpis` | GET | Tenant-specific KPIs |
| `/api/tenants/{id}/production` | GET | Tenant production data |
| `/api/tenants/{id}/production/submit` | POST | Submit tenant production data |
| `/api/corporate/summary` | GET | Cross-tenant aggregated summary |
| `/api/reports/latest` | GET | Download latest PDF |
| `/api/websocket/status` | GET | WebSocket connection status |

### 8.2 Query Parameters
* `days`: Number of days for historical data (default: 7)
* `machine_id`: Filter by specific machine (optional)

## 9. Automated Reporting

### 9.1 Schedule
* **Daily Reports**: 08:00 AM automatic generation
* **Immediate**: On container startup
* **Manual**: `docker exec mfg_reporting_pipeline python main.py`

### 9.2 Report Contents
* Performance summary KPIs
* Ensemble ML Predicted Downtime with confidence
* LSTM/Prophet Time-Series Forecasts (7-30 days)
* AI Strategic Insights (full analysis)
* AI Maintenance Recommendations
* AI Quality Improvement Plan
* Production by shift charts
* Downtime trend analysis
* Machine comparison charts
* Advanced ML visualizations (4 charts)
* Forecasting visualizations with confidence intervals
* Anomaly detection results

### 9.3 Email Distribution
PDF automatically emailed if SMTP configured in `.env`

## 10. System Management

### 10.1 Commands
```bash
# Stop all services
docker compose down

# Restart specific service
docker compose restart api-server

# View logs
docker logs -f mfg_api_server
docker logs -f mfg_plc_sim

# Manual report generation (with AI and forecasting)
docker exec mfg_reporting_pipeline python main.py

# Check AI insights file
docker exec mfg_reporting_pipeline cat reports/ai_insights.txt

# Test multi-tenant features
curl http://localhost:8000/api/tenants/list
curl http://localhost:8000/api/corporate/summary

# Reset database (CAUTION)
docker compose down -v
```

### 10.2 Monitoring
```bash
# Check service health
docker ps

# API health
curl http://localhost:8000/api/health

# WebSocket status
curl http://localhost:8000/api/websocket/status

# Test ML prediction
curl http://localhost:8000/api/ml/forecast

# Test advanced forecasting (requires TensorFlow and Prophet)
curl http://localhost:8000/api/ml/forecast-advanced

# Test AI insights (requires Groq API key)
curl http://localhost:8000/api/ai/insights

# Test tenant features
curl http://localhost:8000/api/tenants/list
```

## 11. Troubleshooting

### 11.1 Common Issues

**Dashboard Not Loading**
```bash
docker ps | grep dashboard
docker logs mfg_dashboard
curl -I http://localhost:8080
```

**WebSocket Issues**
```bash
docker logs mfg_api_server | grep -i websocket
curl http://localhost:8000/api/websocket/status
# Clear browser cache (Ctrl+Shift+Delete)
```

**ML/AI/Forecasting Features Not Working**
```bash
# Check if advanced ML loaded
docker logs mfg_api_server | grep "Advanced ML"

# Check forecasting availability
docker logs mfg_reporting_pipeline | grep "forecasting"

# Verify dependencies
docker exec mfg_api_server pip list | grep -E "xgboost|tensorflow|prophet"

# Check AI API key
docker exec mfg_api_server printenv | grep GROQ
```

**Multi-Tenant Features Not Working**
```bash
# Check tenant manager initialization
docker logs mfg_api_server | grep -i tenant

# Verify database tables
docker exec mfg_postgres psql -U mfg_user -d manufacturing \
  -c "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE 'tenant%';"
```

**Database Connection Failed**
```bash
docker ps | grep postgres
docker compose restart db
```

**Insufficient Data for ML**
```bash
# ML requires 5+ records
docker logs mfg_reporting_pipeline | grep "Insufficient data"
```

### 11.2 Complete Reset
```bash
docker compose down -v --rmi all
docker compose up -d --build
```

## 12. Performance Tuning

**WebSocket Update Frequency** (`api_server.py`):
```python
await asyncio.sleep(5)  # Adjust to 10 for less frequent updates
```

**ML Training Data Window** (API call):
```bash
curl http://localhost:8000/api/ml/forecast?days=60  # More historical data
```

**Forecasting Parameters** (API call):
```bash
curl http://localhost:8000/api/ml/forecast-advanced?days=30  # Longer forecast horizon
```

**AI Response Length** (`ai_insights.py`):
```python
"max_tokens": 2000  # Increase for more detailed insights
```

**Database Connection Pool** (`api_server.py`):
```python
engine = create_engine(..., pool_size=10, max_overflow=20)
```

**Tenant Query Optimization** (`tenant_manager.py`):
```python
# Use indexed queries and limit result sets
```

## 13. Security Considerations

**Production Deployment:**
- Use reverse proxy (nginx/Caddy) with TLS for ports 8000/8080
- Strong passwords: `DB_PASSWORD=$(openssl rand -base64 32)`
- Keep PostgreSQL (5432) and OPC-UA (4840) internal
- Use Docker secrets instead of `.env` for sensitive data
- Rotate Groq API keys regularly
- Implement rate limiting on AI endpoints
- Validate tenant IDs in all multi-tenant endpoints
- Use row-level security for tenant data isolation
- Audit log all cross-tenant queries

## 14. Future Enhancements

### Implemented 
- [x] OPC-UA integration with live PLCs
- [x] PostgreSQL persistence
- [x] Real-time WebSocket dashboard
- [x] RESTful API
- [x] Ensemble ML predictive maintenance (RF + XGBoost)
- [x] Anomaly detection (Isolation Forest)
- [x] AI-powered strategic insights (Groq API)
- [x] AI maintenance planning
- [x] AI quality analysis
- [x] LSTM/Prophet time-series forecasting
- [x] Multi-tenant data isolation
- [x] Tenant management API
- [x] Corporate cross-tenant analytics

### Planned 
- [ ] Grafana dashboards with tenant filtering

### Scalability Roadmap
- Horizontal scaling for multiple plants with tenant sharding
- Redis caching for KPIs, ML predictions, and tenant metadata
- Apache Kafka for event streaming with tenant partitioning
- TimescaleDB for time-series optimization per tenant
- Model versioning and A/B testing per tenant
- Distributed forecasting for high-volume tenants

## 15. Architecture Diagram

```
┌──────────────────┐ Port 4840 (OPC-UA)
│ PLC Simulator    │
│ (3 Machines)     │
└────────┬─────────┘
         │
    ┌────▼─────────────────┐
    │ Docker Network       │
    │ (mfg_network)        │
    └────┬─────────────────┘
         │
    ┌────▼─────────┐   ┌─────────────────────┐
    │ PostgreSQL   │◄──┤ API Server          │ Port 8000
    │ Port 5432    │   │ FastAPI             │ (REST + WS)
    │ Multi-Tenant │   │ + ML Engine         │ (ML + AI)
    │ Schemas      │   │ + AI Engine         │ + Multi-Tenant
    └──────────────┘   │ + Forecasting       │ + Forecasting
                       │ + Tenant Manager    │
                       └──────┬──────────────┘
                              │
                       ┌──────▼──────────┐
                       │ Dashboard       │ Port 8080
                       │ Nginx+HTML      │ (Live Charts)
                       └─────────────────┘
                              │
                       ┌──────▼──────────┐
                       │ Groq API        │ (External)
                       │ Llama 3.3 70B   │
                       └─────────────────┘
```

## 16. Getting Started Without AI (Optional)

If you don't have a Groq API key, the system will work perfectly without AI features:
- All ML predictions still function (ensemble models, anomaly detection)
- Time-series forecasting available (LSTM/Prophet)
- Multi-tenant features fully functional
- Basic rule-based insights are generated as fallback
- Dashboard and reporting continue normally

To add AI later, simply add `GROQ_API_KEY` to `.env` and restart:
```bash
docker compose restart api-server reporting-app
```

For forecasting without TensorFlow/Prophet (lighter deployment):
- Basic ensemble ML still works
- Simple linear forecasting used as fallback
- All other features remain available
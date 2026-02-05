# Automated Manufacturing Intelligence System (AMIS)

## 1. Project Overview
An enterprise-grade manufacturing analytics platform featuring real-time data collection, AI-powered predictive maintenance, and automated reporting with strategic insights.

**Key Features:**
- **Real-time Data Collection**: Live OPC-UA industrial PLC data streaming
- **Advanced ML Engine**: Ensemble models (Random Forest + XGBoost) for predictive maintenance
- **AI Strategic Insights**: Groq-powered analysis for actionable recommendations
- **Live Dashboard**: WebSocket-powered analytics with interactive visualizations
- **RESTful API**: Comprehensive REST API with production data, KPIs, ML, and AI endpoints
- **Database Persistence**: PostgreSQL with 7-day rolling historical data
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
* **API Server** (`src/api_server.py`): FastAPI with REST, WebSocket, ML, and AI forecast endpoints
* **Advanced ML Engine** (`src/ml_engine_advanced.py`): **NEW** - Ensemble models with anomaly detection
* **AI Insights** (`src/ai_insights.py`): **NEW** - Groq-powered strategic analysis
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
│   ├── ml_engine_advanced.py   # Advanced ML engine
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
* **AI**: Groq API (Llama 3.3 70B) for strategic insights
* **Visualizations**: Chart.js (web), Matplotlib, Seaborn (PDF)
* **Reporting**: FPDF, XlsxWriter
* **Industrial Protocol**: asyncua (OPC-UA)
* **Database**: PostgreSQL 15 with SQLAlchemy
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

# Advanced ML forecast (NEW)
curl http://localhost:8000/api/ml/forecast?days=30

# AI strategic insights (NEW)
curl http://localhost:8000/api/ai/insights?days=7

# AI maintenance plan (NEW)
curl http://localhost:8000/api/ai/maintenance-plan?days=7

# Get live machine data
curl http://localhost:8000/api/machines/live

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

### 6.2 AI-Powered Strategic Insights 
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

### 6.3 Advanced Visualizations
The ML engine generates comprehensive visualizations (`chart_ml_advanced.png`):
- **Feature Importance**: Top factors affecting downtime
- **Model Accuracy**: Actual vs predicted scatter plot
- **Anomaly Detection**: Timeline with flagged anomalies
- **Downtime Forecast**: 5-period prediction with confidence bands

All visualizations are automatically included in PDF reports.

## 7. API Reference

### 7.1 REST Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/api/health` | GET | System health check |
| `/api/production` | GET | Production data with filters |
| `/api/kpis` | GET | Calculated KPIs |
| `/api/machines` | GET | Machine statistics |
| `/api/machines/live` | GET | Real-time OPC-UA data |
| `/api/ml/forecast` | GET | **Ensemble ML downtime prediction** |
| `/api/ai/insights` | GET | **AI strategic analysis**  |
| `/api/ai/maintenance-plan` | GET | **AI maintenance schedule**  |
| `/api/reports/latest` | GET | Download latest PDF |
| `/api/websocket/status` | GET | WebSocket connection status |

### 7.2 Query Parameters
* `days`: Number of days for historical data (default: 7)
* `machine_id`: Filter by specific machine (optional)

## 8. Automated Reporting

### 8.1 Schedule
* **Daily Reports**: 08:00 AM automatic generation
* **Immediate**: On container startup
* **Manual**: `docker exec mfg_reporting_pipeline python main.py`

### 8.2 Report Contents
* Performance summary KPIs
* **Ensemble ML Predicted Downtime** with confidence
* **AI Strategic Insights** (full analysis)
* **AI Maintenance Recommendations**
* **AI Quality Improvement Plan**
* Production by shift charts
* Downtime trend analysis
* Machine comparison charts
* **Advanced ML visualizations** (4 charts)
* **Anomaly detection results**

### 8.3 Email Distribution
PDF automatically emailed if SMTP configured in `.env`

## 9. System Management

### 9.1 Commands
```bash
# Stop all services
docker compose down

# Restart specific service
docker compose restart api-server

# View logs
docker logs -f mfg_api_server
docker logs -f mfg_plc_sim

# Manual report generation (with AI)
docker exec mfg_reporting_pipeline python main.py

# Check AI insights file
docker exec mfg_reporting_pipeline cat reports/ai_insights.txt

# Reset database (CAUTION)
docker compose down -v
```

### 9.2 Monitoring
```bash
# Check service health
docker ps

# API health
curl http://localhost:8000/api/health

# WebSocket status
curl http://localhost:8000/api/websocket/status

# Test ML prediction
curl http://localhost:8000/api/ml/forecast

# Test AI insights (requires Groq API key)
curl http://localhost:8000/api/ai/insights
```

## 10. Troubleshooting

### 10.1 Common Issues

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

**ML/AI Features Not Working**
```bash
# Check if advanced ML loaded
docker logs mfg_api_server | grep "Advanced ML"

# Verify dependencies
docker exec mfg_api_server pip list | grep -E "xgboost|aiohttp"

# Check AI API key
docker exec mfg_api_server printenv | grep GROQ
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

### 10.2 Complete Reset
```bash
docker compose down -v --rmi all
docker compose up -d --build
```

## 11. Performance Tuning

**WebSocket Update Frequency** (`api_server.py`):
```python
await asyncio.sleep(5)  # Adjust to 10 for less frequent updates
```

**ML Training Data Window** (API call):
```bash
curl http://localhost:8000/api/ml/forecast?days=60  # More historical data
```

**AI Response Length** (`ai_insights.py`):
```python
"max_tokens": 2000  # Increase for more detailed insights
```

**Database Connection Pool** (`api_server.py`):
```python
engine = create_engine(..., pool_size=10, max_overflow=20)
```

## 12. Security Considerations

**Production Deployment:**
- Use reverse proxy (nginx/Caddy) with TLS for ports 8000/8080
- Strong passwords: `DB_PASSWORD=$(openssl rand -base64 32)`
- Keep PostgreSQL (5432) and OPC-UA (4840) internal
- Use Docker secrets instead of `.env` for sensitive data
- Rotate Groq API keys regularly
- Implement rate limiting on AI endpoints

## 13. Future Enhancements

### Implemented 
- [x] OPC-UA integration with live PLCs
- [x] PostgreSQL persistence
- [x] Real-time WebSocket dashboard
- [x] RESTful API
- [x] **Ensemble ML predictive maintenance (RF + XGBoost)**
- [x] **Anomaly detection (Isolation Forest)**
- [x] **AI-powered strategic insights (Groq API)**
- [x] **AI maintenance planning**
- [x] **AI quality analysis**

### Planned 
- [ ] LSTM/Prophet for time-series forecasting
- [ ] Multi-plant support with tenant isolation
- [ ] Grafana dashboards
- [ ] Alerting system (Slack/Teams)
- [ ] Role-based access control (RBAC)
- [ ] Kubernetes deployment
- [ ] Custom AI model fine-tuning
- [ ] Real-time anomaly alerts

### Scalability Roadmap
- Horizontal scaling for multiple plants
- Redis caching for KPIs and ML predictions
- Apache Kafka for event streaming
- TimescaleDB for time-series optimization
- Model versioning and A/B testing

## 14. Architecture Diagram

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
    ┌────▼─────────┐   ┌─────────────────┐
    │ PostgreSQL   │◄──┤ API Server      │ Port 8000
    │ Port 5432    │   │ FastAPI         │ (REST + WS)
    └──────────────┘   │ + ML Engine     │ (ML + AI)
                       │ + AI Engine     │
                       └──────┬──────────┘
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

## 15. Getting Started Without AI (Optional)

If you don't have a Groq API key, the system will work perfectly without AI features:
- All ML predictions still function (ensemble models, anomaly detection)
- Basic rule-based insights are generated as fallback
- Dashboard and reporting continue normally

To add AI later, simply add `GROQ_API_KEY` to `.env` and restart:
```bash
docker compose restart api-server reporting-app
```

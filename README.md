# Automated Manufacturing Report Generation System (AMRS)

## 1. Project Overview
An enterprise-grade automation suite for manufacturing analytics featuring real-time data collection, predictive maintenance with machine learning, and automated reporting.

**Key Features:**
- **Real-time Data Collection**: Live OPC-UA industrial PLC data streaming
- **ML-Powered Predictions**: Linear regression model for predictive maintenance and downtime forecasting
- **Live Dashboard**: WebSocket-powered analytics with interactive charts
- **RESTful API**: Full REST API with production data, KPIs, and ML endpoints
- **Database Persistence**: PostgreSQL with 7-day rolling historical data
- **Automated Reporting**: Scheduled PDF generation with embedded ML insights and email distribution
- **Containerized Deployment**: Docker Compose orchestration with health checks

## 2. System Architecture

### 2.1 Core Services
* **PLC Simulator** (`plc-simulator`): Simulates 3 manufacturing machines via OPC-UA protocol
* **PostgreSQL Database** (`db`): Persistent storage for production history
* **API Server** (`api-server`): FastAPI with REST + WebSocket streaming + ML endpoints
* **Dashboard** (`dashboard`): Nginx-served interactive web dashboard with live updates
* **Reporting Pipeline** (`reporting-app`): Automated data collection, ML analysis, and report generation

### 2.2 Application Modules
* **Data Ingestion** (`src/ingestion.py`): OPC-UA client with retry logic
* **API Server** (`src/api_server.py`): FastAPI with REST, WebSocket, and ML forecast endpoints
* **ML Engine** (`src/ml_engine.py`): **NEW** - Linear regression model for downtime prediction
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
│   └── chart_ml_forecast.png   # ML prediction visualization
├── src/
│   ├── opcua_simulator.py      # PLC simulator
│   ├── ingestion.py            # OPC-UA + PostgreSQL
│   ├── api_server.py           # FastAPI REST + WebSocket + ML
│   ├── ml_engine.py            # ML predictive engine
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
* **Data**: Pandas, NumPy, scikit-learn
* **ML**: Linear Regression for predictive maintenance
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

# ML forecast endpoint (NEW)
curl http://localhost:8000/api/ml/forecast?days=30

# Get live machine data
curl http://localhost:8000/api/machines/live

# Download latest PDF
curl http://localhost:8000/api/reports/latest --output report.pdf
```

**WebSocket**: ws://localhost:8000/ws
- Real-time data streaming every 5 seconds

## 6. Machine Learning Features 

### 6.1 ML Endpoint
**NEW**: `/api/ml/forecast` - Predictive maintenance analysis

```bash
curl http://localhost:8000/api/ml/forecast?days=30
```

**Response Example:**
```json
{
  "predicted_downtime_next_shift": 25.3,
  "model_confidence_score": 0.87,
  "risk_assessment": "Medium",
  "message": "Prediction based on linear regression of units vs defects."
}
```

### 6.2 ML Model Details
* **Algorithm**: Linear Regression (scikit-learn)
* **Features**: Units Produced, Defective Units
* **Target**: Downtime (minutes)
* **Training**: Automated on-the-fly with recent data
* **Output**: Next shift downtime prediction + confidence score

### 6.3 Risk Assessment
* **Low Risk**: Predicted downtime ≤ 20 minutes
* **Medium Risk**: Predicted downtime 20-40 minutes
* **High Risk**: Predicted downtime > 40 minutes

### 6.4 ML Visualization
The ML engine generates `chart_ml_forecast.png` showing:
- Regression line of defects vs downtime
- Model accuracy (R² score)
- Automatically included in PDF reports

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
| `/api/ml/forecast` | GET | **ML downtime prediction** |
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
* **ML Predicted Downtime** (NEW)
* **ML Model Confidence Score** (NEW)
* Production by shift charts
* Downtime trend analysis
* Machine comparison charts
* **ML forecast visualization** (NEW)

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

# Manual report generation
docker exec mfg_reporting_pipeline python main.py

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

# ML prediction test
curl http://localhost:8000/api/ml/forecast
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

**Database Connection Failed**
```bash
docker ps | grep postgres
docker compose restart db
```

**ML Forecast Returns Zero**
```bash
# Check if sufficient data exists (needs 5+ records)
docker logs mfg_reporting_pipeline | grep "ML"
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

## 13. Future Enhancements

### Implemented 
- [x] OPC-UA integration with live PLCs
- [x] PostgreSQL persistence
- [x] Real-time WebSocket dashboard
- [x] RESTful API
- [x] **ML predictive maintenance**

### Planned 
- [ ] Advanced ML models (LSTM, Prophet) for better predictions
- [ ] Multi-plant support with tenant isolation
- [ ] Grafana dashboards
- [ ] Alerting system (Slack/Teams)
- [ ] Role-based access control (RBAC)
- [ ] Kubernetes deployment

### Scalability Roadmap
- Horizontal scaling for multiple plants
- Redis caching for KPIs
- Apache Kafka for event streaming
- TimescaleDB for time-series optimization

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
    ┌────▼────────┐   ┌─────────────┐
    │ PostgreSQL  │◄──┤ API Server  │ Port 8000
    │ Port 5432   │   │ FastAPI     │ (REST + WS + ML)
    └─────────────┘   └──────┬──────┘
                             │
                      ┌──────▼──────┐
                      │ Dashboard   │ Port 8080
                      │ Nginx+HTML  │ (Live Charts)
                      └─────────────┘
```
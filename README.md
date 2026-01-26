# Automated Manufacturing Report Generation System (AMRS)

## 1. Project Overview
The Automated Manufacturing Report Generation System (AMRS) is an enterprise-grade automation suite designed to streamline production data management. The system handles the end-to-end lifecycle of manufacturing reporting, from live OPC-UA data ingestion and PostgreSQL archival to KPI calculation, data visualization, and automated distribution via SMTP.

**Key Features:**
- **Real-time Data Collection**: Live streaming from OPC-UA industrial PLC simulators
- **Live Dashboard**: Real-time WebSocket-powered analytics dashboard with interactive charts
- **RESTful API**: Full REST API with endpoints for production data, KPIs, and machine statistics
- **WebSocket Streaming**: Real-time data updates every 5 seconds to connected clients
- **Database Persistence**: PostgreSQL storage with 7-day rolling historical data
- **Data Validation**: Automated integrity checks with audit logging
- **Advanced Analytics**: KPI calculation with visual dashboards
- **Automated Reporting**: Scheduled PDF generation with email distribution
- **Containerized Deployment**: Full Docker Compose orchestration with health checks

## 2. System Architecture
The project follows a microservices architecture with four main components:

### 2.1 Core Services
* **PLC Simulator Service** (`plc-simulator`): Simulates 3 manufacturing machines using OPC-UA protocol, generating real-time production metrics every 2 seconds
* **PostgreSQL Database** (`db`): Persistent storage for production history with automatic table initialization
* **API Server** (`api-server`): FastAPI-based REST API with WebSocket support for real-time data streaming
* **Dashboard** (`dashboard`): Nginx-served interactive web dashboard with live charts and KPI cards
* **Reporting Pipeline** (`reporting-app`): Main application orchestrating data collection, validation, KPI calculation, and report generation

### 2.2 Application Modules
* **Data Ingestion Module** (`src/ingestion.py`): Fetches live OPC-UA data with retry logic and merges with historical PostgreSQL records
* **API Server** (`src/api_server.py`): FastAPI application with REST endpoints and WebSocket streaming capabilities
* **Database Initialization** (`init_db.py`): Automatically creates database schema on first startup
* **Validation Module** (`src/validation.py`): Performs data integrity checks, handles missing values, and logs anomalies
* **KPI Engine** (`src/kpi_engine.py`): Calculates critical metrics including Yield Percentage, Defect Rates, and Average Downtime
* **Report Generator** (`src/report_generator.py`): Produces styled PDF summaries with embedded analytical charts and optional email delivery
* **Scheduler** (`src/scheduler.py`): Automated daily report generation at 08:00 AM with database initialization

## 3. Directory Structure
```text
report_automation/
│
├── dashboard/
│   └── dashboard.html      # Real-time web dashboard with charts
│
├── data/
│   ├── raw/                # Input CSV/Excel files (fallback)
│   └── processed/          # Archived historical data
│
├── reports/
│   ├── excel/              # Generated spreadsheets
│   ├── pdf/                # Generated visual reports
│   ├── chart_prod.png      # Production by shift visualization
│   ├── chart_down_trend.png # Downtime trend analysis
│   ├── chart_down_var.png  # Machine downtime comparison
│   └── validation_log.txt  # Data integrity audit trail
│
├── src/
│   ├── opcua_simulator.py  # PLC simulation with 3 machines
│   ├── ingestion.py        # OPC-UA client + PostgreSQL integration
│   ├── api_server.py       # FastAPI REST + WebSocket server
│   ├── validation.py       # Cleaning and error handling
│   ├── kpi_engine.py       # Mathematical calculations
│   ├── report_generator.py # Visuals and document creation
│   └── scheduler.py        # Time-based triggers + DB init
│
├── .env                    # Environment variables and credentials
├── config.yaml             # System path configurations
├── docker-compose.yml      # Multi-container orchestration
├── Dockerfile              # Container build instructions
├── init_db.py              # PostgreSQL schema initialization
├── main.py                 # Primary orchestrator
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

## 4. Technical Specifications

### 4.1 Technology Stack
* **Core Language**: Python 3.9
* **API Framework**: FastAPI with Uvicorn ASGI server
* **Real-time Communication**: WebSockets for live data streaming
* **Data Processing**: Pandas, NumPy
* **Visualizations**: Chart.js (frontend), Matplotlib, Seaborn (PDF reports)
* **Reporting**: FPDF (PDF), XlsxWriter (Excel), OpenPyXL
* **Industrial Protocol**: asyncua (OPC-UA client/server)
* **Database**: PostgreSQL 15 with SQLAlchemy ORM and psycopg adapter
* **Scheduling**: schedule library for automated daily runs
* **Web Server**: Nginx (for dashboard hosting)
* **Deployment**: Docker Compose with health checks and volume persistence

### 4.2 Network Architecture
```
┌──────────────────────┐ Port 4840 (OPC-UA)
│   PLC Simulator      │
│   (3 Machines)       │
└──────────┬───────────┘
           │
    ┌──────▼───────────────────────┐
    │   Docker Network             │
    │   (mfg_network)              │
    └──────┬───────────────────────┘
           │
    ┌──────▼──────────┐   ┌────────────────┐
    │  PostgreSQL DB  │   │ API Server     │ Port 8000 (REST + WS)
    │  Port 5432      │◄──┤ (FastAPI)      │
    └─────────────────┘   └───────┬────────┘
                                  │
                          ┌───────▼────────┐
                          │ Dashboard      │ Port 8080 (HTTP)
                          │ (Nginx + HTML) │
                          └────────────────┘
```

## 5. Setup and Deployment

### 5.1 Prerequisites
* **Docker Engine** 20.10+ installed and running
* **Docker Compose** V2 installed
* **WSL2 environment** (for Windows users)
* Minimum 2GB RAM and 5GB disk space
* Modern web browser (Chrome, Firefox, Edge, Safari)

### 5.2 Environment Configuration
Create a `.env` file in the root directory:

```bash
# Database Configuration
DB_HOST=db
DB_PORT=5432
DB_NAME=manufacturing
DB_USER=mfg_user
DB_PASSWORD=mfg_pass123

# OPC-UA Configuration
OPCUA_SERVER_URL=opc.tcp://plc-simulator:4840/freeopcua/server/

# Email Configuration (Optional)
EMAIL_SENDER=your-email@gmail.com
EMAIL_RECEIVER=manager@factory.com
EMAIL_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

**Note**: For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833) instead of your regular password.

### 5.3 Quick Start

#### Option A: Full System Deployment (Recommended)
```bash
# Clone or navigate to project directory
cd ~/report_automation

# Build and start all services
docker compose up -d --build

# Watch initialization logs
docker logs -f mfg_api_server

# Verify all services are healthy
docker ps
```

Expected output:
```
INFO: Starting FastAPI application...
INFO: Starting live data stream background task...
INFO: Application startup complete.
INFO: Successfully connected to OPC-UA server and fetched 3 machine records
```

#### Option B: Access the Live Dashboard
```bash
# Open your browser and navigate to:
http://localhost:8080

# The dashboard will show:
# - Real-time KPI cards (Total Units, Defects, Yield Rate, Avg Downtime)
# - Live production trend chart
# - Machine performance comparison chart
# - Live machine status table (updates every 5 seconds)
```

#### Option C: Test the API Endpoints
```bash
# Get system health
curl http://localhost:8000/api/health

# Get WebSocket status
curl http://localhost:8000/api/websocket/status

# Get KPIs for last 7 days
curl http://localhost:8000/api/kpis?days=7

# Get production data
curl http://localhost:8000/api/production?days=7

# Get machine statistics
curl http://localhost:8000/api/machines?days=7

# Get live machine data
curl http://localhost:8000/api/machines/live

# Download latest PDF report
curl http://localhost:8000/api/reports/latest --output report.pdf
```

#### Option D: Manual Report Generation
```bash
# Generate a report on-demand
docker exec mfg_reporting_pipeline python main.py

# View the generated PDF
ls -lh reports/pdf/Report.pdf
```

### 5.4 System Management

#### Stop All Services
```bash
docker compose down
```

#### Restart Specific Service
```bash
# Restart API server
docker compose restart api-server

# Restart dashboard
docker compose restart dashboard

# Restart PLC simulator
docker compose restart plc-simulator
```

#### View Real-time Logs
```bash
# All services
docker compose logs -f

# Specific service
docker logs -f mfg_plc_sim          # PLC simulator
docker logs -f mfg_postgres         # Database
docker logs -f mfg_api_server       # API server
docker logs -f mfg_reporting_pipeline # Reporting app
docker logs -f mfg_dashboard        # Dashboard web server
```

#### Check Service Health
```bash
docker ps
```

All containers should show `healthy` or `running` status:
```
CONTAINER ID   STATUS
mfg_plc_sim    Up 5 minutes (healthy)
mfg_postgres   Up 5 minutes (healthy)
mfg_api_server Up 5 minutes
mfg_dashboard  Up 5 minutes
mfg_reporting  Up 5 minutes
```

#### Database Operations
```bash
# Connect to PostgreSQL
docker exec -it mfg_postgres psql -U mfg_user -d manufacturing

# Inside psql:
\dt                          # List tables
SELECT COUNT(*) FROM live_production;  # Count records
SELECT * FROM live_production ORDER BY created_at DESC LIMIT 10;  # Recent records
\q                           # Exit
```

### 5.5 Data Flow Verification

1. **Check PLC Simulator** (generates data every 2 seconds):
```bash
docker logs mfg_plc_sim --tail 10
```

2. **Monitor API Server** (fetches OPC-UA data every 5 seconds):
```bash
docker logs mfg_api_server --tail 20
```

3. **Test WebSocket Connection**:
```bash
# Install wscat if needed: npm install -g wscat
wscat -c ws://localhost:8000/ws

# You should see live updates every 5 seconds:
# {"type":"live_update","timestamp":"...","data":[...]}
```

4. **Monitor Database Growth**:
```bash
watch -n 5 'docker exec mfg_postgres psql -U mfg_user -d manufacturing -c "SELECT COUNT(*) FROM live_production;"'
```

5. **Inspect Generated Reports**:
```bash
ls -lh reports/pdf/
cat reports/validation_log.txt
```

## 6. API Documentation

### 6.1 REST API Endpoints

The API server runs on **port 8000** and provides the following endpoints:

#### System Endpoints
- `GET /` - API information and available endpoints
- `GET /api/health` - Health check with database connectivity status
- `GET /api/websocket/status` - WebSocket connection status and active client count

#### Data Endpoints
- `GET /api/production?days=7` - Get production data (default: last 7 days)
- `GET /api/kpis?days=7` - Get calculated KPIs
- `GET /api/machines?days=7` - Get machine statistics
- `GET /api/machines/live` - Get current live data from OPC-UA
- `GET /api/reports/latest` - Download the latest PDF report

#### Write Endpoints
- `POST /api/production` - Create new production record (for integrations)

#### WebSocket Endpoint
- `WS /ws` - Real-time data streaming (updates every 5 seconds)

### 6.2 WebSocket Protocol

The WebSocket connection broadcasts live machine data every 5 seconds:

```json
{
  "type": "live_update",
  "timestamp": "2026-01-26T10:30:45.123456",
  "data": [
    {
      "production_date": "2026-01-26",
      "machine_id": "OPC-UA-M1",
      "units_produced": 342.0,
      "defective_units": 5.0,
      "downtime_min": 12.5,
      "shift": "Live-Stream"
    },
    ...
  ]
}
```

### 6.3 API Response Examples

**KPI Response:**
```json
{
  "total_units": 15420,
  "total_defects": 287,
  "yield_percentage": 98.14,
  "avg_downtime": 11.34,
  "report_date": "2026-01-26"
}
```

**Machine Stats Response:**
```json
[
  {
    "machine_id": "OPC-UA-M1",
    "units_produced": 5230.0,
    "defective_units": 95.0,
    "downtime_minutes": 342.5
  },
  ...
]
```

## 7. Dashboard Features

### 7.1 Real-time Dashboard (Port 8080)

The interactive dashboard provides:

**Live Status Indicator:**
- Connection status dot (green when connected, red when disconnected)
- Automatic reconnection on connection loss

**Interactive Controls:**
- Time range selector (Last 24 Hours, 7 Days, 30 Days)
- Manual refresh button
- PDF report download button

**KPI Cards (Auto-refresh):**
- Total Units Produced
- Total Defective Units with defect rate
- Yield Rate percentage
- Average Downtime per machine

**Real-time Charts:**
- Production Trend: Line chart showing daily production over selected period
- Machine Performance: Bar chart comparing units produced vs defects per machine

**Live Machine Status Table:**
- Real-time updates every 5 seconds via WebSocket
- Shows units produced, defects, and downtime for each machine
- Hover effects for better interactivity

### 7.2 Dashboard Technology
- **Chart.js 4.4.0**: Professional, animated charts
- **WebSocket**: Real-time data streaming
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional interface with smooth animations

## 8. Data Integrity and Auditing

### 8.1 Validation Pipeline
The system implements comprehensive data quality controls:

* **Missing Value Handling**: Automatically fills null values with zeros and logs corrections
* **Negative Value Detection**: Clips negative production/downtime values and logs violations
* **Logic Validation**: Ensures defective units never exceed total production
* **Audit Trail**: All corrections logged to `reports/validation_log.txt` with timestamps

### 8.2 Database Archival
* **Automatic Storage**: Every OPC-UA data fetch is archived to PostgreSQL with timestamp
* **7-Day Rolling Window**: Reports include both live and historical data from the past week
* **Persistent Volumes**: Data survives container restarts via Docker volumes
* **Transaction Safety**: SQLAlchemy ensures atomic database operations

### 8.3 Monitoring and Logging
```bash
# View validation results
cat reports/validation_log.txt

# Check database archive status
docker exec mfg_postgres psql -U mfg_user -d manufacturing -c \
  "SELECT production_date, machine_id, COUNT(*) as records 
   FROM live_production 
   GROUP BY production_date, machine_id 
   ORDER BY production_date DESC;"

# Monitor WebSocket connections
curl http://localhost:8000/api/websocket/status
```

## 9. Report Outputs

### 9.1 PDF Report Contents
- **Professional Header**: Branded title with dark blue background
- **KPI Summary Table**: 
  - Total Units Produced
  - Total Defects
  - Yield Percentage
  - Average Downtime
  - Report Generation Date
- **Production by Shift Chart**: Bar chart comparing machine performance across shifts
- **Downtime Trend Analysis**: Line graph showing temporal patterns with Gaussian distribution
- **Machine Comparison**: Bar chart highlighting downtime disparities

### 9.2 Sample Output
```bash
reports/
├── pdf/
│   └── Report.pdf              # Main deliverable
├── chart_prod.png              # Shift production visualization
├── chart_down_trend.png        # Downtime trend graph (randomized)
├── chart_down_var.png          # Machine comparison chart
└── validation_log.txt          # Data quality audit log
```

## 10. Scheduling and Automation

The system runs automatically via the built-in scheduler:

* **Daily Reports**: Generated at **08:00 AM** every day
* **Immediate Startup Run**: Generates a report on container start for verification
* **Email Distribution**: Automatically sends PDF to configured recipients (if SMTP enabled)
* **Retry Logic**: OPC-UA connections retry up to 5 times with exponential backoff
* **Real-time Streaming**: WebSocket broadcasts every 5 seconds when clients are connected

### Manual Trigger
```bash
docker exec mfg_reporting_pipeline python main.py
```

## 11. Troubleshooting

### 11.1 Common Issues

**Dashboard Not Loading**
```bash
# Check if dashboard container is running
docker ps | grep dashboard

# Check nginx logs
docker logs mfg_dashboard

# Verify dashboard file exists
docker exec mfg_dashboard ls -la /usr/share/nginx/html/

# Test dashboard directly
curl -I http://localhost:8080
```

**WebSocket Connection Issues**
```bash
# Check API server logs for WebSocket errors
docker logs mfg_api_server | grep -i websocket

# Verify WebSocket endpoint
curl http://localhost:8000/api/websocket/status

# Test WebSocket connection manually
wscat -c ws://localhost:8000/ws

# Common fix: Clear browser cache and reload dashboard
```

**"Object of type date is not JSON serializable" Error**
```bash
# This error was fixed in the latest version
# If you still see it, ensure you have the updated api_server.py

# Rebuild the API server
docker compose build api-server
docker compose up -d api-server

# Verify the fix
docker logs mfg_api_server | grep -i "successfully connected"
```

**Database Connection Failed**
```bash
# Check if database container is running
docker ps | grep postgres

# Verify credentials in .env file
cat .env | grep DB_

# Test database connectivity
docker exec mfg_api_server python -c "from sqlalchemy import create_engine; import os; engine = create_engine(f\"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}\"); engine.connect(); print('DB OK')"

# Restart database
docker compose restart db
```

**OPC-UA Connection Timeout**
```bash
# Verify PLC simulator is healthy
docker logs mfg_plc_sim | tail -20

# Check network connectivity
docker exec mfg_api_server ping plc-simulator -c 3

# Verify OPC-UA endpoint
docker exec mfg_api_server python -c "import asyncio; from asyncua import Client; asyncio.run((lambda: Client(url='opc.tcp://plc-simulator:4840/freeopcua/server/').connect())())"

# Restart PLC simulator
docker compose restart plc-simulator
```

**No PDF Generated**
```bash
# Check for errors in pipeline logs
docker logs mfg_reporting_pipeline

# Verify write permissions
docker exec mfg_reporting_pipeline ls -la reports/pdf/

# Run manually with verbose output
docker exec mfg_reporting_pipeline python main.py
```

**Charts Not Displaying**
```bash
# Check browser console for JavaScript errors (F12)
# Verify API endpoints are responding
curl http://localhost:8000/api/production?days=7
curl http://localhost:8000/api/machines?days=7

# Clear browser cache (Ctrl+Shift+Delete)
# Reload dashboard (Ctrl+F5)
```

### 11.2 Reset Everything
```bash
# Stop all containers
docker compose down

# Remove all data (CAUTION: deletes all historical records)
docker volume rm report_automation_postgres_data

# Remove all images (forces complete rebuild)
docker compose down --rmi all

# Fresh start
docker compose up -d --build
```

### 11.3 Debug Mode
```bash
# Run API server in debug mode with verbose logging
docker compose stop api-server
docker compose run --rm -p 8000:8000 api-server uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --log-level debug

# Run reporting pipeline in foreground
docker compose stop reporting-app
docker compose run --rm reporting-app python main.py
```

## 12. Performance Optimization

### 12.1 Tuning Parameters

**WebSocket Update Frequency** (in `api_server.py`):
```python
await asyncio.sleep(5)  # Change to 10 for less frequent updates
```

**OPC-UA Retry Configuration** (in `ingestion.py`):
```python
async def get_all_opcua_data(max_retries=5, retry_delay=2):
    # Increase max_retries for unreliable networks
    # Increase retry_delay for slower PLC response times
```

**Database Connection Pool** (in `api_server.py`):
```python
engine = create_engine(
    f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}",
    pool_size=10,  # Adjust based on concurrent API requests
    max_overflow=20
)
```

### 12.2 Scaling Recommendations

**For High-Frequency Data Collection:**
- Reduce WebSocket update interval from 5s to 1-2s
- Increase database connection pool size
- Add Redis caching for frequently accessed KPIs

**For Multiple Plants:**
- Deploy separate stacks per plant
- Use centralized PostgreSQL with tenant isolation
- Implement API gateway for aggregated dashboards

**For High Availability:**
- Use PostgreSQL replication (master-slave)
- Deploy multiple API server instances behind load balancer
- Implement health check endpoints for orchestrators

## 13. Security Considerations

### 13.1 Production Deployment

**Never expose these ports publicly without proper security:**
- Port 8000 (API): Use reverse proxy with TLS (nginx/Caddy)
- Port 8080 (Dashboard): Use authentication middleware
- Port 5432 (PostgreSQL): Keep internal, use VPN for remote access
- Port 4840 (OPC-UA): Implement OPC-UA security mode

**Environment Variables:**
```bash
# Use strong passwords in production
DB_PASSWORD=$(openssl rand -base64 32)

# Enable SMTP with app-specific passwords
EMAIL_PASSWORD=your-16-digit-app-password
```

**Docker Security:**
```bash
# Run containers as non-root user
# Add to Dockerfile: USER 1000:1000

# Use secrets instead of environment variables
# Migrate .env to Docker secrets
```

## 14. Future Enhancements

### 14.1 Implemented Features 
- [x] Integration with live PLC/SCADA systems via OPC-UA
- [x] PostgreSQL database for historical data persistence
- [x] Real-time dashboard using FastAPI with WebSocket updates
- [x] RESTful API for third-party integrations
- [x] Interactive charts with Chart.js

### 14.2 Planned Features 
- [ ] Predictive analytics for machine downtime using Machine Learning (LSTM/Prophet)
- [ ] Multi-plant support with tenant isolation
- [ ] Grafana dashboards for advanced monitoring
- [ ] Alerting system for anomaly detection (Slack/Teams integration)
- [ ] Role-based access control (RBAC) for dashboard and API access
- [ ] Export data to Excel/CSV from dashboard
- [ ] Historical data replay for training purposes

### 14.3 Scalability Roadmap
- Kubernetes deployment with Helm charts
- Horizontal scaling for multiple plant locations
- Redis caching for improved query performance
- Apache Kafka for event streaming from multiple PLCs
- Distributed tracing with OpenTelemetry
- TimescaleDB for time-series optimization
- GraphQL API for flexible data queries




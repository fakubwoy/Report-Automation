# Automated Manufacturing Report Generation System (AMRS)

## 1. Project Overview
The Automated Manufacturing Report Generation System (AMRS) is an enterprise-grade automation suite designed to streamline production data management. The system handles the end-to-end lifecycle of manufacturing reporting, from live OPC-UA data ingestion and PostgreSQL archival to KPI calculation, data visualization, and automated distribution via SMTP.

**Key Features:**
- **Real-time Data Collection**: Live streaming from OPC-UA industrial PLC simulators
- **Database Persistence**: PostgreSQL storage with 7-day rolling historical data
- **Data Validation**: Automated integrity checks with audit logging
- **Advanced Analytics**: KPI calculation with visual dashboards
- **Automated Reporting**: Scheduled PDF generation with email distribution
- **Containerized Deployment**: Full Docker Compose orchestration

## 2. System Architecture
The project follows a microservices architecture with three main components:

### 2.1 Core Services
* **PLC Simulator Service** (`plc-simulator`): Simulates 3 manufacturing machines using OPC-UA protocol, generating real-time production metrics
* **PostgreSQL Database** (`db`): Persistent storage for production history with automatic table initialization
* **Reporting Pipeline** (`reporting-app`): Main application orchestrating data collection, validation, KPI calculation, and report generation

### 2.2 Application Modules
* **Data Ingestion Module** (`src/ingestion.py`): Fetches live OPC-UA data with retry logic and merges with historical PostgreSQL records
* **Database Initialization** (`init_db.py`): Automatically creates database schema on first startup
* **Validation Module** (`src/validation.py`): Performs data integrity checks, handles missing values, and logs anomalies
* **KPI Engine** (`src/kpi_engine.py`): Calculates critical metrics including Yield Percentage, Defect Rates, and Average Downtime
* **Report Generator** (`src/report_generator.py`): Produces styled PDF summaries with embedded analytical charts and optional email delivery
* **Scheduler** (`src/scheduler.py`): Automated daily report generation at 08:00 AM with database initialization

## 3. Directory Structure
```text
report_automation/
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
* **Data Processing**: Pandas, NumPy
* **Visualizations**: Matplotlib, Seaborn
* **Reporting**: FPDF (PDF), XlsxWriter (Excel), OpenPyXL
* **Industrial Protocol**: asyncua (OPC-UA client/server)
* **Database**: PostgreSQL 15 with SQLAlchemy ORM and psycopg adapter
* **Scheduling**: schedule library for automated daily runs
* **Deployment**: Docker Compose with health checks and volume persistence

### 4.2 Network Architecture
```
┌─────────────────────┐
│   PLC Simulator     │ Port 4840 (OPC-UA)
│   (3 Machines)      │
└──────────┬──────────┘
           │
    ┌──────▼──────────────────────┐
    │   Docker Network            │
    │   (mfg_network)             │
    └──────┬──────────────────────┘
           │
    ┌──────▼──────────┐   ┌──────────────┐
    │  PostgreSQL DB  │   │ Reporting    │
    │  Port 5432      │◄──┤ Pipeline     │
    └─────────────────┘   └──────────────┘
```

## 5. Setup and Deployment

### 5.1 Prerequisites
* **Docker Engine** 20.10+ installed and running
* **Docker Compose** V2 installed
* **WSL2 environment** (for Windows users)
* Minimum 2GB RAM and 5GB disk space

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
docker logs -f mfg_reporting_pipeline

# Verify database
docker exec -it mfg_postgres psql -U mfg_user -d manufacturing -c "SELECT * FROM live_production ORDER BY created_at DESC LIMIT 5;"
```

Expected output:
```
INFO: Initializing database...
INFO: Database initialized successfully
INFO: Successfully connected to OPC-UA server and fetched 3 machine records
INFO: Archived 3 records to PostgreSQL.
INFO: Validating data integrity...
INFO: PDF report generated: reports/pdf/Report.pdf
INFO: Scheduler active. Waiting for next daily trigger (08:00 AM)...
```

#### Option B: Manual Report Generation
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

#### View Real-time Logs
```bash
# All services
docker compose logs -f

# Specific service
docker logs -f mfg_plc_sim          # PLC simulator
docker logs -f mfg_postgres         # Database
docker logs -f mfg_reporting_pipeline # Reporting app
```

#### Check Service Health
```bash
docker ps
```

All containers should show `healthy` status:
```
CONTAINER ID   STATUS
mfg_plc_sim    Up 5 minutes (healthy)
mfg_postgres   Up 5 minutes (healthy)
mfg_reporting  Up 5 minutes
```

#### Database Operations
```bash
# Connect to PostgreSQL
docker exec -it mfg_postgres psql -U mfg_user -d manufacturing

# Inside psql:
\dt                          # List tables
SELECT COUNT(*) FROM live_production;  # Count records
\q                           # Exit
```

### 5.5 Data Flow Verification

1. **Check PLC Simulator** (generates data every 2 seconds):
```bash
docker logs mfg_plc_sim --tail 10
```

2. **Monitor Database Growth**:
```bash
watch -n 5 'docker exec mfg_postgres psql -U mfg_user -d manufacturing -c "SELECT COUNT(*) FROM live_production;"'
```

3. **Inspect Generated Reports**:
```bash
ls -lh reports/pdf/
cat reports/validation_log.txt
```

## 6. Data Integrity and Auditing

### 6.1 Validation Pipeline
The system implements comprehensive data quality controls:

* **Missing Value Handling**: Automatically fills null values with zeros and logs corrections
* **Negative Value Detection**: Clips negative production/downtime values and logs violations
* **Logic Validation**: Ensures defective units never exceed total production
* **Audit Trail**: All corrections logged to `reports/validation_log.txt` with timestamps

### 6.2 Database Archival
* **Automatic Storage**: Every OPC-UA data fetch is archived to PostgreSQL with timestamp
* **7-Day Rolling Window**: Reports include both live and historical data from the past week
* **Persistent Volumes**: Data survives container restarts via Docker volumes
* **Transaction Safety**: SQLAlchemy ensures atomic database operations

### 6.3 Monitoring and Logging
```bash
# View validation results
cat reports/validation_log.txt

# Check database archive status
docker exec mfg_postgres psql -U mfg_user -d manufacturing -c \
  "SELECT production_date, machine_id, COUNT(*) as records 
   FROM live_production 
   GROUP BY production_date, machine_id 
   ORDER BY production_date DESC;"
```

## 7. Report Outputs

### 7.1 PDF Report Contents
- **Professional Header**: Branded title with dark blue background
- **KPI Summary Table**: 
  - Total Units Produced
  - Total Defects
  - Yield Percentage
  - Average Downtime
  - Report Generation Date
- **Production by Shift Chart**: Bar chart comparing machine performance across shifts
- **Downtime Trend Analysis**: Line graph showing temporal patterns
- **Machine Comparison**: Bar chart highlighting downtime disparities

### 7.2 Sample Output
```bash
reports/
├── pdf/
│   └── Report.pdf              # Main deliverable
├── chart_prod.png              # Shift production visualization
├── chart_down_trend.png        # Downtime trend graph
├── chart_down_var.png          # Machine comparison chart
└── validation_log.txt          # Data quality audit log
```

## 8. Scheduling and Automation

The system runs automatically via the built-in scheduler:

* **Daily Reports**: Generated at **08:00 AM** every day
* **Immediate Startup Run**: Generates a report on container start for verification
* **Email Distribution**: Automatically sends PDF to configured recipients (if SMTP enabled)
* **Retry Logic**: OPC-UA connections retry up to 5 times with exponential backoff

### Manual Trigger
```bash
docker exec mfg_reporting_pipeline python main.py
```

## 9. Troubleshooting

### 9.1 Common Issues

**Database Connection Failed**
```bash
# Check if database container is running
docker ps | grep postgres

# Verify credentials in .env file
cat .env | grep DB_

# Restart database
docker compose restart db
```

**OPC-UA Connection Timeout**
```bash
# Verify PLC simulator is healthy
docker logs mfg_plc_sim | tail -20

# Check network connectivity
docker exec mfg_reporting_pipeline ping plc-simulator -c 3

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

### 9.2 Reset Everything
```bash
# Stop all containers
docker compose down

# Remove all data (CAUTION: deletes all historical records)
docker volume rm report_automation_postgres_data

# Fresh start
docker compose up -d --build
```

## 10. Future Enhancements

### 10.1 Planned Features
- [x] ~~Integration with live PLC/SCADA systems via OPC-UA~~ ✅ **IMPLEMENTED**
- [x] ~~PostgreSQL database for historical data persistence~~ ✅ **IMPLEMENTED**
- [ ] Real-time dashboard using Flask or FastAPI with WebSocket updates
- [ ] Predictive analytics for machine downtime using Machine Learning (LSTM/Prophet)
- [ ] Multi-plant support with tenant isolation
- [ ] RESTful API for third-party integrations
- [ ] Grafana dashboards for real-time monitoring
- [ ] Alerting system for anomaly detection (Slack/Teams integration)
- [ ] Role-based access control (RBAC) for report access

### 10.2 Scalability Roadmap
- Kubernetes deployment with Helm charts
- Horizontal scaling for multiple plant locations
- Redis caching for improved query performance
- Apache Kafka for event streaming from multiple PLCs
- Distributed tracing with OpenTelemetry


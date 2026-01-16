# Automated Manufacturing Report Generation System (AMRS)

## 1. Project Overview
The Automated Manufacturing Report Generation System (AMRS) is an enterprise-grade automation suite designed to streamline production data management. The system handles the end-to-end lifecycle of manufacturing reporting, from raw data ingestion and integrity validation to KPI calculation, data visualization, and automated distribution via SMTP.

## 2. System Architecture
The project follows a modular architecture to ensure scalability and maintainability.

* **Data Ingestion Module**: Reads raw CSV or Excel files and standardizes data types.
* **Validation Module**: Performs data integrity checks, handles missing values, and logs anomalies.
* **KPI Engine**: Calculates critical metrics including Yield Percentage, Defect Rates, and Average Downtime.
* **Report Generator**: Produces formatted Excel workbooks and styled PDF summaries with embedded analytical charts.
* **Automation & Distribution**: Provides scheduling capabilities and automated email delivery to stakeholders.



## 3. Directory Structure
```text
report_automation/
│
├── data/
│   ├── raw/                # Input CSV/Excel files
│   └── processed/          # Archived historical data
│
├── reports/
│   ├── excel/              # Generated spreadsheets
│   └── pdf/                # Generated visual reports
│
├── src/
│   ├── ingestion.py        # Data loading logic
│   ├── validation.py       # Cleaning and error handling
│   ├── kpi_engine.py       # Mathematical calculations
│   ├── report_generator.py # Visuals and document creation
│   └── scheduler.py        # Time-based triggers
│
├── .env                    # Secrets and SMTP credentials
├── config.yaml             # System path configurations
├── Dockerfile              # Containerization instructions
├── main.py                 # Primary orchestrator
└── README.md               # Documentation
```
## 4. Technical Specifications

* Core Language: Python 3.9 

* Data Processing: Pandas, NumPy

* Visualizations: Matplotlib, Seaborn 

* Reporting: FPDF (PDF), XlsxWriter (Excel) 

* Deployment: Docker (Linux-based python:3.9-slim) 

## 5. Setup and Deployment
### 5.1 Prerequisites
* Docker Engine installed and running.

* WSL2 environment (for Windows users).

### 5.2 Environment Configuration
Create a .env file in the root directory to store sensitive credentials:

```Plaintext
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECEIVER=manager@factory.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### 5.3 Execution Instructions
Build the Container Image:

```Bash
docker build -t mfg-system .
```

Execute the Report Pipeline:

```Bash
docker run --env-file .env \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/data:/app/data \
  mfg-system
```
## 6. Data Integrity and Auditing
To meet industrial standards, the system implements the following:


* Validation Logs: All data corrections or missing values are recorded in reports/validation_log.txt for administrative review.

* Data Archiving: Every processed raw file is timestamped and copied to data/processed/ to maintain an immutable audit trail.

* KPI Accuracy: Calculations are verified through automated unit tests in the tests/ directory.

## 7. Future Enhancements
* Integration with live PLC/SCADA systems via OPC-UA.

* Implementation of a real-time dashboard using Flask or FastAPI.

* Predictive analytics for machine downtime using Machine Learning.
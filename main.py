import yaml
import logging
import os
from datetime import datetime
from src.ingestion import load_live_data # Updated Import
from src.validation import validate_data
from src.kpi_engine import calculate_kpis
from src.report_generator import generate_visuals, generate_pdf, send_email
from dotenv import load_dotenv

load_dotenv() # Load DB and Email credentials
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_pipeline():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    try:
        # 1. Ingestion: Now pulling from DB instead of CSV
        df = load_live_data()
        
        if df.empty:
            logging.info("Pipeline skipped: No new data to process.")
            return
        
        # 2. Validation
        df = validate_data(df)
        
        # 3. KPI Calculations
        summary, machine_stats = calculate_kpis(df)
        
        # 4. Visualization & PDF Generation
        charts = generate_visuals(df)
        generate_pdf(summary, charts, config['paths']['pdf_output'])
        
        # 5. Archive: Save the live snapshot to a CSV for auditing
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        archive_name = f"data/processed/live_snapshot_{ts}.csv"
        df.to_csv(archive_name, index=False)
        logging.info(f"Live data snapshot archived to {archive_name}")
        
        # 6. Email Delivery
        if os.getenv('EMAIL_PASSWORD'):
            send_email(config['paths']['pdf_output'])
            logging.info("Report emailed successfully.")

    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")

if __name__ == "__main__":
    run_pipeline()
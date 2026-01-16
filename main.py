import yaml
import logging
import shutil
import os
from datetime import datetime
from src.ingestion import load_raw_data
from src.validation import validate_data
from src.kpi_engine import calculate_kpis
from src.report_generator import generate_visuals, generate_pdf, send_email

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_pipeline():
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    try:
        # 1. Ingestion
        df = load_raw_data(config['paths']['raw_data'])
        
        # 2. Validation
        df = validate_data(df)
        
        # 3. KPI Calculations
        summary, machine_stats = calculate_kpis(df)
        
        # 4. Visualization & PDF Generation
        charts = generate_visuals(df)
        generate_pdf(summary, charts, config['paths']['pdf_output'])
        
        # 5. Archive Raw Data to processed folder
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        archive_name = f"data/processed/data_archive_{ts}.csv"
        shutil.copy(config['paths']['raw_data'], archive_name)
        logging.info(f"Raw data archived to {archive_name}")
        
        # 6. Email Delivery
        if os.getenv('EMAIL_PASSWORD'):
            send_email(config['paths']['pdf_output'])
            logging.info("Report emailed successfully.")

        logging.info("Process Complete. Check /reports and /data/processed folders.")

    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")

if __name__ == "__main__":
    run_pipeline()
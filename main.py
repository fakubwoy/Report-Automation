import logging
import pandas as pd
from src.ingestion import load_live_data
from src.validation import validate_data
from src.kpi_engine import calculate_kpis
from src.report_generator import generate_visuals, generate_pdf, send_email

# Ensure logging is initialized at the top level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_pipeline():
    try:
        # 1. Ingestion (Historical + Live)
        df, raw_live_data, engine = load_live_data()
        
        # 2. Archive to DB (using the engine from ingestion)
        if raw_live_data:
            try:
                live_df_to_save = pd.DataFrame(raw_live_data)
                live_df_to_save.to_sql('live_production', engine, if_exists='append', index=False)
                logging.info(f"Archived {len(raw_live_data)} records to PostgreSQL.")
            except Exception as db_e:
                logging.warning(f"DB Archive failed, but continuing report: {db_e}")

        # 3. Validation
        df = validate_data(df)
        
        # 4. KPI Calculations
        summary, machine_stats = calculate_kpis(df)
        
        # 5. Reporting
        charts = generate_visuals(df)
        generate_pdf(summary, charts, "reports/pdf/Report.pdf")
        
        # 6. Email
        send_email("reports/pdf/Report.pdf")
        logging.info("Pipeline execution complete.")

    except Exception as e:
        # Fixed: logging is now defined
        logging.error(f"Pipeline Failed: {e}")

if __name__ == "__main__":
    run_pipeline()
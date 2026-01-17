import schedule
import time
import logging
import sys
import os

# Add the parent directory to sys.path so it can find main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import run_pipeline

def start_scheduler():
    # Initialize database first
    logging.info("Initializing database...")
    try:
        from init_db import init_database
        init_database()
    except Exception as e:
        logging.warning(f"Database initialization failed (will use fallback data): {e}")
    
    logging.info("Starting initial system check and report generation...")
    # Run once immediately on startup to verify everything works
    try:
        run_pipeline()
    except Exception as e:
        logging.error(f"Initial pipeline run failed: {e}")

    # Schedule to run daily at 08:00 AM
    schedule.every().day.at("08:00").do(run_pipeline)
    
    logging.info("Scheduler active. Waiting for next daily trigger (08:00 AM)...")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    start_scheduler()
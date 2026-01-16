import schedule
import time
import logging
from main import run_pipeline

def start_scheduler():
    # Schedule to run daily at 08:00 AM
    schedule.every().day.at("08:00").do(run_pipeline)
    
    logging.info("Scheduler started. Waiting for scheduled tasks...")
    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute
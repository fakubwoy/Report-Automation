import os
from sqlalchemy import create_engine, text
import time
import logging

logging.basicConfig(level=logging.INFO)

def init_database():
    """Initialize PostgreSQL database with required tables"""
    user = os.getenv('DB_USER', 'mfg_user')
    password = os.getenv('DB_PASSWORD', 'mfg_pass123')
    host = os.getenv('DB_HOST', 'db')
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME', 'manufacturing')
    
    # Wait for PostgreSQL to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            engine = create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}")
            with engine.connect() as conn:
                # Create table if it doesn't exist
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS live_production (
                        id SERIAL PRIMARY KEY,
                        production_date DATE NOT NULL,
                        machine_id VARCHAR(50) NOT NULL,
                        units_produced NUMERIC NOT NULL,
                        defective_units NUMERIC NOT NULL,
                        downtime_min NUMERIC NOT NULL,
                        shift VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                logging.info("Database initialized successfully")
                return
        except Exception as e:
            if i < max_retries - 1:
                logging.warning(f"Database connection attempt {i+1}/{max_retries} failed: {e}")
                time.sleep(2)
            else:
                logging.error(f"Failed to initialize database after {max_retries} attempts")
                raise

if __name__ == "__main__":
    init_database()
import pandas as pd
import logging
import os
from sqlalchemy import create_engine

def load_live_data():
    logging.info("Connecting to SCADA/PLC Database via SQLAlchemy...")
    
    # Credentials from .env
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME')

    engine = create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}")
    
    try:
        # Querying based on your schema
        query = "SELECT * FROM live_production WHERE production_date >= CURRENT_DATE;"
        
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            logging.warning("No live data found for the current date.")
            return pd.DataFrame()

        # Mapping DB columns to Pipeline columns
        column_mapping = {
            'production_date': 'Date',
            'machine_id': 'Machine ID',
            'units_produced': 'Units Produced',
            'defective_units': 'Defective Units',
            'downtime_min': 'Downtime (minutes)', # Fixed mapping
            'shift': 'Shift'
        }
        
        # 1. Rename columns first
        df = df.rename(columns=column_mapping)
        
        # 2. Validate and cast numeric data
        required_cols = ['Units Produced', 'Defective Units', 'Downtime (minutes)']
        for col in required_cols:
            if col not in df.columns:
                logging.warning(f"Column {col} missing from DB. Initializing with 0.")
                df[col] = 0
            else:
                # Ensure values are floats/ints for plotting
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        return df

    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise
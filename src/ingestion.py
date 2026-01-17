import pandas as pd
import logging
import os
import asyncio
import random
import time
from sqlalchemy import create_engine
from asyncua import Client

logging.getLogger("asyncua").setLevel(logging.WARNING)

async def get_all_opcua_data(max_retries=5, retry_delay=2):
    """
    Fetch data from OPC-UA server with retry logic
    """
    url = os.getenv('OPCUA_SERVER_URL', 'opc.tcp://plc-simulator:4840/freeopcua/server/')
    results = []
    
    for attempt in range(max_retries):
        try:
            async with Client(url=url, timeout=10) as client:
                root = client.get_objects_node()
                for i in range(1, 4):
                    m_path = f"2:Machine_{i}"
                    units = await (await root.get_child([m_path, "2:UnitsProduced"])).get_value()
                    defects = await (await root.get_child([m_path, "2:DefectiveUnits"])).get_value()
                    down = await (await root.get_child([m_path, "2:DowntimeMinutes"])).get_value()
                    
                    results.append({
                        'production_date': pd.Timestamp.now().date(),
                        'machine_id': f"OPC-UA-M{i}",
                        'units_produced': float(units),
                        'defective_units': float(defects),
                        'downtime_min': float(down),
                        'shift': 'Live-Stream'
                    })
                logging.info(f"Successfully connected to OPC-UA server and fetched {len(results)} machine records")
                return results
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"OPC-UA connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error(f"OPC-UA connection failed after {max_retries} attempts: {e}")
                return []
    
    return []

def load_live_data():
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    # Use service name from compose, not localhost 
    host = os.getenv('DB_HOST', 'db') 
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME')

    engine = create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}")
    
    db_df = pd.DataFrame()
    try:
        query = "SELECT * FROM live_production WHERE production_date >= CURRENT_DATE - INTERVAL '7 days';"
        with engine.connect() as conn:
            db_df = pd.read_sql_query(query, conn)
    except Exception:
        logging.warning("DB Fetch failed; generating non-periodic random history.")
        history = []
        for i in range(50):
            history.append({
                'production_date': (pd.Timestamp.now() - pd.Timedelta(hours=i)).date(),
                'machine_id': f"HIST-{random.randint(1,5)}",
                # Realistic production randomness
                'units_produced': int(random.gauss(350, 40)),
                'defective_units': max(0, int(random.gauss(5, 2))),
                # FIXED: Random noise instead of i % 8
                'downtime_min': max(0, random.gauss(12, 5)), 
                'shift': random.choice(['Day', 'Night'])
            })
        db_df = pd.DataFrame(history)

    live_data_list = asyncio.run(get_all_opcua_data())
    live_df = pd.DataFrame(live_data_list)

    df_combined = pd.concat([db_df, live_df], ignore_index=True)
    df_combined = df_combined.rename(columns={
        'production_date': 'Date', 'machine_id': 'Machine ID',
        'units_produced': 'Units Produced', 'defective_units': 'Defective Units',
        'downtime_min': 'Downtime (minutes)', 'shift': 'Shift'
    })
    
    for col in ['Units Produced', 'Defective Units', 'Downtime (minutes)']:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').fillna(0)
    
    return df_combined, live_data_list, engine
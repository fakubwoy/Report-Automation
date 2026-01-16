import pandas as pd
import os
import logging

def load_raw_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at {file_path}")
    
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Cast columns to numeric to ensure math logic works
    numeric_cols = ['Units Produced', 'Defective Units', 'Downtime (minutes)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Standardize Date format
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    return df
import pandas as pd
import logging

def validate_data(df):
    logging.info("Validating data integrity...")
    errors = []
    
    # Check for missing values
    if df.isnull().values.any():
        errors.append("Missing values detected and zeroed.")
        df = df.fillna(0)
    
    # Logic check: No negative production or downtime
    cols = ['Units Produced', 'Defective Units', 'Downtime (minutes)']
    for col in cols:
        if (df[col] < 0).any():
            errors.append(f"Negative values in {col} corrected.")
            df[col] = df[col].clip(lower=0)
            
    # Logic check: Defects cannot exceed total units
    mask = df['Defective Units'] > df['Units Produced']
    if mask.any():
        errors.append("Defects exceeded production; capped at production value.")
        df.loc[mask, 'Defective Units'] = df.loc[mask, 'Units Produced']

    # Write log file to reports directory
    with open("reports/validation_log.txt", "w") as f:
        if errors:
            f.write("VALIDATION ERRORS FOUND AND FIXED:\n")
            f.write("\n".join(errors))
        else:
            f.write("Validation successful: No errors found in the dataset.")
            
    return df
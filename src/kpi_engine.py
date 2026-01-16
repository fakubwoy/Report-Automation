import pandas as pd

def calculate_kpis(df):
    # Overall KPIs
    total_prod = df['Units Produced'].sum()
    total_defects = df['Defective Units'].sum()
    
    summary = {
        'Total Units': int(total_prod),
        'Total Defects': int(total_defects),
        'Yield %': round(((total_prod - total_defects) / total_prod) * 100, 2) if total_prod > 0 else 0,
        'Avg Downtime (min)': round(df['Downtime (minutes)'].mean(), 2),
        'Report Date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    # Machine Stats
    machine_stats = df.groupby('Machine ID').agg({
        'Units Produced': 'sum',
        'Defective Units': 'sum',
        'Downtime (minutes)': 'sum'
    }).reset_index()
    
    return summary, machine_stats
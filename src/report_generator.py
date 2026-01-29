import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import logging
import smtplib
from email.message import EmailMessage

def generate_visuals(df):
    sns.set_theme(style="whitegrid")
    chart_paths = []
    
    # 1. Production Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='Shift', y='Units Produced', hue='Machine ID')
    plt.title("Production by Shift")
    plt.legend(title='Machine ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    path_prod = "reports/chart_prod.png"
    plt.savefig(path_prod)
    plt.close()
    chart_paths.append(path_prod)

    # 2. FIXED: Random Downtime Trend
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df.reset_index(), x='index', y='Downtime (minutes)', marker='o', color='red')
    plt.title("Randomized Downtime Trend (Gaussian Distribution)")
    plt.xlabel("Data Sequence (History + Live)")
    plt.tight_layout()
    path_trend = "reports/chart_down_trend.png"
    plt.savefig(path_trend)
    plt.close()
    chart_paths.append(path_trend)

    # 3. Downtime Comparison (Fixed Warning) 
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x='Machine ID', y='Downtime (minutes)', hue='Machine ID', palette='Reds_d', legend=False)
    plt.xticks(rotation=45)
    plt.title("Downtime by Machine")
    plt.tight_layout()
    path_var = "reports/chart_down_var.png"
    plt.savefig(path_var)
    plt.close()
    chart_paths.append(path_var)
    
    return chart_paths

def send_email(file_path):
    # Check if we even have a password/server before trying to connect
    if not os.getenv('EMAIL_PASSWORD') or not os.getenv('SMTP_SERVER'):
        logging.info("Skipping email delivery: SMTP credentials not configured.")
        return

    try:
        msg = EmailMessage()
        msg['Subject'] = f"Manufacturing Report: {pd.Timestamp.now().strftime('%Y-%m-%d')}"
        msg['From'] = os.getenv('EMAIL_SENDER')
        msg['To'] = os.getenv('EMAIL_RECEIVER')
        msg.set_content("Automated report attached.")

        with open(file_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename="Report.pdf")

        with smtplib.SMTP(os.getenv('SMTP_SERVER'), int(os.getenv('SMTP_PORT')), timeout=10) as s:
            s.starttls()
            s.login(os.getenv('EMAIL_SENDER'), os.getenv('EMAIL_PASSWORD'))
            s.send_message(msg)
            logging.info("Report emailed successfully.")
    except Exception as e:
        logging.warning(f"Email delivery failed (Network issues?): {e}")

def generate_pdf(summary, charts, output_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Professional Header
    pdf.set_fill_color(33, 47, 61) 
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "MFG AUTOMATED REPORT", ln=True, align='C')
    
    # KPI Table
    pdf.ln(25)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Performance Summary - {summary.get('Report Date', 'N/A')}", ln=True)
    
    pdf.set_font("Arial", '', 10)
    for k, v in summary.items():
        pdf.cell(60, 8, f"{k}:", border=1)
        pdf.cell(60, 8, f"{v}", border=1, ln=True)
        
    # Add Charts (including ML if present)
    for chart in charts:
        pdf.ln(8)
        # Check if we need a new page for the last chart to avoid cutoff
        if pdf.get_y() > 200: 
            pdf.add_page()
        
        # Add a title for the ML chart specifically if it's the 4th one
        if "ml_forecast" in chart:
             pdf.set_font("Arial", 'B', 11)
             pdf.cell(0, 10, "Predictive Maintenance Analysis (ML)", ln=True)
             
        pdf.image(chart, x=10, w=190)
        
    pdf.output(output_path)
    logging.info(f"PDF report generated: {output_path}")
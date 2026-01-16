import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import smtplib
import os
from email.message import EmailMessage

def generate_visuals(df):
    sns.set_theme(style="whitegrid")
    
    # Production Chart
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x='Shift', y='Units Produced', hue='Machine ID')
    plt.title("Production by Shift")
    plt.savefig("reports/chart_prod.png")
    plt.close()

    # Downtime Chart
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x='Date', y='Downtime (minutes)', marker='o', color='red')
    plt.title("Downtime Trend")
    plt.savefig("reports/chart_down.png")
    plt.close()
    
    return ["reports/chart_prod.png", "reports/chart_down.png"]

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
    pdf.cell(0, 10, f"Performance Summary - {summary['Report Date']}", ln=True)
    
    pdf.set_font("Arial", '', 10)
    for k, v in summary.items():
        pdf.cell(60, 8, f"{k}:", border=1)
        pdf.cell(60, 8, f"{v}", border=1, ln=True)
        
    # Add Charts
    for chart in charts:
        pdf.ln(5)
        pdf.image(chart, x=15, w=170)
        
    pdf.output(output_path)

def send_email(file_path):
    msg = EmailMessage()
    msg['Subject'] = f"Manufacturing Report: {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    msg['From'] = os.getenv('EMAIL_SENDER')
    msg['To'] = os.getenv('EMAIL_RECEIVER')
    msg.set_content("Automated report attached.")

    with open(file_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename="Report.pdf")

    with smtplib.SMTP(os.getenv('SMTP_SERVER'), int(os.getenv('SMTP_PORT'))) as s:
        s.starttls()
        s.login(os.getenv('EMAIL_SENDER'), os.getenv('EMAIL_PASSWORD'))
        s.send_message(msg)
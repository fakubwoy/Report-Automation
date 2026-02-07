import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import logging
import smtplib
from email.message import EmailMessage

def generate_visuals(df):
    """Generate production visualizations"""
    sns.set_theme(style="whitegrid")
    chart_paths = []
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
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
    logging.info(f"Generated production chart: {path_prod}")

    # 2. Downtime Trend
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df.reset_index(), x='index', y='Downtime (minutes)', marker='o', color='red')
    plt.title("Randomized Downtime Trend (Gaussian Distribution)")
    plt.xlabel("Data Sequence (History + Live)")
    plt.tight_layout()
    path_trend = "reports/chart_down_trend.png"
    plt.savefig(path_trend)
    plt.close()
    chart_paths.append(path_trend)
    logging.info(f"Generated downtime trend chart: {path_trend}")

    # 3. Downtime Comparison
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x='Machine ID', y='Downtime (minutes)', hue='Machine ID', palette='Reds_d', legend=False)
    plt.xticks(rotation=45)
    plt.title("Downtime by Machine")
    plt.tight_layout()
    path_var = "reports/chart_down_var.png"
    plt.savefig(path_var)
    plt.close()
    chart_paths.append(path_var)
    logging.info(f"Generated downtime comparison chart: {path_var}")
    
    return chart_paths

def send_email(file_path):
    """Send email with report attachment"""
    # Check if we have the necessary credentials
    smtp_server = os.getenv('SMTP_SERVER')
    email_password = os.getenv('EMAIL_PASSWORD')
    email_sender = os.getenv('EMAIL_SENDER')
    email_receiver = os.getenv('EMAIL_RECEIVER')
    smtp_port = os.getenv('SMTP_PORT', '587')
    
    # Log configuration status
    logging.info("Email Configuration Status:")
    logging.info(f"  SMTP Server: {'✓ Configured' if smtp_server else '✗ Missing'}")
    logging.info(f"  Email Password: {'✓ Configured' if email_password else '✗ Missing'}")
    logging.info(f"  Sender Email: {email_sender if email_sender else '✗ Missing'}")
    logging.info(f"  Receiver Email: {email_receiver if email_receiver else '✗ Missing'}")
    
    if not all([smtp_server, email_password, email_sender, email_receiver]):
        logging.info("⚠️  Skipping email delivery: SMTP credentials not fully configured.")
        logging.info("To enable email reports, set these environment variables:")
        if not smtp_server:
            logging.info("  - SMTP_SERVER (e.g., smtp.gmail.com)")
        if not smtp_port:
            logging.info("  - SMTP_PORT (e.g., 587)")
        if not email_sender:
            logging.info("  - EMAIL_SENDER (your email address)")
        if not email_password:
            logging.info("  - EMAIL_PASSWORD (your email password/app password)")
        if not email_receiver:
            logging.info("  - EMAIL_RECEIVER (recipient email address)")
        return

    if not os.path.exists(file_path):
        logging.error(f"❌ Cannot send email: Report file not found at {file_path}")
        return

    try:
        logging.info(f"Attempting to send email to {email_receiver}...")
        
        msg = EmailMessage()
        msg['Subject'] = f"Manufacturing Report: {pd.Timestamp.now().strftime('%Y-%m-%d')}"
        msg['From'] = email_sender
        msg['To'] = email_receiver
        msg.set_content("Automated manufacturing report is attached.\n\nThis report contains production KPIs, ML predictions, and AI-powered insights.")

        # Attach the PDF
        with open(file_path, 'rb') as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype='application', subtype='pdf', filename="Manufacturing_Report.pdf")

        # Send email
        with smtplib.SMTP(smtp_server, int(smtp_port), timeout=30) as server:
            server.starttls()
            server.login(email_sender, email_password)
            server.send_message(msg)
            logging.info(f"✅ Report emailed successfully to {email_receiver}")
            
    except smtplib.SMTPAuthenticationError as auth_e:
        logging.error(f"❌ Email authentication failed: {auth_e}")
        logging.error("Check your EMAIL_SENDER and EMAIL_PASSWORD credentials")
    except smtplib.SMTPException as smtp_e:
        logging.error(f"❌ SMTP error occurred: {smtp_e}")
    except Exception as e:
        logging.warning(f"⚠️  Email delivery failed: {e}")
        logging.info("Report was still generated successfully and is available locally")

def generate_pdf(summary, charts, output_path):
    """Generate PDF report with charts and KPIs"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
        
        # Add Charts
        for chart in charts:
            if not os.path.exists(chart):
                logging.warning(f"Chart not found: {chart}")
                continue
                
            pdf.ln(8)
            # Check if we need a new page for the last chart to avoid cutoff
            if pdf.get_y() > 200: 
                pdf.add_page()
            
            # Add a title for the advanced ML chart
            if "ml_advanced" in chart:
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 10, "Machine Learning Analysis Dashboard", ln=True)
            elif "ml_forecast" in chart:
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 10, "Machine Learning Forecast", ln=True)
                
            pdf.image(chart, x=10, w=190)
            logging.info(f"Added chart to PDF: {chart}")
            
        pdf.output(output_path)
        logging.info(f"✅ PDF report generated successfully: {output_path}")
        
    except Exception as e:
        logging.error(f"❌ PDF generation failed: {e}", exc_info=True)
        raise
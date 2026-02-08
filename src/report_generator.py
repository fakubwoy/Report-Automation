import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import logging
import smtplib
from email.message import EmailMessage

class PDF(FPDF):
    """Custom PDF class with automatic footer on every page"""
    def footer(self):
        """Add footer to every page automatically"""
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Manufacturing Intelligence Report | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Page {self.page_no()} of {{nb}}", align='C')

def sanitize_text(text):
    """Remove or replace characters that FPDF cannot encode in Latin-1"""
    if not isinstance(text, str):
        text = str(text)
    
    # Replace common problematic Unicode characters with ASCII equivalents
    replacements = {
        '\u2022': '-',  # Bullet point
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201C': '"',  # Left double quote
        '\u201D': '"',  # Right double quote
        '\u2026': '...', # Ellipsis
        '\u00B0': ' deg', # Degree symbol
    }
    
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Remove any remaining non-Latin-1 characters
    try:
        text.encode('latin-1')
    except UnicodeEncodeError:
        # If still can't encode, replace problematic characters
        text = text.encode('latin-1', errors='replace').decode('latin-1')
    
    return text

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

def generate_chart_insights(chart_name, summary, df=None):
    """Generate contextual insights for each chart"""
    insights = {
        'chart_prod.png': {
            'title': 'Production Performance by Shift',
            'insight': f"Total production across all shifts: {summary.get('Total Units', 'N/A')} units. "
                      f"Overall yield stands at {summary.get('Yield %', 'N/A')}%, indicating strong quality control."
        },
        'chart_down_trend.png': {
            'title': 'Downtime Trend Analysis',
            'insight': f"Average downtime is {summary.get('Avg Downtime (min)', 'N/A')} minutes per cycle. "
                      f"Monitor spikes to identify maintenance opportunities and improve OEE."
        },
        'chart_down_var.png': {
            'title': 'Machine-Level Downtime Comparison',
            'insight': f"Machines with highest downtime require priority attention for maintenance scheduling. "
                      f"Total defective units: {summary.get('Total Defects', 'N/A')}, focus on quality improvements."
        },
        'ml_advanced': {
            'title': 'Machine Learning Analysis Dashboard',
            'insight': f"ML Model Confidence: {summary.get('ML Confidence', 'N/A')}. "
                      f"Risk Level: {summary.get('Risk Level', 'Medium')}. Anomalies detected: {summary.get('Anomalies Detected', 0)}. "
                      f"Predicted downtime: {summary.get('ML Predicted Downtime', 'N/A')}."
        },
        'ml_forecast': {
            'title': 'Advanced Time-Series Forecasting',
            'insight': f"Next period forecast: {summary.get('Forecast Next Period', 'N/A')}. "
                      f"7-day average forecast: {summary.get('7-Day Avg Forecast', 'N/A')}. "
                      f"Use these predictions for proactive resource planning."
        }
    }
    
    # Match chart by filename
    for key, value in insights.items():
        if key in chart_name:
            return value
    
    # Default insight
    return {
        'title': 'Analysis Chart',
        'insight': 'This visualization provides additional production metrics and analysis.'
    }

def generate_pdf(summary, charts, output_path):
    """Generate PDF report with charts, insights, and professional formatting"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use custom PDF class with automatic footer
        pdf = PDF()
        pdf.alias_nb_pages()  # Enable {nb} placeholder for total pages
        pdf.add_page()
        
        # Professional Header with gradient effect
        pdf.set_fill_color(33, 47, 61) 
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 22)
        pdf.ln(8)
        pdf.cell(0, 12, "MANUFACTURING INTELLIGENCE REPORT", ln=True, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"Generated: {summary.get('Report Date', 'N/A')}", ln=True, align='C')
        
        # Executive Summary Section
        pdf.ln(12)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Executive Summary", ln=True, fill=True)
        
        pdf.ln(2)
        pdf.set_font("Arial", '', 10)
        
        # Key metrics in a clean 2-column layout
        key_metrics = [
            ('Total Units', summary.get('Total Units', 'N/A')),
            ('Yield Rate', f"{summary.get('Yield %', 'N/A')}%"),
            ('Total Defects', summary.get('Total Defects', 'N/A')),
            ('Avg Downtime', f"{summary.get('Avg Downtime (min)', 'N/A')} min"),
        ]
        
        for i in range(0, len(key_metrics), 2):
            # First column
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(45, 8, sanitize_text(f"{key_metrics[i][0]}:"), border=1, align='L')
            pdf.set_font("Arial", '', 10)
            pdf.cell(45, 8, sanitize_text(str(key_metrics[i][1])), border=1, align='C')
            
            # Second column (if exists)
            if i + 1 < len(key_metrics):
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(45, 8, sanitize_text(f"{key_metrics[i+1][0]}:"), border=1, align='L')
                pdf.set_font("Arial", '', 10)
                pdf.cell(45, 8, sanitize_text(str(key_metrics[i+1][1])), border=1, align='C')
            
            pdf.ln()
        
        # Performance Analysis Summary
        pdf.ln(6)
        pdf.set_fill_color(245, 250, 255)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 7, "Performance Analysis", ln=True, fill=True)
        
        pdf.set_font("Arial", '', 9)
        pdf.set_text_color(40, 40, 40)
        
        # Calculate quality rate
        total_units = summary.get('Total Units', 0)
        total_defects = summary.get('Total Defects', 0)
        yield_pct = summary.get('Yield %', 0)
        
        # Performance insights
        if isinstance(yield_pct, (int, float)):
            if yield_pct >= 99:
                quality_assessment = "Excellent - exceeding industry standards"
            elif yield_pct >= 97:
                quality_assessment = "Good - meeting quality targets"
            elif yield_pct >= 95:
                quality_assessment = "Acceptable - some improvement needed"
            else:
                quality_assessment = "Below target - immediate action required"
        else:
            quality_assessment = "Quality data available"
        
        avg_downtime = summary.get('Avg Downtime (min)', 0)
        if isinstance(avg_downtime, (int, float)):
            if avg_downtime < 10:
                downtime_assessment = "Optimal - minimal disruption"
            elif avg_downtime < 20:
                downtime_assessment = "Moderate - within acceptable range"
            elif avg_downtime < 30:
                downtime_assessment = "Elevated - review maintenance schedule"
            else:
                downtime_assessment = "Critical - requires immediate attention"
        else:
            downtime_assessment = "Downtime data available"
        
        performance_text = f"""Quality Performance: {quality_assessment}. With {total_defects} defective units out of {total_units} produced, the manufacturing process demonstrates {'strong' if yield_pct >= 97 else 'adequate'} quality control.

Operational Efficiency: {downtime_assessment}. Current downtime levels {'support' if avg_downtime < 20 else 'impact'} overall equipment effectiveness (OEE).

Key Focus Areas: {'Continue current practices' if yield_pct >= 97 and avg_downtime < 20 else 'Prioritize quality improvements and downtime reduction'} to maintain competitive advantage."""
        
        pdf.multi_cell(0, 4.5, sanitize_text(performance_text))
        
        # ML/AI Insights Section
        if 'ML Predicted Downtime' in summary:
            pdf.ln(6)
            pdf.set_fill_color(230, 245, 255)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, "AI & Machine Learning Insights", ln=True, fill=True)
            
            ml_metrics = []
            if 'ML Predicted Downtime' in summary:
                ml_metrics.append(('Predicted Downtime', summary.get('ML Predicted Downtime')))
            if 'ML Confidence' in summary:
                ml_metrics.append(('Model Confidence', summary.get('ML Confidence')))
            if 'Risk Level' in summary:
                ml_metrics.append(('Risk Assessment', summary.get('Risk Level')))
            if 'Anomalies Detected' in summary:
                ml_metrics.append(('Anomalies Found', str(summary.get('Anomalies Detected'))))
            if 'Forecast Next Period' in summary:
                ml_metrics.append(('Next Period Forecast', summary.get('Forecast Next Period')))
            if '7-Day Avg Forecast' in summary:
                ml_metrics.append(('7-Day Avg Forecast', summary.get('7-Day Avg Forecast')))
            
            pdf.set_font("Arial", '', 9)
            # Display ML metrics in 2-column layout for compactness
            for i in range(0, len(ml_metrics), 2):
                pdf.cell(50, 6, sanitize_text(f"{ml_metrics[i][0]}:"), border=1)
                pdf.cell(40, 6, sanitize_text(str(ml_metrics[i][1])), border=1)
                
                if i + 1 < len(ml_metrics):
                    pdf.cell(50, 6, sanitize_text(f"{ml_metrics[i+1][0]}:"), border=1)
                    pdf.cell(40, 6, sanitize_text(str(ml_metrics[i+1][1])), border=1)
                
                pdf.ln()
            
            # AI Interpretation
            pdf.ln(3)
            pdf.set_font("Arial", '', 9)
            pdf.set_text_color(40, 40, 40)
            
            risk_level = summary.get('Risk Level', 'Medium')
            confidence = summary.get('ML Confidence', 'N/A')
            
            ai_interpretation = f"""AI Analysis: The predictive model shows {confidence} confidence in forecasting upcoming operational trends. Risk level is assessed as {risk_level}, {'indicating stable operations' if risk_level == 'Low' else 'suggesting proactive measures may be beneficial'}. Anomaly detection has identified {summary.get('Anomalies Detected', 0)} irregular patterns that warrant investigation for process optimization."""
            
            pdf.multi_cell(0, 4.5, sanitize_text(ai_interpretation))
        
        # Recommendations Section
        pdf.ln(6)
        pdf.set_fill_color(255, 250, 240)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 7, "Key Recommendations", ln=True, fill=True)
        
        pdf.set_font("Arial", '', 9)
        pdf.set_text_color(40, 40, 40)
        
        recommendations = []
        if isinstance(yield_pct, (int, float)) and yield_pct < 98:
            recommendations.append("- Investigate root causes of defects to improve yield above 98%")
        if isinstance(avg_downtime, (int, float)) and avg_downtime > 15:
            recommendations.append("- Implement predictive maintenance to reduce downtime below 15 minutes")
        if summary.get('Anomalies Detected', 0) > 5:
            recommendations.append("- Review detected anomalies for potential process improvements")
        
        if not recommendations:
            recommendations.append("- Maintain current operational excellence")
            recommendations.append("- Continue monitoring for early detection of issues")
        
        recommendations.append("- Utilize ML forecasts for proactive resource planning")
        recommendations.append("- Review detailed analytics in following sections for deeper insights")
        
        for rec in recommendations:
            pdf.multi_cell(0, 4.5, sanitize_text(rec))
        
        # Charts and Insights Section
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, "Detailed Analytics & Visualizations", ln=True, fill=True)
        
        for chart in charts:
            if not os.path.exists(chart):
                logging.warning(f"Chart not found: {chart}")
                continue
            
            # Get insights for this chart
            chart_info = generate_chart_insights(os.path.basename(chart), summary)
            
            # Check if we need a new page
            if pdf.get_y() > 200:
                pdf.add_page()
            
            pdf.ln(8)
            
            # Chart Title
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(33, 47, 61)
            pdf.cell(0, 8, sanitize_text(chart_info['title']), ln=True)
            
            # Chart Insight (before image for context)
            pdf.set_font("Arial", '', 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, sanitize_text(chart_info['insight']))
            pdf.ln(2)
            
            # Determine image size based on chart type
            img_width = 180
            img_x = 15
            
            # Larger size for ML dashboard
            if "ml_advanced" in chart or "ml_forecast" in chart:
                img_width = 185
                img_x = 12
            
            # Add the chart image
            pdf.image(chart, x=img_x, w=img_width)
            pdf.ln(5)
            
            # Add a subtle divider line
            pdf.set_draw_color(200, 200, 200)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            
            logging.info(f"Added chart with insights: {chart}")
        
        # Footer is now automatically added to every page by the PDF class
        pdf.output(output_path)
        logging.info(f"✅ PDF report generated successfully: {output_path}")
        
    except Exception as e:
        logging.error(f"❌ PDF generation failed: {e}", exc_info=True)
        raise
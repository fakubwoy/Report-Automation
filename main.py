import logging
import pandas as pd
from src.ingestion import load_live_data_async
from src.validation import validate_data
from src.kpi_engine import calculate_kpis
from src.report_generator import generate_visuals, generate_pdf, send_email
import asyncio
import os

# Try to use advanced ML engine, fallback to basic if not available
try:
    from src.ml_engine_advanced import perform_advanced_ml_analysis
    from src.ai_insights import get_comprehensive_ai_insights
    ADVANCED_ML_AVAILABLE = True
    logging.info("Advanced ML and AI engines loaded")
except ImportError:
    from src.ml_engine import perform_ml_analysis
    ADVANCED_ML_AVAILABLE = False
    logging.warning("Using basic ML engine - advanced features unavailable")

# Try to use advanced forecasting
try:
    from src.forecasting_engine import perform_advanced_forecasting
    FORECASTING_AVAILABLE = True
    logging.info("Advanced forecasting engine loaded (LSTM/Prophet)")
except ImportError:
    FORECASTING_AVAILABLE = False
    logging.warning("Advanced forecasting unavailable - install tensorflow and prophet")

# Ensure logging is initialized at the top level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def run_pipeline_async():
    """Async version for AI integration"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs("reports/pdf", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # 1. Ingestion (Historical + Live)
        logging.info("Loading production data...")
        df, raw_live_data, engine = await load_live_data_async()
        
        # 2. Archive to DB
        if raw_live_data:
            try:
                live_df_to_save = pd.DataFrame(raw_live_data)
                live_df_to_save.to_sql('live_production', engine, if_exists='append', index=False)
                logging.info(f"Archived {len(raw_live_data)} records to PostgreSQL.")
            except Exception as db_e:
                logging.warning(f"DB Archive failed, but continuing report: {db_e}")

        # 3. Validation
        logging.info("Validating data...")
        df = validate_data(df)
        
        # 4. KPI Calculations
        logging.info("Calculating KPIs...")
        summary, machine_stats = calculate_kpis(df)
        
        # 5. Advanced ML Analysis
        logging.info("Running Advanced ML & AI Analysis...")
        charts = []
        
        if ADVANCED_ML_AVAILABLE:
            # Advanced ML with multiple models
            pred_downtime, pred_score, ml_insights, ml_chart_path = perform_advanced_ml_analysis(df)
            
            # Add ML stats to summary
            summary['ML Predicted Downtime'] = f"{pred_downtime:.1f} min"
            summary['ML Confidence'] = f"{pred_score*100:.1f}%"
            summary['Risk Level'] = ml_insights.get('risk_level', 'Unknown')
            summary['Anomalies Detected'] = ml_insights.get('anomalies_detected', 0)
            
            # Get AI-powered insights
            try:
                kpi_summary = {
                    'total_units': summary.get('Total Units', 0),
                    'total_defects': summary.get('Total Defects', 0),
                    'yield_percentage': summary.get('Yield %', 0),
                    'avg_downtime': summary.get('Avg Downtime (min)', 0)
                }
                
                ai_insights = await get_comprehensive_ai_insights(
                    ml_insights,
                    df.to_dict('records')[:20],  # Sample data
                    kpi_summary,
                    machine_stats.to_dict('records') if hasattr(machine_stats, 'to_dict') else machine_stats
                )
                
                # Add AI insights to summary
                if ai_insights and 'strategic_insights' in ai_insights:
                    summary['AI Analysis'] = "Generated (see report)"
                    
                    # Save AI insights to file for the report
                    with open('reports/ai_insights.txt', 'w') as f:
                        f.write("=== AI STRATEGIC INSIGHTS ===\n\n")
                        f.write(ai_insights['strategic_insights'].get('full_analysis', 'N/A'))
                        f.write("\n\n=== AI MAINTENANCE PLAN ===\n\n")
                        f.write(str(ai_insights.get('maintenance_plan', 'N/A')))
                        f.write("\n\n=== AI QUALITY INSIGHTS ===\n\n")
                        f.write(str(ai_insights.get('quality_insights', 'N/A')))
                    
                    logging.info("AI insights saved to reports/ai_insights.txt")
                    
            except Exception as ai_e:
                logging.warning(f"AI insights generation failed: {ai_e}")
            
            # Generate visualizations
            logging.info("Generating visualizations...")
            charts = generate_visuals(df)
            
            # CRITICAL FIX: Add the ML chart with the correct filename
            if ml_chart_path and os.path.exists(ml_chart_path):
                charts.append(ml_chart_path)
                logging.info(f"Added ML chart: {ml_chart_path}")
            else:
                logging.warning("ML chart not generated")
            
            # Advanced Time-Series Forecasting
            if FORECASTING_AVAILABLE:
                try:
                    logging.info("Running advanced time-series forecasting...")
                    lstm_forecast, prophet_forecast, ensemble_forecast, forecast_chart = perform_advanced_forecasting(
                        df, target_col='Downtime (minutes)', forecast_periods=7
                    )
                    
                    if forecast_chart and os.path.exists(forecast_chart):
                        charts.append(forecast_chart)
                        logging.info(f"Added forecast chart: {forecast_chart}")
                        
                        # Add forecast summary to report
                        if ensemble_forecast is not None and len(ensemble_forecast) > 0:
                            summary['Forecast Next Period'] = f"{ensemble_forecast[0]:.1f} min"
                            summary['7-Day Avg Forecast'] = f"{ensemble_forecast.mean():.1f} min"
                    
                except Exception as forecast_e:
                    logging.warning(f"Forecasting failed but continuing: {forecast_e}")
                    
        else:
            # Basic ML fallback
            from src.ml_engine import perform_ml_analysis
            pred_downtime, pred_score, ml_chart_path = perform_ml_analysis(df)
            
            summary['ML Predicted Downtime'] = f"{pred_downtime:.1f} min"
            summary['ML Confidence'] = f"{pred_score:.2f}"
            
            logging.info("Generating visualizations...")
            charts = generate_visuals(df)
            
            if ml_chart_path and os.path.exists(ml_chart_path):
                charts.append(ml_chart_path)
                logging.info(f"Added ML chart: {ml_chart_path}")

        # 6. Generate PDF Report
        logging.info("Generating PDF report...")
        report_path = "reports/pdf/Report.pdf"
        generate_pdf(summary, charts, report_path)
        
        if os.path.exists(report_path):
            logging.info(f"✅ PDF report successfully generated: {report_path}")
        else:
            logging.error("❌ PDF report generation failed - file does not exist")
        
        # 7. Email
        logging.info("Attempting to send email...")
        send_email(report_path)
        
        logging.info("✅ Pipeline execution complete with ML & AI insights.")

    except Exception as e:
        logging.error(f"Pipeline Failed: {e}", exc_info=True)

def run_pipeline():
    """Synchronous wrapper for backward compatibility"""
    try:
        # Run async pipeline
        asyncio.run(run_pipeline_async())
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_pipeline()
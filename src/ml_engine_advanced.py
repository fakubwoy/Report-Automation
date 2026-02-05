import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO)

class AdvancedMLEngine:
    """
    Advanced ML Engine with multiple models and AI-powered insights
    """
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.feature_importance = {}
        
    def prepare_features(self, df):
        """
        Advanced feature engineering
        """
        try:
            # Ensure we have the required columns
            required_cols = ['Units Produced', 'Defective Units', 'Downtime (minutes)']
            for col in required_cols:
                if col not in df.columns:
                    logging.error(f"Missing column: {col}")
                    return None, None
            
            # Create derived features
            df = df.copy()
            df['Defect_Rate'] = df['Defective Units'] / (df['Units Produced'] + 1)  # Avoid division by zero
            df['Quality_Score'] = 1 - df['Defect_Rate']
            df['Production_Efficiency'] = df['Units Produced'] / (df['Downtime (minutes)'] + 1)
            
            # Rolling statistics (if we have enough data)
            if len(df) >= 5:
                df['Units_MA3'] = df['Units Produced'].rolling(window=3, min_periods=1).mean()
                df['Defects_MA3'] = df['Defective Units'].rolling(window=3, min_periods=1).mean()
                df['Downtime_Trend'] = df['Downtime (minutes)'].rolling(window=3, min_periods=1).mean()
            else:
                df['Units_MA3'] = df['Units Produced']
                df['Defects_MA3'] = df['Defective Units']
                df['Downtime_Trend'] = df['Downtime (minutes)']
            
            # Feature matrix
            feature_cols = [
                'Units Produced', 'Defective Units', 'Defect_Rate', 
                'Quality_Score', 'Production_Efficiency',
                'Units_MA3', 'Defects_MA3', 'Downtime_Trend'
            ]
            
            X = df[feature_cols].fillna(0)
            y = df['Downtime (minutes)']
            
            return X, y
            
        except Exception as e:
            logging.error(f"Feature preparation failed: {e}")
            return None, None
    
    def train_ensemble_models(self, X, y):
        """
        Train multiple models for ensemble prediction
        """
        try:
            if len(X) < 5:
                logging.warning("Insufficient data for training. Need at least 5 samples.")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train_scaled, y_train)
            rf_score = self.rf_model.score(X_test_scaled, y_test)
            
            # Train XGBoost
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.xgb_model.fit(X_train_scaled, y_train)
            xgb_score = self.xgb_model.score(X_test_scaled, y_test)
            
            # Feature importance from Random Forest
            self.feature_importance = dict(zip(
                X.columns,
                self.rf_model.feature_importances_
            ))
            
            logging.info(f"Random Forest R² Score: {rf_score:.3f}")
            logging.info(f"XGBoost R² Score: {xgb_score:.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            return False
    
    def detect_anomalies(self, X):
        """
        Detect anomalies in production data
        """
        try:
            if len(X) < 5:
                return np.array([])
            
            # Train Isolation Forest
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            X_scaled = self.scaler.transform(X)
            anomalies = self.anomaly_detector.fit_predict(X_scaled)
            
            # -1 means anomaly, 1 means normal
            anomaly_indices = np.where(anomalies == -1)[0]
            
            logging.info(f"Detected {len(anomaly_indices)} anomalies in {len(X)} samples")
            
            return anomaly_indices
            
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            return np.array([])
    
    def predict_ensemble(self, X_future):
        """
        Make predictions using ensemble of models
        """
        try:
            X_scaled = self.scaler.transform(X_future)
            
            # Get predictions from both models
            rf_pred = self.rf_model.predict(X_scaled)
            xgb_pred = self.xgb_model.predict(X_scaled)
            
            # Ensemble: weighted average (RF: 40%, XGB: 60%)
            ensemble_pred = 0.4 * rf_pred + 0.6 * xgb_pred
            
            # Calculate confidence based on agreement
            agreement = 1 - abs(rf_pred - xgb_pred) / (abs(rf_pred) + abs(xgb_pred) + 1)
            confidence = float(np.mean(agreement))
            
            return float(ensemble_pred[0]), confidence
            
        except Exception as e:
            logging.error(f"Ensemble prediction failed: {e}")
            return 0.0, 0.0
    
    def generate_ml_insights(self, df, prediction, confidence, anomalies):
        """
        Generate ML-based insights and recommendations
        """
        insights = {
            'prediction': prediction,
            'confidence': confidence,
            'anomalies_detected': len(anomalies),
            'feature_importance': self.feature_importance,
            'risk_level': self._assess_risk(prediction),
            'recommendations': self._generate_recommendations(df, prediction, anomalies)
        }
        
        return insights
    
    def _assess_risk(self, predicted_downtime):
        """Assess operational risk level"""
        if predicted_downtime <= 15:
            return 'Low'
        elif predicted_downtime <= 30:
            return 'Medium'
        elif predicted_downtime <= 45:
            return 'High'
        else:
            return 'Critical'
    
    def _generate_recommendations(self, df, prediction, anomalies):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Downtime recommendation
        if prediction > 30:
            recommendations.append({
                'type': 'maintenance',
                'priority': 'High',
                'message': 'Schedule preventive maintenance to reduce predicted downtime',
                'action': 'Review machine performance logs'
            })
        
        # Anomaly recommendation
        if len(anomalies) > 0:
            recommendations.append({
                'type': 'anomaly',
                'priority': 'Medium',
                'message': f'{len(anomalies)} anomalous patterns detected in recent data',
                'action': 'Investigate unusual production patterns'
            })
        
        # Quality recommendation
        if 'Defect_Rate' in df.columns:
            avg_defect_rate = df['Defect_Rate'].mean()
            if avg_defect_rate > 0.05:  # More than 5% defect rate
                recommendations.append({
                    'type': 'quality',
                    'priority': 'High',
                    'message': f'Average defect rate at {avg_defect_rate*100:.1f}% exceeds threshold',
                    'action': 'Inspect quality control processes'
                })
        
        return recommendations
    
    def create_advanced_visualizations(self, df, X, y, prediction, anomalies):
        """
        Create comprehensive ML visualizations
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Advanced ML Analytics Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Feature Importance
            if self.feature_importance:
                sorted_features = sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                features, importance = zip(*sorted_features)
                
                axes[0, 0].barh(features, importance, color='#3498db')
                axes[0, 0].set_xlabel('Importance Score')
                axes[0, 0].set_title('Feature Importance (Random Forest)')
                axes[0, 0].grid(axis='x', alpha=0.3)
            
            # 2. Actual vs Predicted
            if self.rf_model and len(y) > 0:
                X_scaled = self.scaler.transform(X)
                predictions = self.rf_model.predict(X_scaled)
                
                axes[0, 1].scatter(y, predictions, alpha=0.6, color='#2ecc71')
                axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 
                               'r--', lw=2, label='Perfect Prediction')
                axes[0, 1].set_xlabel('Actual Downtime (min)')
                axes[0, 1].set_ylabel('Predicted Downtime (min)')
                axes[0, 1].set_title('Model Accuracy: Actual vs Predicted')
                axes[0, 1].legend()
                axes[0, 1].grid(alpha=0.3)
            
            # 3. Anomaly Detection
            if len(anomalies) > 0 and 'Downtime (minutes)' in df.columns:
                time_index = range(len(df))
                axes[1, 0].plot(time_index, df['Downtime (minutes)'], 
                               'b-', label='Normal', linewidth=2)
                axes[1, 0].scatter(anomalies, df.iloc[anomalies]['Downtime (minutes)'],
                                  color='red', s=100, label='Anomalies', 
                                  marker='X', zorder=5)
                axes[1, 0].set_xlabel('Time Sequence')
                axes[1, 0].set_ylabel('Downtime (minutes)')
                axes[1, 0].set_title(f'Anomaly Detection ({len(anomalies)} anomalies found)')
                axes[1, 0].legend()
                axes[1, 0].grid(alpha=0.3)
            
            # 4. Prediction Forecast
            if 'Downtime (minutes)' in df.columns:
                recent_data = df['Downtime (minutes)'].tail(10)
                future_steps = 5
                forecast = [prediction] * future_steps
                
                combined_data = list(recent_data) + forecast
                combined_index = range(len(combined_data))
                
                axes[1, 1].plot(combined_index[:len(recent_data)], 
                               combined_data[:len(recent_data)],
                               'b-', linewidth=2, label='Historical')
                axes[1, 1].plot(combined_index[len(recent_data)-1:], 
                               combined_data[len(recent_data)-1:],
                               'r--', linewidth=2, label='Forecast', marker='o')
                axes[1, 1].axvline(x=len(recent_data)-1, color='gray', 
                                  linestyle=':', alpha=0.5)
                axes[1, 1].set_xlabel('Time Sequence')
                axes[1, 1].set_ylabel('Downtime (minutes)')
                axes[1, 1].set_title('Downtime Forecast (Next 5 Periods)')
                axes[1, 1].legend()
                axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            plot_path = "reports/chart_ml_advanced.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logging.error(f"Visualization creation failed: {e}")
            return None


async def get_ai_insights(ml_insights, production_summary):
    """
    Get AI-powered insights using Claude API
    """
    try:
        # Prepare context for AI
        context = {
            'predicted_downtime': ml_insights['prediction'],
            'confidence': ml_insights['confidence'],
            'risk_level': ml_insights['risk_level'],
            'anomalies': ml_insights['anomalies_detected'],
            'top_features': dict(list(ml_insights['feature_importance'].items())[:3]),
            'production_summary': production_summary
        }
        
        prompt = f"""Analyze this manufacturing data and provide strategic insights:

Production Context:
- Predicted Next Shift Downtime: {context['predicted_downtime']:.1f} minutes
- Model Confidence: {context['confidence']*100:.1f}%
- Risk Level: {context['risk_level']}
- Anomalies Detected: {context['anomalies']}
- Top Influencing Factors: {context['top_features']}

Production Summary:
{json.dumps(production_summary, indent=2)}

Please provide:
1. Root cause analysis of predicted downtime
2. Specific actionable recommendations
3. Potential cost impact if issues not addressed
4. Preventive measures for the next 48 hours

Keep response concise and focused on actionable insights."""

        # Call Claude API
        response = await fetch("https://api.anthropic.com/v1/messages", {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            })
        })
        
        if response.ok:
            data = await response.json()
            ai_analysis = data['content'][0]['text']
            return ai_analysis
        else:
            logging.warning("AI insights unavailable - API call failed")
            return None
            
    except Exception as e:
        logging.error(f"AI insights generation failed: {e}")
        return None


def perform_advanced_ml_analysis(df):
    """
    Main function to perform comprehensive ML analysis
    
    Returns:
        - prediction: Ensemble prediction for next shift downtime
        - confidence: Model confidence score
        - insights: Complete ML insights dictionary
        - plot_path: Path to advanced visualization
    """
    try:
        logging.info("Starting Advanced ML Analysis...")
        
        # Initialize engine
        engine = AdvancedMLEngine()
        
        # Prepare features
        X, y = engine.prepare_features(df)
        
        if X is None or len(X) < 5:
            logging.warning("Insufficient data for ML analysis")
            return 0.0, 0.0, {}, None
        
        # Train models
        training_success = engine.train_ensemble_models(X, y)
        
        if not training_success:
            logging.warning("Model training incomplete")
            return 0.0, 0.0, {}, None
        
        # Detect anomalies
        anomalies = engine.detect_anomalies(X)
        
        # Prepare future prediction (using recent averages + 10% increase)
        future_features = X.tail(1).copy()
        for col in ['Units Produced', 'Defective Units']:
            if col in future_features.columns:
                future_features[col] *= 1.1
        
        # Make ensemble prediction
        prediction, confidence = engine.predict_ensemble(future_features)
        
        # Generate insights
        insights = engine.generate_ml_insights(df, prediction, confidence, anomalies)
        
        # Create visualizations
        plot_path = engine.create_advanced_visualizations(
            df, X, y, prediction, anomalies
        )
        
        # Log results
        logging.info(f"ML Analysis Complete:")
        logging.info(f"  - Predicted Downtime: {prediction:.2f} min")
        logging.info(f"  - Confidence: {confidence*100:.1f}%")
        logging.info(f"  - Risk Level: {insights['risk_level']}")
        logging.info(f"  - Anomalies: {len(anomalies)}")
        
        return prediction, confidence, insights, plot_path
        
    except Exception as e:
        logging.error(f"Advanced ML Analysis Failed: {e}", exc_info=True)
        return 0.0, 0.0, {}, None


# Backward compatibility function
def perform_ml_analysis(df):
    """
    Simple wrapper for backward compatibility
    """
    prediction, confidence, insights, plot_path = perform_advanced_ml_analysis(df)
    return prediction, confidence, plot_path
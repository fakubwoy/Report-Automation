import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Try importing LSTM dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    LSTM_AVAILABLE = True
    logging.info("TensorFlow/LSTM available")
except ImportError:
    LSTM_AVAILABLE = False
    logging.warning("TensorFlow not available - LSTM forecasting disabled")

# Try importing Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logging.info("Prophet available")
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available - Prophet forecasting disabled")

logging.basicConfig(level=logging.INFO)


class TimeSeriesForecaster:
    """
    Advanced time-series forecasting using LSTM and Prophet
    """
    
    def __init__(self):
        self.lstm_model = None
        self.prophet_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 10  # Look back 10 time steps
        self.forecast_horizon = 7  # Forecast 7 periods ahead
        
    def prepare_lstm_data(self, data, target_col='Downtime (minutes)'):
        """
        Prepare time-series data for LSTM
        
        Args:
            data: DataFrame with time-series data
            target_col: Column to forecast
            
        Returns:
            X_train, y_train, X_test, y_test, scaler
        """
        try:
            if len(data) < self.sequence_length + 5:
                logging.warning(f"Insufficient data for LSTM (need at least {self.sequence_length + 5} samples)")
                return None, None, None, None, None
            
            # Extract target values
            values = data[target_col].values.reshape(-1, 1)
            
            # Scale data
            scaled_data = self.scaler.fit_transform(values)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length), 0])
                y.append(scaled_data[i + self.sequence_length, 0])
            
            X = np.array(X)
            y = np.array(y)
            
            # Reshape for LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Train/test split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logging.info(f"LSTM data prepared: Train={len(X_train)}, Test={len(X_test)}")
            return X_train, y_train, X_test, y_test, scaled_data
            
        except Exception as e:
            logging.error(f"LSTM data preparation failed: {e}")
            return None, None, None, None, None
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM neural network
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            logging.info("LSTM model architecture built")
            return model
            
        except Exception as e:
            logging.error(f"LSTM model building failed: {e}")
            return None
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """
        Train LSTM model
        """
        try:
            if X_train is None or len(X_train) == 0:
                logging.warning("No training data for LSTM")
                return None, None
            
            # Build model
            self.lstm_model = self.build_lstm_model((X_train.shape[1], 1))
            
            if self.lstm_model is None:
                return None, None
            
            # Early stopping to prevent overfitting
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            train_loss = self.lstm_model.evaluate(X_train, y_train, verbose=0)[0]
            test_loss = self.lstm_model.evaluate(X_test, y_test, verbose=0)[0]
            
            logging.info(f"LSTM Training Complete - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            return self.lstm_model, history
            
        except Exception as e:
            logging.error(f"LSTM training failed: {e}")
            return None, None
    
    def forecast_lstm(self, data, target_col='Downtime (minutes)', periods=7):
        """
        Generate LSTM forecast
        
        Args:
            data: Historical data
            target_col: Column to forecast
            periods: Number of periods to forecast
            
        Returns:
            forecast_values, confidence_lower, confidence_upper
        """
        try:
            if self.lstm_model is None:
                logging.warning("LSTM model not trained")
                return None, None, None
            
            # Get last sequence from data
            values = data[target_col].values.reshape(-1, 1)
            scaled_data = self.scaler.transform(values)
            
            # Get last sequence_length values
            last_sequence = scaled_data[-self.sequence_length:]
            current_sequence = last_sequence.reshape(1, self.sequence_length, 1)
            
            # Generate forecast
            forecast_scaled = []
            for _ in range(periods):
                # Predict next value
                next_pred = self.lstm_model.predict(current_sequence, verbose=0)[0, 0]
                forecast_scaled.append(next_pred)
                
                # Update sequence (rolling window)
                current_sequence = np.append(current_sequence[0, 1:, 0], next_pred)
                current_sequence = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Inverse transform to get actual values
            forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
            forecast_values = self.scaler.inverse_transform(forecast_scaled).flatten()
            
            # Calculate confidence intervals (approximation based on historical variance)
            historical_std = data[target_col].std()
            confidence_lower = forecast_values - 1.96 * historical_std
            confidence_upper = forecast_values + 1.96 * historical_std
            
            logging.info(f"LSTM forecast generated for {periods} periods")
            return forecast_values, confidence_lower, confidence_upper
            
        except Exception as e:
            logging.error(f"LSTM forecasting failed: {e}")
            return None, None, None
    
    def train_prophet(self, data, target_col='Downtime (minutes)', date_col='Date'):
        """
        Train Prophet model
        
        Args:
            data: DataFrame with date and target columns
            target_col: Column to forecast
            date_col: Date column name
        """
        try:
            # Prepare data for Prophet (needs 'ds' and 'y' columns)
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime(data[date_col]),
                'y': data[target_col]
            })
            
            # Initialize and train Prophet
            self.prophet_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            
            self.prophet_model.fit(prophet_data)
            
            logging.info("Prophet model trained successfully")
            return self.prophet_model
            
        except Exception as e:
            logging.error(f"Prophet training failed: {e}")
            return None
    
    def forecast_prophet(self, periods=7, freq='D'):
        """
        Generate Prophet forecast
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'H' for hourly)
            
        Returns:
            forecast DataFrame with predictions and confidence intervals
        """
        try:
            if self.prophet_model is None:
                logging.warning("Prophet model not trained")
                return None
            
            # Create future dataframe
            future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq)
            
            # Generate forecast
            forecast = self.prophet_model.predict(future)
            
            logging.info(f"Prophet forecast generated for {periods} periods")
            return forecast
            
        except Exception as e:
            logging.error(f"Prophet forecasting failed: {e}")
            return None
    
    def create_forecast_visualization(self, data, lstm_forecast=None, prophet_forecast=None, 
                                     target_col='Downtime (minutes)', output_path='reports/chart_forecast.png'):
        """
        Create comprehensive forecast visualization comparing LSTM and Prophet
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.patch.set_facecolor('#1e2337')
            
            for ax in axes.flat:
                ax.set_facecolor('#141829')
                ax.spines['bottom'].set_color('#2a2f4a')
                ax.spines['top'].set_color('#2a2f4a')
                ax.spines['left'].set_color('#2a2f4a')
                ax.spines['right'].set_color('#2a2f4a')
                ax.tick_params(colors='#9ca3af')
                ax.xaxis.label.set_color('#e4e7eb')
                ax.yaxis.label.set_color('#e4e7eb')
                ax.title.set_color('#e4e7eb')
            
            # Historical data
            historical_values = data[target_col].values
            historical_indices = range(len(historical_values))
            
            # Plot 1: Historical Data with Trend
            axes[0, 0].plot(historical_indices, historical_values, 
                          'b-', linewidth=2, label='Historical', marker='o', markersize=4)
            
            # Add moving average
            if len(historical_values) >= 5:
                ma = pd.Series(historical_values).rolling(window=5, min_periods=1).mean()
                axes[0, 0].plot(historical_indices, ma, 
                              'g--', linewidth=2, label='5-Period MA', alpha=0.7)
            
            axes[0, 0].set_xlabel('Time Period', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel(target_col, fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Historical Data with Trend', fontsize=12, fontweight='bold', pad=10)
            axes[0, 0].legend(loc='upper left', fontsize=9)
            axes[0, 0].grid(alpha=0.3, linestyle='--')
            
            # Plot 2: LSTM Forecast
            if lstm_forecast is not None and lstm_forecast[0] is not None:
                forecast_vals, conf_lower, conf_upper = lstm_forecast
                forecast_indices = range(len(historical_values), len(historical_values) + len(forecast_vals))
                
                # Plot historical
                axes[0, 1].plot(historical_indices[-20:], historical_values[-20:], 
                              'b-', linewidth=2.5, label='Historical', marker='o', markersize=5)
                
                # Plot forecast
                axes[0, 1].plot(forecast_indices, forecast_vals, 
                              'r--', linewidth=2.5, label='LSTM Forecast', marker='s', markersize=6)
                
                # Confidence interval
                axes[0, 1].fill_between(forecast_indices, conf_lower, conf_upper,
                                       alpha=0.2, color='red', label='95% Confidence')
                
                # Separator
                axes[0, 1].axvline(x=len(historical_values)-1, color='gray', 
                                  linestyle=':', alpha=0.6, linewidth=2)
                
                axes[0, 1].set_xlabel('Time Period', fontsize=11, fontweight='bold')
                axes[0, 1].set_ylabel(target_col, fontsize=11, fontweight='bold')
                axes[0, 1].set_title(f'LSTM Forecast ({len(forecast_vals)} Periods)', 
                                    fontsize=12, fontweight='bold', pad=10)
                axes[0, 1].legend(loc='upper left', fontsize=9)
                axes[0, 1].grid(alpha=0.3, linestyle='--')
            else:
                axes[0, 1].text(0.5, 0.5, 'LSTM Forecast\nNot Available', 
                              ha='center', va='center', fontsize=14, color='#9ca3af',
                              transform=axes[0, 1].transAxes)
            
            # Plot 3: Prophet Forecast
            if prophet_forecast is not None:
                # Get forecast portion only
                forecast_portion = prophet_forecast.tail(self.forecast_horizon)
                
                axes[1, 0].plot(historical_indices[-20:], historical_values[-20:], 
                              'b-', linewidth=2.5, label='Historical', marker='o', markersize=5)
                
                forecast_indices = range(len(historical_values), 
                                       len(historical_values) + len(forecast_portion))
                
                axes[1, 0].plot(forecast_indices, forecast_portion['yhat'].values, 
                              'g--', linewidth=2.5, label='Prophet Forecast', marker='^', markersize=6)
                
                # Confidence interval
                axes[1, 0].fill_between(forecast_indices, 
                                       forecast_portion['yhat_lower'].values,
                                       forecast_portion['yhat_upper'].values,
                                       alpha=0.2, color='green', label='95% Confidence')
                
                # Separator
                axes[1, 0].axvline(x=len(historical_values)-1, color='gray', 
                                  linestyle=':', alpha=0.6, linewidth=2)
                
                axes[1, 0].set_xlabel('Time Period', fontsize=11, fontweight='bold')
                axes[1, 0].set_ylabel(target_col, fontsize=11, fontweight='bold')
                axes[1, 0].set_title(f'Prophet Forecast ({len(forecast_portion)} Periods)', 
                                    fontsize=12, fontweight='bold', pad=10)
                axes[1, 0].legend(loc='upper left', fontsize=9)
                axes[1, 0].grid(alpha=0.3, linestyle='--')
            else:
                axes[1, 0].text(0.5, 0.5, 'Prophet Forecast\nNot Available', 
                              ha='center', va='center', fontsize=14, color='#9ca3af',
                              transform=axes[1, 0].transAxes)
            
            # Plot 4: Comparison
            if lstm_forecast is not None and lstm_forecast[0] is not None and prophet_forecast is not None:
                forecast_vals_lstm = lstm_forecast[0]
                forecast_vals_prophet = prophet_forecast.tail(self.forecast_horizon)['yhat'].values
                
                comparison_indices = range(1, min(len(forecast_vals_lstm), len(forecast_vals_prophet)) + 1)
                
                axes[1, 1].plot(comparison_indices, forecast_vals_lstm[:len(comparison_indices)], 
                              'r-', linewidth=2.5, label='LSTM', marker='s', markersize=6)
                axes[1, 1].plot(comparison_indices, forecast_vals_prophet[:len(comparison_indices)], 
                              'g-', linewidth=2.5, label='Prophet', marker='^', markersize=6)
                
                # Ensemble (average)
                ensemble = (forecast_vals_lstm[:len(comparison_indices)] + 
                           forecast_vals_prophet[:len(comparison_indices)]) / 2
                axes[1, 1].plot(comparison_indices, ensemble, 
                              'b--', linewidth=3, label='Ensemble (Avg)', marker='o', markersize=7)
                
                axes[1, 1].set_xlabel('Forecast Period', fontsize=11, fontweight='bold')
                axes[1, 1].set_ylabel('Predicted ' + target_col, fontsize=11, fontweight='bold')
                axes[1, 1].set_title('Model Comparison & Ensemble', fontsize=12, fontweight='bold', pad=10)
                axes[1, 1].legend(loc='upper left', fontsize=9)
                axes[1, 1].grid(alpha=0.3, linestyle='--')
            else:
                axes[1, 1].text(0.5, 0.5, 'Model Comparison\nNot Available', 
                              ha='center', va='center', fontsize=14, color='#9ca3af',
                              transform=axes[1, 1].transAxes)
            
            plt.suptitle('Advanced Time-Series Forecasting Dashboard', 
                        fontsize=16, fontweight='bold', color='#e4e7eb', y=0.995)
            plt.tight_layout()
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e2337')
            plt.close()
            
            logging.info(f"Forecast visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Forecast visualization creation failed: {e}", exc_info=True)
            return None


def perform_advanced_forecasting(df, target_col='Downtime (minutes)', forecast_periods=7):
    """
    Main function to perform advanced time-series forecasting
    
    Returns:
        - lstm_forecast: LSTM predictions tuple (values, lower, upper)
        - prophet_forecast: Prophet forecast DataFrame
        - ensemble_forecast: Combined forecast
        - chart_path: Path to visualization
    """
    try:
        logging.info("Starting Advanced Time-Series Forecasting...")
        
        forecaster = TimeSeriesForecaster()
        forecaster.forecast_horizon = forecast_periods
        
        lstm_forecast = None
        prophet_forecast = None
        
        # LSTM Forecasting
        if LSTM_AVAILABLE and len(df) >= 20:
            logging.info("Training LSTM model...")
            X_train, y_train, X_test, y_test, scaled_data = forecaster.prepare_lstm_data(df, target_col)
            
            if X_train is not None:
                model, history = forecaster.train_lstm(X_train, y_train, X_test, y_test, epochs=50)
                
                if model is not None:
                    lstm_forecast = forecaster.forecast_lstm(df, target_col, forecast_periods)
        else:
            logging.warning("LSTM forecasting skipped - TensorFlow not available or insufficient data")
        
        # Prophet Forecasting
        if PROPHET_AVAILABLE and len(df) >= 10:
            logging.info("Training Prophet model...")
            
            # Ensure date column exists
            if 'Date' not in df.columns:
                df['Date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
            
            model = forecaster.train_prophet(df, target_col, 'Date')
            
            if model is not None:
                prophet_forecast = forecaster.forecast_prophet(forecast_periods)
        else:
            logging.warning("Prophet forecasting skipped - Prophet not available or insufficient data")
        
        # Create ensemble forecast
        ensemble_forecast = None
        if lstm_forecast is not None and lstm_forecast[0] is not None and prophet_forecast is not None:
            lstm_vals = lstm_forecast[0]
            prophet_vals = prophet_forecast.tail(forecast_periods)['yhat'].values
            
            # Simple average ensemble
            min_len = min(len(lstm_vals), len(prophet_vals))
            ensemble_forecast = (lstm_vals[:min_len] + prophet_vals[:min_len]) / 2
            
            logging.info(f"Ensemble forecast created: {ensemble_forecast}")
        
        # Create visualization
        chart_path = forecaster.create_forecast_visualization(
            df, lstm_forecast, prophet_forecast, target_col
        )
        
        logging.info("Advanced forecasting complete")
        return lstm_forecast, prophet_forecast, ensemble_forecast, chart_path
        
    except Exception as e:
        logging.error(f"Advanced forecasting failed: {e}", exc_info=True)
        return None, None, None, None
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def perform_ml_analysis(df):
    """
    Trains a simple regression model to predict downtime based on units produced and defects.
    Returns:
        - prediction (float): Predicted downtime for the next shift
        - model_score (float): R^2 score of the model
        - plot_path (str): Path to the generated ML visualization
    """
    try:
        # 1. Data Preparation
        # Ensure we have numeric data and drop NaNs
        data = df[['Units Produced', 'Defective Units', 'Downtime (minutes)']].dropna()
        
        # We need at least a few rows to train
        if len(data) < 5:
            logging.warning("Not enough data for ML analysis. Returning defaults.")
            return 0.0, 0.0, None

        X = data[['Units Produced', 'Defective Units']]
        y = data['Downtime (minutes)']

        # 2. Train Model (Simple Linear Regression)
        # In a real scenario, you'd persist this model, but for this lightweight app,
        # retraining on small datasets on-the-fly is fast and seamless.
        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)

        # 3. Predict for "Next Shift" 
        # We use the average of current performance + 10% load as a hypothetical "Next Shift" scenario
        avg_units = data['Units Produced'].mean() * 1.1
        avg_defects = data['Defective Units'].mean() * 1.1
        
        prediction_input = pd.DataFrame([[avg_units, avg_defects]], columns=['Units Produced', 'Defective Units'])
        predicted_downtime = model.predict(prediction_input)[0]

        # 4. Generate Visualization (Actual vs Predicted Regression Line)
        plt.figure(figsize=(10, 5))
        
        # Plotting the relationship between Defects and Downtime as a proxy visualization
        sns.regplot(x='Defective Units', y='Downtime (minutes)', data=data, 
                    scatter_kws={'color': 'blue', 'alpha': 0.6}, line_kws={'color': 'red'})
        
        plt.title(f"ML Insight: Defect Impact on Downtime (Model Accuracy: {score:.2f})")
        plt.xlabel("Defective Units")
        plt.ylabel("Downtime (minutes)")
        plt.tight_layout()
        
        plot_path = "reports/chart_ml_forecast.png"
        plt.savefig(plot_path)
        plt.close()

        return round(predicted_downtime, 2), round(score, 2), plot_path

    except Exception as e:
        logging.error(f"ML Engine Failed: {e}")
        return 0.0, 0.0, None
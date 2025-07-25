import os
import glob
import pandas as pd
import numpy as np
import time  # Import the time module

def calculate_rmse(predictions, actuals):
    """
    Calculate Root Mean Squared Error (RMSE) between predictions and actual values.
    """
    return np.sqrt(((predictions - actuals) ** 2).mean())

def calculate_r_squared(predictions, actuals):
    """
    Custom implementation of R-squared (coefficient of determination).
    Handles NaN values by dropping them.
    """
    # Drop NaN values from predictions and actuals
    valid_mask = ~np.isnan(predictions) & ~np.isnan(actuals)
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    # If no valid data remains, return NaN
    if len(predictions) == 0 or len(actuals) == 0:
        return np.nan

    # Calculate R-squared
    ss_total = ((actuals - actuals.mean()) ** 2).sum()
    ss_residual = ((actuals - predictions) ** 2).sum()
    return 1 - (ss_residual / ss_total)

def generate_metrics_file(data_dir, output_file):
    """
    Generate a metrics file containing RMSE and R-squared for beta forecasts.

    Args:
        data_dir (str): Directory containing the beta forecast files (e.g., 'data/US/factors/').
        output_file (str): Path to save the generated metrics file.
    """
    start_time = time.time()  # Start the timer

    # Find all beta forecast files in the directory
    forecast_files = glob.glob(os.path.join(data_dir, "beta*_forecasts_*.csv"))
    metrics = []  # To store the results

    print(f"Found {len(forecast_files)} files to process...")

    # Process each file
    for file_path in forecast_files:
        # Extract model name and target variable from the filename
        file_name = os.path.basename(file_path)
        target_variable = file_name.split("_")[0]  # e.g., 'beta1', 'beta2', 'beta3'
        model_name = file_name.replace(f"{target_variable}_forecasts_", "").replace(".csv", "")  # Extract model name

        # Load the forecast file
        df = pd.read_csv(file_path)

        # Calculate RMSE and R-squared for each ExecutionDate and Horizon
        grouped = df.groupby(["ExecutionDate", "Horizon"])
        for (execution_date, horizon), group in grouped:
            rmse = calculate_rmse(group["Prediction"], group["Actual"])
            r_squared = calculate_r_squared(group["Prediction"], group["Actual"])
            metrics.append({
                "ExecutionDate": execution_date,
                "Horizon": horizon,
                "Model": model_name,
                "TargetVariable": target_variable,
                "Metric": "RMSE",
                "Value": rmse
            })
            metrics.append({
                "ExecutionDate": execution_date,
                "Horizon": horizon,
                "Model": model_name,
                "TargetVariable": target_variable,
                "Metric": "R-squared",
                "Value": r_squared
            })

    # Convert metrics to a DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_file, index=False)
    print(f"Metrics file saved to {output_file}")

    # End the timer and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to generate metrics file: {elapsed_time:.2f} seconds")

# Example usage
data_dir = r"C:\git\backtest-baam\data\US\factors"  # Directory containing the beta forecast files
output_file = r"C:\git\backtest-baam\data\US\metrics\factors_metrics.csv"  # Output metrics file
generate_metrics_file(data_dir, output_file)
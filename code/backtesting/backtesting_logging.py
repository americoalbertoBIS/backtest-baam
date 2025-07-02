import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import mlflow
import glob 
import re

import os
os.chdir(r'C:\git\backtest-baam\code')

from visualization.matplotlib_plots import plot_forecasts_with_actuals

def clean_model_name(model_name):
    """
    Cleans a model name to make it suitable for file names.

    Args:
        model_name (str): Original model name.

    Returns:
        str: Cleaned model name suitable for file names.
    """
    # Replace special characters and spaces with underscores
    cleaned_name = re.sub(r"[^\w\s]", "_", model_name)  # Replace non-alphanumeric characters with _
    cleaned_name = re.sub(r"\s+", "_", cleaned_name)    # Replace spaces with underscores
    cleaned_name = re.sub(r"_+", "_", cleaned_name)     # Remove multiple underscores
    return cleaned_name.strip("_")  # Remove leading/trailing underscores

def check_existing_results(country, save_dir, target_col, model_name, method_name):
    """
    Checks if the model has already been backtested by looking for its full predictions file
    in the country-specific folder.

    Args:
        country (str): Country name or code.
        save_dir (str): Directory where results are saved.
        target_col (str): Target column for the model.
        model_name (str): Name of the model.
        method_name (str): Name of the backtesting method.

    Returns:
        bool: True if the model has already been backtested, False otherwise.
    """
    # Clean the model name for file naming
    cleaned_model_name = clean_model_name(model_name)

    # Construct the country-specific folder path
    country_save_dir = os.path.join(save_dir, country)

    # Search for the full predictions file pattern in the country folder
    file_pattern = os.path.join(country_save_dir, f"{target_col}_forecasts_{cleaned_model_name}.csv")
    matching_files = glob.glob(file_pattern)

    # Check if any matching file exists
    if matching_files:
        print(f"Found existing predictions file(s) for model '{model_name}', method '{method_name}': {matching_files}")
        return True

    print(f"No existing predictions file found for model '{model_name}', method '{method_name}'.")
    return False

def setup_mlflow(target_col):
    """
    Sets up MLflow for tracking experiments.

    Args:
        target_col (str): Target column for the experiment.
    """
    mlflow.set_tracking_uri(r"sqlite:///C:/git/backtest-baam/mlflow/mlflow.db")
    experiment_name = f"{target_col}_parallel_backtest"
    mlflow.set_experiment(experiment_name)


def log_backtest_results(df, target_col, model_name, method_name, horizons, predictions, actuals, df_predictions=None, save_dir="results"):
    """
    Logs backtest results to MLflow, calculates metrics, and saves predictions and visualizations.

    Args:
        df (pd.DataFrame): Input data.
        target_col (str): Target column.
        model_name (str): Name of the model.
        method_name (str): Name of the backtesting method.
        horizons (list): Forecast horizons.
        predictions (dict): Forecasted values.
        actuals (dict): Actual values.
        df_predictions (pd.DataFrame, optional): DataFrame containing predictions (default is None).
        save_dir (str): Directory to save results (default is "results").

    Returns:
        pd.DataFrame: DataFrame of metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_metrics = []

    for horizon in horizons:
        if predictions[horizon] and actuals[horizon]:
            valid_indices = ~np.isnan(predictions[horizon]) & ~np.isnan(actuals[horizon])
            filtered_predictions = np.array(predictions[horizon])[valid_indices]
            filtered_actuals = np.array(actuals[horizon])[valid_indices]

            if len(filtered_predictions) == 0 or len(filtered_actuals) == 0:
                continue

            mse = mean_squared_error(filtered_actuals, filtered_predictions)
            r2 = r2_score(filtered_actuals, filtered_predictions)
            rmse = mse ** 0.5

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{model_name} - {method_name} - Horizon {horizon} months - {timestamp}"

            with mlflow.start_run(nested=True, run_name=run_name):
                mlflow.log_param("Model", model_name)
                mlflow.log_param("Method", method_name)
                mlflow.log_param("Horizon", horizon)
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("R²", r2)
                mlflow.log_metric("RMSE", rmse)

                df_results = pd.DataFrame({'Actuals': filtered_actuals, 'Predictions': filtered_predictions})
                results_file = os.path.join(save_dir, f"results_{model_name}_{method_name}_{horizon}months_{timestamp}.csv")
                #df_results.to_csv(results_file, index=False)
                #mlflow.log_artifact(results_file)

                metrics = pd.DataFrame({
                    'Model': [model_name],
                    'Method': [method_name],
                    'Horizon': [horizon],
                    'MSE': [mse],
                    'R²': [r2],
                    'RMSE': [rmse],
                    'Timestamp': [timestamp]
                })
                #all_metrics.append(metrics)

    if df_predictions is not None:
        predictions_file = os.path.join(save_dir, f"full_predictions_{model_name}_{method_name}_{timestamp}.csv")
        df_predictions.to_csv(predictions_file, index=False)
        mlflow.log_artifact(predictions_file)

        plot_file = os.path.join(r"C:\git\backtest-baam\graphs", f"forecast_vs_actuals_{model_name}_{method_name}_{timestamp}.png")
        realized_beta = df[target_col]
        plot_forecasts_with_actuals(target_col, df_predictions, realized_beta, model=model_name, save_path=plot_file)
        mlflow.log_artifact(plot_file)

    return None #pd.concat(all_metrics)
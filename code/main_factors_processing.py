

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from tqdm import tqdm

import os 
os.chdir(r'C:\git\backtest-baam\code')

from data_preparation.data_loader import DataLoaderYC
from modeling.yield_curve_modeling import YieldCurveModel

from backtesting.factors_processing import FactorsProcessor

CONFIDENCE_LEVEL = 0.05  # 5% for 95% confidence level
MONTHS_IN_YEAR = 12      # Number of months in a year
SAVE_DIR = r"L:\RMAS\Users\Alberto\backtest-baam\data"
LOG_DIR = r"C:\git\backtest-baam\logs"
MLFLOW_TRACKING_URI = r"sqlite:///C:/git/backtest-baam/mlflow/mlflow.db"

def process_execution_date(country, model_name, model_config, execution_date, yield_curve_model, model_params):
    """
    Process a single execution date for a given country and model.

    Args:
        country (str): The country being processed (e.g., "US").
        model_name (str): The name of the model being processed (e.g., "AR_1").
        model_config (dict): The configuration for the selected model.
        execution_date (datetime): The execution date being processed.
        yield_curve_model (YieldCurveModel): The yield curve model instance.
        model_params (dict): The model parameters.
    """
    #print(f"Processing execution date: {execution_date} for model: {model_name} in country: {country}")
    
    # Create an instance of YieldCurveProcessor
    processor = FactorsProcessor(
        country=country,
        model_name=model_name,
        model_config=model_config,
        execution_date=execution_date,
        yield_curve_model=yield_curve_model,
        model_params=model_params
    )

    # Process the selected beta combination for the current execution date
    processor.load_betas()  # Load forecasted and simulated betas
    processor.compute_simulated_observed_yields()  # Compute yields using simulations
    processor.compute_observed_yields()  # Process observed yields
    processor.compute_predicted_yields()  # Compute predicted yields
    processor.align_observed_and_predicted_yields()  # Align observed and predicted yields
    processor.compute_rmse_r_squared()  # Compute RMSE and R-squared
    processor.compute_var_cvar_vol()  # Compute VaR, CVaR, and returns
    processor.save_results()  # Save results

# =============================================================================
#     # Log results with MLflow
#     run_name = f"{country}_{model_name}"
#     with mlflow.start_run(run_name=run_name):
#         metrics_file = processor.results_dir / f"metrics_timeseries_{model_name}.csv"
#         metrics_df = pd.read_csv(metrics_file)
#         for metric in ["RMSE", "R-Squared"]:
#             for maturity in yield_curve_model.uniqueTaus:
#                 metric_values = metrics_df[
#                     (metrics_df["Metric"] == metric) & (metrics_df["Maturity (Years)"] == maturity)
#                 ]["Value"]
#                 mlflow.log_metric(f"{metric}_Mean_{maturity}y", metric_values.mean())
# 
#         # Log the metrics file as an artifact
#         mlflow.log_artifact(metrics_file)
# =============================================================================
                    


models_configurations = {
     "AR_1": {
         "beta1": "AR_1",
         "beta2": "AR_1",
         "beta3": "AR_1"
     },
    "AR_1_Output_Gap_Direct_Inflation_UCSV": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta2": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta3": "AR_1_Output_Gap_Direct_Inflation_UCSV"
    },
    "Mixed_Model": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta2": "AR_1_Output_Gap_Direct",
        "beta3": "AR_1"
    }
}

#country = 'US'
#model_name = 'AR_1'
#model_config = models_configurations[model_name]

max_workers = max(1, multiprocessing.cpu_count() // 2)

if __name__ == "__main__":
    #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Define the experiment name (e.g., based on the project name)
    #experiment_name = "Yield Curve Backtest"
    #mlflow.set_experiment(experiment_name)  # Set the experiment name

    countries = ['US', 'EA']  # Add other countries if needed (e.g., 'EA', 'UK')
    data_loader = DataLoaderYC(r'L:\RMAS\Resources\BAAM\OpenBAAM\Private\Data\BaseDB.mat')

    for country in countries:
        # Initialize data loader and yield curve model
        _, _, _ = data_loader.load_data()
        if country == 'EA':
            selectedCurveName, selected_curve_data, modelParams = data_loader.process_data('DE')
        else:
            selectedCurveName, selected_curve_data, modelParams = data_loader.process_data(country)
        modelParams.update({'minMaturity': 0.08, 'maxMaturity': 10, 'lambda1fixed': 0.7173})
        yield_curve_model = YieldCurveModel(selected_curve_data, modelParams)

        # Iterate over the selected beta combinations (from models_configurations)
        for model_name, model_config in models_configurations.items():
            print(f"Processing model: {model_name} for country: {country}")

            # Get all execution dates for the current country and model combination
            execution_dates_file = SAVE_DIR + f'/{country}/factors/beta1_forecasts_{model_config["beta1"]}.csv'
            execution_dates = pd.read_csv(execution_dates_file)['ExecutionDate'].unique()

            # Convert execution dates to datetime
            execution_dates = pd.to_datetime(execution_dates)

            # Partition execution dates into non-overlapping chunks for each worker
            execution_dates_chunks = np.array_split(execution_dates, max_workers)

            # Parallelize processing of execution date chunks
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_execution_date,
                        country,
                        model_name,
                        model_config,
                        execution_date,
                        yield_curve_model,
                        modelParams
                    )
                    for chunk in execution_dates_chunks
                    for execution_date in chunk
                ]
            
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Execution Dates"):
                    try:
                        future.result()
                    except Exception as e:
                        #logging.error(f"Error processing execution date: {e}", exc_info=True)
                        print(f"Error processing execution date: {e}")



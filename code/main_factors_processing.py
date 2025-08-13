

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import logging
from tqdm import tqdm
from pathlib import Path
import os 
os.chdir(r'C:\git\backtest-baam\code')

from data_preparation.data_loader import DataLoaderYC
from modeling.yield_curve_modeling import YieldCurveModel
from modeling.evaluation_metrics import calculate_out_of_sample_metrics
from backtesting.factors_processing import FactorsProcessor

CONFIDENCE_LEVEL = 0.05  # 5% for 95% confidence level
MONTHS_IN_YEAR = 12      # Number of months in a year
SAVE_DIR = r"L:\RMAS\Users\Alberto\backtest-baam\data"
LOG_DIR = r"C:\git\backtest-baam\logs"
MLFLOW_TRACKING_URI = r"sqlite:///C:/git/backtest-baam/mlflow/mlflow.db"

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "main_factors_processing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compute_and_save_out_of_sample_metrics(df_predictions, output_dir):
    """
    Compute out-of-sample metrics (e.g., RMSE, R-squared) for all maturities and save them to separate files.

    Args:
        df_predictions (pd.DataFrame): DataFrame containing predictions, actuals, horizons, and execution dates.
        output_dir (str or Path): Directory where the metrics files will be saved.
        model_name (str): Name of the model (used in file naming).
    """
    # Initialize lists to store metrics for all maturities
    outofsample_metrics_by_horizon = []
    outofsample_metrics_by_exec_date = []
    outofsample_metrics = []

    # Iterate over maturities
    for maturity in df_predictions['maturity'].unique():
        # Filter predictions for the current maturity
        temp = df_predictions[df_predictions['maturity'] == maturity].copy()

        # Calculate metrics for the current maturity
        outofsample_metrics_temp = calculate_out_of_sample_metrics(temp)

        # Add maturity as a column to each set of metrics
        outofsample_metrics_temp["by_horizon"]['maturity'] = maturity
        outofsample_metrics_temp["by_execution_date"]['maturity'] = maturity
        outofsample_metrics_temp["by_row"]['maturity'] = maturity

        # Append metrics to the corresponding lists
        outofsample_metrics_by_horizon.append(outofsample_metrics_temp["by_horizon"])
        outofsample_metrics_by_exec_date.append(outofsample_metrics_temp["by_execution_date"])
        outofsample_metrics.append(outofsample_metrics_temp["by_row"])

    # Combine metrics across all maturities
    metrics_by_horizon = pd.concat(outofsample_metrics_by_horizon, ignore_index=True)
    metrics_by_execution_date = pd.concat(outofsample_metrics_by_exec_date, ignore_index=True)
    metrics_by_row = pd.concat(outofsample_metrics, ignore_index=True)

    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics to CSV files
    print("Saving out-of-sample metrics to files...")

    metrics_by_horizon_file = output_dir / f"outofsample_metrics_by_horizon.csv"
    metrics_by_horizon.to_csv(metrics_by_horizon_file, index=False)

    metrics_by_execution_date_file = output_dir / f"outofsample_metrics_by_execution_date.csv"
    metrics_by_execution_date.to_csv(metrics_by_execution_date_file, index=False)

    metrics_by_row_file = output_dir / f"outofsample_metrics_by_row.csv"
    metrics_by_row.to_csv(metrics_by_row_file, index=False)

    print("Out-of-sample metrics saved successfully.")

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
    processor.save_simulated_yields_long_format()
    processor.compute_observed_yields()  # Process observed yields
    processor.compute_predicted_yields()  # Compute predicted yields
    processor.align_observed_and_predicted_yields()  # Align observed and predicted yields

    # calculate and save returns
    processor.calculate_and_save_returns()  # Save monthly and annual returns
    processor.compute_var_cvar_vol()

    aligned_data = pd.concat(
                        [processor.aligned_observed_yields_df.stack(), processor.aligned_predicted_yields_df.stack()],
                        axis=1,
                        keys=["actual", "prediction"]
                        ).dropna()  # Drop rows where either observed or predicted values are missing

    # Construct the predictions DataFrame
    predictions = pd.DataFrame({
        "horizon": (aligned_data.index.get_level_values(0) - execution_date).days // 30,
        "actual": aligned_data["actual"].values,
        "prediction": aligned_data["prediction"].values,
        "execution_date": execution_date,
        "forecasted_date": aligned_data.index.get_level_values(0),
        "maturity": aligned_data.index.get_level_values(1)
    }).reset_index(drop=True)
    
    return predictions
    #processor.compute_rmse_r_squared()  # Compute RMSE and R-squared
    processor.compute_and_save_out_of_sample_metrics() # Compute and save out-of-sample metrics
      # Compute VaR, CVaR, and returns
    #processor.save_results()  # Save results

def main():
    """
    Main function to process factors for multiple countries, models, and execution dates.
    """
    # Initialize data loader
    data_loader = DataLoaderYC(r'L:\RMAS\Resources\BAAM\OpenBAAM\Private\Data\BaseDB.mat')

    # Define the countries and models to process
    countries = ['US']  # Add other countries if needed (e.g., 'UK') , 'EA'
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
    models_configurations = {
        "AR_1": {
            "beta1": "AR_1",
            "beta2": "AR_1",
            "beta3": "AR_1"
        },
    }
    # Determine the number of workers for parallel processing
    max_workers = max(1, multiprocessing.cpu_count() // 2)
    logging.info(f"Using {max_workers} workers for parallel processing.")

    all_predictions = []  # List to store predictions for all execution dates

    for country in countries:
        logging.info(f"Starting processing for country: {country}")

        # Load data for the country
        _, _, _ = data_loader.load_data()
        if country == 'EA':
            selectedCurveName, selected_curve_data, modelParams = data_loader.process_data('DE')
        else:
            selectedCurveName, selected_curve_data, modelParams = data_loader.process_data(country)

        # Update model parameters
        modelParams.update({'minMaturity': 0.08, 'maxMaturity': 10, 'lambda1fixed': 0.7173})
        yield_curve_model = YieldCurveModel(selected_curve_data, modelParams)

        # Iterate over the selected beta combinations (from models_configurations)
        for model_name, model_config in models_configurations.items():
            logging.info(f"Processing model: {model_name} for country: {country}")

            # Get all execution dates for the current country and model combination
            execution_dates_file = Path(SAVE_DIR) / country / "factors" / model_config["beta1"] / "beta1" / "forecasts.csv"
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

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model_name} for {country}"):
                    try:
                        predictions = future.result()
                        all_predictions.append(predictions)  # Collect predictions
                    except Exception as e:
                        logging.error(f"Error in parallel processing: {e}", exc_info=True)

            # Combine all predictions into a single DataFrame
            df_predictions = pd.concat(all_predictions, ignore_index=True)

            compute_and_save_out_of_sample_metrics(df_predictions, Path(SAVE_DIR) / country / "yields" / model_name)

    logging.info("Out-of-sample metrics saved successfully.")

if __name__ == "__main__":
    main()



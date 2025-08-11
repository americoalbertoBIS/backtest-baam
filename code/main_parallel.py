#%%
import os
import logging
from data_preparation.data_loader import DataLoader
from backtesting.backtesting_pipeline import run_all_backtests_parallel  # Updated to use the parallelized version
from backtesting.backtesting_logging import check_existing_results
from backtesting.config_models import models  # Import models configuration

import warnings
warnings.filterwarnings("ignore")

#%%

def setup_logging(log_dir, country):
    """
    Sets up logging to write to both a file and the console.

    Args:
        log_dir (str): Directory to save log files.
        country (str): Country name or code.
    """
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    log_file = os.path.join(log_dir, f"{country}.log")  # Log file for the specific country

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add a file handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Add a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Logs will be written to: {log_file}")

#%%

def main(country, model_name_to_test=None, target_col_to_test=None):
    """
    Main execution pipeline for backtesting for a specific country.
    """
    try:
        # Step 1: Define parameters
        variable_list = ['GDP', 'IP', 'CPI']  # List of macroeconomic variables
        shadow_flag = True  # Whether to use shadow rotation for betas
        save_dir = r"C:\git\backtest-baam\data"  # Directory to save results
        horizons = [6, 12, 24, 36, 48, 60]  # Forecast horizons
        horizons = range(1, max(horizons)+1)
        target_columns = ['beta1', 'beta2', 'beta3']  # Target columns for backtesting
        num_simulations = 1000  # Number of simulations for each execution date
        max_workers = os.cpu_count() // 2  # Use half of the available CPU cores for parallel processing

        # Step 2: Initialize DataLoader
        logging.info("Initializing DataLoader...")
        data_loader = DataLoader(country=country, variable_list=variable_list, shadow=shadow_flag)

        # Step 3: Load combined data
        logging.info("Loading data...")
        df_combined = data_loader.get_data()

        # Step 4: Filter models and target columns
        models_to_run = [model for model in models if model["name"] == model_name_to_test] if model_name_to_test else models[5:]
        target_columns_to_run = [target_col_to_test] if target_col_to_test else target_columns

        # Step 4: Loop through models and target columns
        logging.info("Running backtests...")
        for model_config in models_to_run:
            model_name = model_config["name"]
            logging.info(f"Running backtests for model: {model_name}")

            for target_col in target_columns_to_run:
                #logging.info(f"Checking existing results for target column: {target_col} under model: {model_name}")

                # Step 4.1: Check for existing results
                backtest_type = "Expanding Window"  # Or another method name if applicable
                #if check_existing_results(country, save_dir, target_col, model_name, backtest_type):
                #    logging.info(f"Results already exist for model '{model_name}', target '{target_col}'. Skipping...")
                #    continue  # Skip this combination if results already exist

                # Step 4.2: Run backtests
                logging.info(f"Running backtests for target column: {target_col} under model: {model_name}")
                run_all_backtests_parallel(
                    country=country,
                    df=df_combined,
                    horizons=horizons,
                    target_col=target_col,
                    save_dir=save_dir,
                    model_config=model_config,  # Pass a single model configuration
                    num_simulations=num_simulations,
                    max_workers=max_workers
                )

        logging.info("Pipeline execution completed for country: %s", country)

    except Exception as e:
        logging.error(f"An error occurred during pipeline execution for country {country}: {e}")

#%%

if __name__ == "__main__":
    # Define the countries to process
    countries = ['US'] #  'US', 'UK' EA
    test = False
    if test:
        model_name_to_test = "AR(1) + Inflation (UCSV) - MRM"
        target_col_to_test = "beta3"
    else:
        model_name_to_test = None
        target_col_to_test = None

    # Define the log directory
    log_dir = r"C:\git\backtest-baam\logs"

    # Run the pipeline for each country
    for country in countries:
        setup_logging(log_dir, country)  # Initialize logging for the specific country
        main(country,
             model_name_to_test=model_name_to_test, 
             target_col_to_test=target_col_to_test)
# %%
# mlflow ui --backend-store-uri sqlite:///C:/git/backtest-baam/mlflow/mlflow.db --port 8000
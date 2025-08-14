#%%
import os
import logging
from data_preparation.data_loader import DataLoader
from backtesting.backtesting_pipeline import run_all_backtests
from backtesting.config_models import models  # Import models configuration

import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#%%

def main():
    """
    Main execution pipeline for backtesting.
    """
    try:
        # Step 1: Define parameters
        country = 'US'
        variable_list = ['GDP', 'IP', 'CPI']  # List of macroeconomic variables
        shadow_flag = True  # Whether to use shadow rotation for betas
        save_dir = r"C:\git\backtest-baam\data"  # Directory to save results
        horizons = [6, 12, 24, 36, 48, 60]  # Forecast horizons
        target_columns = ['beta1', 'beta2', 'beta3']  # Target columns for backtesting

        # Step 2: Initialize DataLoader
        logger.info("Initializing DataLoader...")
        data_loader = DataLoader(country=country, variable_list=variable_list, shadow=shadow_flag)

        # Step 3: Load combined data
        logger.info("Loading data...")
        df_combined = data_loader.get_data()

        # Step 4: Run backtests for each target column
        logger.info("Running backtests...")
        for target_col in target_columns:
            logger.info(f"Running backtests for target column: {target_col}")
            run_all_backtests(country, df_combined, horizons, target_col, save_dir=save_dir, models=models)

        logger.info("Pipeline execution completed.")

    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")

#%%
main()

#%%
if __name__ == "__main__":
    main()
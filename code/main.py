import os
os.chdir(r'C:\git\backtest-baam\code')

import pandas as pd

from data_preparation.data_loader import DataLoader
from backtesting.backtesting_pipeline import run_all_backtests

def main():
    """
    Main execution pipeline for backtesting.
    """
    # Step 1: Define parameters
    country = 'US'
    variable_list = ['GDP', 'IP', 'CPI']  # List of macroeconomic variables
    shadow_flag = True  # Whether to use shadow rotation for betas
    save_dir = r"K:\RMAS\Users\Alberto\BacktestBAAM\data\temp"  # Directory to save results
    horizons = [6, 12, 24, 36, 48, 60]  # Forecast horizons
    target_columns = ['beta1', 'beta2', 'beta3']  # Target columns for backtesting

    # Step 2: Initialize DataLoader
    print("Initializing DataLoader...")
    data_loader = DataLoader(country=country, variable_list=variable_list, shadow=shadow_flag)

    # Step 3: Load combined data
    print("Loading data...")
    df_combined = data_loader.get_data()

    # Step 4: Run backtests for each target column
    print("Running backtests...")
    for target_col in target_columns:
        print(f"Running backtests for target column: {target_col}")
        run_all_backtests(country, df_combined, horizons, target_col, save_dir=save_dir)

    # Step 5: Reconstruct yields and evaluate forecasts
    #print("Reconstructing yields and evaluating forecasts...")
    #tau = 1.0  # Example value for tau, adjust as needed
    #maturities = [1, 2, 3, 5, 7, 10, 20, 30]  # Example maturities

    # Read predictions for all betas
    #df_predictions_beta1 = pd.read_csv(os.path.join(save_dir, f'full_predictions_{country}_beta1_shadow.csv'))
    #df_predictions_beta2 = pd.read_csv(os.path.join(save_dir, f'full_predictions_{country}_beta2_shadow.csv'))
    #df_predictions_beta3 = pd.read_csv(os.path.join(save_dir, f'full_predictions_{country}_beta3_shadow.csv'))

    # Combine predictions for further analysis (if needed)
    # Example: df_all_predictions = [df_predictions_beta1, df_predictions_beta2, df_predictions_beta3]

    print("Pipeline execution completed.")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 14:13:32 2025

@author: al005366
"""

import os
os.chdir(r'C:\git\backtest-baam\code')
import logging
from data_preparation.data_loader import DataLoader
from backtesting.backtesting_pipeline import run_all_backtests_parallel  # Updated to use the parallelized version
from backtesting.backtesting_logging import check_existing_results
from backtesting.config_models import models  # Import models configuration
from backtesting.backtesting_pipeline import generate_execution_dates
from backtesting.backtesting_logging import clean_model_name
from config_paths import QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH
from data_preparation.conensus_forecast import ConsensusForecast

import pandas as pd
import numpy as np

def generate_bootstrap_indices(
    available_dates, num_simulations, max_horizon, 
    bootstrap_type="iid", block_length=6, half_life=12
):
    """
    Generate bootstrapped indices for different bootstrapping types.
    """
    indices = []
    n = len(available_dates)
    if bootstrap_type == "iid":
        for _ in range(num_simulations):
            indices.append(np.random.choice(available_dates, size=max_horizon, replace=True))
    elif bootstrap_type == "block":
        for _ in range(num_simulations):
            sim_indices = []
            i = 0
            while i < max_horizon:
                start = np.random.randint(0, n - block_length + 1)
                block = available_dates[start:start+block_length]
                sim_indices.extend(block)
                i += block_length
            indices.append(sim_indices[:max_horizon])
    elif bootstrap_type == "half_life":
        weights = np.exp(-np.log(2) * np.arange(n)[::-1] / half_life)
        weights /= weights.sum()
        for _ in range(num_simulations):
            indices.append(np.random.choice(available_dates, size=max_horizon, replace=True, p=weights))
    else:
        raise ValueError("Unknown bootstrap_type")
    return np.array(indices)

def generate_and_save_bootstrap_indices(
    df, execution_dates, num_simulations, max_horizon, save_path,
    bootstrap_type="iid", block_length=6, half_life=12, save_csv = True
):
    """
    For each execution date, generate a long-format CSV of bootstrapped indices.
    Columns: ExecutionDate, SimulationID, Horizon, BootDate
    """
    records = []
    for exec_date in execution_dates:
        available_dates = df.loc[:exec_date].index
        boot_indices = generate_bootstrap_indices(
            available_dates, num_simulations, max_horizon, 
            bootstrap_type=bootstrap_type, block_length=block_length, half_life=half_life
        )
        for sim in range(num_simulations):
            for h, boot_date in enumerate(boot_indices[sim], 1):
                records.append({
                    "ExecutionDate": exec_date,
                    "SimulationID": sim,
                    "Horizon": h,
                    "BootDate": boot_date
                })
    df_boot = pd.DataFrame(records)
    if save_csv:
        df_boot.to_csv(save_path, index=False)
    return df_boot

country = 'US'
# Step 1: Define parameters
variable_list = ['GDP', 'IP', 'CPI']  # List of macroeconomic variables
shadow_flag = True  # Whether to use shadow rotation for betas
save_dir = r"C:\git\backtest-baam\data"  # Directory to save results
horizons = [6, 12, 24, 36, 48, 60]  # Forecast horizons
horizons = range(1, max(horizons)+1)
target_columns = ['beta1', 'beta2', 'beta3']  # Target columns for backtesting
num_simulations = 1000  # Number of simulations for each execution date
max_workers = os.cpu_count() // 2  # Use half of the available CPU cores for parallel processing
model_name_to_test=None
target_col_to_test=None

# Step 2: Initialize DataLoader
logging.info("Initializing DataLoader...")
data_loader = DataLoader(country=country, variable_list=variable_list, shadow=shadow_flag)

# Step 3: Load combined data
logging.info("Loading data...")
df_combined = data_loader.get_data()

# Step 4: Filter models and target columns
models_to_run = [model for model in models if model["name"] == model_name_to_test] if model_name_to_test else models
target_columns_to_run = [target_col_to_test] if target_col_to_test else target_columns

# Step 4: Loop through models and target columns
logging.info("Running backtests...")
for model_config in models_to_run:
    model_name = model_config["name"]
    logging.info(f"Running backtests for model: {model_name}")
            
    if model_config['params'].get("macro_forecast") == "consensus":
        consensus_forecast = ConsensusForecast(QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH)
        df_consensus_gdp, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} GDP")
        df_consensus_inf, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} INF")
    else:
        df_consensus_gdp, df_consensus_inf = None, None
    
    # --- 2. Generate execution dates for this model ---
    execution_dates = generate_execution_dates(
        data=df_combined,
        consensus_df=df_consensus_gdp if model_config['params'].get("macro_forecast") == "consensus" else None,
        execution_date_column="forecast_date" if model_config['params'].get("macro_forecast") == "consensus" else None,
        min_years=3,
        macro_forecast=model_config['params'].get("macro_forecast")
    )
    
    bootstrap_csv_path = os.path.join(
        save_dir, country, "factors", clean_model_name(model_name), "bootstrapped_indices.csv"
    )
    import time
    start_time = time.time()
    df_boot = generate_and_save_bootstrap_indices(
        df=df_combined,
        execution_dates=execution_dates,
        num_simulations=num_simulations,
        max_horizon=max(horizons),
        save_path=bootstrap_csv_path,
        bootstrap_type="iid",  # or "iid", "half_life" block
        block_length=None,
        half_life=None
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    df_boot_block = generate_and_save_bootstrap_indices(
        df=df_combined,
        execution_dates=execution_dates,
        num_simulations=num_simulations,
        max_horizon=max(horizons),
        save_path=bootstrap_csv_path,
        bootstrap_type="block",  # or "iid", "half_life" block
        block_length=12,
        half_life=None,
        save_csv = False
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    df_boot_hl = generate_and_save_bootstrap_indices(
        df=df_combined,
        execution_dates=execution_dates,
        num_simulations=num_simulations,
        max_horizon=max(horizons),
        save_path=bootstrap_csv_path,
        bootstrap_type="half_life",  # or "iid", "half_life" block
        block_length=None,
        half_life=36,
        save_csv = False
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

from modeling.time_series_modeling import AR1Model

ar_1_model = AR1Model()

ar_1fit = ar_1_model.fit(df_combined, 'beta1')

boot_df = pd.read_csv(bootstrap_csv_path, parse_dates=["ExecutionDate", "BootDate"])
boot_df_exec = df_boot_block[df_boot_block["ExecutionDate"] == '2025-01-01']

bootstrapped_errors = np.random.choice(ar_1fit.resid, size=(num_simulations, max(horizons)), replace=True)
bootstrapped_errors_matrix = np.zeros((num_simulations, max(horizons)))

for sim in range(num_simulations):
    sim_boot_dates = boot_df_exec[boot_df_exec["SimulationID"] == sim].sort_values("Horizon")["BootDate"]
    bootstrapped_errors_matrix[sim, :] = ar_1fit.resid.reindex(sim_boot_dates.values).values

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

flattened_data = bootstrapped_errors_matrix.flatten()

# Plot the distribution
sns.histplot(flattened_data, kde=True, bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Flattened 2D Array')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

bootstrap_dates_exec = boot_df_exec 
#bootstrapped_errors = np.zeros(max(horizons))
sim_id = 300
sim_boot_dates = bootstrap_dates_exec[bootstrap_dates_exec["SimulationID"] == sim_id].sort_values("Horizon")["BootDate"]
bootstrapped_errors = ar_1fit.resid.reindex(sim_boot_dates.values).values

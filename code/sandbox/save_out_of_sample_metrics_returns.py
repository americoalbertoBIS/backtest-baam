# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 17:32:47 2025

@author: al005366
"""

import pandas as pd
import os
os.chdir(r'C:\git\backtest-baam\code')
from backtesting.config_models import models_configurations
from main_factors_processing import compute_and_save_out_of_sample_metrics

SAVE_DIR = r'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data_joint'
from pathlib import Path

countries = ['UK']
models = list(models_configurations.keys())

for country in countries:
    print(country)
    for model_name in models:
        print(model_name)
        
        df = pd.read_csv(rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data_joint\{country}\returns\estimated_returns\{model_name}\annual_metrics.csv')
        #df = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual_metrics.csv')
        
        selected_metric = ['Expected Annual Returns', 'Observed Annual Return']
        df_sel = df[df['metric'].isin(selected_metric)]
        
        df_preprocessed = df_sel.pivot_table(
            index=['maturity_years', 'execution_date', 'horizon_years'],
            columns='metric',
            values='value'
        ).reset_index()
        
        df_preprocessed = df_preprocessed.rename(
            columns={
                'Observed Annual Return': 'actual',
                'Expected Annual Returns': 'prediction'
            }
        )
        
        # Drop rows where "actual" or "prediction" is NaN
        df_preprocessed = df_preprocessed.dropna(subset=['actual', 'prediction'])
        
        # Step 2: Add "forecast_date" column
        # Convert execution_date to datetime if not already
        df_preprocessed['execution_date'] = pd.to_datetime(df_preprocessed['execution_date'])
        
        # Calculate forecast_date as execution_date + horizon_years (converted to days)
        df_preprocessed['forecast_date'] = df_preprocessed['execution_date'] + pd.to_timedelta(
            df_preprocessed['horizon_years'] * 365, unit='D'
        )
        
        df_preprocessed = df_preprocessed.rename(columns={'maturity_years': 'maturity', 'horizon_years': 'horizon'})
        
        # Step 3: Calculate RMSE and other metrics using the provided functions
        compute_and_save_out_of_sample_metrics(df_preprocessed, Path(SAVE_DIR) / country / "returns" / "estimated_returns" / model_name)
        
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Define the countries, models, and selected maturities
selected_maturities = [0.25, 2, 5, 10]
selected_country = 'US'  # Change to 'EA' if needed
save_dir = Path(SAVE_DIR) / selected_country / "returns" / "estimated_returns"

# Step 2: Initialize a DataFrame to store RMSE results across models
rmse_data = pd.DataFrame()
models = ['AR_1','Mixed_Model', 'Mixed_Model_curvMacro', 'AR_1_Output_Gap_Direct_Inflation_UCSV']

# Step 3: Loop through models and load the RMSE by horizon file
for model_name in models:
    metrics_file = save_dir / model_name / "outofsample_metrics_by_horizon.csv"
    if metrics_file.exists():
        metrics_by_horizon = pd.read_csv(metrics_file)
        metrics_by_horizon['model'] = model_name  # Add the model name as a column
        rmse_data = pd.concat([rmse_data, metrics_by_horizon], ignore_index=True)

# Step 4: Filter data for the selected maturities
filtered_data = rmse_data[rmse_data['maturity'].isin(selected_maturities)]

# Step 5: Create subplots for each maturity
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=False, sharey=False)
axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

# Step 6: Plot RMSE by horizon for each maturity in a separate subplot
for i, maturity in enumerate(selected_maturities):
    ax = axes[i]
    maturity_data = filtered_data[filtered_data['maturity'] == maturity]
    
    for model_name in maturity_data['model'].unique():
        model_data = maturity_data[maturity_data['model'] == model_name]
        ax.plot(
            model_data['horizon'], 
            model_data['rmse'], 
            label=f"Model {model_name}", 
            marker='o'
        )
    
    ax.set_title(f"Maturity: {maturity} years", fontsize=12)
    ax.set_xlabel("Horizon (years)", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

# Step 7: Adjust layout and show the plot
fig.suptitle(f"RMSE by Horizon for Selected Maturities ({selected_country})", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the suptitle
plt.show()
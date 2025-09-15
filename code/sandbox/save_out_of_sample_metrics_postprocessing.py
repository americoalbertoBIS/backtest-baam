# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:38:24 2025

@author: al005366
"""

import pandas as pd
import os
os.chdir(r'C:\git\backtest-baam\code')
#from modeling.evaluation_metrics import calculate_out_of_sample_metrics
from backtesting.config_models import models_configurations
from main_factors_processing import compute_and_save_out_of_sample_metrics

SAVE_DIR = r'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data_joint'
from pathlib import Path
    
countries = ['EA', 'US']
models = list(models_configurations.keys())

for country in countries:
    print(country)
    for model_name in models:
        print(model_name)
        
        data = pd.read_csv(rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data_joint\{country}\yields\estimated_yields\{model_name}\forecasts.csv')

        mean_simulated_yields_and_actual_aligned = data.dropna(subset=["mean_simulated", "actual"]).copy()
        mean_simulated_yields_and_actual_aligned = mean_simulated_yields_and_actual_aligned.rename(columns={"mean_simulated": "prediction"})
        
        # Optionally, keep only required columns for metrics
        df_predictions = mean_simulated_yields_and_actual_aligned[["horizon", "actual", 
                                                                   "prediction", "execution_date", 
                                                                   "forecast_date", "maturity"]].reset_index(drop=True)

            
        compute_and_save_out_of_sample_metrics(df_predictions, Path(SAVE_DIR) / country / "yields" / "estimated_yields" / model_name)

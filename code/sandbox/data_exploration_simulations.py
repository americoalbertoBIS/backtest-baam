# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 09:18:07 2025

@author: al005366
"""

import pandas as pd

df_sim = pd.read_parquet(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\factors\AR_1_Output_Gap_Direct_Inflation_UCSV\beta1\simulations.parquet', 
                         engine = 'pyarrow')


execution_dates = df_sim['execution_date'].unique()

execution_date = execution_dates[200]

df_sim_exec_date = df_sim[df_sim['execution_date']==execution_date].copy()
df_sim_exec_date = df_sim_exec_date.sort_values(by=['simulation_id','forecast_date'])

df_pred = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\factors\AR_1_Output_Gap_Direct_Inflation_UCSV\beta1\forecasts.csv')
df_pred['execution_date'] = pd.to_datetime(df_pred['execution_date'])
df_pred['forecast_date'] = pd.to_datetime(df_pred['forecast_date'])
df_pred_exec_date = df_pred[df_pred['execution_date']==execution_date].copy()
df_pred_exec_date = df_pred_exec_date.dropna()
df_sim_t = df_sim_exec_date.pivot(index='forecast_date', columns='simulation_id', values='simulated_value')

import os
os.chdir(r'C:\git\backtest-baam\code')

from data_preparation.data_loader import DataLoader

country = 'US'
variable_list = None  # List of macroeconomic variables
shadow_flag = True  # Whether to use shadow rotation for betas
data_loader = DataLoader(country=country, variable_list=variable_list, shadow=shadow_flag)
df_betas = data_loader.get_betas()

# Extract the starting value from df_betas
starting_value = df_betas['beta1'][df_betas.index == execution_date].iloc[0]

# Define the earlier date (e.g., the date corresponding to execution_dates[-1])
earlier_date = execution_date

# Create the new row as a dataframe with the earlier date as the index
new_row = pd.DataFrame(
    data=[[starting_value] * df_sim_t.shape[1]],  # Create a single row with the starting value
    index=[earlier_date],                        # Set the index as the earlier date
    columns=df_sim_t.columns                     # Use the same column names as df_sim_t
)

# Concatenate the new row and the original dataframe
df_sim_t = pd.concat([new_row, df_sim_t])


import matplotlib.pyplot as plt

# Set the figure size to make the plot larger
plt.figure(figsize=(10, 6))  # Width=12 inches, Height=8 inches

# Plot the 1000 lines in light grey
for column in df_sim_t.columns:
    plt.plot(df_sim_t.index, df_sim_t[column], color='lightgrey', linewidth=0.5)

# Plot the prediction line in black
plt.plot(df_pred_exec_date['forecast_date'], df_pred_exec_date['prediction'], color='black', linewidth=2, label='prediction')
plt.plot(df_betas['beta1'][(df_betas.index >= execution_date)&(df_betas.index <= df_pred_exec_date['forecast_date'].iloc[-1])], label='actual')
# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add labels, title, and legend
plt.xlabel('Date', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title(f'Execution date = {str(execution_date).split(" ")[0]}', fontsize=16)
plt.legend(fontsize=12)

# Show the plot
plt.show()


plt.plot(df_pred_exec_date['forecast_date'], df_pred_exec_date['prediction'], color='black', linewidth=2, label='prediction')
plt.plot(df_betas['beta1'])



# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:20:10 2025

@author: al005366
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data\US\yields\US_observed_yields_forecasts_AR_1.csv')
# Convert forecasted_date and execution_date to datetime for proper sorting and grouping
df['forecasted_date'] = pd.to_datetime(df['forecasted_date'])
df['execution_date'] = pd.to_datetime(df['execution_date'])

# Sort the DataFrame by forecasted_date
#df = df.sort_values(by='forecasted_date')

# Get unique maturities
maturities = df['maturity'].unique()

# Plot predictions for each maturity and execution_date group
for maturity in maturities:
    subset = df[df['maturity'] == maturity]
    # Create a plot
    plt.figure(figsize=(10, 6))

    # Group by execution_date to avoid connecting unrelated points
    for execution_date, group in subset.groupby('execution_date'):
        plt.plot(group['forecasted_date'], group['prediction'], label=f"{maturity} - {execution_date}", 
                 color='grey', alpha = 0.3)
    
    plt.plot(subset.groupby('forecasted_date')['actual'].last(), color = 'k')
    # Add labels, legend, and title
    plt.xlabel('Forecasted Date')
    plt.ylabel('Prediction')
    plt.title(f'Predictions for {maturity}')
    #plt.legend(title='Maturity and Execution Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
maturity = '0.25 years'
subset = df[df['maturity'] == maturity].copy()
subset = subset[subset['execution_date']>'1989-09-01']
# Create a plot
plt.figure(figsize=(10, 6))

# Group by execution_date to avoid connecting unrelated points
for execution_date, group in subset.groupby('execution_date'):
    plt.plot(group['forecasted_date'], group['prediction'], label=f"{maturity} - {execution_date}", 
             color='grey', alpha = 0.3)

plt.plot(subset.groupby('forecasted_date')['actual'].last(), color = 'k')
# Add labels, legend, and title
plt.xlabel('Forecasted Date')
plt.ylabel('Prediction')
plt.title(f'Predictions for {maturity}')
#plt.legend(title='Maturity and Execution Date', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
        
df = pd.read_csv(r'C:\git\backtest-baam\data\US\beta1_forecasts_AR_1.csv')
# Convert forecasted_date and execution_date to datetime for proper sorting and grouping
df['ForecastDate'] = pd.to_datetime(df['ForecastDate'])
df['ExecutionDate'] = pd.to_datetime(df['ExecutionDate'])

plt.figure(figsize=(10, 6))

# Group by execution_date to avoid connecting unrelated points
for execution_date, group in df.groupby('ExecutionDate'):
    plt.plot(group['ForecastDate'], group['Prediction'], 
             color='grey', alpha = 0.3)

plt.plot(df.groupby('ForecastDate')['Actual'].last(), color = 'k')
# Add labels, legend, and title
plt.xlabel('Forecasted Date')
plt.ylabel('Prediction')
plt.title('Predictions for Beta 1')
#plt.legend(title='Maturity and Execution Date', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

output_file = r"L:\RMAS\Users\Alberto\backtest-baam\data\backtest_results_all_countries_indicators.csv"
df = pd.read_csv(output_file)
df['forecastedDate'] = pd.to_datetime(df['forecastedDate'])
df['executionDate'] = pd.to_datetime(df['executionDate'])
df_str = df[(df['indicator']=='STR')&(df['country']=='US')].copy()

plt.figure(figsize=(10, 6))

# Group by execution_date to avoid connecting unrelated points
for execution_date, group in df_str.groupby('executionDate'):
    plt.plot(group['forecastedDate'], group['AR1_fcst'], 
             color='grey', alpha = 0.3)

plt.plot(df_str.groupby('forecastedDate')['actual'].last(), color = 'k')
# Add labels, legend, and title
plt.xlabel('Forecasted Date')
plt.ylabel('Prediction')
plt.title('Predictions for STR - AR1_fcst')
#plt.legend(title='Maturity and Execution Date', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Group by execution_date to avoid connecting unrelated points
for execution_date, group in df_str.groupby('executionDate'):
    plt.plot(group['forecastedDate'], group['consensus_fcst'], 
             color='grey', alpha = 0.3)

plt.plot(df_str.groupby('forecastedDate')['actual'].last(), color = 'k')
# Add labels, legend, and title
plt.xlabel('Forecasted Date')
plt.ylabel('Prediction')
plt.title('Predictions for STR - consensus_fcst')
#plt.legend(title='Maturity and Execution Date', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

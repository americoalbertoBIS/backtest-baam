# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:37:39 2025

@author: al005366
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set working directory
os.chdir(r'C:\git\yield-curve-app\src')

# Import required modules
from utils.data_processing import DataLoader
from utils.modeling import YieldCurveModel
from utils.nelson_siegel import compute_nsr_shadow_ts_noErr

# Load data
data_loader = DataLoader(r'\\msfsshared\bnkg\RMAS\Resources\BAAM\OpenBAAM\Private\Data\BaseDB.mat')
AllCalcData, countries, sdr_countries = data_loader.load_data()
selectedCurveName, selected_curve_data, modelParams = data_loader.process_data('US')

# Configure model parameters
modelParams.update({'minMaturity': 0.08, 'maxMaturity': 10, 'lambda1fixed': 0.7173})
yield_curve_model = YieldCurveModel(selected_curve_data, modelParams)

# Load forecast and simulation data
df_pred_beta1 = pd.read_csv(r'C:\git\backtest-baam\data\US\beta1_forecasts_AR_1_GDP_Inflation_UCSV.csv')
df_pred_beta2 = pd.read_csv(r'C:\git\backtest-baam\data\US\beta2_forecasts_AR_1_GDP_Inflation_UCSV.csv')
df_pred_beta3 = pd.read_csv(r'C:\git\backtest-baam\data\US\beta3_forecasts_AR_1_GDP_Inflation_UCSV.csv')

df_sim_beta1 = pd.read_parquet(r'C:\git\backtest-baam\data\US\beta1_simulations_AR_1_GDP_Inflation_UCSV.parquet', engine='pyarrow')
df_sim_beta2 = pd.read_parquet(r'C:\git\backtest-baam\data\US\beta2_simulations_AR_1_GDP_Inflation_UCSV.parquet', engine='pyarrow')
df_sim_beta3 = pd.read_parquet(r'C:\git\backtest-baam\data\US\beta3_simulations_AR_1_GDP_Inflation_UCSV.parquet', engine='pyarrow')

execution_dates = df_sim_beta1['ExecutionDate'].unique()
execution_date = execution_dates[0]

# Helper function to subset dataframe
def subset_dataframe(df, execution_date):
    df['ExecutionDate'] = pd.to_datetime(df['ExecutionDate'])
    df['ForecastDate'] = pd.to_datetime(df['ForecastDate'])
    return df[df['ExecutionDate'] == execution_date].copy()

# Subset data for the specific execution date
df_sub_beta1 = subset_dataframe(df_pred_beta1, execution_date)
df_sub_beta2 = subset_dataframe(df_pred_beta2, execution_date)
df_sub_beta3 = subset_dataframe(df_pred_beta3, execution_date)

# Create rotated betas for baseline projection
rotated_betas = np.array([
    df_sub_beta1['Prediction'].dropna().values,
    df_sub_beta2['Prediction'].dropna().values,
    df_sub_beta3['Prediction'].dropna().values
]).T

# Compute baseline observed yields
modelParams['lambda'] = 0.7173
observed_yields_est = compute_nsr_shadow_ts_noErr(
    rotated_betas, yield_curve_model.uniqueTaus, yield_curve_model.invRotationMatrix, modelParams
)

# Create observed yields dataframe for baseline
observed_yields_est_df = pd.DataFrame(
    observed_yields_est,
    index=df_sub_beta1['ForecastDate'].dropna(),
    columns=[f'{tau} years' for tau in yield_curve_model.uniqueTaus]
)

# Process observed yields
dates_num = selected_curve_data['Dates'][0][0]
dates_str = [
    datetime.strftime(datetime.fromordinal(int(d)) - timedelta(days=366), '%Y-%m-%d')
    for d in dates_num
]
observed_yields_df = pd.DataFrame(
    yield_curve_model.yieldsObservedAgg,
    columns=[f'{tau} years' for tau in yield_curve_model.uniqueTaus],
    index=dates_str[-len(yield_curve_model.yieldsObservedAgg):]
)
observed_yields_df.index = pd.to_datetime(observed_yields_df.index)
observed_yields_df_resampled = observed_yields_df.resample('MS').mean()

# Align observed and estimated yields
overlapping_dates = observed_yields_df_resampled.index.intersection(observed_yields_est_df.index)
aligned_observed_yields_df = observed_yields_df_resampled.loc[overlapping_dates]
aligned_observed_yields_est_df = observed_yields_est_df.loc[overlapping_dates]

# Process simulations for the execution date
def filter_and_pivot_simulated_betas(df_sim, execution_date):
    df_sim = df_sim[df_sim['ExecutionDate'] == execution_date].copy()
    df_sim['ForecastDate'] = pd.to_datetime(df_sim['ForecastDate'])
    return df_sim.pivot(index='ForecastDate', columns='SimulationID', values='SimulatedValue')

sim_beta1_pivot = filter_and_pivot_simulated_betas(df_sim_beta1, execution_date)
sim_beta2_pivot = filter_and_pivot_simulated_betas(df_sim_beta2, execution_date)
sim_beta3_pivot = filter_and_pivot_simulated_betas(df_sim_beta3, execution_date)

# Combine simulations into a 3D array
simulated_betas = np.array([
    sim_beta1_pivot.values,
    sim_beta2_pivot.values,
    sim_beta3_pivot.values
]).transpose(1, 0, 2)  # Shape: (forecast dates, 3 betas, simulations)

# Compute observed yields for all simulations
simulated_observed_yields = [
    compute_nsr_shadow_ts_noErr(
        simulated_betas[:, :, sim_id],
        yield_curve_model.uniqueTaus,
        yield_curve_model.invRotationMatrix,
        modelParams
    )
    for sim_id in range(simulated_betas.shape[2])
]

# Combine simulated observed yields into a MultiIndex DataFrame
simulated_observed_yields_df = pd.DataFrame(
    np.stack(simulated_observed_yields, axis=-1).reshape(len(sim_beta1_pivot.index), -1),  # Flatten simulations
    index=sim_beta1_pivot.index,
    columns=pd.MultiIndex.from_product(
        [yield_curve_model.uniqueTaus, range(simulated_betas.shape[2])],  # Maturity and SimulationID
        names=["Maturity", "SimulationID"]
    )
)

# Plot observed, estimated, and simulation data
fig, axes = plt.subplots(5, 3, figsize=(20, 20))
axes = axes.flatten()

for i, column in enumerate(aligned_observed_yields_df.columns):
    ax = axes[i]
    
    # Convert column name to numeric format for matching maturity
    column_numeric = float(column.split()[0])
    
    # Extract all simulations for the current maturity
    simulations_for_maturity = simulated_observed_yields_df.xs(column_numeric, level="Maturity", axis=1)
    
    # Plot simulations as light grey lines
    for simulation_id in simulations_for_maturity.columns:
        ax.plot(simulations_for_maturity.index, simulations_for_maturity[simulation_id], color='lightgrey', linewidth=0.5, alpha=0.6)
    
    # Plot the average of simulations as a blue line
    simulation_avg = simulations_for_maturity.mean(axis=1)  # Compute the mean across simulations
    ax.plot(simulation_avg.index, simulation_avg, color='blue', linewidth=1.5, label='Simulation Avg')
    
    # Plot the baseline projection in black
    ax.plot(aligned_observed_yields_est_df.index, aligned_observed_yields_est_df[column], color='black', linewidth=2, label='Baseline')
    
    # Plot the observed yields in red
    ax.plot(aligned_observed_yields_df.index, aligned_observed_yields_df[column], color='red', linewidth=1.5, label='Observed')
    
    # Add titles and labels
    ax.set_title(column, fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Yield', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

# Remove unused subplots
for j in range(len(aligned_observed_yields_df.columns), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f'Simulations, Baseline, and Observed Yields for Execution Date = {execution_date}', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:22:04 2025

@author: al005366
"""

import pandas as pd
base_path = r'L:\RMAS\Users\Alberto\backtest-baam\data\US\returns\estimated_returns'
data_1  = pd.read_parquet(fr'{base_path}\AR_1\annual\2.0_years\simulations_01122019.parquet')
data_2  = pd.read_parquet(fr'{base_path}\Mixed_Model\annual\2.0_years\simulations_01122019.parquet')
data_3  = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data\US\returns\estimated_returns\AR_1\annual_metrics.csv')

data_bench = data_3[(data_3['Maturity (Years)']==2)&(data_3['Execution Date']=='2019-12-01')&(data_3['Metric']=='Observed Annual Return')].drop_duplicates()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure
plt.figure(figsize=(12, 8))

# Get unique horizons
horizons = data_1['Horizon (Years)'].unique()
horizons = [5]
# Loop through each horizon and plot KDE for both dataframes
for horizon in horizons:
    # Filter data for the current horizon
    data_1_filtered = data_1#[data_1['Horizon (Years)'] == horizon]
    data_2_filtered = data_2#[data_2['Horizon (Years)'] == horizon]
    
    sns.kdeplot(data_bench['Value'], label='Observed Annual Returns over the 5 year horizon', bw_adjust=1)
    
    # Plot KDE for data_1
    sns.kdeplot(data_1_filtered['AnnualReturn'], label=f'Data 1 - Horizon {horizon} Years', bw_adjust=1)
    
    # Plot KDE for data_2
    sns.kdeplot(data_2_filtered['AnnualReturn'], label=f'Data 2 - Horizon {horizon} Years', bw_adjust=1)

# Add title and labels
plt.title('KDE of Annual Returns by Horizon (Years)', fontsize=16)
plt.xlabel('Annual Return', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

plt.figure(figsize=(12, 8))

data_1_filtered = data_1#[data_1['Horizon (Years)'] == horizon]
data_2_filtered = data_2#[data_2['Horizon (Years)'] == horizon]

# Plot KDE for data_1
sns.kdeplot(data_1_filtered['AnnualReturn'], label=f'Data 1 - Horizon {horizon} Years', bw_adjust=1)

# Plot KDE for data_2
sns.kdeplot(data_2_filtered['AnnualReturn'], label=f'Data 2 - Horizon {horizon} Years', bw_adjust=1)

# Add title and labels
plt.title('KDE of Annual Returns by Horizon (Years)', fontsize=16)
plt.xlabel('Annual Return', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import math

# Get unique horizons
horizons = sorted(data_1['Horizon (Years)'].unique())

# Define the grid layout (2 rows x 3 columns)
n_rows = 2
n_cols = 3

# Create subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 10), sharey=True)

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Loop through each horizon and plot KDE for both dataframes
for i, horizon in enumerate(horizons):
    # Filter data for the current horizon
    data_1_filtered = data_1[data_1['Horizon (Years)'] == horizon]
    data_2_filtered = data_2[data_2['Horizon (Years)'] == horizon]
    
    # Plot KDE for data_1 on the current axis
    sns.kdeplot(data_1_filtered['AnnualReturn'], label=f'Data 1', bw_adjust=1, ax=axes[i], color='blue')
    
    # Plot KDE for data_2 on the current axis
    sns.kdeplot(data_2_filtered['AnnualReturn'], label=f'Data 2', bw_adjust=1, ax=axes[i], color='orange')
    
    # Add title and legend for the current subplot
    axes[i].set_title(f'Horizon {horizon} Years', fontsize=14)
    axes[i].legend()
    axes[i].grid(True)

# Hide any unused subplots if horizons < n_rows * n_cols
for j in range(len(horizons), len(axes)):
    axes[j].axis('off')

# Add global x and y labels
fig.text(0.5, 0.04, 'Annual Return', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)

# Adjust layout to fit titles and labels
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure
plt.figure(figsize=(12, 8))

# Plot KDE for observed data
sns.kdeplot(data_bench['Value'], label='Observed 5-Year Cumulative Returns', bw_adjust=1, color='red')

# Plot KDE for simulated data
sns.kdeplot(data_1['AnnualReturn'], label='Simulated Data 1 (Full Horizon)', bw_adjust=1, color='blue')
sns.kdeplot(data_2['AnnualReturn'], label='Simulated Data 2 (Full Horizon)', bw_adjust=1, color='green')

# Add title and labels
plt.title('KDE of 5-Year Cumulative Returns', fontsize=16)
plt.xlabel('Cumulative Return', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

from scipy.stats import ks_2samp

# Perform KS test for data_1
ks_stat_1, p_value_1 = ks_2samp(data_bench['Value'], data_1['AnnualReturn'])

# Perform KS test for data_2
ks_stat_2, p_value_2 = ks_2samp(data_bench['Value'], data_2['AnnualReturn'])

print(f"KS Test (Data 1 vs Observed): Statistic = {ks_stat_1:.4f}, P-Value = {p_value_1:.4f}")
print(f"KS Test (Data 2 vs Observed): Statistic = {ks_stat_2:.4f}, P-Value = {p_value_2:.4f}")

from scipy.stats import wasserstein_distance

# Calculate Wasserstein distance for data_1
wd_1 = wasserstein_distance(data_bench['Value'], data_1['AnnualReturn'])

# Calculate Wasserstein distance for data_2
wd_2 = wasserstein_distance(data_bench['Value'], data_2['AnnualReturn'])

print(f"Wasserstein Distance (Data 1 vs Observed): {wd_1:.4f}")
print(f"Wasserstein Distance (Data 2 vs Observed): {wd_2:.4f}")

# Calculate 95% confidence intervals for simulated data
ci_1_lower = data_1['AnnualReturn'].quantile(0.025)
ci_1_upper = data_1['AnnualReturn'].quantile(0.975)
ci_2_lower = data_2['AnnualReturn'].quantile(0.025)
ci_2_upper = data_2['AnnualReturn'].quantile(0.975)

# Add confidence intervals to the plot
plt.axvline(ci_1_lower, color='blue', linestyle='--', label='Data 1 - 95% CI Lower')
plt.axvline(ci_1_upper, color='blue', linestyle='--', label='Data 1 - 95% CI Upper')
plt.axvline(ci_2_lower, color='green', linestyle='--', label='Data 2 - 95% CI Lower')
plt.axvline(ci_2_upper, color='green', linestyle='--', label='Data 2 - 95% CI Upper')

# Calculate coverage for data_1
coverage_1 = ((data_bench['Value'] >= data_1['AnnualReturn'].min()) & 
              (data_bench['Value'] <= data_1['AnnualReturn'].max())).mean()

# Calculate coverage for data_2
coverage_2 = ((data_bench['Value'] >= data_2['AnnualReturn'].min()) & 
              (data_bench['Value'] <= data_2['AnnualReturn'].max())).mean()

print(f"Coverage (Data 1): {coverage_1:.2%}")
print(f"Coverage (Data 2): {coverage_2:.2%}")


from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

ecdf_observed = ECDF(data_bench['Value'])
ecdf_data_1 = ECDF(data_1['AnnualReturn'])
ecdf_data_2 = ECDF(data_2['AnnualReturn'])

plt.figure(figsize=(12, 8))
plt.plot(ecdf_observed.x, ecdf_observed.y, label='Observed', color='red')
plt.plot(ecdf_data_1.x, ecdf_data_1.y, label='Data 1', color='blue')
plt.plot(ecdf_data_2.x, ecdf_data_2.y, label='Data 2', color='green')
plt.title('Empirical CDF Comparison')
plt.xlabel('Cumulative Return')
plt.ylabel('ECDF')
plt.legend()
plt.grid(True)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:52:14 2025

@author: al005366
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

def diebold_mariano_test(e1, e2, h=1, alternative='less'):
    d = (e1 ** 2) - (e2 ** 2)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    n = len(d)
    # Correction for autocorrelation if h > 1
    if h > 1:
        gamma = [np.cov(d[:-lag], d[lag:])[0, 1] for lag in range(1, h)]
        d_var += 2 * np.sum(gamma)
    dm_stat = d_mean / np.sqrt(d_var / n)
    if alternative == 'two-sided':
        p_value = 2 * norm.sf(np.abs(dm_stat))
    elif alternative == 'less':
        p_value = norm.cdf(dm_stat)
    else:
        p_value = 1 - norm.cdf(dm_stat)
    return dm_stat, p_value

data_ar1 = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint_bootstrap_test\US\yields\estimated_yields\AR_1\forecasts.csv')
data_ar1_bench = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint_bootstrap_test\US\yields\observed_yields\AR_1\forecasts.csv')
data_mixed = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint_bootstrap_test\US\yields\estimated_yields\Mixed_Model\forecasts.csv')
#data_mixed = data_mixed.rename(columns={"mean_simulated": "prediction"})

# Convert execution_date to datetime
data_ar1['execution_date'] = pd.to_datetime(data_ar1['execution_date'])
data_ar1_bench['execution_date'] = pd.to_datetime(data_ar1_bench['execution_date'])
data_mixed['execution_date'] = pd.to_datetime(data_mixed['execution_date'])


data_mixed['error'] = data_mixed['mean_simulated'].sub(data_mixed['actual'])#.dropna()
data_ar1_bench['error'] = data_ar1_bench['prediction'].sub(data_ar1_bench['actual'])#.dropna()
data_ar1['error'] = data_ar1['mean_simulated'].sub(data_ar1['actual'])#.dropna()

results = []

for maturity in data_ar1['maturity'].unique():
    data_ar1_temp = data_ar1[data_ar1['maturity'] == maturity].copy()
    data_mixed_temp = data_mixed[data_mixed['maturity'] == maturity].copy()
    data_ar1_bench_temp = data_ar1_bench[data_ar1_bench['maturity'] == str(maturity) + ' years'].copy()
    
    # Merge datasets on execution_date and horizon
    merged_data = data_ar1_temp[['execution_date', 'horizon', 'error']].rename(columns={'error': 'error_ar1'})
    merged_data = merged_data.merge(
        data_mixed_temp[['execution_date', 'horizon', 'error']].rename(columns={'error': 'error_mixed'}),
        on=['execution_date', 'horizon'],
        how='inner'
    )
    merged_data = merged_data.merge(
        data_ar1_bench_temp[['execution_date', 'horizon', 'error']].rename(columns={'error': 'error_bench'}),
        on=['execution_date', 'horizon'],
        how='inner'
    )
    
    # Drop any rows with missing values (if any)
    merged_data = merged_data.dropna()
    
    # Loop through horizons
    for horizon in merged_data['horizon'].unique():
        # Filter data for the given horizon
        horizon_data = merged_data[merged_data['horizon'] == horizon]
        
        # Extract aligned errors
        e1 = horizon_data['error_ar1']
        e2 = horizon_data['error_mixed']
        e_bench = horizon_data['error_bench']/100
        
        # Perform Diebold-Mariano tests against the benchmark
        # Perform Diebold-Mariano tests against the benchmark
        dm_stat_ar1_vs_bench, p_value_ar1_vs_bench = diebold_mariano_test(e_bench, e1, h=horizon, alternative='less')
        dm_stat_mixed_vs_bench, p_value_mixed_vs_bench = diebold_mariano_test(e_bench, e2, h=horizon, alternative='less')
        
        # Append results to the list
        results.append({
            'maturity': maturity,
            'horizon': horizon,
            'dm_stat_ar1_vs_bench': dm_stat_ar1_vs_bench,
            'p_value_ar1_vs_bench': p_value_ar1_vs_bench,
            'dm_stat_mixed_vs_bench': dm_stat_mixed_vs_bench,
            'p_value_mixed_vs_bench': p_value_mixed_vs_bench
        })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

import matplotlib.pyplot as plt

# Maturities to plot
selected_maturities = [0.25, 2, 5, 10]

# Define colors for DM statistics and their lighter shades for p-values
colors = {
    'ar1_dm': 'blue',
    'mixed_dm': 'red',
    'threshold': 'yellow',
    'ar1_pval': 'lightblue',
    'mixed_pval': 'lightcoral',
    'threshold_pval': 'lightgoldenrodyellow'
}

# Set up the subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

# Loop through each selected maturity and plot its DM statistics and p-values
for i, maturity in enumerate(selected_maturities):
    # Filter results for the current maturity
    filtered_results = results_df[results_df['maturity'] == maturity]
    
    # Create a secondary y-axis for p-values
    ax = axes[i]
    ax2 = ax.twinx()
    
    # Plot DM statistics on the left y-axis
    ax.plot(filtered_results['horizon'], filtered_results['dm_stat_ar1_vs_bench'], 
            label='AR(1) vs Benchmark (DM Stat)', color=colors['ar1_dm'], linestyle='-', marker='o')
    ax.plot(filtered_results['horizon'], filtered_results['dm_stat_mixed_vs_bench'], 
            label='Mixed Model vs Benchmark (DM Stat)', color=colors['mixed_dm'], linestyle='-', marker='x')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Reference line at 0
    
    # Plot p-values on the secondary y-axis with lighter colors
    ax2.plot(filtered_results['horizon'], filtered_results['p_value_ar1_vs_bench'], 
             label='AR(1) vs Benchmark (P-Value)', color=colors['ar1_pval'], linestyle='dotted')
    ax2.plot(filtered_results['horizon'], filtered_results['p_value_mixed_vs_bench'], 
             label='Mixed Model vs Benchmark (P-Value)', color=colors['mixed_pval'], linestyle='dotted')
    ax2.axhline(0.05, color=colors['threshold'], linestyle='--', linewidth=0.8, label='Significance Threshold (0.05)')  # Threshold
    
    # Set axis labels and title
    ax.set_title(f"Maturity {maturity} Years", fontsize=14)
    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_ylabel("DM Statistic", fontsize=12, color='black')
    ax2.set_ylabel("P-Value", fontsize=12, color='black')
    
    # Set grid and limits
    ax.grid(True)
    ax2.set_ylim(0, 1)  # P-value range between 0 and 1

# Add a shared legend below the chart
fig.legend(['AR(1) vs Benchmark (DM Stat)', 'Mixed Model vs Benchmark (DM Stat)', 
            'AR(1) vs Benchmark (P-Value)', 'Mixed Model vs Benchmark (P-Value)', 
            'Significance Threshold (0.05)'], 
           loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=12, ncol=3)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the shared legend
plt.suptitle("DM Statistics and P-Values Across Horizons for Selected Maturities (Simplified Colors)", fontsize=16)

# Show the plot
plt.show()



plt.plot(e1)
plt.plot(e2)
plt.plot(e_bench/100)
# Save results to a CSV file (optional)
results_df.to_csv(r'L:\RMAS\Users\Alberto\backtest-baam\dm_test_results.csv', index=False)

# Print the first few rows of the results
print(results_df.head())

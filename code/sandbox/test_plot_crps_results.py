# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:43:03 2025

@author: al005366
"""

import pandas as pd

data = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_horizon.csv')
data = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_horizon.csv')
data = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_row.csv')
data = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_row.csv')

#data = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\monthly\crps_by_row.csv')
data['forecast_date'] = pd.to_datetime(data['forecast_date'])
data['execution_date'] = pd.to_datetime(data['execution_date'])
unique_maturities = sorted(data["maturity"].unique())
n_maturities = len(unique_maturities)

fig, axes = plt.subplots(nrows=(n_maturities + 2) // 3, ncols=3, figsize=(18, 4 * ((n_maturities + 2) // 3)), sharex=False, sharey=True)
axes = axes.flatten()

for idx, maturity in enumerate(unique_maturities):
    ax = axes[idx]
    maturity_data = data[data["maturity"] == maturity]
    avg_crps_by_horizon = maturity_data.groupby("horizon", as_index=False).agg({"crps": "mean"})
    ax.plot(avg_crps_by_horizon["horizon"], avg_crps_by_horizon["crps"], marker='o')
    #median_crps_by_horizon = maturity_data.groupby("horizon", as_index=False).agg({"crps": "median"})
    #ax.plot(median_crps_by_horizon["horizon"], median_crps_by_horizon["crps"], marker='o')
    ax.set_title(f"Maturity: {maturity} years", fontsize=14)
    ax.set_xlabel("Forecasting Horizon (Months)", fontsize=12)
    ax.set_ylabel("Average CRPS", fontsize=12)
    ax.grid(True)

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Average CRPS Across Horizons for Each Maturity", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, axes = plt.subplots(nrows=(n_maturities + 2) // 3, ncols=3, figsize=(18, 4 * ((n_maturities + 2) // 3)), sharex=False, sharey=True)
axes = axes.flatten()


for idx, maturity in enumerate(unique_maturities):
    ax = axes[idx]
    maturity_data = data[data["maturity"] == maturity]
    avg_crps_by_horizon = maturity_data.groupby("execution_date", as_index=False).agg({"crps": "mean"})
    ax.plot(avg_crps_by_horizon["execution_date"], avg_crps_by_horizon["crps"])
    #median_crps_by_horizon = maturity_data.groupby("horizon", as_index=False).agg({"crps": "median"})
    #ax.plot(median_crps_by_horizon["horizon"], median_crps_by_horizon["crps"], marker='o')
    ax.set_title(f"Maturity: {maturity} years", fontsize=14)
    ax.set_xlabel("Forecasting Horizon (Months)", fontsize=12)
    ax.set_ylabel("Average CRPS", fontsize=12)
    ax.grid(True)

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Average CRPS Across Horizons for Each Maturity", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%

# Load AR_1 and Mixed_Model data
data_ar1 = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\EA\returns\estimated_returns\AR_1\annual\crps_by_horizon.csv')
data_mixed = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\EA\returns\estimated_returns\Mixed_Model\annual\crps_by_horizon.csv')

# Filter maturities to include only 0.25, 2, 5, and 10 years
selected_maturities = [0.25, 2.0, 5.0, 10.0]

# Filter the data for the selected maturities
data_ar1 = data_ar1[data_ar1["maturity"].isin(selected_maturities)]
data_mixed = data_mixed[data_mixed["maturity"].isin(selected_maturities)]

# Define colors for the models
model_colors = {"AR_1": "#aa322f", "Mixed_Model": "#3a6bac"}  # Red for AR_1, Blue for Mixed_Model

# Prepare the subplots
fig, axes = plt.subplots(1, len(selected_maturities), figsize=(18, 6), sharey=True)

# Function to style each subplot
def style_subplot(ax):
    ax.set_facecolor("#d5d6d2")  # Grey background
    ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)  # White gridlines with low zorder
    ax.yaxis.tick_right()  # Move Y-axis to the right
    ax.yaxis.set_label_position("right")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# Plot Average CRPS Across Horizons for Selected Maturities
for idx, maturity in enumerate(selected_maturities):
    ax = axes[idx]
    
    # Filter data for the current maturity
    ar1_data = data_ar1[data_ar1["maturity"] == maturity]
    mixed_data = data_mixed[data_mixed["maturity"] == maturity]
    
    # Calculate average CRPS by horizon
    avg_crps_ar1 = ar1_data.groupby("horizon", as_index=False).agg({"crps": "mean"})
    avg_crps_mixed = mixed_data.groupby("horizon", as_index=False).agg({"crps": "mean"})
    
    # Plot AR_1 and Mixed_Model results
    ax.plot(avg_crps_ar1["horizon"], avg_crps_ar1["crps"], marker='o', color=model_colors["AR_1"], label="AR(1)", linewidth=2.5, zorder=2)
    ax.plot(avg_crps_mixed["horizon"], avg_crps_mixed["crps"], marker='o', color=model_colors["Mixed_Model"], label="Mixed_Model", linewidth=2.5, zorder=2)
    
    # Style subplot
    style_subplot(ax)
    
    # Add title and labels
    ax.set_title(f"Maturity: {maturity}", fontsize=14)
    ax.set_xlabel("Forecasting Horizon (Months)", fontsize=12)
    if idx == 0:  # Add y-axis label only for the first subplot
        ax.set_ylabel("Average CRPS", fontsize=12)

    # Add a single legend for the entire figure
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=2,
        facecolor="white",
        frameon=False,
        fontsize=12
    )

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for the legend
plt.suptitle("Average CRPS Across Horizons for Selected Maturities", fontsize=18)
plt.show()

# Get unique maturities
unique_maturities = sorted(data_ar1["maturity"].unique())
n_maturities = len(unique_maturities)

# Define colors for the models
model_colors = {"AR_1": "#aa322f", "Mixed_Model": "#3a6bac"}  # Red for AR_1, Blue for Mixed_Model

# Plot Average CRPS Across Horizons
fig, axes = plt.subplots(nrows=(n_maturities + 2) // 3, ncols=3, figsize=(18, 4 * ((n_maturities + 2) // 3)), sharex=False, sharey=True)
axes = axes.flatten()

for idx, maturity in enumerate(unique_maturities):
    ax = axes[idx]
    
    # Filter data for the current maturity
    ar1_data = data_ar1[data_ar1["maturity"] == maturity]
    mixed_data = data_mixed[data_mixed["maturity"] == maturity]
    
    # Calculate average CRPS by horizon
    avg_crps_ar1 = ar1_data.groupby("horizon", as_index=False).agg({"crps": "mean"})
    avg_crps_mixed = mixed_data.groupby("horizon", as_index=False).agg({"crps": "mean"})
    
    # Plot AR_1 and Mixed_Model results
    ax.plot(avg_crps_ar1["horizon"], avg_crps_ar1["crps"], marker='o', color=model_colors["AR_1"], label="AR(1)")
    ax.plot(avg_crps_mixed["horizon"], avg_crps_mixed["crps"], marker='o', color=model_colors["Mixed_Model"], label="Mixed_Model")
    
    # Add title and labels
    ax.set_title(f"Maturity: {maturity} years", fontsize=14)
    ax.set_xlabel("Forecasting Horizon (Months)", fontsize=12)
    ax.set_ylabel("Average CRPS", fontsize=12)
    ax.grid(True)

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Average CRPS Across Horizons for Each Maturity", fontsize=18)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=2, fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

data_ar1 = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_execution_date.csv')
data_mixed = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_execution_date.csv')

data_ar1['execution_date'] = pd.to_datetime(data_ar1['execution_date'])
data_mixed['execution_date'] = pd.to_datetime(data_mixed['execution_date'])

# Plot Average CRPS Across Execution Dates
fig, axes = plt.subplots(nrows=(n_maturities + 2) // 3, ncols=3, figsize=(18, 4 * ((n_maturities + 2) // 3)), sharex=False, sharey=True)
axes = axes.flatten()

for idx, maturity in enumerate(unique_maturities):
    ax = axes[idx]
    
    # Filter data for the current maturity
    ar1_data = data_ar1[data_ar1["maturity"] == maturity]
    mixed_data = data_mixed[data_mixed["maturity"] == maturity]
    
    # Calculate average CRPS by execution date
    avg_crps_ar1 = ar1_data.groupby("execution_date", as_index=False).agg({"crps": "mean"})
    avg_crps_mixed = mixed_data.groupby("execution_date", as_index=False).agg({"crps": "mean"})
    
    # Plot AR_1 and Mixed_Model results
    ax.plot(avg_crps_ar1["execution_date"], avg_crps_ar1["crps"], color=model_colors["AR_1"], label="AR(1)")
    ax.plot(avg_crps_mixed["execution_date"], avg_crps_mixed["crps"], color=model_colors["Mixed_Model"], label="Mixed_Model")
    
    # Add title and labels
    ax.set_title(f"Maturity: {maturity} years", fontsize=14)
    ax.set_xlabel("Execution Date", fontsize=12)
    ax.set_ylabel("Average CRPS", fontsize=12)
    ax.grid(True)

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Average CRPS Across Execution Dates for Each Maturity", fontsize=18)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=2, fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

#%%

import seaborn as sns
import matplotlib.pyplot as plt

# Load the AR_1 and Mixed_Model data
data_ar1 = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_horizon.csv')
data_mixed = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_horizon.csv')

# Process data: Ensure dates are datetime and calculate CRPS mean by horizon and maturity
#data_ar1['forecast_date'] = pd.to_datetime(data_ar1['forecast_date'])
#data_mixed['forecast_date'] = pd.to_datetime(data_mixed['forecast_date'])

# Group by horizon and maturity, calculate mean CRPS
horizon_results_ar1 = data_ar1.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})
horizon_results_mixed = data_mixed.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})

# Pivot the DataFrame for heatmaps
heatmap_data_ar1 = horizon_results_ar1.pivot(index="horizon", columns="maturity", values="crps")
heatmap_data_mixed = horizon_results_mixed.pivot(index="horizon", columns="maturity", values="crps")

# Create side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# AR_1 heatmap
sns.heatmap(heatmap_data_ar1, cmap="viridis", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[0])
axes[0].set_title("CRPS Heatmap by Horizon and Maturity - AR(1)", fontsize=14)
axes[0].set_xlabel("Maturity", fontsize=12)
axes[0].set_ylabel("Horizon (Months)", fontsize=12)

# Mixed_Model heatmap
sns.heatmap(heatmap_data_mixed, cmap="viridis", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[1])
axes[1].set_title("CRPS Heatmap by Horizon and Maturity - Mixed_Model", fontsize=14)
axes[1].set_xlabel("Maturity", fontsize=12)
axes[1].set_ylabel("Horizon (Months)", fontsize=12)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Load the AR_1 and Mixed_Model data
data_ar1 = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_horizon.csv')
data_mixed = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_horizon.csv')

# Process data: Ensure dates are datetime and calculate CRPS mean by horizon and maturity
data_ar1['forecast_date'] = pd.to_datetime(data_ar1['forecast_date'])
data_mixed['forecast_date'] = pd.to_datetime(data_mixed['forecast_date'])

# Group by horizon and maturity, calculate mean CRPS
horizon_results_ar1 = data_ar1.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})
horizon_results_mixed = data_mixed.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})

# Pivot the DataFrame for heatmaps
heatmap_data_ar1 = horizon_results_ar1.pivot(index="horizon", columns="maturity", values="crps")
heatmap_data_mixed = horizon_results_mixed.pivot(index="horizon", columns="maturity", values="crps")

# Create side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# AR_1 heatmap (red)
sns.heatmap(heatmap_data_ar1, cmap="Reds", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[0])
axes[0].set_title("CRPS Heatmap by Horizon and Maturity - AR(1)", fontsize=14)
axes[0].set_xlabel("Maturity", fontsize=12)
axes[0].set_ylabel("Horizon (Months)", fontsize=12)

# Mixed_Model heatmap (blue)
sns.heatmap(heatmap_data_mixed, cmap="Blues", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[1])
axes[1].set_title("CRPS Heatmap by Horizon and Maturity - Mixed_Model", fontsize=14)
axes[1].set_xlabel("Maturity", fontsize=12)
axes[1].set_ylabel("Horizon (Months)", fontsize=12)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

#%%


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the AR_1 and Mixed_Model data for horizon and execution date
data_ar1_horizon = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_horizon.csv')
data_mixed_horizon = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_horizon.csv')

data_ar1_exec_date = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_execution_date.csv')
data_mixed_exec_date = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_execution_date.csv')

# Ensure dates are in datetime format
#data_ar1_exec_date['execution_date'] = pd.to_datetime(data_ar1_exec_date['execution_date'])
#data_mixed_exec_date['execution_date'] = pd.to_datetime(data_mixed_exec_date['execution_date'])

# Group data for heatmaps
# Horizon vs Maturity (CRPS mean)
horizon_results_ar1 = data_ar1_horizon.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})
horizon_results_mixed = data_mixed_horizon.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})

# Execution Date vs Maturity (CRPS mean)
exec_date_results_ar1 = data_ar1_exec_date.groupby(["execution_date", "maturity"], as_index=False).agg({"crps": "mean"})
exec_date_results_mixed = data_mixed_exec_date.groupby(["execution_date", "maturity"], as_index=False).agg({"crps": "mean"})

# Pivot the DataFrames for heatmaps
heatmap_data_ar1_horizon = horizon_results_ar1.pivot(index="horizon", columns="maturity", values="crps")
heatmap_data_mixed_horizon = horizon_results_mixed.pivot(index="horizon", columns="maturity", values="crps")

heatmap_data_ar1_exec_date = exec_date_results_ar1.pivot(index="execution_date", columns="maturity", values="crps")
heatmap_data_mixed_exec_date = exec_date_results_mixed.pivot(index="execution_date", columns="maturity", values="crps")

# Create a 2x2 grid for heatmaps
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# AR_1 Heatmap: Horizon vs Maturity
sns.heatmap(heatmap_data_ar1_horizon, cmap="Reds", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[0, 0])
axes[0, 0].set_title("AR(1) - CRPS by Horizon and Maturity", fontsize=14)
axes[0, 0].set_xlabel("Maturity", fontsize=12)
axes[0, 0].set_ylabel("Horizon (Months)", fontsize=12)

# AR_1 Heatmap: Execution Date vs Maturity
sns.heatmap(heatmap_data_ar1_exec_date, cmap="Reds", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[1, 0])
axes[1, 0].set_title("AR(1) - CRPS by Execution Date and Maturity", fontsize=14)
axes[1, 0].set_xlabel("Maturity", fontsize=12)
axes[1, 0].set_ylabel("Execution Date", fontsize=12)

# Mixed_Model Heatmap: Horizon vs Maturity
sns.heatmap(heatmap_data_mixed_horizon, cmap="Blues", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[0, 1])
axes[0, 1].set_title("Mixed_Model - CRPS by Horizon and Maturity", fontsize=14)
axes[0, 1].set_xlabel("Maturity", fontsize=12)
axes[0, 1].set_ylabel("Horizon (Months)", fontsize=12)

# Mixed_Model Heatmap: Execution Date vs Maturity
sns.heatmap(heatmap_data_mixed_exec_date, cmap="Blues", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'}, ax=axes[1, 1])
axes[1, 1].set_title("Mixed_Model - CRPS by Execution Date and Maturity", fontsize=14)
axes[1, 1].set_xlabel("Maturity", fontsize=12)
axes[1, 1].set_ylabel("Execution Date", fontsize=12)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
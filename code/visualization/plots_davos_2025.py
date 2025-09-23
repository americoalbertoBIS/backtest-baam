# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:50:54 2025

@author: al005366
"""

#%%
import pandas as pd

base_folder = r'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data_joint\US'

data_ar1_beta1 = pd.read_csv(fr'{base_folder}\factors\AR_1\beta1\forecasts.csv')
data_ar1_beta2 = pd.read_csv(fr'{base_folder}\factors\AR_1\beta2\forecasts.csv')
data_ar1_beta3 = pd.read_csv(fr'{base_folder}\factors\AR_1\beta3\forecasts.csv')

data_ar1og_beta2 = pd.read_csv(fr'{base_folder}\factors\AR_1_Output_Gap_Direct\beta2\forecasts.csv')
data_ar1ogmrm_beta2 = pd.read_csv(fr'{base_folder}\factors\AR_1_Output_Gap_Direct_MRM\beta2\forecasts.csv')

data_ar1oginf_beta1 = pd.read_csv(fr'{base_folder}\factors\AR_1_Output_Gap_Direct_Inflation_UCSV\beta1\forecasts.csv')
data_ar1oginfmrm_beta1 = pd.read_csv(fr'{base_folder}\factors\AR_1_Output_Gap_Direct_Inflation_UCSV_MRM\beta1\forecasts.csv')

data_ar1oginf_beta3 = pd.read_csv(fr'{base_folder}\factors\AR_1_Output_Gap_Direct_Inflation_UCSV\beta3\forecasts.csv')
data_ar1oginfmrm_beta3 = pd.read_csv(fr'{base_folder}\factors\AR_1_Output_Gap_Direct_Inflation_UCSV_MRM\beta3\forecasts.csv')

import matplotlib.pyplot as plt
import pandas as pd

# Function to plot forecasts with actuals for all models of a single beta
def plot_forecasts_with_actuals(ax, datasets, realized_beta, beta_title, model_colors):
    

    # Iterate over all models for the beta
    for idx, (data, model_label) in enumerate(datasets):
        
        #mask = data.apply(lambda row: row.astype(str).str.contains("<<<<<<<|>>>>>>>|=======").any(), axis=1)
        #data = data[~mask]
        
        unique_execution_dates = data['execution_date'].unique()
        model_color = model_colors[idx % len(model_colors)]  # Assign a unique color for each model

        # Plot all predictions for the model
        for execution_date in unique_execution_dates:
            subset = data[data['execution_date'] == execution_date]
            
            ax.plot(pd.to_datetime(subset['forecast_date']), subset['prediction'], color=model_color, alpha=0.3)

        # Add a label for the model in the legend
        ax.plot([], [], color=model_color, label=model_label)

    # Plot the actual series
    realized_beta_index = pd.to_datetime(realized_beta.index)
    ax.plot(realized_beta_index, realized_beta.values, color='black', linewidth=2, label="Actual")
    #ax.plot(realized_beta.index, realized_beta.values, color='black', linewidth=2, label="Actual")

    # Add zero reference line
    ax.axhline(0, color="black", linewidth=1.5, linestyle="-")  # Solid black line for zero reference

    # Add title, labels, and grid
    #ax.set_title(beta_title, fontsize=12, loc='center', pad=10)
    
    #ax.set_xlabel("", fontsize=10)
    #ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, color="white", linestyle='-', linewidth=1)  # Solid white gridlines

    # Move Y-axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Remove all spines except for X and Y axes
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Set background color for the plot area
    ax.set_facecolor("#d5d6d2")  # Grey background for the graph

# Prepare the data
datasets = [
    (data_ar1_beta1, "AR(1)"),
    (data_ar1oginf_beta1, "AR(1) + output gap + inflation (Consensus)"),
    (data_ar1oginfmrm_beta1, "AR(1) + output gap + inflation (MRM)")
]
datasets_beta2 = [
    (data_ar1_beta2, "AR(1)"),
    (data_ar1og_beta2, "AR(1) + output gap (Consensus)"),
    (data_ar1ogmrm_beta2, "AR(1) + output gap (MRM)")
]
datasets_beta3 = [
    (data_ar1_beta3, "AR(1)"),
    (data_ar1oginf_beta3, "AR(1) + output gap + inflation (Consensus)"),
    (data_ar1oginfmrm_beta3, "AR(1) + output gap + inflation (MRM)")
]

for data, _ in datasets + datasets_beta2 + datasets_beta3:
    mask = data.apply(lambda row: row.astype(str).str.contains("<<<<<<<|>>>>>>>|=======").any(), axis=1)
    data = data[~mask]
    data['forecast_date'] = pd.to_datetime(data['forecast_date'])
    data['execution_date'] = pd.to_datetime(data['execution_date'])

# Sort by execution_date and forecast_date
for i, (data, label) in enumerate(datasets):
    datasets[i] = (data.sort_values(by=['execution_date', 'forecast_date']), label)
for i, (data, label) in enumerate(datasets_beta2):
    datasets_beta2[i] = (data.sort_values(by=['execution_date', 'forecast_date']), label)
for i, (data, label) in enumerate(datasets_beta3):
    datasets_beta3[i] = (data.sort_values(by=['execution_date', 'forecast_date']), label)

# Calculate the Actual series by taking the mean grouped by forecast_date
def get_actual_series(data):
    return data.groupby('forecast_date')['actual'].mean()

# Subset the data to start from the year 2000
start_date = '2000-01-01'
for i, (data, label) in enumerate(datasets):
    datasets[i] = (data[data['forecast_date'] >= start_date], label)
for i, (data, label) in enumerate(datasets_beta2):
    datasets_beta2[i] = (data[data['forecast_date'] >= start_date], label)
for i, (data, label) in enumerate(datasets_beta3):
    datasets_beta3[i] = (data[data['forecast_date'] >= start_date], label)

# Get the Actual series for each beta
actual_series_beta1 = get_actual_series(datasets[0][0])
actual_series_beta2 = get_actual_series(datasets_beta2[0][0])
actual_series_beta3 = get_actual_series(datasets_beta3[0][0])

actual_series_beta1 = actual_series_beta1[actual_series_beta1.index >= start_date]
actual_series_beta2 = actual_series_beta2[actual_series_beta2.index >= start_date]
actual_series_beta3 = actual_series_beta3[actual_series_beta3.index >= start_date]

# Custom colors for the models
model_colors = ["#aa322f", "#3a6bac", "#eaa121"]  # Updated color palette

# Create a 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Set the font globally
plt.rcParams['font.family'] = "Segoe UI"

# Plot Shadow short rate
plot_forecasts_with_actuals(
    ax=axes[0],
    datasets=datasets,
    realized_beta=actual_series_beta1,
    beta_title="Shadow short rate",
    model_colors=model_colors
)

# Plot Slope
plot_forecasts_with_actuals(
    ax=axes[1],
    datasets=datasets_beta2,
    realized_beta=actual_series_beta2,
    beta_title="Slope",
    model_colors=model_colors
)

# Plot Curvature
plot_forecasts_with_actuals(
    ax=axes[2],
    datasets=datasets_beta3,
    realized_beta=actual_series_beta3,
    beta_title="Curvature",
    model_colors=model_colors
)

# Add legends to each subplot
for ax in axes:
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),
        ncol=1,  # Legend in one column
        facecolor="white",  # Explicitly set white background for the legend
        frameon=False,  # Remove borders from the legend box
        fontsize=12
    )

# Adjust layout and show the plot
plt.tight_layout()
#plt.savefig(r"L:\RMAS\Users\Alberto\AMAP 2025\graphs\forecast_vs_actuals.svg", format="svg")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def calculate_rmse_by_horizon(data, start_date='1990-01-01'):
    """
    Calculate RMSE for a dataset grouped by Horizon from the specified execution_date onward.

    Parameters:
        data (pd.DataFrame): The dataset containing 'execution_date', 'prediction', 'Actual', and 'Horizon'.
        start_date (str): The start date to filter the data (default is '1990-01-01').

    Returns:
        dict: A dictionary with Horizon as keys and RMSE as values.
    """
    # Filter the data to include only rows where execution_date >= start_date
    filtered_data = data[pd.to_datetime(data['execution_date']) >= start_date]

    # Ensure there are no NaN values in 'prediction' or 'Actual'
    filtered_data = filtered_data.dropna(subset=['prediction', 'actual'])

    # Group by Horizon and calculate RMSE for each group
    rmse_by_horizon = {}
    for horizon, group in filtered_data.groupby('horizon'):
        rmse = np.sqrt(np.mean((group['prediction'] - group['actual'])**2))
        rmse_by_horizon[horizon] = rmse

    return rmse_by_horizon

# Calculate RMSE by Horizon for each dataset and organize into separate groups
rmse_results_beta1 = {model_label: calculate_rmse_by_horizon(data) for data, model_label in datasets}
rmse_results_beta2 = {model_label: calculate_rmse_by_horizon(data) for data, model_label in datasets_beta2}
rmse_results_beta3 = {model_label: calculate_rmse_by_horizon(data) for data, model_label in datasets_beta3}

# Plotting the RMSE by Horizon
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Custom colors for the models
model_colors = ["#aa322f", "#3a6bac", "#eaa121"]  # Updated color palette

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

# Beta 1 (Shadow short rate)
for idx, (model_label, rmse_by_horizon) in enumerate(rmse_results_beta1.items()):
    horizons = sorted(rmse_by_horizon.keys())
    rmse_values = [rmse_by_horizon[h] for h in horizons]
    axes[0].plot(horizons, rmse_values, label=model_label, color=model_colors[idx % len(model_colors)], linewidth=2, zorder=2)
    # Add dots every 12 horizons
    dot_horizons = [h for h in horizons if h % 12 == 0]
    dot_rmse_values = [rmse_by_horizon[h] for h in dot_horizons]
    axes[0].scatter(dot_horizons, dot_rmse_values, color=model_colors[idx % len(model_colors)], s=40, zorder=3)

style_subplot(axes[0])
axes[0].set_xlabel("Horizon (months)", fontsize=10)

# Beta 2 (Slope)
for idx, (model_label, rmse_by_horizon) in enumerate(rmse_results_beta2.items()):
    horizons = sorted(rmse_by_horizon.keys())
    rmse_values = [rmse_by_horizon[h] for h in horizons]
    axes[1].plot(horizons, rmse_values, label=model_label, color=model_colors[idx % len(model_colors)], linewidth=2, zorder=2)
    # Add dots every 12 horizons
    dot_horizons = [h for h in horizons if h % 12 == 0]
    dot_rmse_values = [rmse_by_horizon[h] for h in dot_horizons]
    axes[1].scatter(dot_horizons, dot_rmse_values, color=model_colors[idx % len(model_colors)], s=40, zorder=3)

style_subplot(axes[1])
axes[1].set_xlabel("Horizon (months)", fontsize=10)

# Beta 3 (Curvature)
for idx, (model_label, rmse_by_horizon) in enumerate(rmse_results_beta3.items()):
    horizons = sorted(rmse_by_horizon.keys())
    rmse_values = [rmse_by_horizon[h] for h in horizons]
    axes[2].plot(horizons, rmse_values, label=model_label, color=model_colors[idx % len(model_colors)], linewidth=2, zorder=2)
    # Add dots every 12 horizons
    dot_horizons = [h for h in horizons if h % 12 == 0]
    dot_rmse_values = [rmse_by_horizon[h] for h in dot_horizons]
    axes[2].scatter(dot_horizons, dot_rmse_values, color=model_colors[idx % len(model_colors)], s=40, zorder=3)

style_subplot(axes[2])
axes[2].set_xlabel("Horizon (months)", fontsize=10)

# Add legends to each subplot
for ax in axes:
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),
        ncol=1,  # Legend in one column
        facecolor="white",  # Explicitly set white background for the legend
        frameon=False,  # Remove borders from the legend box
        fontsize=12
    )

# Adjust layout and show the plot
#plt.savefig(r"L:\RMAS\Users\Alberto\AMAP 2025\graphs\rmse_factors.svg", format="svg")
plt.show()

#%%


import pandas as pd

base_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US'

data_ar1 = pd.read_csv(fr'{base_folder}\yields\estimated_yields\AR_1\forecasts.csv')
data_mixed = pd.read_csv(fr'{base_folder}\yields\estimated_yields\Mixed_Model\forecasts.csv')
data_ar1bench = pd.read_csv(fr'{base_folder}\yields\observed_yields\AR_1\forecasts.csv')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Helper function to calculate RMSE
def calculate_rmse(data, start_date, end_date=None, scale_benchmark=False):
    """
    Calculate RMSE from the forecasts data filtered by start and end dates.

    Parameters:
        data (pd.DataFrame): The dataset containing 'execution_date', 'Horizon', 'prediction', and 'actual'.
        start_date (str): The start date to filter the data.
        end_date (str): The end date to filter the data (optional).
        scale_benchmark (bool): Whether to divide 'actual' and 'prediction' by 100 for the benchmark model.

    Returns:
        pd.DataFrame: A DataFrame with 'Horizon' and 'RMSE' columns.
    """
    # Convert execution_date to datetime if not already
    data['execution_date'] = pd.to_datetime(data['execution_date'])

    # Filter the data by execution_date
    filtered_data = data[data['execution_date'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_data = filtered_data[
            filtered_data['execution_date'] <= pd.to_datetime(end_date)
        ]

    # Scale actual and prediction values for benchmark models if required
    if scale_benchmark:
        filtered_data['actual'] = filtered_data['actual'] / 100
        filtered_data['prediction'] = filtered_data['prediction'] / 100

    # Drop rows with missing Actual or prediction values
    filtered_data = filtered_data.dropna(subset=["actual", "prediction"])

    # Calculate RMSE by Horizon
    rmse_data = (
        filtered_data
        .groupby("Horizon")
        .apply(lambda x: np.sqrt(((x["prediction"] - x["actual"]) ** 2).mean()))
        .reset_index(name="RMSE")
    )
    return rmse_data

# Load raw data
base_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US'
data_ar1 = pd.read_csv(fr'{base_folder}\yields\estimated_yields\AR_1\forecasts.csv')
data_mixed = pd.read_csv(fr'{base_folder}\yields\estimated_yields\Mixed_Model\forecasts.csv')
data_ar1bench = pd.read_csv(fr'{base_folder}\yields\observed_yields\AR_1\forecasts.csv')

# Normalize column names for consistency
data_ar1bench = data_ar1bench.rename(columns={"horizon": "Horizon"})

# Define maturities to analyze
maturities = ['2.0 years', '5.0 years', '10.0 years']

# Calculate RMSE for each model and maturity
rmse_results = {}
for maturity in maturities:
    print(f"Calculating RMSE for maturity: {maturity}")
    rmse_results[maturity] = {
        "AR(1) factors": calculate_rmse(data_ar1[data_ar1['maturity'] == maturity], start_date='1990-01-01'),
        "Macro-based approach": calculate_rmse(data_mixed[data_mixed['maturity'] == maturity], start_date='1990-01-01'),
        "AR(1) benchmark": calculate_rmse(data_ar1bench[data_ar1bench['maturity'] == maturity], start_date='1990-01-01', scale_benchmark=True),
    }

# Prepare the 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Custom colors for the models
model_colors = ["#aa322f", "#3a6bac", "#eaa121"]

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

# Plot RMSE by Horizon for each maturity
for idx, maturity in enumerate(maturities):
    ax = axes[idx]
    
    # Plot RMSE for each model
    for model_idx, (model_label, rmse_data) in enumerate(rmse_results[maturity].items()):
        ax.plot(
            rmse_data["Horizon"],
            rmse_data["RMSE"] * 100,  # Multiply RMSE by 100 for percentage interpretation
            label=model_label,
            color=model_colors[model_idx % len(model_colors)],
            linewidth=2.5,  # Slightly thicker lines
            zorder=2
        )
        # Add dots every 12 horizons
        dot_horizons = rmse_data["Horizon"][rmse_data["Horizon"] % 12 == 0]
        dot_rmse_values = rmse_data["RMSE"][rmse_data["Horizon"] % 12 == 0] * 100
        ax.scatter(
            dot_horizons,
            dot_rmse_values,
            color=model_colors[model_idx % len(model_colors)],
            s=40,
            zorder=3
        )
    
    # Style the subplot
    style_subplot(ax)
    ax.set_xlabel("Horizon (months)", fontsize=10)
    
    # Drop Y-axis label and title
    if idx == 0:
        ax.set_ylabel("")  # Remove Y-axis label
    ax.set_title("")  # Remove title

# Add legends to each subplot
for ax in axes:
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),
        ncol=1,  # Legend in one column
        facecolor="white",  # Explicitly set white background for the legend
        frameon=False,  # Remove borders from the legend box
        fontsize=12
    )

# Adjust layout and show the plot
plt.tight_layout()
#plt.savefig(r"L:\RMAS\Users\Alberto\AMAP 2025\graphs\rmse_by_horizon_yields.svg", format="svg")
plt.show()


#%%



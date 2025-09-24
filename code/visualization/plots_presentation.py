# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:50:54 2025

@author: al005366
"""


graphs_folder = r'L:\RMAS\Users\Alberto\final_presentation_CMA\graphs'


#%%
import matplotlib.pyplot as plt

# Parameters for the expanding window
n_passes = 5  # Number of backtesting passes
initial_train_size = 5  # Initial training window size
forecast_horizon = 2  # Forecasting window size
total_time = initial_train_size + n_passes + forecast_horizon  # Total time series length

# Custom colors for the bars
train_color = "#3a6bac"  # Blue for training
forecast_color = "#aa322f"  # Red for forecasting

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Style the subplot
def style_subplot(ax):
    ax.set_facecolor("#d5d6d2")  # Grey background
    ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)  # White gridlines with low zorder
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# Plot the expanding window concept
for pass_num in range(n_passes):
    # Calculate training and forecasting window ranges
    train_start = 0
    train_end = initial_train_size + pass_num  # Training window expands by 1 step each pass
    forecast_start = train_end + 1
    forecast_end = forecast_start + forecast_horizon - 1

    # Plot training window (blue bar)
    ax.barh(y=pass_num, width=train_end - train_start + 1, left=train_start, color=train_color, label="Training" if pass_num == 0 else "", zorder=2)

    # Plot forecasting window (red bar)
    ax.barh(y=pass_num, width=forecast_end - forecast_start + 1, left=forecast_start, color=forecast_color, label="Forecasting" if pass_num == 0 else "", zorder=2)

# Add title, labels, and legend
ax.set_title("Expanding window backtesting concept", fontsize=16)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Execution Date", fontsize=12)
ax.set_yticks(range(n_passes))  # Correct range for y-axis ticks
ax.set_yticklabels([f"Pass {i + 1}" for i in range(n_passes)], fontsize=10)

# Adjust x-axis ticks to increment by 1 and include the last number
ax.set_xticks(range(0, total_time + 1, 1))  # Step size of 1 for x-axis ticks
ax.set_xlim(0, total_time)  # Explicitly set the x-axis limits to include the final value
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

# Apply the custom style to the subplot
style_subplot(ax)

# Add legend
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=2,  # Legend in two columns
    facecolor="white",  # Explicitly set white background for the legend
    frameon=False,  # Remove borders from the legend box
    fontsize=12
)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(r"L:\RMAS\Users\Alberto\final_presentation_CMA\graphs\expanding_backtesting_window_concept.svg", format="svg")
plt.show()

#%%

import matplotlib.pyplot as plt

# Parameters for the walk-forward validation
n_passes = 5  # Number of backtesting passes
train_size = 5  # Fixed training window size
forecast_horizon = 2  # Forecasting window size
total_time = train_size + n_passes * forecast_horizon  # Total time series length

# Custom colors for the bars
train_color = "#3a6bac"  # Blue for training
forecast_color = "#aa322f"  # Red for forecasting

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Style the subplot
def style_subplot(ax):
    ax.set_facecolor("#d5d6d2")  # Grey background
    ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)  # White gridlines with low zorder
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# Plot the walk-forward concept
for pass_num in range(n_passes):
    # Calculate training and forecasting window ranges
    train_start = pass_num * forecast_horizon
    train_end = train_start + train_size - 1
    forecast_start = train_end + 1
    forecast_end = forecast_start + forecast_horizon - 1

    # Plot training window (blue bar)
    ax.barh(y=pass_num, width=train_end - train_start + 1, left=train_start, color=train_color, label="Training" if pass_num == 0 else "", zorder=2)

    # Plot forecasting window (red bar)
    ax.barh(y=pass_num, width=forecast_end - forecast_start + 1, left=forecast_start, color=forecast_color, label="Forecasting" if pass_num == 0 else "", zorder=2)

# Add title, labels, and legend
ax.set_title("Walk-Forward Validation Concept", fontsize=16)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Execution Date", fontsize=12)
ax.set_yticks(range(n_passes))  # Correct range for y-axis ticks
ax.set_yticklabels([f"Pass {i + 1}" for i in range(n_passes)], fontsize=10)

# Adjust x-axis ticks to increment by 1 and include the last number
ax.set_xticks(range(0, total_time + 1, 1))  # Step size of 1 for x-axis ticks
ax.set_xlim(0, total_time)  # Explicitly set the x-axis limits to include the final value
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

# Apply the custom style to the subplot
style_subplot(ax)

# Add legend
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=2,  # Legend in two columns
    facecolor="white",  # Explicitly set white background for the legend
    frameon=False,  # Remove borders from the legend box
    fontsize=12
)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(r"L:\RMAS\Users\Alberto\final_presentation_CMA\graphs\walk_forward_window_concept.svg", format="svg")
plt.show()

#%%
import matplotlib.pyplot as plt

# Parameters for both concepts
n_passes = 5  # Number of backtesting passes
initial_train_size = 5  # Initial training window size for expanding window
train_size = 5  # Fixed training window size for walk-forward
forecast_horizon = 2  # Forecasting window size
total_time_expanding = initial_train_size + n_passes + forecast_horizon  # Total time for expanding window
total_time_walk_forward = train_size + n_passes * forecast_horizon  # Total time for walk-forward

# Custom colors for the bars
train_color = "#3a6bac"  # Blue for training
forecast_color = "#aa322f"  # Red for forecasting

# Create a 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Function to style the subplots
def style_subplot(ax):
    ax.set_facecolor("#d5d6d2")  # Grey background
    ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)  # White gridlines with low zorder
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# Plot Expanding Window Backtesting
ax = axes[0]
for pass_num in range(n_passes):
    # Calculate training and forecasting window ranges
    train_start = 0
    train_end = initial_train_size + pass_num  # Training window expands by 1 step each pass
    forecast_start = train_end + 1
    forecast_end = forecast_start + forecast_horizon - 1

    # Plot training window (blue bar)
    ax.barh(y=pass_num, width=train_end - train_start + 1, left=train_start, color=train_color, label="Training" if pass_num == 0 else "", zorder=2)

    # Plot forecasting window (red bar)
    ax.barh(y=pass_num, width=forecast_end - forecast_start + 1, left=forecast_start, color=forecast_color, label="Forecasting" if pass_num == 0 else "", zorder=2)

# Add title, labels
ax.set_title("Expanding window", fontsize=16)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Execution date", fontsize=12)
ax.set_yticks(range(n_passes))  # Correct range for y-axis ticks
ax.set_yticklabels([f"{i + 1}" for i in range(n_passes)], fontsize=10)
ax.set_xticks(range(0, total_time_expanding + 1, 1))  # Step size of 1 for x-axis ticks
ax.set_xlim(0, total_time_expanding)  # Explicitly set the x-axis limits to include the final value
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
style_subplot(ax)

# Plot Walk-Forward Validation
ax = axes[1]
for pass_num in range(n_passes):
    # Calculate training and forecasting window ranges
    train_start = pass_num * forecast_horizon
    train_end = train_start + train_size - 1
    forecast_start = train_end + 1
    forecast_end = forecast_start + forecast_horizon - 1

    # Plot training window (blue bar)
    ax.barh(y=pass_num, width=train_end - train_start + 1, left=train_start, color=train_color, label="Training" if pass_num == 0 else "", zorder=2)

    # Plot forecasting window (red bar)
    ax.barh(y=pass_num, width=forecast_end - forecast_start + 1, left=forecast_start, color=forecast_color, label="Forecasting" if pass_num == 0 else "", zorder=2)

# Add title, labels
ax.set_title("Walk-forward", fontsize=16)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("")  # No y-label for the second plot
ax.set_yticks(range(n_passes))  # Correct range for y-axis ticks
ax.set_yticklabels([f"{i + 1}" for i in range(n_passes)], fontsize=10)
ax.set_xticks(range(0, total_time_walk_forward + 1, 1))  # Step size of 1 for x-axis ticks
ax.set_xlim(0, total_time_walk_forward)  # Explicitly set the x-axis limits to include the final value
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
style_subplot(ax)

# Add single legend to the figure
fig.legend(

handles=[
        plt.Line2D([0], [0], color=train_color, lw=4, label="Training"),
        plt.Line2D([0], [0], color=forecast_color, lw=4, label="Forecasting")
    ],    loc='lower center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,  # Legend in two columns
    facecolor="white",  # Explicitly set white background for the legend
    frameon=False,  # Remove borders from the legend box
    fontsize=16
)

# Adjust layout and save the figure
plt.tight_layout()  # Adjust rect to make space for the legend
plt.savefig(r"L:\RMAS\Users\Alberto\final_presentation_CMA\graphs\combined_backtesting_concepts.svg", format="svg")
plt.show()

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
plt.savefig(r"L:\RMAS\Users\Alberto\final_presentation_CMA\graphs\forecast_vs_actuals_factors.svg", format="svg")
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
plt.tight_layout()
plt.savefig(rf"{graphs_folder}\rmse_factors_US.svg", format="svg")
plt.show()

#%%


import pandas as pd
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
        filtered_data['actual'] = filtered_data['actual'] /100
        filtered_data['prediction'] = filtered_data['prediction'] /100

    # Drop rows with missing Actual or prediction values
    filtered_data = filtered_data.dropna(subset=["actual", "prediction"])

    # Calculate RMSE by Horizon
    rmse_data = (
        filtered_data
        .groupby("horizon")
        .apply(lambda x: np.sqrt(((x["prediction"] - x["actual"]) ** 2).mean()))
        .reset_index(name="rmse")
    )
    return rmse_data

# Load raw data
base_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US'
measure = 'yields'
estimated_path = fr'{base_folder}\{measure}\estimated_{measure}'
observed_path = fr'{base_folder}\{measure}\observed_{measure}'

if measure == 'returns':
    freq = 'annual' # annual, monthly
    
    data_ar1 = pd.read_csv(fr'{estimated_path}\AR_1\{freq}\forecasts.csv')
    data_mixed = pd.read_csv(fr'{estimated_path}\Mixed_Model\{freq}\forecasts.csv')
    data_mixedCurvMacro = pd.read_csv(fr'{estimated_path}\Mixed_Model_curvMacro\{freq}\forecasts.csv')
    #data_ar1bench = pd.read_csv(fr'{observed_path}\AR_1\{freq}\forecasts.csv')

else:
    data_ar1 = pd.read_csv(fr'{estimated_path}\AR_1\forecasts.csv')
    data_mixed = pd.read_csv(fr'{estimated_path}\Mixed_Model\forecasts.csv')
    data_mixedCurvMacro = pd.read_csv(fr'{estimated_path}\Mixed_Model_curvMacro\forecasts.csv')
    data_ar1bench = pd.read_csv(fr'{observed_path}\AR_1\forecasts.csv')

    # Normalize column names for consistency
    data_mixed = data_mixed.rename(columns={"mean_simulated": "prediction"})
    data_mixed['maturity'] = data_mixed['maturity'].astype(str) + ' years'
    data_ar1 = data_ar1.rename(columns={"mean_simulated": "prediction"})
    data_ar1['maturity'] = data_ar1['maturity'].astype(str) + ' years'
    data_mixedCurvMacro = data_mixedCurvMacro.rename(columns={"mean_simulated": "prediction"})
    data_mixedCurvMacro['maturity'] = data_mixedCurvMacro['maturity'].astype(str) + ' years'

# Define maturities to analyze
maturities = ['0.25 years', '2.0 years', '5.0 years', '10.0 years']

# Calculate RMSE for each model and maturity
rmse_results = {}
for maturity in maturities:
    print(f"Calculating RMSE for maturity: {maturity}")
    rmse_results[maturity] = {
        "AR(1) factors": calculate_rmse(data_ar1[data_ar1['maturity'] == maturity], start_date='1990-01-01'),
        "Macro-based approach": calculate_rmse(data_mixed[data_mixed['maturity'] == maturity], start_date='1990-01-01'),
        "Macro-based approach (macro-curv)": calculate_rmse(data_mixedCurvMacro[data_mixedCurvMacro['maturity'] == maturity], start_date='1990-01-01'),
        "AR(1) benchmark": calculate_rmse(data_ar1bench[data_ar1bench['maturity'] == maturity], start_date='1990-01-01', scale_benchmark=True),
    }

# Prepare the 1x3 subplot
fig, axes = plt.subplots(1, len(maturities), figsize=(18, 6), sharey=True)

# Custom colors for the models
model_colors = ["#aa322f", "#3a6bac", "#eaa121", "#633d83"]

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
    for model_idx, (model_label, rmse_data) in enumerate(rmse_results[maturity].items()):
        # Plot line for all horizons
        ax.plot(
            rmse_data["horizon"],
            rmse_data["rmse"],
            label=model_label,
            color=model_colors[model_idx % len(model_colors)],
            linewidth=2.5,
            zorder=2
        )
        # Add dots for every horizon if returns and annual, otherwise every 12 horizons
        if measure == 'returns' and freq == 'annual':
            dot_horizons = rmse_data["horizon"]  # Include all horizons
            dot_rmse_values = rmse_data["rmse"]
        else:
            dot_horizons = rmse_data["horizon"][rmse_data["horizon"] % 12 == 0]  # Every 12 horizons
            dot_rmse_values = rmse_data["rmse"][rmse_data["horizon"] % 12 == 0]
        ax.scatter(
            dot_horizons,
            dot_rmse_values,
            color=model_colors[model_idx % len(model_colors)],
            s=40,  # Dot size
            zorder=3
        )
    # Style subplot
    style_subplot(ax)
    ax.set_xlabel("Horizon (years)", fontsize=10)
    ax.set_ylabel("", fontsize=10)

    # Show only integer values on x-axis if frequency is annual
    if freq == 'annual':
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add legend only for the first panel
    if idx == 0:
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.4),
            ncol=1,
            facecolor="white",
            frameon=False,
            fontsize=12
        )
        


plt.tight_layout()
plt.savefig(rf"{graphs_folder}\rmse_yields_US_with_legends_sharey_st.svg", format="svg")
plt.show()

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function to calculate RMSE by execution date
def calculate_rmse_by_execution_date(data, start_date, end_date=None, scale_benchmark=False):
    """
    Calculate RMSE from the forecasts data grouped by execution date.

    Parameters:
        data (pd.DataFrame): The dataset containing 'execution_date', 'Horizon', 'prediction', and 'actual'.
        start_date (str): The start date to filter the data.
        end_date (str): The end date to filter the data (optional).
        scale_benchmark (bool): Whether to divide 'actual' and 'prediction' by 100 for the benchmark model.

    Returns:
        pd.DataFrame: A DataFrame with 'execution_date' and 'RMSE' columns.
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

    # Calculate RMSE by execution date
    rmse_data = (
        filtered_data
        .groupby("execution_date")
        .apply(lambda x: np.sqrt(((x["prediction"] - x["actual"]) ** 2).mean()))
        .reset_index(name="rmse")
    )
    return rmse_data

# Define maturities to analyze
maturities = ['0.25 years', '2.0 years', '5.0 years', '10.0 years']

# Calculate RMSE for each model and maturity by execution date
rmse_results_by_execution_date = {}
for maturity in maturities:
    print(f"Calculating RMSE for maturity: {maturity}")
    rmse_results_by_execution_date[maturity] = {
        "AR(1) factors": calculate_rmse_by_execution_date(data_ar1[data_ar1['maturity'] == maturity], start_date='1990-01-01'),
        "Macro-based approach": calculate_rmse_by_execution_date(data_mixed[data_mixed['maturity'] == maturity], start_date='1990-01-01'),
        #"Macro-based approach (macro-curv)": calculate_rmse_by_execution_date(data_mixedCurvMacro[data_mixedCurvMacro['maturity'] == maturity], start_date='1990-01-01'),
        "AR(1) benchmark": calculate_rmse_by_execution_date(data_ar1bench[data_ar1bench['maturity'] == maturity], start_date='1990-01-01', scale_benchmark=True),
    }

# Prepare the 1x4 subplot
fig, axes = plt.subplots(1, len(maturities), figsize=(18, 6), sharey=False)

# Custom colors for the models
model_colors = ["#aa322f", "#3a6bac", "#eaa121", "#633d83"]

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

# Plot RMSE by Execution Date for each maturity
for idx, maturity in enumerate(maturities):
    ax = axes[idx]
    for model_idx, (model_label, rmse_data) in enumerate(rmse_results_by_execution_date[maturity].items()):
        # Plot line for all execution dates
        ax.plot(
            rmse_data["execution_date"],
            rmse_data["rmse"],
            label=model_label,
            color=model_colors[model_idx % len(model_colors)],
            linewidth=1.5,
            zorder=2
        )
        # Add dots for each execution date
    # Style subplot
    style_subplot(ax)
    ax.set_xlabel("Execution Date", fontsize=10)
    ax.set_ylabel("", fontsize=10)

    # Add legend only for the first panel
    if idx == 0:
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.4),
            ncol=1,
            facecolor="white",
            frameon=False,
            fontsize=12
        )

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(rf"{graphs_folder}\rmse_yields_by_execution_date_nosharey_US.svg", format="svg")
plt.show()


#%%
# ...existing code...
# Load yields data
base_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US'
measure = 'yields'
estimated_path = fr'{base_folder}\{measure}\estimated_{measure}'

# Load data for Mixed_Model and AR_1
data_mixed_yields = pd.read_csv(fr'{estimated_path}\Mixed_Model\forecasts.csv')
data_ar1_yields = pd.read_csv(fr'{estimated_path}\AR_1\forecasts.csv')

# Normalize column names for consistency
data_mixed_yields = data_mixed_yields.rename(columns={"mean_simulated": "prediction"})
data_mixed_yields['maturity'] = data_mixed_yields['maturity'].astype(str) + ' years'
data_mixed_yields['forecast_date'] = pd.to_datetime(data_mixed_yields['forecast_date'])
data_mixed_yields['execution_date'] = pd.to_datetime(data_mixed_yields['execution_date'])

data_ar1_yields = data_ar1_yields.rename(columns={"mean_simulated": "prediction"})
data_ar1_yields['maturity'] = data_ar1_yields['maturity'].astype(str) + ' years'
data_ar1_yields['forecast_date'] = pd.to_datetime(data_ar1_yields['forecast_date'])
data_ar1_yields['execution_date'] = pd.to_datetime(data_ar1_yields['execution_date'])

# Define maturities
maturities = ['2.0 years', '5.0 years', '10.0 years']

# Plot setup
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_colors = {"Mixed_Model": "#3a6bac", "AR_1": "#aa322f"}  # Blue for Mixed_Model, Red for AR_1

for idx, maturity in enumerate(maturities):
    ax = axes[idx]
    # Filter Mixed_Model data from 1990 onward
    df_mixed = data_mixed_yields[(data_mixed_yields['maturity'] == maturity) & (data_mixed_yields['forecast_date'] >= '1990-01-01')].copy()
    df_ar1 = data_ar1_yields[(data_ar1_yields['maturity'] == maturity) & (data_ar1_yields['forecast_date'] >= '1990-01-01')].copy()

    # Plot all predictions for Mixed_Model
    for exec_date in df_mixed['execution_date'].unique():
        subset = df_mixed[df_mixed['execution_date'] == exec_date]
        ax.plot(subset['forecast_date'], subset['prediction']*100, color=model_colors["Mixed_Model"], alpha=0.3)

    # Plot all predictions for AR_1
    for exec_date in df_ar1['execution_date'].unique():
        subset = df_ar1[df_ar1['execution_date'] == exec_date]
        ax.plot(subset['forecast_date'], subset['prediction']*100, color=model_colors["AR_1"], alpha=0.3)

    # Add legend entries
    ax.plot([], [], color=model_colors["Mixed_Model"], label="Macro-based approach")
    ax.plot([], [], color=model_colors["AR_1"], label="AR(1) factors")

    # Plot actuals (mean by forecast_date)
    actuals = (
        df_mixed.groupby('forecast_date')['actual']
        .mean()
    )
    actuals.index = pd.to_datetime(actuals.index)
    ax.plot(actuals.index, actuals.values*100, color='black', linewidth=2, label='Actual')

    # Style the plot
    #ax.set_title(f"{maturity} yields", fontsize=12)
    ax.set_facecolor("#d5d6d2")
    ax.grid(True, color="white", linestyle='-', linewidth=1)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    #ax.axhline(0, color="black", linewidth=1.5, linestyle="-")

# Add legends to all subplots
for ax in axes:
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),
        ncol=1,
        facecolor="white",
        frameon=False,
        fontsize=12
    )

plt.tight_layout()
plt.savefig(rf"{graphs_folder}\forecast_vs_actuals_yields.svg", format="svg")
plt.show()
# ...existing code...
#%%
#%%
# ...existing code...
# Load yields data
base_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US'
measure = 'returns'
estimated_path = fr'{base_folder}\{measure}\estimated_{measure}'

# Load data for Mixed_Model and AR_1
data_mixed_yields = pd.read_csv(fr'{estimated_path}\Mixed_Model\monthly\forecasts.csv')
data_ar1_yields = pd.read_csv(fr'{estimated_path}\AR_1\monthly\forecasts.csv')

# Normalize column names for consistency
data_mixed_yields = data_mixed_yields.rename(columns={"mean_simulated": "prediction"})
data_mixed_yields['forecast_date'] = pd.to_datetime(data_mixed_yields['forecast_date'])
data_mixed_yields['execution_date'] = pd.to_datetime(data_mixed_yields['execution_date'])

data_ar1_yields = data_ar1_yields.rename(columns={"mean_simulated": "prediction"})
data_ar1_yields['forecast_date'] = pd.to_datetime(data_ar1_yields['forecast_date'])
data_ar1_yields['execution_date'] = pd.to_datetime(data_ar1_yields['execution_date'])

# Define maturities
maturities = ['2.0 years', '5.0 years', '10.0 years']

# Plot setup
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_colors = {"Mixed_Model": "#3a6bac", "AR_1": "#aa322f"}  # Blue for Mixed_Model, Red for AR_1

for idx, maturity in enumerate(maturities):
    ax = axes[idx]
    # Filter Mixed_Model data from 1990 onward
    df_mixed = data_mixed_yields[(data_mixed_yields['maturity'] == maturity) & (data_mixed_yields['forecast_date'] >= '2015-01-01')].copy()
    df_ar1 = data_ar1_yields[(data_ar1_yields['maturity'] == maturity) & (data_ar1_yields['forecast_date'] >= '2015-01-01')].copy()

    # Plot all predictions for Mixed_Model
    for exec_date in df_mixed['execution_date'].unique():
        subset = df_mixed[df_mixed['execution_date'] == exec_date]
        ax.plot(subset['forecast_date'], subset['prediction'], color=model_colors["Mixed_Model"], alpha=0.3)

    # Plot all predictions for AR_1
    for exec_date in df_ar1['execution_date'].unique():
        subset = df_ar1[df_ar1['execution_date'] == exec_date]
        ax.plot(subset['forecast_date'], subset['prediction'], color=model_colors["AR_1"], alpha=0.3)

    # Add legend entries
    ax.plot([], [], color=model_colors["Mixed_Model"], label="Macro-based approach")
    ax.plot([], [], color=model_colors["AR_1"], label="AR(1) factors")

    # Plot actuals (mean by forecast_date)
    actuals = (
        df_mixed.groupby('forecast_date')['actual']
        .last()
    )#.shift(-12)
    actuals.index = pd.to_datetime(actuals.index)
    ax.plot(actuals.index, actuals.values, color='black', linewidth=2, label='Actual')

    # Style the plot
    ax.set_title(f"{maturity} yields", fontsize=12)
    ax.set_facecolor("#d5d6d2")
    ax.grid(True, color="white", linestyle='-', linewidth=1)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.axhline(0, color="black", linewidth=1.5, linestyle="-")

# Add legends to all subplots
for ax in axes:
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),
        ncol=1,
        facecolor="white",
        frameon=False,
        fontsize=12
    )

plt.tight_layout()
plt.show()
# ...existing code...

#%% CONSENSUS

import pandas as pd
data_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\consensus_backtest'
master_rmse_horizon = pd.read_csv(rf"{data_folder}\rmse_horizon_all_countries_indicators.csv")

# Plot
import matplotlib.pyplot as plt

def plot_rmse_by_horizon_all_countries(master_rmse_horizon):
    countries = master_rmse_horizon['country'].unique()
    indicators = ["GDP", "CPI", "STR", "LTR"]
    model_colors = ["#aa322f", "#3a6bac"]  # AR(1), Consensus

    def style_subplot(ax):
        ax.set_facecolor("#d5d6d2")
        ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    for country in countries:
        fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)
        for i, indicator in enumerate(indicators):
            ax = axes[i]
            df = master_rmse_horizon[
                (master_rmse_horizon['country'] == country) &
                (master_rmse_horizon['indicator'] == indicator)
            ]
            if df.empty:
                ax.set_title(f"{indicator}\n(no data)")
                style_subplot(ax)
                continue
            horizons = df['horizon']
            ax.plot(horizons, df['RMSE_AR1'], label="AR(1)", color=model_colors[0], linewidth=2, zorder=2)
            ax.plot(horizons, df['RMSE_consensus'], label="Consensus", color=model_colors[1], linewidth=2, zorder=2)
            # Dots every 1 year (12 months)
            dot_horizons = horizons[horizons % 1 == 0] if horizons.dtype.kind in 'if' else horizons
            ax.scatter(dot_horizons, df.loc[horizons % 1 == 0, 'RMSE_AR1'], color=model_colors[0], s=40, zorder=3)
            ax.scatter(dot_horizons, df.loc[horizons % 1 == 0, 'RMSE_consensus'], color=model_colors[1], s=40, zorder=3)
            ax.set_title(indicator, fontsize=14)
            ax.set_xlabel("Horizon (years)", fontsize=12)
            #if i == 0:
            #    ax.set_ylabel("RMSE", fontsize=12)
            style_subplot(ax)
            if i == 0:
                ax.legend(
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.4),
                    ncol=1,
                    facecolor="white",
                    frameon=False,
                    fontsize=12
                )
        #fig.suptitle(f"RMSE by Horizon for {country}", fontsize=12)
        plt.tight_layout()
        plt.savefig(rf"{graphs_folder}\{country}_consensus_backtest.svg", format="svg")
        plt.show()

# Usage:
plot_rmse_by_horizon_all_countries(master_rmse_horizon)            

#%%
data_ar1 = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_horizon.csv')
data_mixed = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_horizon.csv')

# Filter maturities to include only 0.25, 2, 5, and 10 years
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
    
    # Calculate average CRPS by horizon (convert horizon to years)
    avg_crps_ar1 = ar1_data.groupby("horizon", as_index=False).agg({"crps": "mean"})
    avg_crps_ar1["horizon"] = avg_crps_ar1["horizon"] / 12  # Convert horizon to years
    
    avg_crps_mixed = mixed_data.groupby("horizon", as_index=False).agg({"crps": "mean"})
    avg_crps_mixed["horizon"] = avg_crps_mixed["horizon"] / 12  # Convert horizon to years
    
    # Plot AR_1 and Mixed_Model results
    ax.plot(avg_crps_ar1["horizon"], avg_crps_ar1["crps"], marker='o', color=model_colors["AR_1"], label="AR(1)", linewidth=2.5, zorder=2)
    ax.plot(avg_crps_mixed["horizon"], avg_crps_mixed["crps"], marker='o', color=model_colors["Mixed_Model"], label="Mixed_Model", linewidth=2.5, zorder=2)
    
    # Style subplot
    style_subplot(ax)
    
    # Add title and labels
    ax.set_xlabel("Horizon (years)", fontsize=12)
    #if idx == 0:  # Add y-axis label only for the first subplot
        #ax.set_ylabel("Average CRPS", fontsize=12)

    # Force integer x-axis ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add legend only for the first panel
    if idx == 0:
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            facecolor="white",
            frameon=False,
            fontsize=12
        )

# Adjust layout and show the plot
plt.tight_layout()  # Leave space for the legend
plt.savefig(rf"{graphs_folder}\crps_by_horizon_sharey_US.svg", format="svg")
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Define the style function
def style_subplot(ax):
    ax.set_facecolor("#d5d6d2")  # Grey background
    ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)  # White gridlines with low zorder
    ax.yaxis.tick_right()  # Move Y-axis to the right
    ax.yaxis.set_label_position("right")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# Generate data for the empirical CDF
fcst_thresholds = np.linspace(0, 7, 700)
empirical_cdf = xr.DataArray(
    coords={"temperature": fcst_thresholds},
    data=[0] * 120 + [0.1] * 80 + [0.2] * 70 + [0.3] * 20 + [0.7] * 30 + [0.8] * 60 + [0.9] * 120 + [1] * 200,
)
observed_cdf = np.heaviside(fcst_thresholds - 4.5, 1)  # Step function for the observation

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the observed CDF (step function)
ax.plot(fcst_thresholds, observed_cdf, label="Observation", color="#aa322f", linewidth=2.5, zorder=2)

# Plot the empirical CDF (based on the ensemble forecast)
ax.plot(fcst_thresholds, empirical_cdf, label="Forecast", color="#3a6bac", linewidth=2.5, zorder=2)

# Fill the area for CRPS
ax.fill_between(fcst_thresholds, empirical_cdf, observed_cdf, color="#eaa121", alpha=0.4, label="CRPS Area", zorder=1)

# Style the subplot
style_subplot(ax)

# Add labels, title, and legend
ax.set_title("Empirical CDF", fontsize=16)
ax.set_xlabel("Return", fontsize=12)
ax.set_ylabel("Probability", fontsize=12)
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.4),
    fontsize=12,
    facecolor="white",
    frameon=False
)

# Show the plot
plt.tight_layout()
plt.savefig(rf"{graphs_folder}\crps_concept.svg", format="svg")
plt.show()

#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Mixed_Model data
data_mixed_horizon = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_horizon.csv')
data_mixed_exec_date = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_execution_date.csv')

# Group data for heatmaps
# Horizon vs Maturity (CRPS mean)
horizon_results_mixed = data_mixed_horizon.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})
horizon_results_mixed['horizon'] = horizon_results_mixed['horizon']/12
horizon_results_mixed['horizon'] = horizon_results_mixed['horizon'].astype(int)
# Execution Date vs Maturity (CRPS mean)
exec_date_results_mixed = data_mixed_exec_date.groupby(["execution_date", "maturity"], as_index=False).agg({"crps": "mean"})

# Pivot the DataFrames for heatmaps
heatmap_data_mixed_horizon = horizon_results_mixed.pivot(index="horizon", columns="maturity", values="crps")

# Convert execution_date to datetime and format as year-month
exec_date_results_mixed["execution_date"] = pd.to_datetime(exec_date_results_mixed["execution_date"])
exec_date_results_mixed["execution_date"] = exec_date_results_mixed["execution_date"].dt.strftime('%Y-%m')

# Pivot the execution date results
heatmap_data_mixed_exec_date = exec_date_results_mixed.pivot(index="execution_date", columns="maturity", values="crps")

# Define the style function
def style_subplot(ax):
    ax.set_facecolor("#d5d6d2")  # Grey background
    #ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)  # White gridlines with low zorder
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# Create a 1x2 grid for Mixed_Model heatmaps
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Mixed_Model Heatmap: Horizon vs Maturity
sns.heatmap(heatmap_data_mixed_horizon.T, cmap="coolwarm", annot=False, fmt=".4f",  ax=axes[0])
#axes[0].set_title("CRPS by horizon and maturity", fontsize=14)
axes[0].set_xlabel("Horizon (years)", fontsize=12)
axes[0].set_ylabel("Maturity", fontsize=12)
style_subplot(axes[0])

# Invert the y-axis for the horizon heatmap
axes[0].invert_yaxis()

# Mixed_Model Heatmap: Execution Date vs Maturity
sns.heatmap(heatmap_data_mixed_exec_date.T, cmap="coolwarm", annot=False, fmt=".4f",  ax=axes[1])
#axes[1].set_title("CRPS by execution date and maturity", fontsize=14)
axes[1].set_xlabel("Execution date", fontsize=12)
axes[1].set_ylabel("", fontsize=12)

# Ensure all y-axis labels show only year and month
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=12)

# Invert the y-axis for the execution date heatmap
axes[1].invert_yaxis()

# Style the subplot
style_subplot(axes[1])

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(rf"{graphs_folder}\crps_heatmaps.svg", format="svg")
plt.show()



#%%

import pandas as pd
import matplotlib.pyplot as plt

# Load the AR_1 and Mixed_Model data
data_ar1 = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\AR_1\annual\crps_by_execution_date.csv')
data_mixed = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual\crps_by_execution_date.csv')

# Convert execution_date to datetime
data_ar1["execution_date"] = pd.to_datetime(data_ar1["execution_date"])
data_mixed["execution_date"] = pd.to_datetime(data_mixed["execution_date"])

# Define the maturities to include
selected_maturities = [0.25, 2.0, 5.0, 10.0]

# Filter the data for the selected maturities
data_ar1 = data_ar1[data_ar1["maturity"].isin(selected_maturities)]
data_mixed = data_mixed[data_mixed["maturity"].isin(selected_maturities)]

# Define the style function
def style_subplot(ax):
    ax.set_facecolor("#d5d6d2")  # Grey background
    ax.grid(True, color="white", linestyle='-', linewidth=1, zorder=0)  # White gridlines with low zorder
    ax.yaxis.tick_right()  # Move Y-axis to the right
    ax.yaxis.set_label_position("right")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# Prepare the subplots
fig, axes = plt.subplots(
    nrows=1, 
    ncols=len(selected_maturities),  # 1x4 layout
    figsize=(18, 6), 
    sharex=False, 
    sharey=False
)

# Plot Average CRPS Across Execution Dates for Selected Maturities
for idx, maturity in enumerate(selected_maturities):
    ax = axes[idx]
    
    # Filter data for the current maturity
    ar1_data = data_ar1[data_ar1["maturity"] == maturity]
    mixed_data = data_mixed[data_mixed["maturity"] == maturity]
    
    # Calculate average CRPS by execution date
    avg_crps_ar1 = ar1_data.groupby("execution_date", as_index=False).agg({"crps": "mean"})
    avg_crps_mixed = mixed_data.groupby("execution_date", as_index=False).agg({"crps": "mean"})
    
    # Plot AR_1 and Mixed_Model results
    ax.plot(
        avg_crps_ar1["execution_date"], 
        avg_crps_ar1["crps"], 
        color="#aa322f", 
        linewidth=2.5, 
        label="AR(1)"
    )
    ax.plot(
        avg_crps_mixed["execution_date"], 
        avg_crps_mixed["crps"], 
        color="#3a6bac", 
        linewidth=1.5, 
        label="Mixed_Model"
    )
    
    # Style subplot
    style_subplot(ax)
    
    # Add title and labels
    #ax.set_title(f"Maturity: {maturity} years", fontsize=14)
    ax.set_xlabel("Execution date", fontsize=12)
    #if idx == 0:  # Add y-axis label only for the first subplot
        #ax.set_ylabel("Average CRPS", fontsize=12)
    if idx == 0:
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.4),
            ncol=1,
            facecolor="white",
            frameon=False,
            fontsize=12
        )
# Add a unified title for the figure
#fig.suptitle("Average CRPS Across Execution Dates for Selected Maturities", fontsize=18)


# Adjust layout and show the plot
plt.tight_layout()  # Leave space for the legend
plt.savefig(rf"{graphs_folder}\crps_by_execution_date_US.svg", format="svg")
plt.show()
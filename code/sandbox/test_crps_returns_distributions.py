# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:06:01 2025

@author: al005366
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from scores.probability import crps_for_ensemble
import matplotlib.pyplot as plt

# Base folder containing simulation data
base_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_test\EA\returns\estimated_returns\AR_1\monthly\simulations'

# Observation data
data_obs = pd.read_csv(rf'L:\RMAS\Users\Alberto\backtest-baam\data_test\EA\returns\estimated_returns\AR_1\monthly\forecasts.csv')
data_obs["forecast_date"] = pd.to_datetime(data_obs["forecast_date"])

# Initialize a results DataFrame
results = []

# Iterate through maturity folders
for maturity_folder in os.listdir(base_folder):
    maturity_path = os.path.join(base_folder, maturity_folder)

    # Check if it's a folder
    if os.path.isdir(maturity_path):
        print(f"Processing maturity: {maturity_folder}")

        # Iterate through Parquet files (one for each execution date)
        for file in os.listdir(maturity_path):
            if file.endswith(".parquet"):
                execution_date = file.split("_")[-1].replace(".parquet", "")  # Extract execution date
                execution_date = pd.to_datetime(execution_date, format="%d%m%Y")  # Parse to datetime

                print(f"  Processing execution date: {execution_date}")

                # Load simulation data
                data_sim = pd.read_parquet(os.path.join(maturity_path, file))
                data_sim = data_sim.sort_values(by=["forecast_date", "simulation_id"])
                data_sim["forecast_date"] = pd.to_datetime(data_sim["forecast_date"])
                data_obs["execution_date"] = pd.to_datetime(data_obs["execution_date"])

                # Handle NaN values in simulations
                data_sim = data_sim.dropna(subset=["monthly_returns"])

                # Filter observation data for the current execution date and maturity
                obs_subset = data_obs[
                    (data_obs["execution_date"] == execution_date) &
                    (data_obs["maturity"] == maturity_folder.replace("_years", " years"))
                ]

                if obs_subset.empty:
                    print(f"    No observation data for execution date {execution_date} and maturity {maturity_folder}. Skipping...")
                    continue

                # Create xarray.DataArray for ensemble_forecast
                ensemble_forecast = xr.DataArray(
                    data=data_sim.pivot(index="forecast_date", columns="simulation_id", values="monthly_returns").values,
                    dims=["time", "ensemble_member"],
                    coords={
                        "time": data_sim["forecast_date"].unique(),
                        "ensemble_member": data_sim["simulation_id"].unique(),
                    },
                )

                # Aggregate observed data to ensure unique forecast_date
                obs_unique = obs_subset.groupby("forecast_date", as_index=False).agg({"actual": "mean"})

                # Create xarray.DataArray for observations
                obs_array = xr.DataArray(
                    data=obs_unique["actual"].values,
                    dims=["time"],
                    coords={"time": obs_unique["forecast_date"].values},
                )

                # Align ensemble_forecast and obs_array
                obs_array, ensemble_forecast = xr.align(obs_array, ensemble_forecast, join="inner")

                # Compute CRPS using empirical CDF
                crps_values = crps_for_ensemble(
                    ensemble_forecast, obs_array, ensemble_member_dim="ensemble_member", method="ecdf", preserve_dims="time"
                )

                # Store detailed results for each forecast_date
                for i, forecast_date in enumerate(ensemble_forecast.time.values):
                    results.append({
                        "execution_date": execution_date,
                        "maturity": maturity_folder,
                        "forecast_date": forecast_date,
                        "crps": crps_values[i].item(),  # CRPS for this forecast_date
                    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
# Compute the horizon in months
results_df["horizon"] = (results_df["forecast_date"].dt.year - results_df["execution_date"].dt.year) * 12 + (results_df["forecast_date"].dt.month - results_df["execution_date"].dt.month)
# Filter results for a specific execution date and maturity
# Extract the numeric part of the maturity column
results_df["maturity"] = results_df["maturity"].str.extract(r"([\d\.]+)").astype(float)

# Aggregate CRPS by horizon and maturity
horizon_results = results_df.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})
# Pivot the DataFrame for the heatmap
heatmap_data = horizon_results.pivot(index="horizon", columns="maturity", values="crps")

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="viridis", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'})

# Customize the plot
plt.title("CRPS Heatmap by Horizon and Maturity", fontsize=16)
plt.xlabel("Maturity", fontsize=14)
plt.ylabel("Horizon (Months)", fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()

# Aggregate CRPS by maturity
bar_data = results_df.groupby(["maturity"], as_index=False).agg({"crps": "mean"})

# Bar plot
plt.figure(figsize=(14, 6))
sns.barplot(data=bar_data, x="maturity", y="crps", palette="viridis")

# Customize the plot
plt.title("Average CRPS by Maturity", fontsize=16)
plt.xlabel("Maturity", fontsize=14)
plt.ylabel("Average CRPS", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

execution_dates_to_plot = results_df["execution_date"].unique()
maturities_to_plot = results_df["maturity"].unique()

for execution_date_to_plot in execution_dates_to_plot:
    for maturity_to_plot in maturities_to_plot:
        subset = results_df[
            (results_df["execution_date"] == execution_date_to_plot) &
            (results_df["maturity"] == maturity_to_plot)
        ]
        
        # Plot CRPS trends over forecast dates
        plt.figure(figsize=(12, 6))
        plt.plot(subset["forecast_date"], subset["crps"], marker='o', label=f"{maturity_to_plot} - {execution_date_to_plot}")
        plt.title(f"CRPS Trends Over Forecast Dates ({maturity_to_plot}, Execution Date: {execution_date_to_plot})", fontsize=16)
        plt.xlabel("Forecast Date", fontsize=14)
        plt.ylabel("CRPS", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()




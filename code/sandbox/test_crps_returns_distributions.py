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
import seaborn as sns
import time
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import seaborn as sns
 
 
root_folder = r"\\msfsshared\BNKG\\RMAS\Users\Alberto\backtest-baam\data_joint\US"
base_folder = rf'{root_folder}\returns\estimated_returns\Mixed_Model\monthly\simulations'
 
data_obs = pd.read_csv(rf'{root_folder}\returns\estimated_returns\Mixed_Model\monthly\forecasts.csv')
data_obs["forecast_date"] = pd.to_datetime(data_obs["forecast_date"])
  
def process_file(maturity_folder, maturity_path, file, data_obs):
    results = []
    if not file.endswith(".parquet"):
        return results
    execution_date = file.split("_")[-1].replace(".parquet", "")
    execution_date = pd.to_datetime(execution_date, format="%d%m%Y")
    data_sim = pd.read_parquet(os.path.join(maturity_path, file))
    data_sim = data_sim.sort_values(by=["forecast_date", "simulation_id"])
    data_sim["forecast_date"] = pd.to_datetime(data_sim["forecast_date"])
    data_obs["execution_date"] = pd.to_datetime(data_obs["execution_date"])
    data_sim = data_sim.dropna(subset=["monthly_returns"])
    obs_subset = data_obs[
        (data_obs["execution_date"] == execution_date) &
        (data_obs["maturity"] == maturity_folder.replace("_years", " years"))
    ]
    if obs_subset.empty:
        return results
    ensemble_forecast = xr.DataArray(
        data=data_sim.pivot(index="forecast_date", columns="simulation_id", values="monthly_returns").values,
        dims=["time", "ensemble_member"],
        coords={
            "time": data_sim["forecast_date"].unique(),
            "ensemble_member": data_sim["simulation_id"].unique(),
        },
    )
    obs_unique = obs_subset.groupby("forecast_date", as_index=False).agg({"actual": "mean"})
    obs_array = xr.DataArray(
        data=obs_unique["actual"].values,
        dims=["time"],
        coords={"time": obs_unique["forecast_date"].values},
    )
    obs_array, ensemble_forecast = xr.align(obs_array, ensemble_forecast, join="inner")
    crps_values = crps_for_ensemble(
        ensemble_forecast, obs_array, ensemble_member_dim="ensemble_member", method="ecdf", preserve_dims="time"
    )
    for i, forecast_date in enumerate(ensemble_forecast.time.values):
        results.append({
            "execution_date": execution_date,
            "maturity": maturity_folder,
            "forecast_date": forecast_date,
            "crps": crps_values[i].item(),
        })
    return results
 
results = []
 
start_time = time.time()
 
maturity_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
for maturity_folder in tqdm(maturity_folders, desc="Maturities"):
    maturity_path = os.path.join(base_folder, maturity_folder)
    parquet_files = [file for file in os.listdir(maturity_path) if file.endswith(".parquet")]
 
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, maturity_folder, maturity_path, file, data_obs)
            for file in parquet_files
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"{maturity_folder}", leave=False):
            results.extend(f.result())
 
end_time = time.time()
print(f"\nCompleted in {end_time - start_time:.2f} seconds.")
 
results_df = pd.DataFrame(results)
# Convert results to a DataFrame
results_df = pd.DataFrame(results)
# Compute the horizon in months
results_df["horizon"] = (results_df["forecast_date"].dt.year - results_df["execution_date"].dt.year) * 12 + (results_df["forecast_date"].dt.month - results_df["execution_date"].dt.month)
# Filter results for a specific execution date and maturity
# Extract the numeric part of the maturity column
results_df["maturity"] = results_df["maturity"].str.extract(r"([\d\.]+)").astype(float)

#%% heatmap
# Aggregate CRPS by horizon and maturity
horizon_results = results_df.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "median"})
# Pivot the DataFrame for the heatmap
heatmap_data = horizon_results.pivot(index="horizon", columns="maturity", values="crps")
# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="viridis", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'})
# Customize the plot
plt.title("CRPS Heatmap by Horizon and Maturity - MEDIAN", fontsize=16)
plt.xlabel("Maturity", fontsize=14)
plt.ylabel("Horizon (Months)", fontsize=14)
plt.tight_layout()
# Show the plot
plt.show()

#%% heatmap
# Aggregate CRPS by horizon and maturity
horizon_results = results_df.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "mean"})
exec_date_results = results_df.groupby(["execution_date", "maturity"], as_index=False).agg({"crps": "mean"})
# Pivot the DataFrame for the heatmap
heatmap_data = horizon_results.pivot(index="horizon", columns="maturity", values="crps")
# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="viridis", annot=False, fmt=".4f", cbar_kws={'label': 'CRPS'})
# Customize the plot
plt.title("CRPS Heatmap by Horizon and Maturity - MEAN", fontsize=16)
plt.xlabel("Maturity", fontsize=14)
plt.ylabel("Horizon (Months)", fontsize=14)
plt.tight_layout()
# Show the plot
plt.show()

#%% by horizon and executrion date for each maturity
unique_maturities = sorted(results_df["maturity"].unique())
n_maturities = len(unique_maturities)

fig, axes = plt.subplots(nrows=(n_maturities + 2) // 3, ncols=3, figsize=(18, 4 * ((n_maturities + 2) // 3)), sharex=False, sharey=False)
axes = axes.flatten()

for idx, maturity in enumerate(unique_maturities):
    ax = axes[idx]
    maturity_data = results_df[results_df["maturity"] == maturity]
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

fig, axes = plt.subplots(nrows=(n_maturities + 2) // 3, ncols=3, figsize=(18, 4 * ((n_maturities + 2) // 3)), sharex=False, sharey=False)
axes = axes.flatten()

for idx, maturity in enumerate(unique_maturities):
    ax = axes[idx]
    maturity_data = results_df[results_df["maturity"] == maturity]
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
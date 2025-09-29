# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:42:01 2025

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
 
def process_file(maturity_folder, maturity_path, file, data_obs):
    results = []
    if not file.endswith(".parquet"):
        return results
    execution_date = file.split("_")[-1].replace(".parquet", "")
    execution_date = pd.to_datetime(execution_date, format="%d%m%Y")
    data_sim = pd.read_parquet(os.path.join(maturity_path, file))
    
    if "forecast_date" not in data_sim.columns and 'horizon' in data_sim.columns and freq == 'annual':
        data_sim["execution_date"] = pd.to_datetime(data_sim["execution_date"])
        # If horizon is not integer, convert
        data_sim["horizon"] = data_sim["horizon"].astype(int)
        # Create forecast_date by adding horizon years to execution_date
        data_sim["forecast_date"] = data_sim.apply(
            lambda row: row["execution_date"] + pd.DateOffset(years=row["horizon"]), axis=1
        )
    
    returns_col = 'simulated_value'
    # Check for required columns
    required_cols = {"forecast_date", "simulation_id", returns_col}
    missing_cols = required_cols - set(data_sim.columns)
    if missing_cols:
        print(f"Missing columns {missing_cols} in file: {os.path.join(maturity_path, file)}")
        return results

    data_sim = data_sim.sort_values(by=["forecast_date", "simulation_id"])
    data_sim["forecast_date"] = pd.to_datetime(data_sim["forecast_date"])
    data_obs["execution_date"] = pd.to_datetime(data_obs["execution_date"])
    data_sim = data_sim.dropna(subset=[returns_col])
    obs_subset = data_obs[
        (data_obs["execution_date"] == execution_date) &
        (data_obs["maturity"] == maturity_folder.replace("_years", " years"))
    ]
    if obs_subset.empty:
        return results
    ensemble_forecast = xr.DataArray(
        data=data_sim.pivot(index="forecast_date", columns="simulation_id", values=returns_col).values,
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

country = 'EA'
root_folder = rf"\\msfsshared\BNKG\\RMAS\Users\Alberto\backtest-baam\data_joint\{country}"
returns_folder = os.path.join(root_folder, "returns", "estimated_returns")
results_store = {}

for model in ['Mixed_Model_curvMacro']: # os.listdir(returns_folder) # 'AR_1', 'Mixed_Model'
    model_path = os.path.join(returns_folder, model)
    if not os.path.isdir(model_path):
        continue
    for freq in ['annual', 'monthly']:
        freq_path = os.path.join(model_path, freq)
        if not os.path.isdir(freq_path):
            continue
        sim_path = os.path.join(freq_path, "simulations")
        forecasts_path = os.path.join(freq_path, "forecasts.csv")
        if not os.path.exists(sim_path) or not os.path.exists(forecasts_path):
            continue

        print(f"Processing Model: {model}, Frequency: {freq}")
        data_obs = pd.read_csv(forecasts_path)
        data_obs["forecast_date"] = pd.to_datetime(data_obs["forecast_date"])

        results = []
        start_time = time.time()
        maturity_folders = [f for f in os.listdir(sim_path) if os.path.isdir(os.path.join(sim_path, f))]
        for maturity_folder in tqdm(maturity_folders, desc="Maturities"):
            maturity_path = os.path.join(sim_path, maturity_folder)
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
        results_df["horizon"] = (results_df["forecast_date"].dt.year - results_df["execution_date"].dt.year) * 12 + \
                                (results_df["forecast_date"].dt.month - results_df["execution_date"].dt.month)
        results_df["maturity"] = results_df["maturity"].str.extract(r"([\d\.]+)").astype(float)

        # Aggregate CRPS by horizon and maturity
        horizon_results = results_df.groupby(["horizon", "maturity"], as_index=False).agg({"crps": "median"})
        exec_date_results = results_df.groupby(["execution_date", "maturity"], as_index=False).agg({"crps": "mean"})

        # Store results in dictionary
        key = f"{model}_{freq}"
        results_store[key] = {
            "results_df": results_df,
            "horizon_results": horizon_results,
            "exec_date_results": exec_date_results
        }

        # Save to disk
        results_df.to_csv(os.path.join(freq_path, "crps_by_row.csv"), index=False)
        horizon_results.to_csv(os.path.join(freq_path, "crps_by_horizon.csv"), index=False)
        exec_date_results.to_csv(os.path.join(freq_path, "crps_by_execution_date.csv"), index=False)

        print(f"Saved results for {key} to {freq_path}")
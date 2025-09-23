# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:05:29 2025

@author: al005366
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from scores.probability import crps_for_ensemble

base_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_test\US\returns\estimated_returns'
frequencies = ["monthly", "annual"]
models = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
results_horizon = []
results_execution = []

for model in models:
    for freq in frequencies:
        sim_base_folder = os.path.join(base_folder, model, freq, "simulations")
        forecasts_file = os.path.join(base_folder, model, freq, "forecasts.csv")
        if not os.path.exists(forecasts_file):
            print(f"Missing forecasts for {model} {freq}")
            continue
        data_obs = pd.read_csv(forecasts_file)
        data_obs["forecast_date"] = pd.to_datetime(data_obs["forecast_date"])
        data_obs["execution_date"] = pd.to_datetime(data_obs["execution_date"])
        maturities = [f for f in os.listdir(sim_base_folder) if os.path.isdir(os.path.join(sim_base_folder, f))]
        for maturity_folder in maturities:
            maturity_path = os.path.join(sim_base_folder, maturity_folder)
            parquet_files = [f for f in os.listdir(maturity_path) if f.endswith(".parquet")]
            for file in parquet_files:
                execution_date_str = file.split("_")[-1].replace(".parquet", "")
                try:
                    execution_date = pd.to_datetime(execution_date_str, format="%d%m%Y")
                except Exception:
                    continue
                data_sim = pd.read_parquet(os.path.join(maturity_path, file))
                data_sim = data_sim.sort_values(by=["forecast_date", "simulation_id"])
                data_sim["forecast_date"] = pd.to_datetime(data_sim["forecast_date"])
                data_sim = data_sim.dropna(subset=["monthly_returns"])
                obs_subset = data_obs[
                    (data_obs["execution_date"] == execution_date) &
                    (data_obs["maturity"] == maturity_folder.replace("_years", " years"))
                ]
                if obs_subset.empty:
                    continue
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
                    horizon = (pd.to_datetime(forecast_date).year - execution_date.year) * 12 + \
                              (pd.to_datetime(forecast_date).month - execution_date.month)
                    results_horizon.append({
                        "model": model,
                        "frequency": freq,
                        "maturity": maturity_folder,
                        "execution_date": execution_date,
                        "forecast_date": forecast_date,
                        "horizon": horizon,
                        "crps": crps_values[i].item()
                    })
                    results_execution.append({
                        "model": model,
                        "frequency": freq,
                        "maturity": maturity_folder,
                        "execution_date": execution_date,
                        "crps": crps_values[i].item()
                    })

# Save CRPS by horizon (grouped by model, frequency, maturity, horizon)
df_horizon = pd.DataFrame(results_horizon)
df_horizon_grouped = df_horizon.groupby(["model", "frequency", "maturity", "horizon"], as_index=False).agg({"crps": "mean"})
df_horizon_grouped.to_csv("crps_by_horizon.csv", index=False)

# Save CRPS by execution date (grouped by model, frequency, maturity, execution_date)
df_execution = pd.DataFrame(results_execution)
df_execution_grouped = df_execution.groupby(["model", "frequency", "maturity", "execution_date"], as_index=False).agg({"crps": "mean"})
df_execution_grouped.to_csv("crps_by_execution_date.csv", index=False)

print("CRPS calculation complete. Files saved: crps_by_horizon.csv, crps_by_execution_date.csv")
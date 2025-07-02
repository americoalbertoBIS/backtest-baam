# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 09:21:12 2025

@author: al005366
"""

import pandas as pd
import matplotlib.pyplot as plt

# Filter RMSE data
rmse_df = metrics_df[metrics_df["Metric"] == "RMSE"]
rmse_df = rmse_df[rmse_df['Horizon (Years)']!=0.5]
# RMSE by Horizon
rmse_by_horizon = (
    rmse_df.groupby("Horizon (Years)")["Value"]
    .mean()
    .reset_index()
    .rename(columns={"Value": "RMSE"})
)

# RMSE by Execution Date
rmse_by_execution_date = (
    rmse_df.groupby(["Execution Date","Maturity (Years)"])["Value"]
    .mean()
    .reset_index()
    .rename(columns={"Value": "RMSE"})
)

# Plot RMSE by Horizon
def plot_rmse_by_horizon(rmse_by_horizon, country):
    """
    Plot RMSE by horizon for a single forecast type.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_by_horizon["Horizon (Years)"], rmse_by_horizon["RMSE"], label="RMSE", marker="o")
    plt.xlabel("Horizon (Years)")
    plt.ylabel("RMSE")
    plt.title(f"RMSE Across Horizons ({country})")
    plt.legend()
    plt.grid()
    plt.show()

# Plot RMSE by Execution Date
def plot_rmse_by_execution_date(rmse_by_execution_date, country):
    """
    Plot RMSE across execution dates for a single forecast type.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_by_execution_date["Execution Date"], rmse_by_execution_date["RMSE"]*100, label="RMSE")
    plt.xlabel("Execution Date")
    plt.ylabel("RMSE")
    plt.title(f"Time Series of RMSE Across Execution Dates ({country})")
    plt.legend()
    plt.grid()
    plt.show()

def plot_rmse_by_execution_date_by_maturity(rmse_by_execution_date, rmse_exec, maturity, country):
    """
    Plot RMSE across execution dates for a single forecast type.
    """
    plt.figure(figsize=(10, 6))
    rmse_by_execution_date = rmse_by_execution_date[rmse_by_execution_date['Maturity (Years)']==maturity]
    plt.plot(rmse_by_execution_date["Execution Date"], rmse_by_execution_date["RMSE"]*100, label="RMSE_factors_AR1")
    plt.plot(rmse_exec["executionDate"], rmse_exec["RMSE_AR1"], label="RMSE_AR1")
    plt.plot(rmse_exec["executionDate"], rmse_exec["RMSE_consensus"], label="RMSE_consensus")
    
    plt.xlabel("Execution Date")
    plt.ylabel("RMSE")
    plt.title(f"RMSE across Execution Date for {country} for maturity {maturity}")
    plt.legend()
    plt.grid()
    plt.show()
    
# Example usage
country = "Example Country"
indicator = "Example Indicator"

plot_rmse_by_horizon(rmse_by_horizon, country)

plot_rmse_by_execution_date(rmse_by_execution_date, country)

plot_rmse_by_execution_date_by_maturity(rmse_by_execution_date, rmse_exec, 0.25, country)

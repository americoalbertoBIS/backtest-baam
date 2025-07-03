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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_rmse(metrics_df):
    """
    Create a 3D plot for RMSE values.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing RMSE values with columns:
            - "Maturity (Years)"
            - "Horizon (Years)"
            - "Metric"
            - "Value"
    """
    # Filter the DataFrame for RMSE values
    rmse_df = metrics_df[metrics_df["Metric"] == "RMSE"]
    rmse_df['Value'] = pd.to_numeric(rmse_df['Value'])*100
    # Extract unique maturities and horizons
    maturities = rmse_df["Maturity (Years)"].unique()
    horizons = rmse_df["Horizon (Years)"].unique()

    # Create a meshgrid for maturities and horizons
    X, Y = np.meshgrid(maturities, horizons)

    # Map RMSE values to the grid
    Z = np.zeros_like(X, dtype=float)
    for i, horizon in enumerate(horizons):
        for j, maturity in enumerate(maturities):
            # Get RMSE value for the current maturity and horizon
            value = rmse_df[
                (rmse_df["Maturity (Years)"] == maturity) &
                (rmse_df["Horizon (Years)"] == horizon)
            ]["Value"].values 
            Z[i, j] = value[0] if len(value) > 0 else np.nan  # Handle missing values

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8)

    # Add labels and title
    ax.set_xlabel("Maturity (Years)", fontsize=12)
    ax.set_ylabel("Horizon (Years)", fontsize=12)
    ax.set_zlabel("RMSE", fontsize=12)
    ax.set_title("3D Plot of RMSE", fontsize=14)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="RMSE")

    # Show the plot
    plt.tight_layout()
    plt.show()                    



# Plot 3D RMSE
plot_3d_rmse(metrics_df)

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

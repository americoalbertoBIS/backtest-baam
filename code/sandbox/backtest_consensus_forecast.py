# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 13:36:04 2025

@author: al005366
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Set working directory
os.chdir(r'C:\git\backtest-baam\code')

# Import custom modules
from data_preparation.data_loader import DataLoader
from data_preparation.conensus_forecast import ConsensusForecast
from modeling.macro_modeling import convert_gdp_to_monthly
from config_paths import QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH
from medconn import read_fred



# Constants
PUBLICATION_LAGS = {"GDP": 3, "CPI": 1, "STR": 0, "LTR": 0}  # Publication lags by indicator
HIGHLIGHT_STEPS = 120  # Steps for highlighting in plots


# ================================
# Utility Functions
# ================================

def replace_last_n_with_nan(series, n):
    """
    Replaces the last 'n' non-NaN values in a series with NaNs.
    """
    non_nan_indices = series.dropna().index[-n:]
    series.loc[non_nan_indices] = np.nan
    return series


def calculate_rmse(x, forecast_col, actual_col):
    """
    Calculate RMSE for a forecast column and actual column.
    """
    return np.sqrt(np.mean((x[forecast_col] - x[actual_col]) ** 2))


def plot_forecast_with_realized(country, indicator, df_forecasts, realized, highlight_steps=HIGHLIGHT_STEPS):
    """
    Plot forecasted vs. realized values.

    Parameters:
    - country: Country code (e.g., 'US').
    - indicator: Indicator name (e.g., 'GDP').
    - df_forecasts: DataFrame of forecasts with columns ['forecast_date', 'forecasted_month', 'monthly_forecast'].
    - realized: Series of realized values.
    - highlight_steps: Steps for highlighting forecast lines.
    """
    plt.figure(figsize=(8, 5))
    df_forecasts = df_forecasts.dropna()
    unique_forecast_dates = df_forecasts['forecast_date'].unique()

    for i, forecast_date in enumerate(unique_forecast_dates):
        sub_df = df_forecasts[df_forecasts['forecast_date'] == forecast_date].dropna()
        sub_df.index = pd.to_datetime(sub_df['forecasted_month'])
        sub_df = sub_df.resample('YS').last()

        projection_dates = pd.to_datetime(sub_df['forecasted_month'])
        projection_values = sub_df['monthly_forecast']

        if i % highlight_steps == 0:
            plt.plot(projection_dates, projection_values * 12, linewidth=2,
                     label=f"Forecast {pd.to_datetime(forecast_date).strftime('%Y-%m')}")
        else:
            plt.plot(projection_dates, projection_values * 12, color='grey', alpha=0.1)

    plt.plot(realized[realized.index > unique_forecast_dates[0]], color='k', linewidth=2, label="Realized")
    plt.title(f"Forecasted vs Realized {indicator} for {country}")
    plt.xlabel("Date")
    plt.ylabel("%")
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_rmse_by_horizon(rmse_by_horizon, country, indicator):
    """
    Plot RMSE by horizon for AR(1) and consensus forecasts.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_by_horizon["horizon"], rmse_by_horizon["RMSE_AR1"], label="AR(1) RMSE", marker="o")
    plt.plot(rmse_by_horizon["horizon"], rmse_by_horizon["RMSE_consensus"], label="Consensus RMSE", marker="x")
    plt.xlabel("Horizon (Years)")
    plt.ylabel("RMSE")
    plt.title(f"RMSE Across Horizons ({country} - {indicator})")
    plt.legend()
    plt.grid()
    plt.show()


def plot_rmse_by_execution_date(rmse_by_execution_date, country, indicator):
    """
    Plot RMSE across execution dates for AR(1) and consensus forecasts.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_by_execution_date["executionDate"], rmse_by_execution_date["RMSE_AR1"], label="AR(1) RMSE")
    plt.plot(rmse_by_execution_date["executionDate"], rmse_by_execution_date["RMSE_consensus"], label="Consensus RMSE")
    plt.xlabel("Execution Date")
    plt.ylabel("RMSE")
    plt.title(f"Time Series of RMSE Across Execution Dates ({country} - {indicator})")
    plt.legend()
    plt.grid()
    plt.show()


import math

def plot_rmse_by_horizon_and_execution(rmse_by_horizon_and_execution, country, indicator):
    """
    Plot RMSE by horizon and execution date for AR(1) and consensus forecasts using a grid of subplots.

    Parameters:
    - rmse_by_horizon_and_execution: DataFrame with RMSE values for each horizon and execution date.
    - country: Country code (e.g., 'US').
    - indicator: Indicator name (e.g., 'GDP').
    """
    horizons = rmse_by_horizon_and_execution["horizon"].unique()
    num_horizons = len(horizons)

    # Determine grid size (rows and columns) to make the layout as square as possible
    num_cols = math.ceil(math.sqrt(num_horizons))  # Number of columns in the grid
    num_rows = math.ceil(num_horizons / num_cols)  # Number of rows in the grid

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), constrained_layout=True)
    axes = np.array(axes)  # Ensure axes is always a 2D array for easier indexing

    # Flatten axes for easy iteration (handles 1D and 2D cases)
    axes = axes.flatten()

    for i, horizon in enumerate(horizons):
        ax = axes[i]
        subset = rmse_by_horizon_and_execution[rmse_by_horizon_and_execution["horizon"] == horizon]
        
        # Plot RMSE for AR(1) and Consensus
        ax.plot(subset["executionDate"], subset["RMSE_AR1"], label="AR(1) RMSE")
        ax.plot(subset["executionDate"], subset["RMSE_consensus"], label="Consensus RMSE")
        
        # Customize subplot
        ax.set_title(f"Horizon {horizon} Years")
        ax.set_xlabel("Execution Date")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid()

    # Hide unused subplots if there are empty slots in the grid
    for j in range(num_horizons, len(axes)):
        axes[j].axis("off")

    # Add a common title for the entire figure
    fig.suptitle(f"Time Series of RMSE by Horizon ({country} - {indicator})", fontsize=16)
    plt.show()
    
# ================================
# Data Preparation
# ================================

def prepare_macro_data(country):
    """
    Prepare macroeconomic data for a given country.
    """
    variable_list = ['GDP', 'IP', 'CPI']
    data_loader = DataLoader(country=country, variable_list=variable_list, shadow=True)
    return data_loader.get_macro_data()


def prepare_forecast_data(country, indicator):
    """
    Prepare consensus forecast data for a given country and indicator.
    """
    consensus_forecast = ConsensusForecast(QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH)
    return consensus_forecast.get_consensus_forecast(country_var=f"{country} {indicator}")


# ================================
# Backtesting Function
# ================================

def run_backtest_with_projections(country, indicator, df, df_consensus, forecast_horizon=120):
    """
    Run a backtest with AR(1) and consensus forecasts.

    Parameters:
    - country: Country code (e.g., 'US').
    - indicator: Indicator name (e.g., 'GDP').
    - df: DataFrame with base macro data.
    - df_consensus: DataFrame with consensus forecast data.
    - forecast_horizon: Number of months for the forecast horizon.

    Returns:
    - final_df: DataFrame with forecasts and realized values.
    - rmse_by_horizon: RMSE by horizon (full sample).
    - rmse_by_execution_date: RMSE by execution date (across all horizons).
    - rmse_by_horizon_and_execution: RMSE by horizon and execution date.
    """
    all_results = []
    df_consensus = df_consensus.dropna()
    execution_dates = df_consensus['forecast_date'].unique()
    publication_lags = PUBLICATION_LAGS[indicator]

    for execution_date in execution_dates:
        temp_exec = pd.DataFrame()
        temp_exec[f'{country}_{indicator}'] = df[f'{country}_{indicator}'][df.index <= execution_date].dropna().copy()
        if publication_lags != 0:
            temp_exec[f'{country}_{indicator}'] = replace_last_n_with_nan(temp_exec[f'{country}_{indicator}'], publication_lags)
        train_data = temp_exec.loc[:execution_date].dropna().copy()

        # Fit AR(1) model
        target_col = f'{country}_{indicator}'
        series = train_data[target_col].dropna().copy()
        lagged_series = series.shift(1).dropna()
        X = sm.add_constant(lagged_series)
        y = series.loc[X.index]
        model = sm.OLS(y, X).fit()

        # Generate AR(1) forecasts iteratively
        forecasts = []
        last_value = series.iloc[-1]
        for _ in range(forecast_horizon):
            next_forecast = model.params["const"] + model.params[series.name] * last_value
            forecasts.append(next_forecast)
            last_value = next_forecast

        # Resample AR(1) forecasts to annual frequency
        forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq="MS")
        series_fcst = pd.Series(forecasts, index=forecast_index)
        series_fcst_y = series_fcst.resample('YS').last()

        # Align with consensus forecasts
        df_forecast_date = df_consensus[df_consensus['forecast_date'] == execution_date].dropna()
        df_forecast_date = df_forecast_date[['forecasted_month', 'monthly_forecast']]
        df_forecast_date.set_index('forecasted_month', inplace=True)
        df_forecast_y = df_forecast_date['monthly_forecast'].resample('YS').last() * 12

        # Align with realized values
        temp_y = df[df.index > execution_date].resample('YS').mean().iloc[:len(series_fcst_y)]
        aligned_data = pd.concat([series_fcst_y, df_forecast_y, temp_y[f'{country}_{indicator}']], axis=1,
                                  keys=["AR1_fcst", "consensus_fcst", "actual"])
        aligned_data = aligned_data.dropna()

        # Add metadata columns
        aligned_data["executionDate"] = execution_date
        aligned_data["forecastedDate"] = aligned_data.index
        aligned_data["horizon"] = (aligned_data.index.year - execution_date.year)# + 1
        aligned_data["indicator"] = indicator
        all_results.append(aligned_data.reset_index(drop=True))

    final_df = pd.concat(all_results, ignore_index=True)

    # Compute RMSE metrics
    rmse_by_horizon = final_df.groupby("horizon").apply(
        lambda x: pd.Series({
            "RMSE_AR1": calculate_rmse(x, "AR1_fcst", "actual"),
            "RMSE_consensus": calculate_rmse(x, "consensus_fcst", "actual")
        })
    ).reset_index()

    rmse_by_execution_date = final_df.groupby("executionDate").apply(
        lambda x: pd.Series({
            "RMSE_AR1": calculate_rmse(x, "AR1_fcst", "actual"),
            "RMSE_consensus": calculate_rmse(x, "consensus_fcst", "actual")
        })
    ).reset_index()

    rmse_by_horizon_and_execution = final_df.groupby(["executionDate", "horizon"]).apply(
        lambda x: pd.Series({
            "RMSE_AR1": calculate_rmse(x, "AR1_fcst", "actual"),
            "RMSE_consensus": calculate_rmse(x, "consensus_fcst", "actual")
        })
    ).reset_index()

    return final_df, rmse_by_horizon, rmse_by_execution_date, rmse_by_horizon_and_execution


# ================================
# Main Script
# ================================

data, meta = read_fred(['TB3MS', 'IR3TIB01DEM156N', 'IR3TIB01GBM156N',
                        'DGS10', 'IRLTLT01DEM156N', 'IRLTLT01GBM156N'])
data = data.resample('MS').mean()
data.columns = ['US_STR', 'EA_STR', 'UK_STR',
                'US_LTR', 'EA_LTR', 'UK_LTR']
data.index = pd.to_datetime(data.index)
data = data.replace({pd.NA: np.nan})
data = data.astype(float)

# Define indicators and their exceptions
indicators = ["GDP", "CPI", "STR", "LTR"]
macro = ["GDP", "CPI"]  # Indicators requiring special transformations

master_df = pd.DataFrame()
master_rmse_horizon = pd.DataFrame()
master_rmse_exec = pd.DataFrame()
master_rmse_horizon_exec = pd.DataFrame()

for country in ['US', 'EA', 'UK']:
    # Prepare macro data
    df_macro = prepare_macro_data(country)

    # Apply transformations for GDP and CPI
    if "GDP" in indicators:
        df_macro[f'{country}_GDP'] = df_macro.filter(regex='GDP').dropna().pct_change(4, fill_method=None) * 100
    if "CPI" in indicators:
        df_macro[f'{country}_CPI'] = df_macro.filter(regex='CPI').dropna().pct_change(12, fill_method=None) * 100

    # Loop over indicators
    for indicator in indicators:
        print(f"Running backtest for {country} - {indicator}")

        # Prepare forecast data
        try:
            if indicator == 'CPI':
                df_consensus, _ = prepare_forecast_data(country, 'INF')
            else:
                df_consensus, _ = prepare_forecast_data(country, indicator)
        except Exception as e:
            print(f"Error preparing consensus forecast data for {country} - {indicator}: {e}")
            continue

        # Select the appropriate data source
        if indicator in macro:
            df = df_macro
            realized_series = df_macro[f"{country}_{indicator}"].dropna()
        else:
            df = data
            realized_series = data[f"{country}_{indicator}"].dropna()

        # Run backtest
        try:
            final_df, rmse_horizon, rmse_exec, rmse_horizon_exec = run_backtest_with_projections(
                country, indicator, df, df_consensus
            )
        except Exception as e:
            print(f"Error running backtest for {country} - {indicator}: {e}")
            continue
        
        rmse_horizon['country'] = country
        rmse_horizon['indicator'] = indicator

        rmse_exec['country'] = country
        rmse_exec['indicator'] = indicator

        rmse_horizon_exec['country'] = country
        rmse_horizon_exec['indicator'] = indicator
        
        master_rmse_horizon = pd.concat([master_rmse_horizon, rmse_horizon], ignore_index=True)
        master_rmse_exec = pd.concat([master_rmse_exec, rmse_exec], ignore_index=True)
        master_rmse_horizon_exec = pd.concat([master_rmse_horizon_exec, rmse_horizon_exec], ignore_index=True)
        
        final_df['country'] = country
        master_df = pd.concat([master_df, final_df], ignore_index=True)
        
        # Plot forecasts vs. realized values
        try:
            plot_forecast_with_realized(
                country=country,
                indicator=indicator,
                df_forecasts=df_consensus,
                realized=realized_series
            )
        except Exception as e:
            print(f"Error plotting forecasts for {country} - {indicator}: {e}")
            continue

        # Plot RMSE metrics
        try:
            plot_rmse_by_horizon(rmse_horizon, country, indicator)
            plot_rmse_by_execution_date(rmse_exec, country, indicator)
            plot_rmse_by_horizon_and_execution(rmse_horizon_exec, country, indicator)
        except Exception as e:
            print(f"Error plotting RMSE metrics for {country} - {indicator}: {e}")
            continue

output_file = r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\backtest_results_all_countries_indicators.csv"
master_df.to_csv(output_file, index=False)

master_rmse_horizon.to_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_horizon_all_countries_indicators.csv", index=False)
master_rmse_exec.to_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_exec_all_countries_indicators.csv", index=False)
master_rmse_horizon_exec.to_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_horizon_exec_all_countries_indicators.csv", index=False)

#%%

import pandas as pd
data_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\consensus_backtest'
master_df = pd.read_csv(rf"{data_folder}\backtest_results_all_countries_indicators.csv")
master_rmse_horizon = pd.read_csv(rf"{data_folder}\rmse_horizon_all_countries_indicators.csv")
master_rmse_exec = pd.read_csv(rf"{data_folder}\rmse_exec_all_countries_indicators.csv")
master_rmse_horizon_exec = pd.read_csv(rf"{data_folder}\rmse_horizon_exec_all_countries_indicators.csv")

# Example for US GDP
country = "US"
indicator = "GDP"

# Filter for the country/indicator
rmse_horizon = master_rmse_horizon[(master_rmse_horizon['country'] == country) & (master_rmse_horizon['indicator'] == indicator)]
rmse_exec = master_rmse_exec[(master_rmse_exec['country'] == country) & (master_rmse_exec['indicator'] == indicator)]
rmse_exec['executionDate'] = pd.to_datetime(rmse_exec['executionDate'])
rmse_horizon_exec = master_rmse_horizon_exec[(master_rmse_horizon_exec['country'] == country) & (master_rmse_horizon_exec['indicator'] == indicator)]
rmse_horizon_exec['executionDate'] = pd.to_datetime(rmse_horizon_exec['executionDate'])
# Plot
plot_rmse_by_horizon(rmse_horizon, country, indicator)
plot_rmse_by_execution_date(rmse_exec, country, indicator)
plot_rmse_by_horizon_and_execution(rmse_horizon_exec, country, indicator)

# Get all unique countries and indicators
countries = master_rmse_horizon['country'].unique()
indicators = master_rmse_horizon['indicator'].unique()

for country in countries:
    for indicator in indicators:
        print(f"Plotting for {country} - {indicator}")
        rmse_horizon = master_rmse_horizon[
            (master_rmse_horizon['country'] == country) & 
            (master_rmse_horizon['indicator'] == indicator)
        ]
        rmse_exec = master_rmse_exec[
            (master_rmse_exec['country'] == country) & 
            (master_rmse_exec['indicator'] == indicator)
        ]
        rmse_exec['executionDate'] = pd.to_datetime(rmse_exec['executionDate'])
        rmse_horizon_exec = master_rmse_horizon_exec[
            (master_rmse_horizon_exec['country'] == country) & 
            (master_rmse_horizon_exec['indicator'] == indicator)
        ]
        rmse_horizon_exec['executionDate'] = pd.to_datetime(rmse_horizon_exec['executionDate'])
        if not rmse_horizon.empty:
            plot_rmse_by_horizon(rmse_horizon, country, indicator)
        if not rmse_exec.empty:
            plot_rmse_by_execution_date(rmse_exec, country, indicator)
        if not rmse_horizon_exec.empty:
            plot_rmse_by_horizon_and_execution(rmse_horizon_exec, country, indicator)

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
        #fig.suptitle(f"RMSE by Horizon for {country}", fontsize=18)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

# Usage:
plot_rmse_by_horizon_all_countries(master_rmse_horizon)            
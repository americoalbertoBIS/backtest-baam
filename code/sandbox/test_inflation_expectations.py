# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:27:29 2025

@author: al005366
"""

import pandas as pd
import numpy as np
import matlab.engine

import medengine as me

import os
os.chdir(r'C:\git\backtest-baam\code')

from data_preparation.time_series_helpers import ar_extrapol, extrapolate_series
from data_preparation.data_transformations import FREQCHG_Q2M_EXO, replace_last_n_with_nan, convert_mom_to_yoy
from modeling.time_series_modeling import AR1Model

MATLAB_MAIN_FOLDER = r'C:\git\vAppDesigner\src'

import pandas as pd
import numpy as np

from data_preparation.data_loader import DataLoader

country = 'US'
variable_list = ['GDP', 'IP', 'CPI']  # List of macroeconomic variables
shadow_flag = True  # Whether to use shadow rotation for betas
save_dir = r"C:\git\backtest-baam\data"  # Directory to save results
horizons = [6, 12, 24, 36, 48, 60]  # Forecast horizons
target_columns = ['beta1', 'beta2', 'beta3']  # Target columns for backtesting
num_simulations = 1000  # Number of simulations for each execution date
max_workers = os.cpu_count() // 2  # Use half of the available CPU cores for parallel processing

# Step 2: Initialize DataLoader
data_loader = DataLoader(country=country, variable_list=variable_list, shadow=shadow_flag)

# Step 3: Load combined data
df_combined = data_loader.get_data()

from data_preparation.conensus_forecast import ConsensusForecast
from config_paths import QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH

consensus_forecast = ConsensusForecast(QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH)
df_consensus_inf, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} INF")


def inflation_ucsv_matlab(series):

    # Scaling
    # scaling = 100 if np.max(series) < 1 else 1
    # series *= scaling

    # Filter valid data
    first_valid_index = series.first_valid_index()
    series = series.loc[first_valid_index:]

    # AR extrapolation
    series_extrap = ar_extrapol(series.to_numpy(copy=True), AROrder=1, DiffFlag=0)

    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(MATLAB_MAIN_FOLDER))
    series_matlab = matlab.double(series.values.tolist())
    fit_result = eng.UCSVFit(series_matlab)
    eng.quit()

    ucsv_baam = np.array(fit_result)

    return ucsv_baam

def inflation_expectations(country, data, consensus_df, execution_date, method="default", macro_forecast="consensus"):
    """
    Calculates inflation expectations for both training and testing data using the specified method.

    Args:
        country (str): Country name or code.
        data (pd.DataFrame): Combined dataset (both train and test data).
        consensus_df (pd.DataFrame): Consensus forecast data.
        execution_date (datetime): Execution date for the calculation.
        method (str): Method for calculating inflation expectations ("default" or "ucsv").
        macro_forecast (str): Macro forecast method ("consensus" or "ar_1"). Defaults to "consensus".

    Returns:
        tuple: Inflation expectations for training data and test data.
    """
    # Split into train and test data
    train_data = data[data.index <= execution_date].copy()
    test_data = data[data.index > execution_date].copy()

    # Step 1: Prepare CPI data
    train_data = replace_last_n_with_nan(train_data, 1)  # Replace NaNs for the last observation
    cpi_mom_train = train_data.pct_change(fill_method=None).iloc[1:] * 100  # MoM CPI growth rates

    # Step 2: Generate inflation forecasts
    if macro_forecast == "ar_1":
        # Fit AR(1) model on CPI MoM growth rates
        ar1_model_inflation = AR1Model()
        fitted_inflation_model = ar1_model_inflation.fit(pd.DataFrame(cpi_mom_train).dropna(), target_col=f"{country}_CPI")
        inflation_growth_forecasts = ar1_model_inflation.forecast(
            model=fitted_inflation_model,
            steps=60,  # Forecast for the test data horizon
            train_data=pd.DataFrame(cpi_mom_train).dropna(),
            target_col=f"{country}_CPI"
        )
        forecast_dates = pd.date_range(start=test_data.first_valid_index(), periods=60, freq="MS")
        df_forecast_date = pd.DataFrame(inflation_growth_forecasts,
                                   index=forecast_dates, 
                                   columns = ['monthly_forecast'])

    elif macro_forecast == "consensus":
        # Use consensus forecasts
        forecast_date = consensus_df[consensus_df["forecast_date"] <= execution_date]["forecast_date"].max()
        df_forecast_date = consensus_df[consensus_df["forecast_date"] == forecast_date]
        df_forecast_date = df_forecast_date[["forecasted_month", "monthly_forecast"]]
        df_forecast_date.set_index("forecasted_month", inplace=True)
        df_forecast_date.index = pd.to_datetime(df_forecast_date.index)
        #test_data["cpi_mom_with_forecast"] = df_forecast_date["monthly_forecast"]

    # Combine train and test data with forecasts
    # combined_data = pd.concat([cpi_mom_train, test_data["cpi_mom_with_forecast"]], axis=0)

    # Step 3: Calculate inflation expectations
    if method == "ucsv":
        # Use UCSV method to calculate inflation expectations independently
        ucsv_fit = inflation_ucsv_matlab(pd.DataFrame(cpi_mom_train))
        df_ucsv_fit = pd.DataFrame(ucsv_fit.flatten(), index = cpi_mom_train.index, columns = ['ucsv'])
        # Attach consensus forecasts to fill missing values after UCSV calculations
        temp = pd.concat([df_forecast_date['monthly_forecast'], df_ucsv_fit], axis = 1)
        inflation_expectations_full = pd.DataFrame(temp['ucsv'].fillna(temp['monthly_forecast']))
        inflation_expectations_full.columns = ['cpi_mom_with_forecast']
    else:
        # Default method
        temp = pd.concat([cpi_mom_train, df_forecast_date['monthly_forecast']], axis=1).dropna(how='all')
        inflation_expectations_full['cpi_mom_with_forecast'] = temp[f'{country}_CPI'].fillna(temp['monthly_forecast'])
        
    # Step 4: Convert to YoY Inflation
    inflation_yoy = convert_mom_to_yoy(inflation_expectations_full['cpi_mom_with_forecast'], "YoY_inflation")
    return inflation_yoy[inflation_yoy.index>=train_data.first_valid_index()], inflation_yoy.loc[test_data.index]

train_inflation, test_inflation = inflation_expectations(
    country="US",
    data=df_combined["US_CPI"],
    consensus_df=None,  # No consensus data needed for AR(1)
    execution_date=pd.Timestamp("2023-01-01"),
    method="ucsv",
    macro_forecast="ar_1"
)
inflation_yoy_train = inflation_yoy.copy()
inflation_yoy_test = inflation_yoy.reindex(test_data.index)
train_data[inflation_col] = inflation_yoy_train
test_data[inflation_col] = inflation_yoy_test
        
        
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:26:46 2025

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
country = 'EA'
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
        
def convert_gdp_to_monthly(df, country, method="FREQCHG", lamb=None, one_sided=False):
    """
    Converts GDP data from quarterly to monthly frequency using the specified method.

    Args:
        df (pd.DataFrame): DataFrame containing GDP and IP data.
        country (str): Country code to extract data for (e.g., 'US').
        method (str): Method to use for conversion ('FREQCHG' or 'Chow-Lin').
        lamb (float, optional): Lambda value for HP filter (if applicable).
        one_sided (bool, optional): Whether to use a one-sided HP filter (if applicable).

    Returns:
        pd.DataFrame: Monthly GDP series.
    """
    # Step 1: Extrapolate the IP series using AR(1) and percentage differences
    ip_extrapolated = ar_extrapol(df[f"{country}_IP"].values, AROrder=1, DiffFlag=2)

    if method == "FREQCHG":
        # Use the FREQCHG method
        gdp_monthly = FREQCHG_Q2M_EXO(df[f"{country}_GDP"].values, ip_extrapolated)
    elif method == "Chow-Lin":
        # Use the Chow-Lin method
        low_freq_ts = pd.DataFrame(df[f"{country}_GDP"]).dropna().resample("QS").last()
        high_freq_ts = pd.DataFrame(ip_extrapolated, index=df.index)
        disaggregated_series, _ = me.chow_lin(low_freq_ts, high_freq_ts, "sum")
        # Combine high-frequency IP and disaggregated GDP for extrapolation
        df_temp = pd.concat([high_freq_ts, disaggregated_series], axis=1)
        # this is to inlcude missing values at the end of the disaggregated series
        gdp_monthly = df_temp.iloc[:,1].values.ravel()
    else:
        raise ValueError("Invalid method. Choose either 'FREQCHG' or 'Chow-Lin'.")

    # Step 2: Extrapolate to fill missing values
    gdp_monthly_extrap = extrapolate_series(gdp_monthly, ip_extrapolated)

    # Step 3: Convert to MoM % change
    gdp_mom_extrap = pd.Series(gdp_monthly_extrap)
    gdp_mom_extrap.index = df.index
    gdp_mom_extrap = gdp_mom_extrap.pct_change(fill_method=None)

    # Step 4: Return the result
    return gdp_mom_extrap, gdp_monthly_extrap

def OUTPUTGAPdirect(para):
    """
    Python translation of OUTPUTGAPdirect from MATLAB for INRetrievalModel == 0.
    This function calculates output gap, potential GDP, and other series based on actual historical data.
    
    Parameters:
    - para: A dictionary of parameters including initial values for the output gap, alpha, and beta.
    
    Returns:
    - Output gap, potential GDP growth rate, level of GDP, and level of potential GDP.
    """
    
    PerfectForesightInsample = False  # 12-month horizon

    OGinitial = para['FDOGInitialValue']
    alpha = para['FDOGalpha']
    beta = para['FDOGbeta']

    # Assume the GDPreturn series is already provided in the parameters
    GDPreturn = para['GDPGrowth']

    try:
        # Initialize arrays to store GDP, potential GDP (PGDP), and output gap estimates (OGest)
        OGest = np.full_like(GDPreturn, np.nan)
        rPGDP = np.full_like(GDPreturn, np.nan)
        PGDP = np.full_like(GDPreturn, np.nan)
        GDP = np.full_like(GDPreturn, np.nan)
        nObs = len(GDPreturn)

        # determine the first valid observation
        FirstObs = max([np.where(~np.isnan(GDPreturn))[0][0]][0], 1)
        
        # Set initial GDP value and calculate cumulative GDP
        GDP[FirstObs-1] = 100  # Initial GDP value
        GDP[FirstObs-1:] = np.cumprod([GDP[FirstObs-1]] + list(GDPreturn[FirstObs:] + 1))#.reshape(-1, 1)  # Cumulative product of GDP

        # Initialize potential GDP (PGDP) and output gap estimates (OGest)
        PGDP[FirstObs] = GDP[FirstObs] / (1 + OGinitial / 100)
        OGest[FirstObs] = OGinitial
        rPGDP[FirstObs] = np.nanmean(GDPreturn[FirstObs:nObs])  # Initial rPGDP

        # Iterate through observations and compute the output gap, rPGDP, and PGDP
        for t in range(FirstObs + 1, nObs):
            rPGDP[t] = rPGDP[t-1] - alpha * (rPGDP[t-1] - GDPreturn[t-1]) + beta * OGest[t-1]
            PGDP[t] = PGDP[t-1] * (1 + rPGDP[t])
            OGest[t] = (GDP[t] / PGDP[t] - 1) * 100

        # Store results
        dict_data_results = {
            'GDPhistoric': GDP,
            'rPGDPhistoric': rPGDP,
            'PGDPhistoric': PGDP,
            'OGesthistoric': OGest
        }

        # Optionally compute Perfect Foresight (if applicable)
        if PerfectForesightInsample:
            m1 = pd.Series(GDPreturn).rolling(window=12).mean().shift(-12)
            dict_data_results['GDPreturnPerfectForesight'] = pd.Series()
            dict_data_results['GDPreturnPerfectForesight'] = m1.fillna(m1.iloc[-1])

        # Return all possible outputs (Output gap, Potential GDP Growth, Level GDP, Level Potential GDP)
        return OGest, rPGDP, GDP, PGDP, dict_data_results

    except Exception as e:
        Error = f'OUTPUTGAPdirect: Unspecified error. {str(e)}'
        print(Error)
        return None, None, None, None, dict_data_results
    
def output_gap(country, data, consensus_df, execution_date, method="direct", macro_forecast="consensus"):
    """
    Calculates the output gap for both training and testing data using the specified method.

    Args:
        country (str): Country name or code.
        data (pd.DataFrame): Combined dataset (both train and test data).
        consensus_df (pd.DataFrame): Consensus GDP forecasts.
        execution_date (datetime): Execution date.
        method (str): Method for output gap calculation ("direct", "HP", or YoY conversion).
        macro_forecast (str): Macro forecast method ("consensus" or "ar_1"). Defaults to "consensus".

    Returns:
        tuple: Output gap (or YoY growth) for training data and test data.
    """
    # Split into train and test data
    train_data = data[data.index <= execution_date].copy()
    test_data = data[data.index > execution_date].copy()

    # Step 1: Prepare data subset and extrapolated GDP growth rates
    train_data[f'{country}_GDP'] = replace_last_n_with_nan(train_data[f'{country}_GDP'], 3)
    train_data[f'{country}_IP'] = replace_last_n_with_nan(train_data[f'{country}_IP'], 1)

    gdp_mom_extrap, gdp_monthly_extrap = convert_gdp_to_monthly(train_data, country=f'{country}')
    df_gdp_mom_extrap = pd.DataFrame(gdp_mom_extrap, columns=['gdp_mom'])
    gdp_monthly_extrap = pd.DataFrame(gdp_monthly_extrap, columns=['gdp_level'])
    gdp_monthly_extrap.index = train_data.index

    # Step 2: Generate GDP growth forecasts
    if macro_forecast == "ar_1":
        # Fit AR(1) model on extrapolated GDP MoM growth rates
        ar1_model_gdp = AR1Model()
        fitted_gdp_model = ar1_model_gdp.fit(df_gdp_mom_extrap, target_col='gdp_mom')
        gdp_growth_forecasts = ar1_model_gdp.forecast(
            model=fitted_gdp_model,
            steps=60,  # Forecast for the test data horizon
            train_data=df_gdp_mom_extrap,
            target_col='gdp_mom'
        )
        # Assign AR(1)-based forecasts to the test data
        test_data['gdp_mom_with_forecast'] = gdp_growth_forecasts
    else:
        # Use consensus forecasts
        forecast_date = consensus_df[consensus_df['forecast_date'] <= execution_date]['forecast_date'].max()
        df_forecast_date = consensus_df[consensus_df['forecast_date'] == forecast_date]
        df_forecast_date = df_forecast_date[['forecasted_month', 'monthly_forecast']]
        df_forecast_date.set_index('forecasted_month', inplace=True)
        test_data['gdp_mom_with_forecast'] = df_forecast_date['monthly_forecast'] / 100

    # Combine train and test data with forecasts
    df_gdp_mom_extrap['gdp_mom_with_forecast'] = df_gdp_mom_extrap['gdp_mom']
    combined_data = pd.concat([df_gdp_mom_extrap, test_data[['gdp_mom_with_forecast']]], axis=0)

    # Step 3: Calculate output gap or YoY growth
    if method == "direct":
        para = {
            'FDOGInitialValue': 0,
            'FDOGalpha': 0.02,
            'FDOGbeta': 0.000002,
            'GDPGrowth': combined_data['gdp_mom_with_forecast'].dropna().values
        }
        OGest, _, _, _, _ = OUTPUTGAPdirect(para)
        output_gap_full = pd.Series(OGest, index=combined_data['gdp_mom_with_forecast'].dropna().index[:len(OGest)])
        return output_gap_full.loc[df_gdp_mom_extrap.dropna().index], output_gap_full.loc[test_data.index]

    elif method == "hp_filter":
        combined_data['growth_factor'] = 1 + (combined_data['gdp_mom_with_forecast'] / 100)
        last_observed_gdp = gdp_monthly_extrap.iloc[-1]
        combined_data['reconstructed_gdp'] = last_observed_gdp.values * combined_data['growth_factor'].cumprod()
        gdpTrend = me.hp_filter(combined_data['reconstructed_gdp'].dropna(), one_sided="kalman", lambda_values=1600000)
        gdpCycle = np.log(combined_data['reconstructed_gdp'].dropna()) - np.log(gdpTrend)
        output_gap_full = pd.Series(gdpCycle, index=combined_data.index)
        return output_gap_full.loc[train_data.index], output_gap_full.loc[test_data.index]

    else:
        # Convert MoM to YoY growth rates
        gdp_yoy = convert_mom_to_yoy(combined_data['gdp_mom_with_forecast'], 'gdp_yoy')
        gdp_yoy.index = combined_data['gdp_mom_with_forecast'].index
        return None, gdp_yoy.dropna()

def generate_execution_dates(data, consensus_df=None, execution_date_column="forecast_date", min_years=3, macro_forecast="consensus"):
    """
    Generates a list of valid execution dates for backtesting.

    Args:
        data (pd.DataFrame): Input dataset with a DateTime index.
        consensus_df (pd.DataFrame, optional): Consensus dataset with forecast dates.
        execution_date_column (str): Column in the consensus dataset that specifies execution dates.
        min_years (int): Minimum number of years of historical data required for training.
        macro_forecast (str): Macro forecast method ("consensus" or "ar_1").

    Returns:
        list: List of valid execution dates.
    """
    # Ensure the dataset has enough historical data
    min_start_index = min_years * 12  # Convert years to months
    first_valid_index = data.dropna().first_valid_index()  # First valid index after dropping NaNs

    if first_valid_index is None:
        raise ValueError("The dataset contains no valid data after dropping missing values.")

    # Ensure sufficient historical data is available for training
    if len(data.loc[first_valid_index:]) < min_start_index:
        raise ValueError(f"Not enough data. At least {min_years} years of valid data are required.")

    # Handle execution dates based on macro_forecast method
    if macro_forecast == "consensus":
        if consensus_df is None or execution_date_column not in consensus_df.columns:
            raise ValueError(f"Consensus dataset must be provided with a '{execution_date_column}' column for consensus forecasts.")

        # Use the forecast dates from the consensus dataset
        execution_dates = consensus_df[execution_date_column].sort_values().unique()

        # Filter execution dates to ensure sufficient training data
        execution_dates = [date for date in execution_dates if date >= data.index[min_start_index]]

    elif macro_forecast == "ar_1":
        # Generate execution dates starting after the first valid index and minimum training period
        execution_dates = data.loc[first_valid_index:].index[min_start_index:]

    else:
        raise ValueError(f"Unknown macro_forecast method: {macro_forecast}")

    return execution_dates

from data_preparation.conensus_forecast import ConsensusForecast
from config_paths import QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH
consensus_forecast = ConsensusForecast(QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH)
df_consensus_gdp, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} GDP")
            
execution_dates = generate_execution_dates(
    data=df_combined,
    consensus_df=None,
    execution_date_column=None,
    min_years=3,
    macro_forecast="ar_1"
)

execution_date = execution_dates[0]

df = df_combined.copy()
train_data_exp = df.loc[:execution_date].copy()  # Historical data up to the execution date
future_dates = pd.date_range(
    start=train_data_exp.index[-1] + pd.DateOffset(months=1),
    periods=max(horizons),
    freq="MS",
)
test_data_exp = pd.DataFrame(index=future_dates)  # Future forecast horizons
combined_data = pd.concat([train_data_exp, test_data_exp], axis=0)

output_gap_train, output_gap_test = output_gap(
            country=country,
            data=combined_data,
            consensus_df=None,
            execution_date=execution_date,
            method="direct",
            macro_forecast="ar_1"
        )

train_data_exp['OG'] = output_gap_train
test_data_exp['OG'] = output_gap_test



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
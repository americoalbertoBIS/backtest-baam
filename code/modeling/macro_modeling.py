
import pandas as pd
import numpy as np
import matlab.engine

import medengine as me

import os
os.chdir(r'C:\git\backtest-baam\code')

from data_preparation.time_series_helpers import ar_extrapol, extrapolate_series
from data_preparation.data_transformations import FREQCHG_Q2M_EXO, replace_last_n_with_nan, convert_mom_to_yoy

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

def output_gap(country, data, consensus_df, execution_date, method="direct"):
    data_subset = data[data.index <= execution_date].copy()
    data_subset[f'{country}_GDP'] = replace_last_n_with_nan(data_subset[f'{country}_GDP'], 3)
    data_subset[f'{country}_IP'] = replace_last_n_with_nan(data_subset[f'{country}_IP'], 1)

    gdp_mom_extrap, gdp_monthly_extrap = convert_gdp_to_monthly(data_subset, country=f'{country}')
    df_gdp_mom_extrap = pd.DataFrame(gdp_mom_extrap, columns=['gdp_mom'])
    gdp_monthly_extrap = pd.DataFrame(gdp_monthly_extrap, columns=['gdp_level'])
    gdp_monthly_extrap.index = data_subset.index

    forecast_date = consensus_df[consensus_df['forecast_date'] <= execution_date]['forecast_date'].max()
    df_forecast_date = consensus_df[consensus_df['forecast_date'] == forecast_date]
    df_forecast_date = df_forecast_date[['forecasted_month', 'monthly_forecast']]
    df_forecast_date.set_index('forecasted_month', inplace=True)
    
    if method == "direct":
        df_gdp_with_lags_and_consensus = pd.concat([df_gdp_mom_extrap, df_forecast_date['monthly_forecast'] / 100], axis=1).dropna(how='all')
        df_gdp_with_lags_and_consensus['gdp_mom_with_consensus'] = df_gdp_with_lags_and_consensus['gdp_mom'].fillna(df_gdp_with_lags_and_consensus['monthly_forecast'])

        para = {
            'FDOGInitialValue': 0,
            'FDOGalpha': 0.02,
            'FDOGbeta': 0.000002,
            'GDPGrowth': df_gdp_with_lags_and_consensus['gdp_mom_with_consensus'].dropna().values
        }
        OGest, _, _, _, _ = OUTPUTGAPdirect(para)
        output_gap_full = pd.Series(OGest, index=df_gdp_with_lags_and_consensus.index[:len(OGest)])

        return output_gap_full, df_gdp_with_lags_and_consensus['gdp_mom_with_consensus']
        
    elif method == "hp_filter":
        df_forecast_date['growth_factor'] = 1 + (df_forecast_date['monthly_forecast'] / 100)
        last_observed_gdp = gdp_monthly_extrap.iloc[-1]
        # Apply cumulative product to reconstruct GDP levels
        df_forecast_date['reconstructed_gdp'] = last_observed_gdp.values * df_forecast_date['growth_factor'].cumprod()
        # Combine observed GDP levels and reconstructed levels
        df_gdp_with_lags_and_consensus = pd.concat([gdp_monthly_extrap, df_forecast_date['reconstructed_gdp']], axis=1)
        df_gdp_with_lags_and_consensus['gdp_level_with_consensus'] = df_gdp_with_lags_and_consensus['gdp_level'].fillna(
            df_gdp_with_lags_and_consensus['reconstructed_gdp']
        )

        gdpTrend = me.hp_filter(df_gdp_with_lags_and_consensus['gdp_level_with_consensus'].dropna(), one_sided="kalman", lambda_values=1600000)
        gdpCycle = pd.DataFrame(np.log(df_gdp_with_lags_and_consensus['gdp_level_with_consensus'].dropna()) - np.log(gdpTrend)) * 100
        gdpCycle.columns = ['ygap_HP_RT']
        output_gap_full = gdpCycle['ygap_HP_RT']
    
        return output_gap_full, df_gdp_with_lags_and_consensus['gdp_level_with_consensus'].dropna()
    
    else:
        df_gdp_with_lags_and_consensus = pd.concat([df_gdp_mom_extrap, df_forecast_date['monthly_forecast'] / 100], axis=1).dropna(how='all')
        df_gdp_with_lags_and_consensus['gdp_mom_with_consensus'] = df_gdp_with_lags_and_consensus['gdp_mom'].fillna(df_gdp_with_lags_and_consensus['monthly_forecast'])
        gdp_yoy = convert_mom_to_yoy(df_gdp_with_lags_and_consensus['gdp_mom_with_consensus'], 'gdp_yoy')
        gdp_yoy.index = df_gdp_with_lags_and_consensus['gdp_mom_with_consensus'].index
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

def inflation_expectations(country, data, consensus_df, execution_date, method="default"):
    """
    Calculates inflation expectations using the specified method.

    Args:
        data (pd.DataFrame): Input data containing CPI series.
        consensus_df (pd.DataFrame): Consensus forecast data.
        execution_date (datetime): Execution date for the calculation.
        method (str): Method for calculating inflation expectations ("default" or "ucsv").

    Returns:
        pd.Series: Year-on-year inflation expectations.
    """
    # Step 1: Prepare CPI data
    data_subset = pd.DataFrame(data[data.index <= execution_date].copy())
    data_subset = replace_last_n_with_nan(data_subset, 1)
    cpi_mom_infl = data_subset.pct_change(fill_method=None).iloc[1:] * 100

    # Step 2: Integrate Consensus Forecast
    forecast_date = consensus_df[consensus_df['forecast_date'] <= execution_date]['forecast_date'].max()
    
    df_forecast_date = consensus_df[consensus_df['forecast_date'] == forecast_date]
    df_forecast_date = df_forecast_date[['forecasted_month', 'monthly_forecast']]
    df_forecast_date.set_index('forecasted_month', inplace=True)
    df_forecast_date.index = pd.to_datetime(df_forecast_date.index)

    # Step 3: Calculate Inflation Expectations
    if method == "ucsv":
        # Use UCSV method
        ucsv_fit = inflation_ucsv_matlab(cpi_mom_infl)
        temp = pd.DataFrame()
        temp['ucsv_o'] = ucsv_fit.flatten()
        temp.index = cpi_mom_infl.index
        temp = pd.concat([df_forecast_date['monthly_forecast'], temp], axis=1)
        temp['cpi_mom_with_consensus'] = temp['ucsv_o'].fillna(temp['monthly_forecast'])
    else:
        # Default method
        temp = pd.concat([cpi_mom_infl, df_forecast_date['monthly_forecast']], axis=1).dropna(how='all')
        temp['cpi_mom_with_consensus'] = temp[f'{country}_CPI'].fillna(temp['monthly_forecast'])

    # Step 4: Convert to YoY Inflation
    inflation_yoy = convert_mom_to_yoy(temp['cpi_mom_with_consensus'], 'ucsv_baam')
    #inflation_yoy = convert_mom_to_yoy(temp['cpi_mom_with_consensus'].values, col_name='YoY_inflation')
    inflation_yoy.index = temp['cpi_mom_with_consensus'].index

    return inflation_yoy
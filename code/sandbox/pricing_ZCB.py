# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:25:19 2025

@author: al005366
"""


import os
os.chdir(r'C:\git\backtest-baam\code')
import pandas as pd
import matplotlib.pyplot as plt

from modeling.time_series_modeling import AR1Model
from data_preparation.data_loader import DataLoaderYC
from modeling.yield_curve_modeling import YieldCurveModel

country ='US'

# Load the yield curve data
data_loader = DataLoaderYC(r'L:\\RMAS\\Resources\\BAAM\\OpenBAAM\\Private\\Data\\BaseDB.mat')
_, _, _ = data_loader.load_data()
if country == 'EA':
    selectedCurveName, selected_curve_data, modelParams = data_loader.process_data('DE')
else:
    selectedCurveName, selected_curve_data, modelParams = data_loader.process_data(country)
# Update model parameters for the yield curve model
modelParams.update({'minMaturity': 0.08, 'maxMaturity': 10, 'lambda1fixed': 0.7173})
yield_curve_model = YieldCurveModel(selected_curve_data, modelParams)

# Extract observed yields and convert dates
dates_str = yield_curve_model.dates_str
observed_yields = yield_curve_model.yieldsObservedAgg
maturities = yield_curve_model.uniqueTaus

# Create a DataFrame for observed yields
observed_yields_df = pd.DataFrame(
    observed_yields,
    columns=[f'{tau} years' for tau in maturities],
    index=dates_str[-len(observed_yields):]
)
observed_yields_df.index = pd.to_datetime(observed_yields_df.index)
observed_yields_df_resampled = observed_yields_df.resample('MS').mean()

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

# ...existing code...

def zcb_components(yield_ser,
                   maturity,
                   freq=12,
                   carry_method='approx'):
    """
    Compute ZCB price, price-return and carry from a yield series in decimal form.

    Args:
      yield_ser : pd.Series
        Yields as decimals (e.g. 0.0427 for 4.27%), indexed by date.
      maturity  : float
        Time-to-maturity in years (e.g. 0.25).
      freq      : int, default=12
        Number of observations per year (e.g. 12 for monthly).
      carry_method : {'approx','exact'}, default='exact'
        ‘approx’: carryₜ = yₜ₋₁ / freq  
        ‘exact’ : carryₜ = ((1+yₜ)^(–(T–1/freq)) – (1+yₜ)^(–T)) / (1+yₜ)^(–T) * freq

    Returns:
      DataFrame with columns:
        Price     : Pₜ = (1 + yₜ)^(−T)
        PriceRet  : Rₜ = Pₜ / Pₜ₋₁ − 1
        Carry     : annualized carry component
        TotalRet  : PriceRet + Carry
    """
    # input checks
    if not isinstance(yield_ser, pd.Series):
        raise TypeError("yield_ser must be a pandas Series")
    if yield_ser.empty:
        raise ValueError("yield_ser is empty")
    try:
        T0 = float(maturity)
    except:
        raise ValueError("maturity must be convertible to float")
    if T0 < 0:
        raise ValueError("maturity must be non-negative")
    if freq <= 0 or not float(freq).is_integer():
        raise ValueError("freq must be a positive integer")
    if carry_method not in ('approx', 'exact'):
        raise ValueError("carry_method must be 'approx' or 'exact'")

    # 1) price at fixed maturity T0
    y = yield_ser.astype(float)
    price = (1 + y) ** (-T0)

    # 2) price-only return
    price_ret = price / price.shift(1) - 1

    # 3) carry
    if carry_method == 'approx':
        carry = y.shift(1) / freq
    else:
        Tm1 = max(T0 - 1/freq, 0.0)
        P_roll = (1 + y) ** (-Tm1)
        carry = (P_roll - price) / price * freq

    total_ret = price_ret + carry

    return pd.DataFrame({
        'Price':     price,
        'PriceRet':  price_ret,
        'Carry':     carry,
        'TotalRet':  total_ret
    }, index=yield_ser.index)

def seq_annual_arith_ret(df, freq=12):
    """
    Given a DataFrame or Series of periodic returns, compute the sequential
    arithmetic‐annual returns by summing each block of 'freq' observations.

    Args:
      df   : pd.DataFrame or pd.Series containing a column 'TotalRet' (or is the Series itself)
      freq : int, number of periods per year (e.g. 12 for monthly)

    Returns:
      pd.Series of length floor(len(df)/freq), indexed by the last date in each block,
      containing the block‐sum of returns.
    """
    # if they passed the whole DataFrame, pull out the TotalRet column
    if isinstance(df, pd.DataFrame):
        if 'TotalRet' not in df:
            raise KeyError("DataFrame must contain a 'TotalRet' column")
        ret = df['TotalRet'].values
        idx = df.index
    else:
        # assume it's a pd.Series
        ret = df.values
        idx = df.index

    # only keep full years
    n = len(ret)
    years = n // freq
    if years == 0:
        return pd.Series([], dtype=float)

    # reshape and sum
    ret_matrix = ret[:years*freq].reshape(years, freq)
    annual_ret = ret_matrix.sum(axis=1)

    # pick the date at the end of each block
    block_dates = [ idx[(i+1)*freq - 1] for i in range(years) ]

    return pd.Series(annual_ret, index=block_dates, name='SeqAnnArithRet')


# — example usage —
# Example usage:
maturity = 0.25
col_name = f"{maturity} years"
y_ser = observed_yields_df_resampled[col_name]

# 1) compute the price+carry table
zcb_df = zcb_components(y_ser/100, maturity=maturity, freq=12, carry_method='approx')

# 2) build sequential annual arithmetic returns
seq_yr =     *100

print(seq_yr.head())

# 3) you can now plot or compare to MATLAB's ARP.sR
seq_yr.plot(title='Sequential Annual Arithmetic Returns')
# Remove or comment out the old functions:
# def calculate_prices_from_yields(...)
# def calculate_returns_from_prices(...)

# Set your threshold and setting as in MATLAB
short_mat_threshold = 10.0
CBShortMatPriceReturn = 0  # set to 1 to always use rolling carry

zcb_df = zcb_components(y_ser, maturity, freq=12)

# Compare to BAAM data
data_baam = pd.read_excel(r'C:\data\returns_10Y.xlsx', sheet_name='Sheet1').dropna().T.iloc[1:,:]
data_baam.columns = ['date', 'return']
data_baam.index = pd.to_datetime(data_baam['date'])
data_baam = data_baam.drop('date', axis = 1)
data_baam = data_baam.resample('MS').mean()

aligned = zcb_df.join(data_baam['return'].rename('baam_ret'), how='inner')
aligned['diff'] = aligned['baam_ret'] - aligned['TotalRet']

plt.plot(data_baam[(data_baam.index<'2026')&(data_baam.index>'2000')])
plt.plot(zcb_df['TotalRet'][zcb_df.index>'2000'])
plt.show()

plt.plot(aligned['diff'])
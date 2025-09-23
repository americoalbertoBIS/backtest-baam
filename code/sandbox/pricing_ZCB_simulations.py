# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 19:18:42 2025

@author: al005366
"""

import pandas as pd
import matplotlib.pyplot as plt

data_sim = pd.read_parquet(
    r'L:\RMAS\Users\Alberto\backtest-baam\data_test\EA\yields\estimated_yields\AR_1\simulations\0.25_years\simulations_01052002.parquet')

# after you read data_sim:
data_sim = data_sim.rename(columns={
    'ForecastDate'  : 'forecast_date',
    'SimulatedValue': 'simulated_value',
    'Maturity'      : 'maturity',
    'SimulationID'  : 'simulation_id',
    'ExecutionDate' : 'execution_date',
    'Model'         : 'model',
    'Horizon'       : 'horizon',
})

# ensure forecast_date is a datetime
data_sim['forecast_date'] = pd.to_datetime(data_sim['forecast_date'])

import numpy as np
import pandas as pd

def compute_sim_zcb_const_maturity_returns(
    sim_df,
    freq=12
):
    """
    For each simulation path and each month, assume you buy a fresh ZCB of 
    fixed maturity (= sim_df.maturity) *at that horizon*, hold one period,
    then roll into the same fixed‐maturity bond again.  Returns never go to zero.
    """
    out = []
    # parse fixed maturity once
    T0 = float(sim_df['maturity'].iloc[0].split()[0])
    
    for sim_id, g in sim_df.groupby('simulation_id', sort=True):
        g = g.sort_values('horizon').reset_index(drop=True)

        # yields in decimals
        y = g['simulated_value'].values   # e.g. 0.0333 for 3.33%
        # build prices at constant maturity
        P = (1 + y) ** (-T0)

        # price‐only return over one month: P_t / P_{t-1} - 1
        price_ret = P / pd.Series(P).shift(1).values - 1

        # carry (annualized) = y_{t-1}/freq
        carry = pd.Series(y).shift(1).values / freq

        df = pd.DataFrame({
            'forecast_date':  g['forecast_date'].values,
            'simulation_id':  sim_id,
            'price_return':   price_ret,
            'carry':          carry,
            'total_return':   price_ret + carry
        })
        out.append(df)

    return pd.concat(out, ignore_index=True)


# usage

# rename columns
# after you read data_sim:
data_sim = data_sim.rename(columns={
    'ForecastDate'  : 'forecast_date',
    'SimulatedValue': 'simulated_value',
    'Maturity'      : 'maturity',
    'SimulationID'  : 'simulation_id',
    'ExecutionDate' : 'execution_date',
    'Model'         : 'model',
    'Horizon'       : 'horizon',
})

data_sim['ForecastDate'] = pd.to_datetime(data_sim['forecast_date'])

sim_results = compute_sim_zcb_const_maturity_returns(data_sim, freq=12)

# plot median
med = sim_results.groupby('forecast_date')['total_return'].median()
plt.figure(figsize=(10,4))
plt.plot(med.index, med.values, label='median')

# fan chart
qs = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]
quantiles = sim_results.groupby('forecast_date')['total_return'].quantile(qs).unstack()
d = quantiles.index
plt.plot(zcb_df['TotalRet'][(zcb_df.index<'2002-07-01')&(zcb_df.index>'2000-01-01')])
plt.fill_between(d, quantiles[0.01], quantiles[0.99], color='red', alpha=0.3)
plt.fill_between(d, quantiles[0.05], quantiles[0.95], color='green', alpha=0.3)
plt.fill_between(d, quantiles[0.25], quantiles[0.75], color='blue', alpha=0.3)
plt.plot(d, quantiles[0.5], color='navy', label='median')
plt.legend()
plt.title('Simulated total returns on a rolling 3m ZCB')
plt.show()
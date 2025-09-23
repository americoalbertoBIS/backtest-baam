import pandas as pd
import numpy as np

# Generate Monthly Series without Extrapolation
def FREQCHG_Q2M_EXO(INSeries, ExoSeries):
    """
    Converts quarterly data to monthly data using an external series (ExoSeries) for volatility adjustment.
    This function uses percentage returns and handles NaNs.
    
    :param INSeries: Input series (quarterly data), pandas Series.
    :param ExoSeries: Exogenous series (monthly data for volatility adjustment), pandas Series.
    :return: Converted monthly series (with interpolated values).
    """
    
    # Ensure INSeries and ExoSeries are pandas Series
    if not isinstance(INSeries, pd.Series):
        INSeries = pd.Series(INSeries)
    if not isinstance(ExoSeries, pd.Series):
        ExoSeries = pd.Series(ExoSeries)

    # Drop missing values and align indices
    iObs = INSeries.dropna().index
    
    # Calculate volatility adjustment factor using percentage changes
    INPDelta = INSeries[iObs].pct_change(fill_method = None)
    ExoPDelta = ExoSeries[iObs].pct_change(fill_method = None)
    VolAdjustRet = INPDelta.dropna().std() / ExoPDelta.dropna().std()

    # Log transformation
    logINSeries = np.log(INSeries)
    logExoSeries = np.log(ExoSeries)

    # Number of observations
    NObs = len(ExoSeries)
    OutSeries = np.full(NObs, np.nan)  # Initialize output series with NaNs

    # Find the first non-missing observation
    FirstObs = max(INSeries.first_valid_index(), ExoSeries.first_valid_index())

    k = FirstObs
    while k < NObs:
        # Find the next valid observation in INSeries
        next_obs_index = logINSeries[k+1:].first_valid_index()

        # Handle the case where no more valid observations are found
        if next_obs_index is None:
            k += 1
            continue

        NextObs = next_obs_index

        # Calculate quarterly and monthly returns
        QRetInstr = logINSeries[NextObs] - logINSeries[k]
        QRetRef = logExoSeries[NextObs] - logExoSeries[k]

        # Calculate monthly returns of the exogenous series between k and NextObs
        MRetRef = logExoSeries[k+1:NextObs+1].values - logExoSeries[k:NextObs].values

        # Adjust monthly returns for volatility
        MRetRefMean = MRetRef.mean()
        MRetRef = (MRetRef - MRetRefMean) * VolAdjustRet + MRetRefMean

        # Distribute the quarterly return difference across the months
        QRetDiff = QRetInstr - QRetRef
        MRetInstr = MRetRef + QRetDiff / len(MRetRef)

        # Cumulative sum of the adjusted log returns
        cumsum_result = np.cumsum([logINSeries[k]] + list(MRetInstr))

        OutSeries[k:NextObs+1] = cumsum_result

        # Move to the next valid observation
        k = NextObs

    # Exponentiate to revert logarithmic transformation
    OutSeries = np.exp(OutSeries)

    return pd.Series(OutSeries)

def mom2yoy_cumulative(df, col_name, periods=12):
    """
    Calculates the YoY inflation expectation using the cumulative product method.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing the MoM series.
    - col_name: str - The name of the column containing the MoM percentage data.
    - periods: int - The number of periods for YoY comparison (default is 12 for monthly YoY).

    Returns:
    - pd.Series - YoY inflation expectation using cumulative product method.
    """
    index_col = (1 + df[col_name] / 100).cumprod()  # Convert MoM % to cumulative index
    yoy_cumulative = index_col.pct_change(periods=periods) * 100  # Calculate YoY % change
    return yoy_cumulative

def mom2yoy_logarithmic(df, periods=12):
    """
    Calculates the YoY inflation expectation using the logarithmic method.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing the MoM series.
    - col_name: str - The name of the column containing the MoM percentage data.
    - periods: int - The number of periods for YoY comparison (default is 12 for monthly YoY).

    Returns:
    - pd.Series - YoY inflation expectation using the logarithmic method.
    """
    log_ret = np.log(1 + df / 100)  # Convert MoM % to log returns
    yoy_log = log_ret.rolling(window=periods).sum()  # Rolling sum over the specified period
    yoy_log = (np.exp(yoy_log) - 1) * 100  # Convert back to percentage
    return yoy_log

def replace_last_n_with_nan(series, n):
    """
    Replaces the last `n` non-NaN values in a series with NaN.
    """
    non_nan_indices = series.dropna(how = 'all').index[-n:]
    series.loc[non_nan_indices] = np.nan
    return series

def convert_mom_to_yoy(mom_series, col_name='YoY_inflation'):
    """
    Converts month-on-month series to year-on-year series.
    """
    df_series = pd.DataFrame(mom_series)
    df_series[col_name] = mom2yoy_logarithmic(df_series)
    return df_series[col_name]

def zcb_components(yields,
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
    if not isinstance(yields, pd.Series):
        raise TypeError("yield_ser must be a pandas Series")
    if yields.empty:
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
    y = yields.astype(float)
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

    return total_ret
    #return pd.DataFrame({
    #    'price':     price,
    #    'price_return':  price_ret,
    #    'carry':     carry,
    #    'total_return':  total_ret
    #}, index=yields.index)

def seq_annual_arith_return(df, freq=12):
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
    
    return pd.Series(annual_ret, index=block_dates)
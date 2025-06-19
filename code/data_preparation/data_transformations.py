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

def mom2yoy_logarithmic(df, col_name, periods=12):
    """
    Calculates the YoY inflation expectation using the logarithmic method.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing the MoM series.
    - col_name: str - The name of the column containing the MoM percentage data.
    - periods: int - The number of periods for YoY comparison (default is 12 for monthly YoY).

    Returns:
    - pd.Series - YoY inflation expectation using the logarithmic method.
    """
    log_ret = np.log(1 + df[col_name] / 100)  # Convert MoM % to log returns
    yoy_log = log_ret.rolling(window=periods).sum()  # Rolling sum over the specified period
    yoy_log = (np.exp(yoy_log) - 1) * 100  # Convert back to percentage
    return yoy_log

def replace_last_n_with_nan(series, n):
    """
    Replaces the last `n` non-NaN values in a series with NaN.
    """
    non_nan_indices = series.dropna().index[-n:]
    series.loc[non_nan_indices] = np.nan
    return series

def convert_mom_to_yoy(mom_series, col_name='YoY_inflation'):
    """
    Converts month-on-month series to year-on-year series.
    """
    df_series = pd.DataFrame(mom_series, columns=['mom'])
    df_series[col_name] = mom2yoy_logarithmic(df_series, 'mom')
    return df_series[col_name]
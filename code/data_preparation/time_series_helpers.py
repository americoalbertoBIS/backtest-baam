import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import data_engine as de

def extrapolate_series(OutSeries, ExoSeries, AROrder=1):
    """
    Extrapolate missing values in OutSeries using an AR process with ExoSeries as an exogenous variable.
    
    Parameters:
    - OutSeries: The input series to be extrapolated (with NaNs at the end).
    - ExoSeries: The exogenous series used for AR-based extrapolation.
    - AROrder: The autoregressive order to be used (default is 1).
    
    Returns:
    - Extrapolated series with NaNs filled in.
    """
    
    # Ensure OutSeries and ExoSeries are pandas Series
    if not isinstance(OutSeries, pd.Series):
        OutSeries = pd.Series(OutSeries)
    if not isinstance(ExoSeries, pd.Series):
        ExoSeries = pd.Series(ExoSeries)
    
    # Exponentiate ExoSeries as per the MATLAB logic
    #ExoSeries = np.exp(ExoSeries)

    # Calculate percentage changes (like PDELTA in MATLAB)
    y = OutSeries.pct_change(fill_method = None)#.dropna()
    exo = ExoSeries.pct_change(fill_method = None)#.dropna()

    # Find first valid observation for both series
    FirstObs = max(y.first_valid_index(), exo.first_valid_index())
    
    # Calculate the number of missing observations at the end of OutSeries
    nMissingObs = len(y) - y.last_valid_index() - 1

    # Proceed with AR model estimation and extrapolation
    if nMissingObs > 0:
        yOut = y.copy()  # This will be used for extrapolation
        exoOrg = exo.copy()  # Original exogenous series copy

        # Slice the valid data for AR estimation
        y = y.loc[FirstObs:y.last_valid_index()]
        exo = exo.loc[FirstObs:y.last_valid_index()]

        # Fit AR model with exogenous variables
        model = AutoReg(y, lags=AROrder, exog=exo)
        results = model.fit()

        # Extract AR and exogenous coefficients (similar to MATLAB's beta_)
        beta_ = results.params
        beta_ = np.array(results.params)
        
        # Loop through the missing observations to extrapolate
        for k in range(0, nMissingObs):
            iObs = len(yOut) - nMissingObs + k  # Adjust for Python's zero-indexing
            
            # Extrapolate using the AR model and exogenous variable
            yOut.iloc[iObs] = beta_[1] * yOut.iloc[iObs - 1] + beta_[2] * exoOrg.iloc[iObs] + beta_[0]
            OutSeries.iloc[iObs] = OutSeries.iloc[iObs - 1] * (1 + yOut.iloc[iObs])

    return OutSeries

def ar_extrapol(INseries, AROrder, DiffFlag):
    """
    Extrapolate missing data using an AR process.
    - INseries: Input time series (numpy array or pandas Series).
    - AROrder: The order of the AR process.
    - DiffFlag: Mode of AR estimation (0: raw series, 1: first differences, 2: percentage differences).
    Returns: Extrapolated time series (INseries with missing values filled).
    """
    
    # Handle differencing
    if DiffFlag == 1:
        y = np.diff(INseries)
    elif DiffFlag == 2:
        y = INseries[1:] / INseries[:-1] - 1
    else:
        y = INseries.copy()  # Create a copy of INseries to avoid in-place modifications
    
    # Identify the first valid observation and count missing data
    FirstObsSeries = np.nanargmin(np.isnan(INseries))  # First non-NaN in INseries
    FirstObs = np.nanargmin(np.isnan(y))  # First non-NaN in y
    nMissingObs = np.sum(np.isnan(INseries))  # Total number of NaNs
    
    # AR model estimation
    if nMissingObs > 0:
        # Prepare valid data for AR model estimation
        y_valid = y[FirstObs:FirstObs + len(y) - nMissingObs]
        
        # Ensure there are enough data points for AR model estimation
        if len(y_valid) > AROrder:
            model = AutoReg(y_valid, lags=AROrder)
            model_fit = model.fit()
            beta_ = model_fit.params
            
            # Extrapolate missing values at the end
            for k in range(nMissingObs):
                iObs = len(y) - nMissingObs + k
                y[iObs] = np.dot(beta_[1:], np.flip(y[iObs - AROrder:iObs])) + beta_[0]
        else:
            raise ValueError("Not enough valid data to estimate AR model.")
    
    # Reconstruct the output series
    OUTseries = np.full(len(INseries), np.nan)
    
    if DiffFlag == 1:
        OUTseries[FirstObsSeries:] = np.cumsum([INseries[FirstObsSeries]] + list(y))
    elif DiffFlag == 2:
        OUTseries[FirstObsSeries] = INseries[FirstObsSeries]
        for k in range(FirstObsSeries + 1, len(INseries)):
            OUTseries[k] = OUTseries[k-1] * (1 + y[k-1])
    else:
        OUTseries[FirstObs:] = np.squeeze(y)
    
    return OUTseries
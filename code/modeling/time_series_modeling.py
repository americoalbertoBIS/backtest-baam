import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

import os
os.chdir(r'C:\git\backtest-baam\code')

def forecast_with_ar1(series, forecast_horizon):
    """
    Fits an AR(1) model to the input series and generates forecasts.

    Args:
        series (pd.Series): Historical data to fit the AR(1) model.
        forecast_horizon (int): Number of periods to forecast.

    Returns:
        pd.Series: Forecasted values for the specified horizon.
    """
    # Drop missing values
    series = series.dropna()

    # Create lagged variable
    lagged_series = series.shift(1).dropna()
    X = sm.add_constant(lagged_series)
    y = series.loc[X.index]

    # Fit AR(1) model
    model = sm.OLS(y, X).fit()

    # Generate forecasts
    forecasts = []
    last_value = series.iloc[-1]
    for _ in range(forecast_horizon):
        next_forecast = model.params["const"] + model.params[series.name] * last_value
        forecasts.append(next_forecast)
        last_value = next_forecast

    forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq="MS")
    return pd.Series(forecasts, index=forecast_index)

def fit_arx_model(data, **kwargs):
    """
    Fits a regression model using predictors and lagged data.

    Args:
        data (pd.Series): Target series.
        **kwargs: Additional predictors as keyword arguments.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Fitted model.
    """
    lagged_data = data.shift(1).dropna()  # Lag the data by 1 period
    predictors = [lagged_data] + [kwargs[key].dropna() for key in kwargs if kwargs[key] is not None]
    X = pd.concat(predictors, axis=1).dropna()  # Combine AR(1) term with additional predictors
    X = sm.add_constant(X)  # Add intercept
    y = data.loc[X.index]  # Align the target variable with the predictors
    model = sm.OLS(y, X)  # Ordinary Least Squares (OLS) regression
    return model.fit()  # Fit the model

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def fit(self, train_data, target_col, **kwargs):
        raise NotImplementedError("The `fit` method must be implemented by subclasses.")
    
class AR1Model(BaseModel):
    def __init__(self):
        super().__init__("AR(1)")

    def fit(self, train_data, target_col, **kwargs):
        """
        Fit an AR(1) model to the training data.

        Args:
            train_data (pd.DataFrame): Training dataset.
            target_col (str): Name of the target column.

        Returns:
            Fitted AR(1) model.
        """
        lagged_target = train_data[target_col].shift(1).dropna()
        X = sm.add_constant(lagged_target)
        y = train_data[target_col].loc[X.index]
        model = sm.OLS(y, X).fit()
        return model

    def forecast(self, model, steps, train_data, target_col):
        """
        Generate iterative AR(1) forecasts.

        Args:
            model: Fitted AR(1) model.
            steps (int): Number of steps to forecast.
            train_data (pd.DataFrame): Training dataset.
            target_col (str): Name of the target column.

        Returns:
            list: Iteratively forecasted values.
        """
        # Automatically get the last observed value from the training data
        last_value = train_data[target_col].iloc[-1]
        
        forecast_values = []
        current_value = last_value

        for _ in range(steps):
            # Forecast using the AR(1) equation: y_t = β0 + β1 * y_t-1
            forecast = model.params[0] + model.params[1] * current_value
            forecast_values.append(forecast)
            current_value = forecast  # Update for the next step

        return forecast_values
    
class ARXModel(BaseModel):
    def __init__(self):
        super().__init__("ARX")

    def fit(self, train_data, target_col, exogenous_vars, **kwargs):
        lagged_target = train_data[target_col].shift(1).dropna()
        exogenous_data = [train_data[var].shift(1).dropna() for var in exogenous_vars]
        X = pd.concat([lagged_target] + exogenous_data, axis=1).dropna()
        X = sm.add_constant(X)
        y = train_data[target_col].loc[X.index]
        model = sm.OLS(y, X).fit()
        return model

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest")

    def fit(self, train_data, target_col, exogenous_vars, **kwargs):
        X = train_data[exogenous_vars].dropna()
        y = train_data[target_col].loc[X.index]
        model = RandomForestRegressor(**kwargs).fit(X, y)
        return model
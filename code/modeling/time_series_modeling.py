import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

import os
os.chdir(r'C:\git\backtest-baam\code')

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
        lagged_target = train_data[target_col].shift(1).dropna()
        X = sm.add_constant(lagged_target)
        y = train_data[target_col].loc[X.index]
        model = sm.OLS(y, X).fit()
        return model
    
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
import os
os.chdir(r'C:\git\backtest-baam\code')

import numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import r2_score

import pandas as pd

def calculate_out_of_sample_metrics(df_predictions):
    """
    Calculates out-of-sample metrics (RMSE, R-squared) by horizon, execution date, and row (observation).

    Args:
        df_predictions (pd.DataFrame): DataFrame containing predictions, actuals, horizons, and execution dates.

    Returns:
        dict: Metrics including RMSE and R-squared by horizon, execution date, and row (observation).
    """
    # Ensure necessary columns are present
    if not {"horizon", "actual", "prediction", "execution_date", "forecast_date"}.issubset(df_predictions.columns):
        raise ValueError("df_predictions must contain 'horizon', 'actual', 'prediction', 'execution_date', and 'forecast_date' columns.")

    # Drop rows with NaN in Actual or Prediction
    df_predictions = df_predictions.dropna(subset=["actual", "prediction"])

    # Calculate residuals and squared errors
    df_predictions["residual"] = df_predictions["actual"] - df_predictions["prediction"]
    df_predictions["squared_error"] = df_predictions["residual"] ** 2
    df_predictions["rmse_row"] = np.sqrt(df_predictions["squared_error"])  # RMSE for each row

    # Metrics by horizon
    metrics_by_horizon = (
        df_predictions.groupby("horizon")
        .apply(lambda group: pd.Series({
            "mse": group["squared_error"].mean(),
            "r_squared": r2_score(group["actual"], group["prediction"]),
            "rmse": np.sqrt(group["squared_error"].mean())
        }))
        .reset_index()
    )

    # RMSE by execution date
    metrics_by_execution_date = (
        df_predictions.groupby("execution_date")
        .apply(lambda group: pd.Series({
            "mse": group["squared_error"].mean(),
            "r_squared": r2_score(group["actual"], group["prediction"]),
            "rmse": np.sqrt(group["squared_error"].mean())
        }))
        .reset_index()
    )

    return {
        "by_horizon": metrics_by_horizon,
        "by_execution_date": metrics_by_execution_date,
        "by_row": df_predictions[["execution_date", "forecast_date", "horizon", "rmse_row"]]  # RMSE for each row
    }

def calculate_rmse(predictions, actuals):
    """
    Calculate Root Mean Squared Error (RMSE) between predictions and actual values.

    Args:
        predictions (np.array): Predicted values.
        actuals (np.array): Actual observed values.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(((predictions - actuals) ** 2).mean())


def calculate_r_squared(predictions, actuals):
    """
    Calculate R-squared (coefficient of determination) between predictions and actual values.

    Args:
        predictions (np.array): Predicted values.
        actuals (np.array): Actual observed values.

    Returns:
        float: R-squared value.
    """
    ss_total = ((actuals - actuals.mean()) ** 2).sum()
    ss_residual = ((actuals - predictions) ** 2).sum()
    return 1 - (ss_residual / ss_total)

def kupiec_pof_test(expected_breaches, actual_breaches, total_observations):
    """
    Perform Kupiec's Proportion of Failures (POF) Test.

    Args:
        expected_breaches (float): Expected proportion of breaches (e.g., confidence level).
        actual_breaches (int): Observed number of breaches.
        total_observations (int): Total number of observations.

    Returns:
        dict: Kupiec POF test results with:
            - "test_statistic": Chi-squared test statistic.
            - "p_value": P-value of the test.
            - "pass": Boolean indicating whether the test passed (p-value > 0.05).
    """
    # Observed proportion of breaches
    observed_breaches = actual_breaches / total_observations

    # Kupiec test statistic (likelihood ratio)
    if actual_breaches == 0:
        return {"test_statistic": 0, "p_value": 1.0, "pass": True}  # No breaches, test passes trivially
    likelihood_ratio = -2 * (
        total_observations * (expected_breaches * np.log(expected_breaches) + (1 - expected_breaches) * np.log(1 - expected_breaches))
        - (actual_breaches * np.log(observed_breaches) + (total_observations - actual_breaches) * np.log(1 - observed_breaches))
    )

    # P-value and test result
    p_value = 1 - chi2.cdf(likelihood_ratio, df=1)
    pass_test = p_value > 0.05  # Test passes if p-value > 0.05

    return {"test_statistic": likelihood_ratio, "p_value": p_value, "pass": pass_test}


def christoffersen_independence_test(breach_sequence):
    """
    Perform Christoffersen's Test for Independence of VaR breaches.

    Args:
        breach_sequence (list): Sequence of 0s (no breach) and 1s (breach).

    Returns:
        dict: Christoffersen's Independence test results with:
            - "test_statistic": Chi-squared test statistic.
            - "p_value": P-value of the test.
            - "pass": Boolean indicating whether the test passed (p-value > 0.05).
    """
    # Transition counts
    n00 = sum((breach_sequence[i] == 0) and (breach_sequence[i + 1] == 0) for i in range(len(breach_sequence) - 1))
    n01 = sum((breach_sequence[i] == 0) and (breach_sequence[i + 1] == 1) for i in range(len(breach_sequence) - 1))
    n10 = sum((breach_sequence[i] == 1) and (breach_sequence[i + 1] == 0) for i in range(len(breach_sequence) - 1))
    n11 = sum((breach_sequence[i] == 1) and (breach_sequence[i + 1] == 1) for i in range(len(breach_sequence) - 1))

    # Transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Likelihood ratio test statistic
    likelihood_ratio = -2 * (
        n00 * np.log(1 - p1) + n01 * np.log(p1) + n10 * np.log(1 - p1) + n11 * np.log(p1)
        - (n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11))
    )

    # P-value and test result
    p_value = 1 - chi2.cdf(likelihood_ratio, df=1)
    pass_test = p_value > 0.05  # Test passes if p-value > 0.05

    return {"test_statistic": likelihood_ratio, "p_value": p_value, "pass": pass_test}


def basel_traffic_light(actual_breaches, total_observations, confidence_level=0.05):
    """
    Perform Basel Traffic Light Approach for VaR backtesting.

    Args:
        actual_breaches (int): Observed number of VaR breaches.
        total_observations (int): Total number of observations.
        confidence_level (float): Confidence level for VaR (default: 0.05).

    Returns:
        str: Traffic light result ("green", "yellow", or "red").
    """
    # Basel thresholds for 99% confidence level (adjust for 95% if needed)
    expected_breaches = total_observations * confidence_level
    green_threshold = expected_breaches + 4 * np.sqrt(expected_breaches * (1 - confidence_level))
    yellow_threshold = expected_breaches + 10 * np.sqrt(expected_breaches * (1 - confidence_level))

    if actual_breaches <= green_threshold:
        return "green"
    elif actual_breaches <= yellow_threshold:
        return "yellow"
    else:
        return "red"
    
def ridge_backtest(es_forecasts, var_forecasts, observed_returns, confidence_level=0.05):
    """
    Perform Acerbi-Szekely Ridge Backtest for Expected Shortfall (ES).

    Args:
        es_forecasts (pd.Series): Predicted ES values.
        var_forecasts (pd.Series): Predicted VaR values.
        observed_returns (pd.Series): Observed portfolio returns.
        confidence_level (float): Confidence level for VaR and ES (default: 0.05).

    Returns:
        dict: Ridge Backtest results with:
            - "test_statistic": The Ridge Backtest statistic.
            - "p_value": The p-value of the test.
            - "pass": Boolean indicating whether the test passed.
    """
    # Calculate the Ridge Backtesting function for each observation
    z_values = (es_forecasts - var_forecasts - (observed_returns + var_forecasts).clip(upper=0)) / es_forecasts

    # Mean of the Ridge Backtesting function
    z_mean = z_values.mean()

    # Monte Carlo simulation to estimate the distribution of the test statistic
    num_simulations = 100000
    simulated_z_means = np.random.normal(loc=0, scale=z_values.std(), size=num_simulations)

    # Calculate p-value
    p_value = (simulated_z_means >= z_mean).mean()

    # Determine if the test passes (p-value > 0.05)
    pass_test = p_value > 0.05

    return {
        "test_statistic": z_mean,
        "p_value": p_value,
        "pass": pass_test
    } 
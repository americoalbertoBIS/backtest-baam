import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import mlflow
from pathlib import Path

os.chdir(r'C:\git\yield-curve-app\src')
# Import required modules
from utils.data_processing import DataLoader
from utils.modeling import YieldCurveModel
from utils.nelson_siegel import compute_nsr_shadow_ts_noErr
from datetime import datetime, timedelta

# Constants
CONFIDENCE_LEVEL = 0.05  # 5% for 95% confidence level
MONTHS_IN_YEAR = 12      # Number of months in a year
SAVE_DIR = r"C:\git\backtest-baam\data"
LOG_DIR = r"C:\git\backtest-baam\logs"
MLFLOW_TRACKING_URI = r"sqlite:///C:/git/backtest-baam/mlflow/mlflow.db"

def calculate_prices_from_yields(yields, maturity):
    """
    Convert yields to zero-coupon bond prices for a given maturity.

    Args:
        yields (pd.DataFrame): DataFrame of yields (rows: dates, columns: simulations).
        maturity (float): Maturity of the bond.

    Returns:
        pd.DataFrame: DataFrame of bond prices.
    """
    return 1 / (1 + yields) ** maturity  # Price = 1 / (1 + Y)^T


def calculate_returns_from_prices(prices, months_in_year=12):
    """
    Calculate monthly and annual returns from bond prices.

    Args:
        prices (pd.DataFrame): DataFrame of bond prices (rows: dates, columns: simulations).
        months_in_year (int): Number of months in a year (default: 12).

    Returns:
        tuple: (monthly_returns, annual_returns)
            - monthly_returns: DataFrame of monthly returns.
            - annual_returns: DataFrame of annual returns (aggregated from monthly returns).
    """
    # Calculate monthly returns
    monthly_returns = prices.pct_change()  # Drop NaN values after percentage change

    # Aggregate monthly returns into annual returns (arithmetic sum)
    annual_returns = monthly_returns.groupby(np.arange(len(monthly_returns)) // months_in_year).sum()

    return monthly_returns, annual_returns

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


class YieldCurveProcessor:
    def __init__(self, country, model, execution_date, yield_curve_model, model_params):
        self.country = country
        self.model = model
        self.execution_date = execution_date
        self.yield_curve_model = yield_curve_model
        self.model_params = model_params
        self.results_dir = Path(SAVE_DIR) / country #/ model
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage for metrics
        self.metrics_data = []

    def load_betas(self):
        """
        Load forecasted and simulated betas for the current model and execution date.
        """
        def subset_dataframe(df, execution_date):
            df['ExecutionDate'] = pd.to_datetime(df['ExecutionDate'])
            df['ForecastDate'] = pd.to_datetime(df['ForecastDate'])
            return df[df['ExecutionDate'] == execution_date].copy()

        # Load forecasted betas
        df_pred_beta1 = pd.read_csv(self.results_dir / f"beta1_forecasts_{self.model}.csv")
        df_pred_beta2 = pd.read_csv(self.results_dir / f"beta2_forecasts_{self.model}.csv")
        df_pred_beta3 = pd.read_csv(self.results_dir / f"beta3_forecasts_{self.model}.csv")

        # Subset for the current execution date
        self.df_pred_beta1 = subset_dataframe(df_pred_beta1, self.execution_date)
        self.df_pred_beta2 = subset_dataframe(df_pred_beta2, self.execution_date)
        self.df_pred_beta3 = subset_dataframe(df_pred_beta3, self.execution_date)

        # Load simulated betas
        self.df_sim_beta1 = pd.read_parquet(self.results_dir / f"beta1_simulations_{self.model}.parquet", engine="pyarrow")
        self.df_sim_beta2 = pd.read_parquet(self.results_dir / f"beta2_simulations_{self.model}.parquet", engine="pyarrow")
        self.df_sim_beta3 = pd.read_parquet(self.results_dir / f"beta3_simulations_{self.model}.parquet", engine="pyarrow")

    def compute_observed_yields(self):
        """
        Process observed yields from the yield curve model.
        """
        # Extract observed yields and convert dates
        #dates_num = self.yield_curve_model.dates
        dates_str = yield_curve_model.dates_str
        observed_yields_df = pd.DataFrame(
            self.yield_curve_model.yieldsObservedAgg,
            columns=[f'{tau} years' for tau in self.yield_curve_model.uniqueTaus],
            index=dates_str[-len(self.yield_curve_model.yieldsObservedAgg):]
        )
        observed_yields_df.index = pd.to_datetime(observed_yields_df.index)
        
        # Resample to monthly frequency
        self.observed_yields_df_resampled = observed_yields_df.resample('MS').mean()
        self.observed_yields_df_resampled /= 100
        
    def compute_predicted_yields(self):
        """
        Compute predicted yields using the Nelson-Siegel model.
        """
        # Combine forecasted betas into a rotated array
        rotated_betas = np.array([
            self.df_pred_beta1['Prediction'].dropna().values,
            self.df_pred_beta2['Prediction'].dropna().values,
            self.df_pred_beta3['Prediction'].dropna().values
        ]).T

        # Compute predicted yields
        self.model_params['lambda'] = 0.7173
        predicted_yields = compute_nsr_shadow_ts_noErr(
            rotated_betas, self.yield_curve_model.uniqueTaus, self.yield_curve_model.invRotationMatrix, self.model_params
        )

        # Create DataFrame for predicted yields
        self.predicted_yields_df = pd.DataFrame(
            predicted_yields,
            index=self.df_pred_beta1['ForecastDate'].dropna(),
            columns=[f'{tau} years' for tau in self.yield_curve_model.uniqueTaus]
        )
        self.predicted_yields_df /= 100
        
    def align_observed_and_predicted_yields(self):
        """
        Align observed and predicted yields by overlapping dates.
        """
        overlapping_dates = self.observed_yields_df_resampled.index.intersection(self.predicted_yields_df.index)
        self.aligned_observed_yields_df = self.observed_yields_df_resampled.loc[overlapping_dates]
        self.aligned_predicted_yields_df = self.predicted_yields_df.loc[overlapping_dates]

    def compute_rmse_r_squared(self):
        """
        Compute RMSE and R-squared for each maturity and specific horizons (e.g., 6, 12, 24, 36, 48, 60 months).
        """
        # Define forecast horizons in months
        forecast_horizons = [6, 12, 24, 36, 48, 60]
    
        for column in self.aligned_observed_yields_df.columns:
            maturity = float(column.split()[0])  # Extract maturity from column name
    
            for horizon in forecast_horizons:
                # Filter observed and predicted yields for the specific horizon
                horizon_end_date = self.execution_date + pd.DateOffset(months=horizon)
                observed_horizon = self.aligned_observed_yields_df.loc[
                    (self.aligned_observed_yields_df.index <= horizon_end_date), column
                ]
                predicted_horizon = self.aligned_predicted_yields_df.loc[
                    (self.aligned_predicted_yields_df.index <= horizon_end_date), column
                ]
    
                # Ensure alignment of observed and predicted data
                overlapping_dates = observed_horizon.index.intersection(predicted_horizon.index)
                observed_horizon = observed_horizon.loc[overlapping_dates]
                predicted_horizon = predicted_horizon.loc[overlapping_dates]
    
                # Skip if no overlapping data
                if len(observed_horizon) == 0 or len(predicted_horizon) == 0:
                    continue
    
                # Calculate RMSE and R-squared
                rmse = calculate_rmse(predicted_horizon.values, observed_horizon.values)
                r_squared = calculate_r_squared(predicted_horizon.values, observed_horizon.values)
    
                # Append metrics to the data list
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon / 12,  # Convert months to years
                    "Metric": "RMSE",
                    "Value": rmse
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon / 12,
                    "Metric": "R-Squared",
                    "Value": r_squared
                })
        
    def compute_var_cvar(self):
        """
        Compute VaR, CVaR, Expected Returns, and Observed Annual Returns for each year of the horizon and maturity.
        """
        for maturity in self.yield_curve_model.uniqueTaus:
            # Extract simulations for the current maturity
            simulations_for_maturity = self.simulated_observed_yields_df.xs(maturity, level="Maturity", axis=1)
    
            # Convert simulated yields to prices
            prices_for_maturity = calculate_prices_from_yields(simulations_for_maturity, maturity)
    
            # Calculate monthly and annual returns for simulated prices
            monthly_returns, annual_returns = calculate_returns_from_prices(prices_for_maturity, months_in_year=MONTHS_IN_YEAR)
    
            # Convert observed yields to prices
            observed_yields = self.aligned_observed_yields_df[f"{maturity} years"]
            observed_prices = calculate_prices_from_yields(observed_yields, maturity)
    
            # Calculate observed monthly returns
            observed_returns = observed_prices.pct_change().dropna()
    
            # Group observed returns into annual periods relative to the execution date
            observed_annual_returns = observed_returns.groupby(
                np.arange(len(observed_returns)) // MONTHS_IN_YEAR
            ).sum()
    
            # Ensure observed annual returns align with simulated annual returns
            observed_annual_returns = observed_annual_returns.iloc[:len(annual_returns)]
    
            # Vectorized calculations
            expected_returns = annual_returns.mean(axis=1)  # Mean across simulations
            var_values = annual_returns.quantile(CONFIDENCE_LEVEL, axis=1)  # VaR (quantile)
            cvar_values = annual_returns[annual_returns.le(var_values, axis=0)].mean(axis=1)  # CVaR (mean below VaR)
    
            # Iterate through each horizon (1 to 5 years)
            for horizon, (expected_return, var, cvar, observed_return) in enumerate(
                zip(expected_returns, var_values, cvar_values, observed_annual_returns), start=1
            ):
                # Skip if the horizon exceeds the available data
                if horizon > 5:
                    break
    
                # Perform Kupiec's POF Test
                annual_return = annual_returns.iloc[horizon - 1]  # Get returns for the specific year
                actual_breaches = len(annual_return[annual_return <= var])
                total_observations = len(annual_return)
                kupiec_results = kupiec_pof_test(CONFIDENCE_LEVEL, actual_breaches, total_observations)
    
                # Perform Christoffersen's Independence Test
                breach_sequence = (annual_return <= var).astype(int).tolist()
                christoffersen_results = christoffersen_independence_test(breach_sequence)
    
                # Perform Basel Traffic Light Test
                basel_result = basel_traffic_light(actual_breaches, total_observations, confidence_level=CONFIDENCE_LEVEL)
    
                # Perform Ridge Backtest for CVaR (ES)
                ridge_results = ridge_backtest(
                    es_forecasts=pd.Series(cvar, index=annual_return.index),
                    var_forecasts=pd.Series(var, index=annual_return.index),
                    observed_returns=annual_return,
                    confidence_level=CONFIDENCE_LEVEL
                )
    
                # Append metrics to the data list
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "VaR",
                    "Value": var
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "CVaR",
                    "Value": cvar
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Expected Returns",
                    "Value": expected_return
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Observed Annual Return",
                    "Value": observed_return
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Kupiec POF Test Statistic",
                    "Value": kupiec_results["test_statistic"]
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Kupiec POF Test P-Value",
                    "Value": kupiec_results["p_value"]
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Kupiec POF Test Pass",
                    "Value": int(kupiec_results["pass"])
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Christoffersen Independence Test Statistic",
                    "Value": christoffersen_results["test_statistic"]
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Christoffersen Independence Test P-Value",
                    "Value": christoffersen_results["p_value"]
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Christoffersen Independence Test Pass",
                    "Value": int(christoffersen_results["pass"])
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Basel Traffic Light",
                    "Value": basel_result
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Ridge Test Statistic",
                    "Value": ridge_results["test_statistic"]
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Ridge Test P-Value",
                    "Value": ridge_results["p_value"]
                })
                self.metrics_data.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Ridge Test Pass",
                    "Value": int(ridge_results["pass"])
                })
        
    def compute_simulated_observed_yields(self):
        """
        Compute simulated observed yields using the Nelson-Siegel model for all simulations.
        """
        def filter_and_pivot_simulated_betas(df_sim, execution_date):
            """
            Filter and pivot the simulated betas DataFrame for the given execution date.
            """
            df_sim = df_sim[df_sim['ExecutionDate'] == execution_date].copy()
            df_sim['ForecastDate'] = pd.to_datetime(df_sim['ForecastDate'])
            return df_sim.pivot(index='ForecastDate', columns='SimulationID', values='SimulatedValue')
    
        # Pivot simulated betas for each beta
        sim_beta1_pivot = filter_and_pivot_simulated_betas(self.df_sim_beta1, self.execution_date)
        sim_beta2_pivot = filter_and_pivot_simulated_betas(self.df_sim_beta2, self.execution_date)
        sim_beta3_pivot = filter_and_pivot_simulated_betas(self.df_sim_beta3, self.execution_date)
    
        # Combine into a 3D array (forecast dates, 3 betas, simulations)
        simulated_betas = np.array([
            sim_beta1_pivot.values,
            sim_beta2_pivot.values,
            sim_beta3_pivot.values
        ]).transpose(1, 0, 2)  # Shape: (forecast dates, 3 betas, simulations)
    
        # Compute observed yields for all simulations
        simulated_observed_yields = [
            compute_nsr_shadow_ts_noErr(
                simulated_betas[:, :, sim_id],
                self.yield_curve_model.uniqueTaus,
                self.yield_curve_model.invRotationMatrix,
                self.model_params
            )
            for sim_id in range(simulated_betas.shape[2])
        ]
    
        # Combine simulated observed yields into a MultiIndex DataFrame
        self.simulated_observed_yields_df = pd.DataFrame(
            np.stack(simulated_observed_yields, axis=-1).reshape(len(sim_beta1_pivot.index), -1),  # Flatten simulations
            index=sim_beta1_pivot.index,
            columns=pd.MultiIndex.from_product(
                [self.yield_curve_model.uniqueTaus, range(simulated_betas.shape[2])],  # Maturity and SimulationID
                names=["Maturity", "SimulationID"]
            )
        )
    
        self.simulated_observed_yields_df /= 100
        
    def save_results(self):
        """
        Save metrics data to a CSV file, appending to existing data without loading the entire file.
        """
        metrics_df = pd.DataFrame(self.metrics_data)
        metrics_file_path = self.results_dir / "metrics_timeseries.csv"
        
        # Append to the file if it exists; otherwise, create a new file
        write_mode = 'a' if metrics_file_path.exists() else 'w'
        header = not metrics_file_path.exists()  # Write header only if the file does not exist
        
        metrics_df.to_csv(metrics_file_path, mode=write_mode, header=header, index=False)
        print(f"Metrics saved to {metrics_file_path}")
    
if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Define the experiment name (e.g., based on the project name)
    experiment_name = "Yield Curve Backtest"
    mlflow.set_experiment(experiment_name)  # Set the experiment name

    countries = ['US'] # , 'EA', 'UK'
    models = ['AR_1'] # _1_Inflation_UCSV', 'AR_1_OutputGap'

    for country in countries:
        data_loader = DataLoader(r'L:\RMAS\Resources\BAAM\OpenBAAM\Private\Data\BaseDB.mat')
        _, _, _ = data_loader.load_data()
        selectedCurveName, selected_curve_data, modelParams = data_loader.process_data(country)
        modelParams.update({'minMaturity': 0.08, 'maxMaturity': 10, 'lambda1fixed': 0.7173})
        yield_curve_model = YieldCurveModel(selected_curve_data, modelParams)

        for model in models:
            execution_dates = pd.read_parquet(SAVE_DIR + f'/{country}/beta1_simulations_{model}.parquet')['ExecutionDate'].unique()

            for execution_date in execution_dates:
                processor = YieldCurveProcessor(country, model, execution_date, yield_curve_model, modelParams)
                processor.load_betas()
                processor.compute_observed_yields()
                processor.compute_predicted_yields()
                processor.align_observed_and_predicted_yields()
                processor.compute_rmse_r_squared()
                processor.compute_simulated_observed_yields()
                processor.compute_var_cvar()
                processor.save_results()

                # Define a unique run name for each execution
                run_name = f"{country}_{model}_{execution_date}"
                with mlflow.start_run(run_name=run_name):
                    metrics_df = pd.read_csv(processor.results_dir / "metrics_timeseries.csv")
                    for metric in ["RMSE", "R-Squared", "VaR", "CVaR", "Kupiec POF Test Pass", "Christoffersen Independence Test Pass", "Ridge Test Pass"]:
                        for maturity in yield_curve_model.uniqueTaus:
                            metric_values = metrics_df[
                                (metrics_df["Metric"] == metric) & (metrics_df["Maturity (Years)"] == maturity)
                            ]["Value"]
                            mlflow.log_metric(f"{metric}_Mean_{maturity}y", metric_values.mean())

                    # Log the metrics file as an artifact
                    mlflow.log_artifact(processor.results_dir / "metrics_timeseries.csv")

                    
# Load metrics_timeseries.csv
metrics_df_raw = pd.read_csv(r"C:\git\backtest-baam\data\US\metrics_timeseries.csv")
metrics_df = metrics_df.iloc[1066:,:]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_rmse(metrics_df):
    """
    Create a 3D plot for RMSE values.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing RMSE values with columns:
            - "Maturity (Years)"
            - "Horizon (Years)"
            - "Metric"
            - "Value"
    """
    # Filter the DataFrame for RMSE values
    rmse_df = metrics_df[metrics_df["Metric"] == "RMSE"]
    rmse_df['Value'] = pd.to_numeric(rmse_df['Value'])*100
    # Extract unique maturities and horizons
    maturities = rmse_df["Maturity (Years)"].unique()
    horizons = rmse_df["Horizon (Years)"].unique()

    # Create a meshgrid for maturities and horizons
    X, Y = np.meshgrid(maturities, horizons)

    # Map RMSE values to the grid
    Z = np.zeros_like(X, dtype=float)
    for i, horizon in enumerate(horizons):
        for j, maturity in enumerate(maturities):
            # Get RMSE value for the current maturity and horizon
            value = rmse_df[
                (rmse_df["Maturity (Years)"] == maturity) &
                (rmse_df["Horizon (Years)"] == horizon)
            ]["Value"].values 
            Z[i, j] = value[0] if len(value) > 0 else np.nan  # Handle missing values

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8)

    # Add labels and title
    ax.set_xlabel("Maturity (Years)", fontsize=12)
    ax.set_ylabel("Horizon (Years)", fontsize=12)
    ax.set_zlabel("RMSE", fontsize=12)
    ax.set_title("3D Plot of RMSE", fontsize=14)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="RMSE")

    # Show the plot
    plt.tight_layout()
    plt.show()                    



# Plot 3D RMSE
plot_3d_rmse(metrics_df)

import matplotlib.pyplot as plt

def plot_var_cvar(yield_curve_processor, maturity, horizon):
    """
    Plot observed returns, VaR, CVaR, and breaches dynamically.

    Args:
        yield_curve_processor (YieldCurveProcessor): Instance of the YieldCurveProcessor class.
        maturity (float): The maturity (in years) to plot.
        horizon (float): The horizon (in years) to plot.
    """
    # Compute observed returns dynamically
    observed_yields = yield_curve_processor.aligned_observed_yields_df[f"{maturity} years"]
    observed_returns = observed_yields.pct_change().dropna()*100  # Percentage change in yields*

    # Compute VaR and CVaR dynamically
    horizon_months = int(horizon * 12)  # Convert horizon to months
    simulated_yields = yield_curve_processor.simulated_observed_yields_df.xs(maturity, level="Maturity", axis=1)
    horizon_end_date = yield_curve_processor.execution_date + pd.DateOffset(months=horizon_months)
    simulated_horizon = simulated_yields.loc[
        (simulated_yields.index <= horizon_end_date)
    ]

    # Compute VaR and CVaR
    var_values = simulated_horizon.quantile(CONFIDENCE_LEVEL, axis=1)
    cvar_values = simulated_horizon[simulated_horizon.le(var_values, axis=0)].mean(axis=1)

    # Align observed returns with VaR and CVaR
    aligned_data = pd.concat([observed_returns, var_values, cvar_values], axis=1, join="inner")
    aligned_data.columns = ["Observed Returns", "VaR", "CVaR"]

    # Identify breaches (observed returns below VaR)
    breaches = aligned_data["Observed Returns"] < aligned_data["VaR"]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(aligned_data.index, aligned_data["Observed Returns"], label="Observed Returns", color="blue", alpha=0.7)
    plt.plot(aligned_data.index, aligned_data["VaR"], label="VaR (Threshold)", color="red", linestyle="--")
    plt.plot(aligned_data.index, aligned_data["CVaR"], label="CVaR (Tail Average)", color="orange", linestyle=":")
    plt.scatter(aligned_data.index[breaches], aligned_data["Observed Returns"][breaches], 
                color="black", label="VaR Breaches", zorder=5)

    # Add labels and legend
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.title(f"VaR and CVaR Visualization (Maturity={maturity}y, Horizon={horizon}y)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Returns", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.6, linestyle="--")
    plt.tight_layout()
    plt.show()
    
# Example: Plot for maturity = 1 year and horizon = 1 year
plot_var_cvar(processor, maturity=1.0, horizon=1.0)
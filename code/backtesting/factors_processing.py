import numpy as np
import pandas as pd

import mlflow
from pathlib import Path
from itertools import zip_longest
from filelock import FileLock

import os
os.chdir(r'C:\git\backtest-baam\code')

from modeling.nelson_siegel import compute_nsr_shadow_ts_noErr
from data_preparation.data_transformations import calculate_prices_from_yields, calculate_returns_from_prices
from modeling.evaluation_metrics import calculate_r_squared, calculate_rmse
from modeling.evaluation_metrics import calculate_out_of_sample_metrics

from datetime import datetime, timedelta

# Constants
CONFIDENCE_LEVEL = 0.05  # 5% for 95% confidence level
MONTHS_IN_YEAR = 12      # Number of months in a year
SAVE_DIR = r"C:\git\backtest-baam\data"
SAVE_DIR = r"L:\RMAS\Users\Alberto\backtest-baam\data"
LOG_DIR = r"C:\git\backtest-baam\logs"
MLFLOW_TRACKING_URI = r"sqlite:///C:/git/backtest-baam/mlflow/mlflow.db"

def load_betas(results_dir, model_config, execution_date):
    """
    Load the beta forecast and simulation files for a specific model configuration.

    Args:
        results_dir (str): Directory where the beta forecast and simulation files are stored.
        model_config (dict): Configuration for the selected model (e.g., "Mixed_Model").
        execution_date (str): Execution date for filtering.

    Returns:
        dict: A dictionary containing DataFrames for forecasted and simulated betas.
    """
    def subset_dataframe(df, execution_date):
        df['ExecutionDate'] = pd.to_datetime(df['ExecutionDate'])
        df['ForecastDate'] = pd.to_datetime(df['ForecastDate'])
        return df[df['ExecutionDate'] == execution_date].copy()

    beta_data = {
        "forecasted": {},
        "simulated": {}
    }

    for beta_name in ["beta1", "beta2", "beta3"]:
        # Get the beta model name from the configuration
        beta_model_name = model_config[beta_name]

        # Load forecasted betas
        forecast_file = f"forecasts.csv"
        forecast_path = results_dir / beta_model_name / beta_name / forecast_file 
        df_forecast = pd.read_csv(forecast_path)
        beta_data["forecasted"][beta_name] = subset_dataframe(df_forecast, execution_date)

        # Load simulated betas
        simulation_file = f"simulations.parquet"
        simulation_path = results_dir / beta_model_name / beta_name / simulation_file
        df_simulation = pd.read_parquet(simulation_path)
        beta_data["simulated"][beta_name] = subset_dataframe(df_simulation, execution_date)

    return beta_data

class FactorsProcessor:
    def __init__(self, country, model_name, model_config, execution_date, yield_curve_model, model_params):
        self.country = country
        self.model_name = model_name  # Name of the model (e.g., "Mixed_Model")
        self.model_config = model_config  # Configuration for the selected model
        self.execution_date = execution_date
        self.yield_curve_model = yield_curve_model
        self.model_params = model_params
        
        # Base directory for the country
        self.base_dir = Path(SAVE_DIR) / country   
             
        # Subdirectories for factors, yields, and metrics
        self.factors_dir = self.base_dir / "factors"
        self.yields_dir = self.base_dir / "yields"
        self.metrics_dir = self.base_dir / "metrics"
        self.returns_dir = self.base_dir / "returns"

        # Ensure all directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.factors_dir.mkdir(parents=True, exist_ok=True)
        self.yields_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage for metrics
        self.metrics_data = []

    def load_betas(self):
        """
        Load forecasted and simulated betas based on the selected model configuration.
        """
        # Load forecasted and simulated betas
        beta_data = load_betas(self.factors_dir, self.model_config, self.execution_date)

        # Assign forecasted and simulated betas to class attributes
        self.df_pred_beta1 = beta_data["forecasted"]["beta1"]
        self.df_pred_beta2 = beta_data["forecasted"]["beta2"]
        self.df_pred_beta3 = beta_data["forecasted"]["beta3"]

        self.df_sim_beta1 = beta_data["simulated"]["beta1"]
        self.df_sim_beta2 = beta_data["simulated"]["beta2"]
        self.df_sim_beta3 = beta_data["simulated"]["beta3"]

    def compute_observed_yields(self):
        """
        Process observed yields from the yield curve model.
        """
        # Extract observed yields and convert dates
        #dates_num = self.yield_curve_model.dates
        dates_str = self.yield_curve_model.dates_str
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
        Compute predicted yields using the Nelson-Siegel model and save them directly to a CSV file.
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

        # Ensure the DataFrame is not empty
        if self.predicted_yields_df.empty:
            print("Predicted yields DataFrame is empty. Skipping.")
            return None

        # Reset the index to make 'forecasted_date' a regular column
        reshaped_df = self.predicted_yields_df.reset_index().melt(
            id_vars='ForecastDate',  # Use the column created by reset_index()
            var_name='maturity',
            value_name='prediction'
        ).rename(columns={'ForecastDate': 'forecasted_date'})

        # Add additional columns
        reshaped_df['execution_date'] = self.execution_date
        
        # Add the horizon column directly from df_pred_beta1
        reshaped_df = reshaped_df.merge(
            self.df_pred_beta1[['ForecastDate', 'Horizon']].rename(columns={'ForecastDate': 'forecasted_date'}),
            on='forecasted_date',
            how='left'
        )
        
        reshaped_df['actual'] = reshaped_df.apply(
            lambda row: self.observed_yields_df_resampled.at[row['forecasted_date'], row['maturity']]
            if row['forecasted_date'] in self.observed_yields_df_resampled.index else np.nan,
            axis=1
        )

        # Ensure the results directory exists
        self.yields_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.yields_dir  / "estimated_yields" / f"{self.model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        output_file_path = model_dir / "forecasts.csv"
        lock_file_path = f"{output_file_path}.lock"

        # Use FileLock to ensure thread-safe writing
        lock = FileLock(lock_file_path)
        with lock:
            reshaped_df.to_csv(
                output_file_path,
                mode='a',  # Append mode
                header=not output_file_path.exists(),  # Write header if the file does not exist
                index=False  # Do not write the index column
            )
        #print(f"Appended predicted yields to {output_file_path}")
        
        return self.predicted_yields_df
        
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
        horizons = [6, 12, 24, 36, 48, 60]  # Forecast horizons
        horizons = range(1, max(horizons)+1)
    
        for column in self.aligned_observed_yields_df.columns:
            maturity = float(column.split()[0])  # Extract maturity from column name
    
            for horizon in horizons:
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
    def calculate_and_save_returns(self):
        """
        Calculate and save monthly and annual returns for a given maturity.

        Args:
            maturity (float): The maturity for which returns are being calculated.
            prices_for_maturity (pd.DataFrame): Simulated prices for the given maturity.
        """
        returns_dir = self.returns_dir / "estimated_returns" / f"{self.model_name}" 
        returns_dir.mkdir(parents=True, exist_ok=True)

        for maturity in self.yield_curve_model.uniqueTaus:
            prices_for_maturity = calculate_prices_from_yields(
                                        self.simulated_observed_yields_df.xs(maturity, level="Maturity", axis=1), 
                                        maturity
                                    )

            # Calculate monthly and annual returns
            monthly_returns, annual_returns = calculate_returns_from_prices(prices_for_maturity, months_in_year=MONTHS_IN_YEAR)

            # Save Monthly Returns
            monthly_returns_long_format = monthly_returns.reset_index().melt(
                id_vars=["ForecastDate"],  # Use ForecastDate as the identifier
                var_name="SimulationID",  # Simulation IDs as variable names
                value_name="MonthlyReturn"  # Monthly returns as values
            )
            monthly_returns_long_format["Maturity"] = f"{maturity} years"
            monthly_returns_long_format["ExecutionDate"] = self.execution_date

            # Save to a Parquet file
            monthly_dir = returns_dir / "monthly" / f"{maturity}_years"
            monthly_dir.mkdir(parents=True, exist_ok=True)
            monthly_file_path = monthly_dir / f"simulations_{self.execution_date.strftime('%d%m%Y')}.parquet"
            monthly_returns_long_format.to_parquet(monthly_file_path, index=False)

            # Save Annual Returns
            annual_returns_long_format = annual_returns.reset_index().melt(
                id_vars=["ForecastDate"],  # Use ForecastDate as the identifier
                var_name="SimulationID",  # Simulation IDs as variable names
                value_name="AnnualReturn"  # Annual returns as values
            )
            annual_returns_long_format["Maturity"] = f"{maturity} years"
            annual_returns_long_format["ExecutionDate"] = self.execution_date

            # Save to a Parquet file
            annual_dir = returns_dir / "annual" / f"{maturity}_years"
            annual_dir.mkdir(parents=True, exist_ok=True)
            annual_file_path = annual_dir / f"simulations_{self.execution_date.strftime('%d%m%Y')}.parquet"
            annual_returns_long_format.to_parquet(annual_file_path, index=False)

    def compute_var_cvar_vol(self):
        """
        Compute VaR, CVaR, Expected Returns, and Observed Monthly and Annual Returns for each year of the horizon and maturity.
        Save monthly and annual metrics to separate files.
        """
        # Initialize lists to store metrics separately for monthly and annual returns
        monthly_metrics = []
        annual_metrics = []

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
            observed_returns = observed_prices.pct_change(fill_method=None).dropna()

            # Group observed returns into annual periods relative to the execution date
            observed_annual_returns = observed_returns.groupby(
                np.arange(len(observed_returns)) // MONTHS_IN_YEAR
            ).sum()

            # Ensure observed annual returns align with simulated annual returns
            observed_annual_returns = observed_annual_returns.iloc[:len(annual_returns)]

            # Vectorized calculations for monthly returns
            monthly_expected_returns = monthly_returns.mean(axis=1)  # Mean across simulations
            monthly_var_values = monthly_returns.quantile(CONFIDENCE_LEVEL, axis=1)  # VaR (quantile)
            monthly_cvar_values = monthly_returns[monthly_returns.le(monthly_var_values, axis=0)].mean(axis=1)  # CVaR
            monthly_volatility = monthly_returns.std(axis=1)  # Volatility (standard deviation)

            # Vectorized calculations for annual returns
            expected_returns = annual_returns.mean(axis=1)  # Mean across simulations
            var_values = annual_returns.quantile(CONFIDENCE_LEVEL, axis=1)  # VaR (quantile)
            cvar_values = annual_returns[annual_returns.le(var_values, axis=0)].mean(axis=1)  # CVaR
            volatility = annual_returns.std(axis=1)  # Volatility (standard deviation)

            # Iterate through monthly horizons (1 to 60 months)
            for horizon, (monthly_return, monthly_var, monthly_cvar, monthly_vol) in enumerate(
                zip_longest(monthly_expected_returns, monthly_var_values, monthly_cvar_values, monthly_volatility, fillvalue=None), 
                start=1
            ):
                # Skip if the horizon exceeds the available data
                if horizon > 60:
                    break

                monthly_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Months)": horizon,
                    "Metric": "Monthly Expected Return",
                    "Value": monthly_return
                })
                monthly_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Months)": horizon,
                    "Metric": "Monthly VaR",
                    "Value": monthly_var
                })
                monthly_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Months)": horizon,
                    "Metric": "Monthly CVaR",
                    "Value": monthly_cvar
                })
                monthly_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Months)": horizon,
                    "Metric": "Monthly Volatility",
                    "Value": monthly_vol
                })

            # Iterate through annual horizons (1 to 5 years)
            for horizon, (expected_return, var, cvar, vol, observed_return) in enumerate(
                zip_longest(expected_returns, var_values, cvar_values, volatility, observed_annual_returns, fillvalue=None), 
                start=1
            ):
                # Skip if the horizon exceeds the available data
                if horizon > 5:
                    break

                annual_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "VaR",
                    "Value": var
                })
                annual_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "CVaR",
                    "Value": cvar
                })
                annual_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Volatility",
                    "Value": vol
                })
                annual_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Expected Returns",
                    "Value": expected_return
                })
                annual_metrics.append({
                    "Maturity (Years)": maturity,
                    "Execution Date": self.execution_date,
                    "Horizon (Years)": horizon,
                    "Metric": "Observed Annual Return",
                    "Value": observed_return
                })

        # Save monthly metrics to a separate file
        monthly_metrics_df = pd.DataFrame(monthly_metrics)
        monthly_file_path = self.returns_dir / "estimated_returns" / f"{self.model_name}" / f"monthly_metrics.csv"
        lock_file_path = f"{monthly_file_path}.lock"
        lock = FileLock(lock_file_path)
        with lock:
            monthly_metrics_df.to_csv(
                monthly_file_path,
                mode='a',  # Append mode
                header=not monthly_file_path.exists(),  # Write header if the file does not exist
                index=False  # Do not write the index column
            )

        # Save annual metrics to a separate file
        annual_metrics_df = pd.DataFrame(annual_metrics)
        annual_file_path = self.returns_dir / "estimated_returns" / f"{self.model_name}" / f"annual_metrics.csv"
        lock_file_path = f"{annual_file_path}.lock"
        lock = FileLock(lock_file_path)
        with lock:
            annual_metrics_df.to_csv(
                annual_file_path,
                mode='a',  # Append mode
                header=not annual_file_path.exists(),  # Write header if the file does not exist
                index=False  # Do not write the index column
            )

    def compute_and_save_out_of_sample_metrics(self, df_predictions):
        """
        Compute out-of-sample metrics (e.g., RMSE, R-squared) for all maturities and save them to separate files.

        Args:
            df_predictions (pd.DataFrame): DataFrame containing predictions, actuals, horizons, and execution dates.
        """
        # Initialize lists to store metrics for all maturities
        outofsample_metrics_by_horizon = []
        outofsample_metrics_by_exec_date = []
        outofsample_metrics = []

        # Iterate over maturities
        for maturity in df_predictions['maturity'].unique():
            # Filter predictions for the current maturity
            temp = df_predictions[df_predictions['maturity'] == maturity].copy()

            # Calculate metrics for the current maturity
            outofsample_metrics_temp = calculate_out_of_sample_metrics(temp)

            # Add maturity as a column to each set of metrics
            outofsample_metrics_temp["by_horizon"]['maturity'] = maturity
            outofsample_metrics_temp["by_execution_date"]['maturity'] = maturity
            outofsample_metrics_temp["by_row"]['maturity'] = maturity

            # Append metrics to the corresponding lists
            outofsample_metrics_by_horizon.append(outofsample_metrics_temp["by_horizon"])
            outofsample_metrics_by_exec_date.append(outofsample_metrics_temp["by_execution_date"])
            outofsample_metrics.append(outofsample_metrics_temp["by_row"])

        # Combine metrics across all maturities
        metrics_by_horizon = pd.concat(outofsample_metrics_by_horizon, ignore_index=True)
        metrics_by_execution_date = pd.concat(outofsample_metrics_by_exec_date, ignore_index=True)
        metrics_by_row = pd.concat(outofsample_metrics, ignore_index=True)

        # Save metrics to CSV files
        print("Saving out-of-sample metrics to files...")

        metrics_by_horizon_file = self.yields_dir / "estimated_yields" / f"{self.model_name}" / f"outofsample_metrics_by_horizon.csv"
        metrics_by_horizon.to_csv(metrics_by_horizon_file, index=False)

        metrics_by_execution_date_file = self.yields_dir / "estimated_yields" / f"{self.model_name}" / f"outofsample_metrics_by_execution_date.csv"
        metrics_by_execution_date.to_csv(metrics_by_execution_date_file, index=False)

        metrics_by_row_file = self.yields_dir / "estimated_yields" / f"{self.model_name}" / f"outofsample_metrics_by_row.csv"
        metrics_by_row.to_csv(metrics_by_row_file, index=False)

        print("Out-of-sample metrics saved successfully.")

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
    
    def save_simulated_yields_long_format(self):
        """
        Save simulated yields in a long-format structure to a Parquet file.
        """
        simulations_dir = self.yields_dir / "estimated_yields" / f"{self.model_name}" / "simulations"
        simulations_dir.mkdir(parents=True, exist_ok=True)

        # Reset the index to include ForecastDate as a column
        reshaped_df = self.simulated_observed_yields_df.reset_index()
        reshaped_df.columns = ['ForecastDate'] + [
        f"{maturity}_{sim_id}" for maturity, sim_id in reshaped_df.columns[1:]
        ]
        # Melt the DataFrame to long format
        long_format_df = reshaped_df.melt(
        id_vars=["ForecastDate"],  # Keep ForecastDate as an identifier
        var_name="Maturity_SimulationID",  # Combine Maturity and SimulationID
        value_name="SimulatedValue"  # Name of the values column
        )
        
        # Split the combined "Maturity_SimulationID" into separate columns
        long_format_df[["Maturity", "SimulationID"]] = long_format_df["Maturity_SimulationID"].str.split("_", expand=True)
        
        # Drop the combined column
        long_format_df = long_format_df.drop(columns=["Maturity_SimulationID"])
        
        # Convert Maturity and SimulationID to appropriate types
        long_format_df["Maturity"] = long_format_df["Maturity"].astype(float)
        long_format_df["SimulationID"] = long_format_df["SimulationID"].astype(int)

        # Add additional columns
        long_format_df["ExecutionDate"] = self.execution_date  # Add execution date
        long_format_df["Model"] = self.model_name  # Add model name
        long_format_df["Horizon"] = (long_format_df["ForecastDate"] - self.execution_date).dt.days // 30  # Calculate horizon in months
        long_format_df["Horizon"] = long_format_df["Horizon"].astype(int)  # Ensure it's an integer
        long_format_df["Maturity"] = long_format_df["Maturity"].astype(float).map(
            lambda x: f"{x} years"
        )  # Convert maturity to a readable string
        
        for maturity in self.yield_curve_model.uniqueTaus:
            maturity_dir = simulations_dir / f"{maturity}_years"
            maturity_dir.mkdir(parents=True, exist_ok=True)

            # Extract simulations for the current maturity
            simulations_for_maturity = long_format_df[long_format_df["Maturity"] == f"{maturity} years"]

            # Save each execution date's simulations as a separate Parquet file
            file_path = maturity_dir / f"simulations_{self.execution_date.strftime('%d%m%Y')}.parquet"

            # Save to Parquet
            simulations_for_maturity.to_parquet(file_path, index=False)

    def save_results(self):
        """
        Save metrics to separate CSV files for yields and returns in the 'metrics' subdirectory.
        """
        metrics_df = pd.DataFrame(self.metrics_data)

        # Separate metrics into yields-related and returns-related
        yields_metrics = metrics_df[metrics_df['Metric'].isin(['RMSE', 'R-Squared'])]
        returns_metrics = metrics_df[metrics_df['Metric'].isin(['VaR', 'CVaR', 'Volatility', 'Expected Returns', 'Observed Annual Return'])]

        # Save yields metrics
        if not yields_metrics.empty:
            yields_file_path = self.metrics_dir / f"yields_metrics_{self.model_name}.csv"
            lock_file_path = f"{yields_file_path}.lock"
            lock = FileLock(lock_file_path)
            with lock:
                yields_metrics.to_csv(
                    yields_file_path,
                    mode='a',  # Append mode
                    header=not yields_file_path.exists(),  # Write header if the file does not exist
                    index=False  # Do not write the index column
                )
            print(f"Yields metrics saved to {yields_file_path}")

        # Save returns metrics
        if not returns_metrics.empty:
            returns_file_path = self.metrics_dir / f"returns_metrics_{self.model_name}.csv"
            lock_file_path = f"{returns_file_path}.lock"
            lock = FileLock(lock_file_path)
            with lock:
                returns_metrics.to_csv(
                    returns_file_path,
                    mode='a',  # Append mode
                    header=not returns_file_path.exists(),  # Write header if the file does not exist
                    index=False  # Do not write the index column
                )
            print(f"Returns metrics saved to {returns_file_path}")
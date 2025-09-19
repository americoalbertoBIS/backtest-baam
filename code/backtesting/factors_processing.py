import numpy as np
import pandas as pd

import mlflow
from pathlib import Path
from itertools import zip_longest
from filelock import FileLock
import logging 

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
SAVE_DIR = r'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data_joint'
LOG_DIR = r"C:\git\backtest-baam\logs"
MLFLOW_TRACKING_URI = r"sqlite:///C:/git/backtest-baam/mlflow/mlflow.db"

class FactorsProcessor:
    def __init__(self, country, model_name, model_config, execution_date, yield_curve_model, model_params, preloaded_data=None):
        """
        Initialize the FactorsProcessor class.

        Args:
            country (str): The country being processed (e.g., "US").
            model_name (str): The name of the model being processed (e.g., "Mixed_Model").
            model_config (dict): Configuration for the selected model.
            execution_date (datetime): The execution date being processed.
            yield_curve_model (YieldCurveModel): The yield curve model instance.
            model_params (dict): The model parameters.
            preloaded_data (dict, optional): Preloaded forecasted and simulated beta data.
        """
        self.country = country
        self.model_name = model_name
        self.model_config = model_config
        self.execution_date = execution_date
        self.yield_curve_model = yield_curve_model
        self.model_params = model_params
        self.preloaded_data = preloaded_data  # Preloaded data for forecasted and simulated betas

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
        Load forecasted and simulated betas based on the selected model configuration using preloaded data.

        This function filters the preloaded beta data for the current execution date and aligns
        the data across all three factors (`beta1`, `beta2`, `beta3`).

        Raises:
            ValueError: If any of the forecasted or simulated beta DataFrames are empty for a valid execution date.
        """
        def subset_dataframe(df, execution_date):
            """
            Filter a DataFrame for the given execution date.

            Args:
                df (pd.DataFrame): The DataFrame to filter.
                execution_date (datetime): The execution date to filter by.

            Returns:
                pd.DataFrame: The filtered DataFrame.
            """
            df['execution_date'] = pd.to_datetime(df['execution_date'])  # Ensure datetime type
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])  # Ensure datetime type
            df = df.sort_values(by=["execution_date", "forecast_date"])  # Sort by ExecutionDate and ForecastDate
            filtered_df = df[df['execution_date'] == execution_date].copy()
            #logging.info(f"Filtered DataFrame for execution date {execution_date}: {filtered_df.shape[0]} rows")
            return filtered_df

        # Ensure preloaded data is available
        if not self.preloaded_data:
            raise ValueError("Preloaded data is missing. Ensure preloaded_data is passed to FactorsProcessor.")

        beta_data = {
            "forecasted": {},
            "simulated": {}
        }

        # Use beta1 as the baseline for execution dates
        baseline_execution_dates = self.preloaded_data["forecasted"]["beta1"]["execution_date"].unique()
        logging.info(f"Baseline execution dates (from beta1): {len(baseline_execution_dates)} dates")

        # Check if the current execution date is in beta1
        if self.execution_date not in baseline_execution_dates:
            logging.warning(f"Execution date {self.execution_date} is not available in beta1. Skipping...")
            raise ValueError(f"Execution date {self.execution_date} is not available in beta1.")

        # Filter and align data for beta1, beta2, and beta3
        for beta_name in ["beta1", "beta2", "beta3"]:
            # Check if preloaded data contains the required keys
            if beta_name not in self.preloaded_data["forecasted"] or beta_name not in self.preloaded_data["simulated"]:
                raise ValueError(f"Preloaded data is missing for '{beta_name}'")

            # Filter forecasted and simulated data for the given execution date
            forecasted_df = subset_dataframe(self.preloaded_data["forecasted"][beta_name], self.execution_date)
            simulated_df = subset_dataframe(self.preloaded_data["simulated"][beta_name], self.execution_date)

            # Check if the execution date exists in the data for the current beta
            if forecasted_df.empty or simulated_df.empty:
                logging.warning(f"Execution date {self.execution_date} is not available in {beta_name}. Skipping...")
                raise ValueError(f"Execution date {self.execution_date} is not available in {beta_name}.")

            # Assign filtered data to beta_data
            beta_data["forecasted"][beta_name] = forecasted_df
            beta_data["simulated"][beta_name] = simulated_df

        # Assign forecasted and simulated betas to class attributes
        self.df_pred_beta1 = beta_data["forecasted"]["beta1"]
        self.df_pred_beta2 = beta_data["forecasted"]["beta2"]
        self.df_pred_beta3 = beta_data["forecasted"]["beta3"]

        self.df_sim_beta1 = beta_data["simulated"]["beta1"]
        self.df_sim_beta2 = beta_data["simulated"]["beta2"]
        self.df_sim_beta3 = beta_data["simulated"]["beta3"]

        # Log success
        logging.info(f"Successfully loaded and aligned betas for execution date {self.execution_date}")

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
            self.df_pred_beta1['prediction'].values,
            self.df_pred_beta2['prediction'].values,
            self.df_pred_beta3['prediction'].values
        ]).T

        # Compute predicted yields
        self.model_params['lambda'] = 0.7173
        predicted_yields = compute_nsr_shadow_ts_noErr(
            rotated_betas, self.yield_curve_model.uniqueTaus, self.yield_curve_model.invRotationMatrix, self.model_params
        )

        # Create DataFrame for predicted yields
        self.predicted_yields_df = pd.DataFrame(
            predicted_yields,
            index=self.df_pred_beta1['forecast_date'].dropna(),
            columns=[f'{tau} years' for tau in self.yield_curve_model.uniqueTaus]
        )
        self.predicted_yields_df /= 100

        # Ensure the DataFrame is not empty
        if self.predicted_yields_df.empty:
            print("Predicted yields DataFrame is empty. Skipping.")
            return None

        # Reset the index to make 'forecasted_date' a regular column
        reshaped_df = self.predicted_yields_df.reset_index().melt(
            id_vars='forecast_date',  # Use the column created by reset_index()
            var_name='maturity',
            value_name='prediction'
        )

        # Add additional columns
        reshaped_df['execution_date'] = self.execution_date
        
        # Add the horizon column directly from df_pred_beta1
        reshaped_df = reshaped_df.merge(
            self.df_pred_beta1[['forecast_date', 'horizon']],
            on='forecast_date',
            how='left'
        )
        
        reshaped_df['actual'] = reshaped_df.apply(
            lambda row: self.observed_yields_df_resampled.at[row['forecast_date'], row['maturity']]
            if row['forecast_date'] in self.observed_yields_df_resampled.index else np.nan,
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
        overlapping_dates = self.observed_yields_df_resampled.index.intersection(self.mean_simulated_yields_df.index)
        self.aligned_observed_yields_df = self.observed_yields_df_resampled.loc[overlapping_dates]
        self.aligned_predicted_yields_df = self.mean_simulated_yields_df.loc[overlapping_dates]

    def calculate_and_save_returns_simulations(self):
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
                                        self.simulated_observed_yields_df.xs(maturity, level="maturity", axis=1), 
                                        maturity
                                    )

            # Calculate monthly and annual returns
            monthly_returns, annual_returns = calculate_returns_from_prices(prices_for_maturity, months_in_year=MONTHS_IN_YEAR)

            # Save Monthly Returns
            monthly_returns_long_format = monthly_returns.reset_index().melt(
                id_vars=["forecast_date"],  # Use ForecastDate as the identifier
                var_name="simulation_id",  # Simulation IDs as variable names
                value_name="monthly_returns"  # Monthly returns as values
            )
            monthly_returns_long_format["maturity"] = f"{maturity} years"
            monthly_returns_long_format["execution_date"] = self.execution_date

            # Save to a Parquet file
            monthly_dir = returns_dir / "monthly" / "simulations" / f"{maturity}_years"
            monthly_dir.mkdir(parents=True, exist_ok=True)
            monthly_file_path = monthly_dir / f"simulations_{self.execution_date.strftime('%d%m%Y')}.parquet"
            monthly_returns_long_format.to_parquet(monthly_file_path, index=False)

            # Save Annual Returns
            annual_returns_long_format = annual_returns.reset_index().melt(
                id_vars=["index"],  # Use ForecastDate as the identifier
                var_name="simulation_id",  # Simulation IDs as variable names
                value_name="annual_returns"  # Annual returns as values
            )
            annual_returns_long_format = annual_returns_long_format.rename(columns={"index": "horizon"})
            annual_returns_long_format["horizon"] = annual_returns_long_format["horizon"]+1
            annual_returns_long_format["maturity"] = f"{maturity} years"
            annual_returns_long_format["execution_date"] = self.execution_date

            # Save to a Parquet file
            annual_dir = returns_dir / "annual" / "simulations" / f"{maturity}_years"
            annual_dir.mkdir(parents=True, exist_ok=True)
            annual_file_path = annual_dir / f"simulations_{self.execution_date.strftime('%d%m%Y')}.parquet"
            annual_returns_long_format.to_parquet(annual_file_path, index=False)

    def get_mean_simulated_returns(self):
        """
        Calculate mean simulated and actual monthly and annual returns for each maturity,
        aligning indices between simulated and observed yields.
        Returns two DataFrames: monthly_all, annual_all.
        """
        results = []
        for maturity in self.yield_curve_model.uniqueTaus:
            # Get simulated and observed yields for this maturity
            simulated_yields = self.simulated_observed_yields_df.xs(maturity, level="maturity", axis=1)
            observed_yields = self.observed_yields_df_resampled[f"{maturity} years"]

            # Align indices
            overlapping_dates = simulated_yields.index.intersection(observed_yields.index)
            aligned_simulated_yields = simulated_yields.loc[overlapping_dates]
            aligned_observed_yields = observed_yields.loc[overlapping_dates]

            # Calculate prices from yields
            simulated_prices = calculate_prices_from_yields(aligned_simulated_yields, maturity)
            observed_prices = calculate_prices_from_yields(aligned_observed_yields, maturity)

            # Monthly returns
            monthly_returns = simulated_prices.pct_change(fill_method=None)
            actual_monthly_returns = observed_prices.pct_change(fill_method=None)

            # Annual returns: group by year starting from forecast_date
            year_index = overlapping_dates.to_series().dt.to_period('A').astype(str)
            annual_returns = monthly_returns.groupby(year_index).sum()
            actual_annual_returns = actual_monthly_returns.groupby(year_index).sum()

            # Align annual returns by year index
            overlapping_years = annual_returns.index.intersection(actual_annual_returns.index)
            aligned_annual_returns = annual_returns.loc[overlapping_years]
            aligned_actual_annual_returns = actual_annual_returns.loc[overlapping_years]

            # Prepare DataFrames
            monthly_df = pd.DataFrame({
                'forecast_date': monthly_returns.index[1:],  # skip first NaN
                'execution_date': self.execution_date,
                'maturity': f"{maturity} years",
                'prediction': monthly_returns.mean(axis=1)[1:].values,
                'actual': actual_monthly_returns[1:].values,
                'horizon': ((monthly_returns.index[1:] - self.execution_date).days // 30)
            })
            annual_df = pd.DataFrame({
                'forecast_date': overlapping_years,
                'execution_date': self.execution_date,
                'maturity': f"{maturity} years",
                'prediction': aligned_annual_returns.mean(axis=1).values,
                'actual': aligned_actual_annual_returns.values,
                'horizon': np.arange(1, len(overlapping_years)+1)
            })
            results.append((monthly_df, annual_df))

        monthly_all = pd.concat([r[0] for r in results], ignore_index=True)
        annual_all = pd.concat([r[1] for r in results], ignore_index=True)
        return monthly_all, annual_all
    
    def compute_var_cvar_vol(self):
        """
        Compute VaR, CVaR, Expected Returns, and Observed Monthly and Annual Returns for each year of the horizon and maturity,
        at multiple confidence levels. Save monthly and annual metrics to separate files.
        """
        # Define confidence levels (as quantiles, e.g. 0.95 for 95%)
        confidence_levels = [0.95, 0.975, 0.99]
        monthly_metrics = []
        annual_metrics = []

        for maturity in self.yield_curve_model.uniqueTaus:
            # Extract simulations for the current maturity
            simulations_for_maturity = self.simulated_observed_yields_df.xs(maturity, level="maturity", axis=1)
            observed_yields = self.observed_yields_df_resampled[f"{maturity} years"]

            # Align time index
            overlapping_dates = observed_yields.index.intersection(simulations_for_maturity.index)
            aligned_observed_yields_df = observed_yields.loc[overlapping_dates]
            aligned_predicted_yields_df = simulations_for_maturity.loc[overlapping_dates]

            # Convert simulated yields to prices
            prices_for_maturity = calculate_prices_from_yields(aligned_predicted_yields_df, maturity)
            monthly_returns, annual_returns = calculate_returns_from_prices(prices_for_maturity, months_in_year=MONTHS_IN_YEAR)

            # Convert observed yields to prices and returns
            observed_prices = calculate_prices_from_yields(aligned_observed_yields_df, maturity)
            observed_returns = observed_prices.pct_change(fill_method=None).dropna()
            observed_annual_returns = observed_returns.groupby(
                np.arange(len(observed_returns)) // MONTHS_IN_YEAR
            ).sum()
            observed_annual_returns = observed_annual_returns.iloc[:len(annual_returns)]

            # Monthly metrics
            monthly_expected_returns = monthly_returns.mean(axis=1)
            monthly_volatility = monthly_returns.std(axis=1)

            # Annual metrics
            expected_returns = annual_returns.mean(axis=1)
            volatility = annual_returns.std(axis=1)

            # Iterate through monthly horizons (1 to 60 months)
            for horizon, (monthly_return, monthly_vol) in enumerate(
                zip_longest(monthly_expected_returns, monthly_volatility, fillvalue=None), start=1
            ):
                if horizon > 60:
                    break
                
                forecast_date = self.execution_date + pd.DateOffset(months=horizon)

                monthly_metrics.append({
                    "maturity_years": maturity,
                    "execution_date": self.execution_date,
                    "horizon": horizon,
                    "forecast_date": forecast_date,
                    "metric": "Monthly Expected Return",
                    "value": monthly_return
                })
                monthly_metrics.append({
                    "maturity_years": maturity,
                    "execution_date": self.execution_date,
                    "horizon": horizon,
                    "forecast_date": forecast_date,
                    "metric": "volatility",
                    "value": monthly_vol
                })
                # Add Observed Monthly Return
                observed_monthly_return = (
                    observed_returns.iloc[horizon - 1]
                    if (horizon - 1) < len(observed_returns)
                    else None
                )
                monthly_metrics.append({
                    "maturity_years": maturity,
                    "execution_date": self.execution_date,
                    "horizon": horizon,
                    "forecast_date": forecast_date,
                    "metric": "Observed Monthly Return",
                    "value": observed_monthly_return
                })

                # VaR and CVaR for each confidence level
                for cl in confidence_levels:
                    var = monthly_returns.quantile(1 - cl, axis=1)[horizon - 1] if horizon - 1 < len(monthly_returns) else None
                    cvar = monthly_returns.iloc[horizon - 1][monthly_returns.iloc[horizon - 1] <= var].mean() if var is not None else None
                    monthly_metrics.append({
                        "maturity_years": maturity,
                        "execution_date": self.execution_date,
                        "horizon": horizon,
                        "forecast_date": forecast_date,
                        "metric": f"VaR {int(cl*100)}",
                        "value": var
                    })
                    monthly_metrics.append({
                        "maturity_years": maturity,
                        "execution_date": self.execution_date,
                        "horizon": horizon,
                        "forecast_date": forecast_date,
                        "metric": f"CVaR {int(cl*100)}",
                        "value": cvar
                    })

            # Iterate through annual horizons (1 to 5 years)
            for horizon, (expected_return, vol, observed_return) in enumerate(
                zip_longest(expected_returns, volatility, observed_annual_returns, fillvalue=None), start=1
            ):
                if horizon > 5:
                    break

                forecast_date = self.execution_date + pd.DateOffset(years=horizon)

                annual_metrics.append({
                    "maturity_years": maturity,
                    "execution_date": self.execution_date,
                    "horizon": horizon,
                    "forecast_date": forecast_date,
                    "metric": "Expected Annual Returns",
                    "value": expected_return
                })
                annual_metrics.append({
                    "maturity_years": maturity,
                    "execution_date": self.execution_date,
                    "horizon": horizon,
                    "forecast_date": forecast_date,
                    "metric": "Volatility",
                    "value": vol
                })
                annual_metrics.append({
                    "maturity_years": maturity,
                    "execution_date": self.execution_date,
                    "horizon": horizon,
                    "forecast_date": forecast_date,
                    "metric": "Observed Annual Return",
                    "value": observed_return
                })

                # VaR and CVaR for each confidence level
                for cl in confidence_levels:
                    var = annual_returns.quantile(1 - cl, axis=1)[horizon - 1] if horizon - 1 < len(annual_returns) else None
                    cvar = annual_returns.iloc[horizon - 1][annual_returns.iloc[horizon - 1] <= var].mean() if var is not None else None
                    annual_metrics.append({
                        "maturity_years": maturity,
                        "execution_date": self.execution_date,
                        "horizon": horizon,
                        "forecast_date": forecast_date,
                        "metric": f"VaR {int(cl*100)}",
                        "value": var
                    })
                    annual_metrics.append({
                        "maturity_years": maturity,
                        "execution_date": self.execution_date,
                        "horizon": horizon,
                        "forecast_date": forecast_date,
                        "metric": f"CVaR {int(cl*100)}",
                        "value": cvar
                    })

        # Save monthly metrics to a separate file
        monthly_metrics_df = pd.DataFrame(monthly_metrics)
        monthly_file_path = self.returns_dir / "estimated_returns" / f"{self.model_name}" / "monthly" / "risk_metrics.csv"
        lock_file_path = f"{monthly_file_path}.lock"
        lock = FileLock(lock_file_path)
        with lock:
            monthly_metrics_df.to_csv(
                monthly_file_path,
                mode='a',
                header=not monthly_file_path.exists(),
                index=False
            )

        # Save annual metrics to a separate file
        annual_metrics_df = pd.DataFrame(annual_metrics)
        annual_file_path = self.returns_dir / "estimated_returns" / f"{self.model_name}" / "annual" / "risk_metrics.csv"
        lock_file_path = f"{annual_file_path}.lock"
        lock = FileLock(lock_file_path)
        with lock:
            annual_metrics_df.to_csv(
                annual_file_path,
                mode='a',
                header=not annual_file_path.exists(),
                index=False
            )

    def compute_simulated_observed_yields(self):
        """
        Compute simulated observed yields using the Nelson-Siegel model for all simulations.
        """
        def filter_and_pivot_simulated_betas(df_sim, execution_date):
            """
            Filter and pivot the simulated betas DataFrame for the given execution date.
            """
            df_sim = df_sim[df_sim['execution_date'] == execution_date].copy()
            df_sim['forecast_date'] = pd.to_datetime(df_sim['forecast_date'])
            return df_sim.pivot(index='forecast_date', columns='simulation_id', values='simulated_value')
    
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
                names=["maturity", "simulation_id"]
            )
        )
    
        self.simulated_observed_yields_df /= 100

        # --- compute the mean simulated yields ---
        # The result is a DataFrame: index=ForecastDate, columns=Maturity
        self.mean_simulated_yields_df = self.simulated_observed_yields_df.T.groupby(level="maturity").mean().T

    def save_simulated_yields_long_format(self):
        """
        Save simulated yields in a long-format structure to a Parquet file.
        """
        simulations_dir = self.yields_dir / "estimated_yields" / f"{self.model_name}" / "simulations"
        simulations_dir.mkdir(parents=True, exist_ok=True)

        # Reset the index to include ForecastDate as a column
        reshaped_df = self.simulated_observed_yields_df.reset_index()
        reshaped_df.columns = ['forecast_date'] + [
        f"{maturity}_{sim_id}" for maturity, sim_id in reshaped_df.columns[1:]
        ]
        # Melt the DataFrame to long format
        long_format_df = reshaped_df.melt(
        id_vars=["forecast_date"],  # Keep ForecastDate as an identifier
        var_name="maturity_simulationid",  # Combine Maturity and SimulationID
        value_name="simulated_value"  # Name of the values column
        )
        
        # Split the combined "Maturity_SimulationID" into separate columns
        long_format_df[["maturity", "simulation_id"]] = long_format_df["maturity_simulationid"].str.split("_", expand=True)
        
        # Drop the combined column
        long_format_df = long_format_df.drop(columns=["maturity_simulationid"])
        
        # Convert Maturity and SimulationID to appropriate types
        long_format_df["maturity"] = long_format_df["maturity"].astype(float)
        long_format_df["simulation_id"] = long_format_df["simulation_id"].astype(int)

        # Add additional columns
        long_format_df["execution_date"] = self.execution_date  # Add execution date
        long_format_df["model"] = self.model_name  # Add model name
        long_format_df["horizon"] = (long_format_df["forecast_date"] - self.execution_date).dt.days // 30  # Calculate horizon in months
        long_format_df["horizon"] = long_format_df["horizon"].astype(int)  # Ensure it's an integer
        long_format_df["maturity"] = long_format_df["maturity"].astype(float).map(
            lambda x: f"{x} years"
        )  # Convert maturity to a readable string
        
        for maturity in self.yield_curve_model.uniqueTaus:
            maturity_dir = simulations_dir / f"{maturity}_years"
            maturity_dir.mkdir(parents=True, exist_ok=True)

            # Extract simulations for the current maturity
            simulations_for_maturity = long_format_df[long_format_df["maturity"] == f"{maturity} years"]

            # Save each execution date's simulations as a separate Parquet file
            file_path = maturity_dir / f"simulations_{self.execution_date.strftime('%d%m%Y')}.parquet"

            # Save to Parquet
            simulations_for_maturity.to_parquet(file_path, index=False)

    def save_mean_simulated_yields_to_forecasts(self):
        """
        Save mean simulated yields and actual observed yields to forecasts.csv in the same format as predicted yields.
        """
        # Reshape mean simulated yields to long format
        reshaped_df = self.mean_simulated_yields_df.reset_index().melt(
            id_vars='forecast_date',
            var_name='maturity',
            value_name='mean_simulated'
        )

        # Add execution date
        reshaped_df['execution_date'] = self.execution_date

        # Add horizon column from df_pred_beta1 (if available)
        if hasattr(self, 'df_pred_beta1') and 'forecast_date' in self.df_pred_beta1.columns and 'horizon' in self.df_pred_beta1.columns:
            reshaped_df = reshaped_df.merge(
                self.df_pred_beta1[['forecast_date', 'horizon']],
                on='forecast_date',
                how='left'
            )

        # Add actual observed yields
        reshaped_df['actual'] = reshaped_df.apply(
                lambda row: self.observed_yields_df_resampled.at[row['forecast_date'],f"{row['maturity']} years"] 
                if row['forecast_date'] in self.observed_yields_df_resampled.index and 
                f"{row['maturity']} years" in self.observed_yields_df_resampled.columns else np.nan,
                axis=1
            )

        # Save to forecasts.csv (same location as predicted yields)
        model_dir = self.yields_dir / "estimated_yields" / f"{self.model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = model_dir / "forecasts.csv"
        lock_file_path = f"{output_file_path}.lock"

        # Use FileLock for thread safety
        
        lock = FileLock(lock_file_path)
        with lock:
            reshaped_df.to_csv(
                output_file_path,
                mode='a',
                header=not output_file_path.exists(),
                index=False
            )
        
        return reshaped_df


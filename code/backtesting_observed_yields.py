import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time

# Example usage
import os
os.chdir(r'C:\git\backtest-baam\code')
from data_preparation.data_loader import DataLoaderYC
from modeling.yield_curve_modeling import YieldCurveModel
from datetime import datetime
    
def process_simulations_inner(args):
    """
    Generate bootstrapped simulations for a single maturity and execution date.

    Args:
        args (tuple): Contains (model_params, residuals, train_data_last_value, forecast_index, num_simulations_chunk, sim_id_start).

    Returns:
        list: Simulation results for the given maturity and execution date.
    """
    model_params, residuals, train_data_last_value, forecast_index, num_simulations_chunk, sim_id_start, maturity, execution_date = args
    simulations_results = []

    for sim_id in range(sim_id_start, sim_id_start + num_simulations_chunk):
        simulated_forecasts = []
        current_value = train_data_last_value

        for horizon in range(1, len(forecast_index) + 1):
            # Add stochastic component (bootstrapped residuals with replacement)
            error = np.random.choice(residuals, replace=True)
            next_simulated_value = (
                model_params["const"] + model_params["y"] * current_value + error
            )
            simulated_forecasts.append(next_simulated_value)
            current_value = next_simulated_value

        # Store simulation results
        simulations_results.extend([
            {
                "maturity": maturity,
                "execution_date": execution_date,
                "forecasted_date": forecast_index[i],
                "horizon": i + 1,
                "simulation_id": sim_id,
                "simulated_value": simulated_forecasts[i]
            }
            for i in range(len(forecast_index))
        ])

    return simulations_results


def process_forecast_and_simulations_outer(args, num_inner_workers):
    """
    Process a single maturity and execution date for forecasts and simulations.

    Args:
        args (tuple): Contains (maturity, series, execution_date, forecast_horizon, num_simulations).
        num_inner_workers (int): Number of workers for inner parallelization.

    Returns:
        (list, list): Deterministic forecasts and simulations results.
    """
    maturity, series, execution_date, forecast_horizon, num_simulations = args
    forecast_results = []
    simulations_results = []

    train_data = series[:execution_date]

    # Create lagged series for AR(1)
    lagged_series = train_data.shift(1).dropna()
    X = sm.add_constant(lagged_series.rename("lagged"))  # Rename the lagged series to "lagged"
    y = train_data.loc[X.index]

    # Validate that X and y are non-empty
    if X.empty or y.empty:
        print(f"Warning: Insufficient data for maturity {maturity} and execution date {execution_date}. Skipping.")
        return [], []  # Return empty results for this task

    # Fit AR(1) model
    model = sm.OLS(y, X).fit()

    # Extract model parameters for pickling
    model_params = {"const": model.params["const"], "lagged": model.params["lagged"]}

    # Get residuals for bootstrapping
    residuals = model.resid

    # Generate deterministic forecasts
    deterministic_forecasts = []
    last_value = train_data.iloc[-1]
    for _ in range(forecast_horizon):
        next_forecast = model_params["const"] + model_params["lagged"] * last_value
        deterministic_forecasts.append(next_forecast)
        last_value = next_forecast

    # Align deterministic forecasts with forecast dates
    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.DateOffset(months=1),
        periods=forecast_horizon,
        freq="MS"
    )
    actuals = series.loc[forecast_index]
    horizons = np.arange(1, forecast_horizon + 1)

    # Store deterministic forecasts
    forecast_results.extend([
        {
            "maturity": maturity,
            "execution_date": execution_date,
            "forecasted_date": forecast_index[i],
            "horizon": horizons[i],
            "prediction": deterministic_forecasts[i],
            "actual": actuals.iloc[i] if i < len(actuals) else np.nan
        }
        for i in range(len(forecast_index))
    ])

    # Split simulations into chunks for inner parallelization
    num_simulations_chunk = max(1, num_simulations // num_inner_workers)
    tasks = [
        (
            model_params, residuals, train_data.iloc[-1], forecast_index,
            num_simulations_chunk, sim_id * num_simulations_chunk,
            maturity, execution_date
        )
        for sim_id in range(num_inner_workers)
    ]

    # Run inner parallelization for simulations
    with ProcessPoolExecutor(max_workers=num_inner_workers) as inner_executor:
        futures = {inner_executor.submit(process_simulations_inner, task): task for task in tasks}

        for future in as_completed(futures):
            simulations_results.extend(future.result())

    return forecast_results, simulations_results


def run_yield_forecasts_and_simulations_parallel(country, observed_yields_df, forecast_horizon=60, num_simulations=1000, num_outer_workers=8, num_inner_workers=4):
    """
    Parallelized generation of deterministic forecasts and simulations for observed yields.

    Args:
        observed_yields_df (pd.DataFrame): DataFrame of observed yields (columns are maturities, index is date).
        forecast_horizon (int): Number of months to forecast.
        num_simulations (int): Number of simulations to generate for each forecast.
        num_outer_workers (int): Number of parallel workers for outer loop.
        num_inner_workers (int): Number of parallel workers for inner loop.

    Returns:
        forecasts_df (pd.DataFrame): Long format DataFrame with deterministic forecasts.
        simulations_df (pd.DataFrame): Long format DataFrame with bootstrapped simulation results.
    """
    start_time = time.time()  # Start the timer

    tasks = []
    forecast_results = []
    simulations_results = []

    # Define the minimum execution date
    min_execution_date = datetime(1990, 1, 1)
    
    # Prepare tasks for outer parallel processing
    for maturity in observed_yields_df.columns:
        series = observed_yields_df[maturity].dropna()
        for execution_date in series.index[:-forecast_horizon]:
            if execution_date < min_execution_date:
                continue
            tasks.append((maturity, series, execution_date, forecast_horizon, num_simulations))

    # Run outer parallel processing
    with ProcessPoolExecutor(max_workers=num_outer_workers) as outer_executor:
        futures = {outer_executor.submit(process_forecast_and_simulations_outer, task, num_inner_workers): task for task in tasks}

        for future in as_completed(futures):
            forecast_res, simulation_res = future.result()
            forecast_results.extend(forecast_res)
            simulations_results.extend(simulation_res)

    # Convert results to DataFrames
    forecasts_df = pd.DataFrame(forecast_results)
    simulations_df = pd.DataFrame(simulations_results)

    # Save results to Parquet
    forecasts_df.to_parquet(f"C:\git\backtest-baam\data\{country}\{country}_observed_yields_forecasts_AR_1.parquet", index=False)
    simulations_df.to_parquet(f"C:\git\backtest-baam\data\{country}\{country}_observed_yields_simulations_AR_1.parquet", index=False)

    end_time = time.time()  # End the timer
    print(f"Total execution time: {end_time - start_time:.2f} seconds")  # Print the execution time

    return forecasts_df, simulations_df


if __name__ == "__main__":

    country = 'US'
    # Load the yield curve data
    data_loader = DataLoaderYC(r'L:\RMAS\Resources\BAAM\OpenBAAM\Private\Data\BaseDB.mat')
    _, _, _ = data_loader.load_data()
    selectedCurveName, selected_curve_data, modelParams = data_loader.process_data(country)

    # Update model parameters for the yield curve model
    modelParams.update({'minMaturity': 0.08, 'maxMaturity': 10, 'lambda1fixed': 0.7173})
    yield_curve_model = YieldCurveModel(selected_curve_data, modelParams)

    # Extract observed yields and convert dates
    dates_str = yield_curve_model.dates_str
    observed_yields = yield_curve_model.yieldsObservedAgg
    maturities = yield_curve_model.uniqueTaus

    # Create a DataFrame for observed yields
    observed_yields_df = pd.DataFrame(
        observed_yields,
        columns=[f'{tau} years' for tau in maturities],
        index=dates_str[-len(observed_yields):]
    )
    observed_yields_df.index = pd.to_datetime(observed_yields_df.index)
    observed_yields_df_resampled = observed_yields_df.resample('MS').mean()
    #observed_yields_df_resampled /= 100
    observed_yields_df_resampled = observed_yields_df_resampled.iloc[:, 1:]
    # Run forecasts and simulations with a timer
    forecasts_df, simulations_df = run_yield_forecasts_and_simulations_parallel(
        country, 
        observed_yields_df_resampled,
        forecast_horizon=60,
        num_simulations=1000,
        num_outer_workers=8,
        num_inner_workers=4
    )
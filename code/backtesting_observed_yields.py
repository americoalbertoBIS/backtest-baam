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
    
import time
import logging


import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='progress.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_forecast_outer(args):
    """
    Process a single maturity and execution date for deterministic forecasts.

    Args:
        args (tuple): Contains (maturity, series, execution_date, forecast_horizon).

    Returns:
        list: Deterministic forecasts for the given maturity and execution date.
    """
    try:
        maturity, series, execution_date, forecast_horizon = args
        forecast_results = []

        # Ensure at least 3 years (36 months) of historical data is available
        min_data_points = 36
        train_data = series[:execution_date]
        if len(train_data) < min_data_points:
            logging.warning(f"Not enough data for maturity {maturity} and execution date {execution_date}. Skipping.")
            return []  # Skip this task

        # Create lagged series for AR(1)
        lagged_series = train_data.shift(1).dropna()
        X = sm.add_constant(lagged_series.rename("lagged"))  # Rename the lagged series to "lagged"
        y = train_data.loc[X.index]

        # Validate that X and y are non-empty
        if X.empty or y.empty:
            logging.warning(f"Insufficient data for maturity {maturity} and execution date {execution_date}. Skipping.")
            return []  # Return empty results for this task

        # Fit AR(1) model
        model = sm.OLS(y, X).fit()

        # Extract model parameters
        model_params = {"const": model.params["const"], "lagged": model.params["lagged"]}

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

        return forecast_results
    except Exception as e:
        logging.error(f"Error in process_forecast_outer: {e}")
        return []


def run_yield_forecasts_parallel(country, observed_yields_df, forecast_horizon=60, num_outer_workers=4):
    """
    Parallelized generation of deterministic forecasts for observed yields.

    Args:
        observed_yields_df (pd.DataFrame): DataFrame of observed yields (columns are maturities, index is date).
        forecast_horizon (int): Number of months to forecast.
        num_outer_workers (int): Number of parallel workers for outer loop.

    Returns:
        forecasts_df (pd.DataFrame): Long format DataFrame with deterministic forecasts.
    """
    start_time = time.time()  # Start the timer

    tasks = []
    forecast_results = []

    # Minimum data points required (3 years of monthly data)
    min_data_points = 36

    # Prepare tasks for parallel processing
    for maturity in observed_yields_df.columns:
        series = observed_yields_df[maturity].dropna()
        for execution_date in series.index:
            # Ensure at least 3 years of data before the execution date
            train_data = series[:execution_date]
            if len(train_data) < min_data_points:
                continue  # Skip this execution date if insufficient data

            tasks.append((maturity, series, execution_date, forecast_horizon))

    total_tasks = len(tasks)  # Total number of tasks
    logging.info(f"Total tasks to process: {total_tasks}")

    completed_tasks = 0  # Counter for completed tasks

    # Run parallel processing
    with ProcessPoolExecutor(max_workers=num_outer_workers) as executor:
        futures = {executor.submit(process_forecast_outer, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                forecast_results.extend(future.result())

                # Update progress
                completed_tasks += 1
                if completed_tasks % 10 == 0 or completed_tasks == total_tasks:
                    logging.info(f"Completed {completed_tasks}/{total_tasks} tasks.")
            except Exception as e:
                logging.error(f"Error in parallel processing: {e}")

    # Convert results to DataFrame
    forecasts_df = pd.DataFrame(forecast_results)

    # Save results to Parquet
    forecasts_df.to_csv(f"C:\\git\\backtest-baam\\data\\{country}\\{country}_observed_yields_forecasts_AR_1.csv", index=False)

    end_time = time.time()  # End the timer
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    return forecasts_df


if __name__ == "__main__":
    country = 'EA'
    # Load the yield curve data
    data_loader = DataLoaderYC(r'L:\\RMAS\\Resources\\BAAM\\OpenBAAM\\Private\\Data\\BaseDB.mat')
    _, _, _ = data_loader.load_data()
    if country == 'EA':
        selectedCurveName, selected_curve_data, modelParams = data_loader.process_data('DE')
    else:
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
    observed_yields_df_resampled = observed_yields_df_resampled.iloc[:, 1:]  # Drop the first column (e.g., 0.08333 years)

    # Run forecasts with a timer
    forecasts_df = run_yield_forecasts_parallel(
        country, 
        observed_yields_df_resampled,
        forecast_horizon=60,
        num_outer_workers=4
    )
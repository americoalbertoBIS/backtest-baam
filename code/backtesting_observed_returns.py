
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import logging
from datetime import datetime

import os
os.chdir(r'C:\git\backtest-baam\code')
from modeling.time_series_modeling import AR1Model
from data_preparation.data_loader import DataLoaderYC
from modeling.yield_curve_modeling import YieldCurveModel

save_dir = r"C:\git\backtest-baam\data"

def extract_residuals(model, execution_date):
    """
    Extracts residuals from the model and stores them in a long format.

    Args:
        model: Fitted regression model (e.g., statsmodels OLS).
        execution_date (datetime): Execution date.
        train_data (pd.DataFrame): Training data used for fitting the model.

    Returns:
        pd.DataFrame: DataFrame containing residuals in long format.
    """
    residuals = model.resid  # Model residuals (in-sample)

    #residuals_list = []
    residuals_df = pd.DataFrame({
        "execution_date": execution_date,
        "date": residuals.index,  # Index of the training data
        "residual": residuals.values
    }).reset_index(drop=True)

    return residuals_df

def extract_insample_metrics(model, execution_date, model_name, target_col):
    """
    Extracts in-sample metrics (R-squared, Adjusted R-squared, RMSE, etc.) in long format.

    Args:
        model: Fitted regression model (e.g., statsmodels OLS).
        execution_date (datetime): Execution date.
        model_name (str): Name of the model.
        target_col (str): Target column being modeled.
        horizons (list): Forecast horizons.
        model_params (dict): Parameters used for the model.

    Returns:
        list: List of dictionaries containing in-sample metrics in long format.
    """
    metrics = []

    # Add coefficients and p-values
    for param, value in model.params.items():
        metrics.append({
            "execution_date": execution_date,
            "indicator": param,
            "metric": "coefficient",
            "model": model_name,
            "target_col": target_col,
            "value": value
        })
        metrics.append({
            "execution_date": execution_date,
            "indicator": param,
            "metric": "p_value",
            "model": model_name,
            "target_col": target_col,
            "value": model.pvalues[param]
        })

    # Adjusted R-squared, and RMSE
    metrics.append({
        "execution_date": execution_date,
        "indicator": "model",
        "metric": "adjusted_r_squared",
        "model": model_name,
        "target_col": target_col,
        "value": model.rsquared_adj if hasattr(model, "rsquared_adj") else np.nan
    })
    metrics.append({
        "execution_date": execution_date,
        "indicator": "model",
        "metric": "n_obs",
        "model": model_name,
        "target_col": target_col,
        "value": model.nobs
    })

    return metrics

def calculate_out_of_sample_metrics(df_predictions):
    """
    Calculates out-of-sample metrics (RMSE, R-squared) by horizon, execution date, and row (observation).

    Args:
        df_predictions (pd.DataFrame): DataFrame containing predictions, actuals, horizons, and execution dates.

    Returns:
        dict: Metrics including RMSE and R-squared by horizon, execution date, and row (observation).
    """
    # Ensure necessary columns are present
    if not {"horizon", "actual", "prediction", "execution_date", "forecasted_date"}.issubset(df_predictions.columns):
        raise ValueError("df_predictions must contain 'horizon', 'actual', 'prediction', 'execution_date', and 'forecasted_date' columns.")

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
        "by_row": df_predictions[["execution_date", "forecasted_date", "horizon", "rmse_row"]]  # RMSE for each row
    }

from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_generate_simulations(
    model, model_name, latest_obs, horizons, execution_date, num_simulations=1000
):
    """
    Parallelized generation of bootstrapped simulations for the specified horizons.

    Args:
        model: Fitted AR(1) model.
        model_name: Name of the model.
        lagged_beta1: Last observed value (AR(1) lagged term).
        horizons: Forecast horizons (list of integers).
        execution_date: Execution date.
        num_simulations: Number of simulations to generate.

    Returns:
        list: Simulation results for all horizons and simulation IDs.
    """
    logging.info(f"Starting parallel simulations for execution date: {execution_date}, model: {model_name}")

    # Extract residuals and bootstrap errors
    try:
        residuals = model.resid
        bootstrapped_errors = np.random.choice(residuals, size=(num_simulations, max(horizons)), replace=True)
    except Exception as e:
        logging.error(f"Error bootstrapping residuals: {e}")
        raise

    simulations = []

    # Use ThreadPoolExecutor to parallelize simulations
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_single_simulation,
                model, model_name, latest_obs, horizons,
                execution_date, sim_id, bootstrapped_errors[sim_id]
            )
            for sim_id in range(num_simulations)
        ]

        for future in as_completed(futures):
            try:
                simulations.extend(future.result())
            except Exception as e:
                logging.error(f"Error in simulation execution: {e}")

    logging.info(f"Completed parallel simulations for execution date: {execution_date}, model: {model_name}")
    return simulations

def process_single_simulation(
    model, model_name, latest_obs, horizons, execution_date, sim_id, bootstrapped_errors
):
    """
    Process a single simulation for a specific execution date.

    Args:
        model: Fitted AR(1) model.
        model_name: Name of the model.
        lagged_beta1: Last observed value (AR(1) lagged term).
        horizons: Forecast horizons (list of integers).
        execution_date: Execution date.
        sim_id: Simulation ID.
        bootstrapped_errors: Bootstrapped errors for this simulation.

    Returns:
        list: Simulation results for all horizons.
    """
    #logging.info(f"Starting simulation {sim_id} for execution date: {execution_date}, model: {model_name}")
    current_value = latest_obs
    simulation_results = []

    try:
        for horizon in range(1, max(horizons) + 1):
            forecast_date = execution_date + pd.DateOffset(months=horizon)

            # Forecast value based on AR(1) equation: y_t = β0 + β1 * y_t-1 + ε_t
            forecast_value = model.params[0] + model.params[1] * current_value  # Intercept + AR(1) term

            # Add bootstrapped error
            forecast_value += bootstrapped_errors[horizon - 1]

            # Update for the next horizon
            current_value = forecast_value

            # Append simulation result
            simulation_results.append({
                "Model": model_name,
                "ExecutionDate": execution_date,
                "ForecastDate": forecast_date,
                "Horizon": horizon,
                "SimulationID": sim_id,
                "SimulatedValue": forecast_value
            })

    except Exception as e:
        logging.error(f"Error in simulation {sim_id} for execution date {execution_date}: {e}")
        raise

    #logging.info(f"Completed simulation {sim_id} for execution date: {execution_date}")
    return simulation_results
    
def process_forecast_outer(args):
    """
    Process a single maturity and execution date for deterministic forecasts.

    Args:
        args (tuple): Contains (maturity, series, execution_date, forecast_horizon).

    Returns:
        list: Deterministic forecasts for the given maturity and execution date.
    """
    try:
        maturity, series, execution_date, forecast_horizon, obs_model_dir = args
        forecast_results = []

        # Ensure at least 3 years (36 months) of historical data is available
        min_data_points = 36
        train_data = pd.DataFrame(series[:execution_date], columns = [maturity])
        if len(train_data) < min_data_points:
            logging.warning(f"Not enough data for maturity {maturity} and execution date {execution_date}. Skipping.")
            return []  # Skip this task

        # fit AR1 model
        ar1_model = AR1Model()
        fitted_model = ar1_model.fit(train_data, target_col = maturity)
        # Generate iterative forecasts
        yld_forecast = ar1_model.forecast(
                model=fitted_model,
                steps=forecast_horizon,  # Forecast for the test data horizon
                train_data=train_data,
                target_col=maturity
            )
        
        latest_observation = train_data[maturity].iloc[-1]  # Last observed value (starting point for simulations)
        simulations = parallel_generate_simulations(
                model=fitted_model,
                model_name="AR(1)",
                latest_obs=latest_observation,
                horizons=np.arange(1, forecast_horizon + 1),
                execution_date=execution_date,
                num_simulations=1000  # Example: 1000 simulations
            )
    
        maturity_dir = os.path.join(obs_model_dir, "simulations", f"{maturity.replace(' ', '_').replace('years', 'maturity')}")
        os.makedirs(maturity_dir, exist_ok=True)
        simulations_file = os.path.join(maturity_dir, f"simulations_{execution_date.strftime('%d%m%Y')}.parquet")
        simulations_df = pd.DataFrame(simulations)
        simulations_df['maturity'] = maturity
        simulations_df.to_parquet(simulations_file, index=False)
        #logging.info(f"Simulations saved for maturity {maturity}, execution date {execution_date} to {simulations_file}")
        
        # Extract residuals (in-sample, based on train_data)
        residuals = extract_residuals(fitted_model, execution_date)
        residuals['maturity'] = maturity
        
        # Extract model metrics
        insample_metrics = extract_insample_metrics(
                model=fitted_model,
                execution_date=execution_date,
                model_name=f'AR_1',
                target_col=maturity
            )

        # Store deterministic forecasts
        forecast_index = pd.date_range(
                start=train_data.index[-1] + pd.DateOffset(months=1),
                periods=60,
                freq="MS"
            )
        
        actuals = series.loc[execution_date+ pd.DateOffset(months=1):]
        horizons = np.arange(1, forecast_horizon + 1)

        forecast_results.extend([
            {
                "maturity": maturity,
                "execution_date": execution_date,
                "forecasted_date": forecast_index[i],
                "horizon": horizons[i],
                "prediction": yld_forecast[i],
                "actual": actuals.iloc[i] if i < len(actuals) else np.nan
            }
            for i in range(len(forecast_index))
        ])

        return forecast_results, residuals, insample_metrics
    except Exception as e:
        logging.error(f"Error in process_forecast_outer: {e}")
        return []


def run_forecasts_parallel(observed_df, obs_model_dir, forecast_horizon=60, num_outer_workers=4):
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
    insample_residuals = []
    insample_metrics = []
    
    # Minimum data points required (3 years of monthly data)
    min_data_points = 36

    # Prepare tasks for parallel processing
    for maturity in observed_df.columns:
        series = observed_df[maturity].dropna()
        for execution_date in series.index:
            # Ensure at least 3 years of data before the execution date
            train_data = series[:execution_date]
            if len(train_data) < min_data_points:
                continue  # Skip this execution date if insufficient data

            tasks.append((maturity, series, execution_date, forecast_horizon, obs_model_dir))

    total_tasks = len(tasks)  # Total number of tasks
    logging.info(f"Total tasks to process: {total_tasks}")

    completed_tasks = 0  # Counter for completed tasks

    # Run parallel processing
    with ProcessPoolExecutor(max_workers=num_outer_workers) as executor:
        futures = {executor.submit(process_forecast_outer, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                forecasts, residuals, metrics = future.result()
                #forecast_results.extend(future.result())
                forecast_results.extend(forecasts)
                insample_residuals.append(residuals)
                insample_metrics.extend(metrics)
                # Update progress
                completed_tasks += 1
                if completed_tasks % 10 == 0 or completed_tasks == total_tasks:
                    logging.info(f"Completed {completed_tasks}/{total_tasks} tasks.")
            except Exception as e:
                logging.error(f"Error in parallel processing: {e}")

    # Convert results to DataFrame
    #forecasts_df = pd.DataFrame(forecast_results)
    # Save results to Parquet
    end_time = time.time()  # End the timer
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    return pd.DataFrame(forecast_results), pd.concat(insample_residuals), pd.DataFrame(insample_metrics)
    #return forecasts_df


if __name__ == "__main__":
    country = 'US' # US EA UK
    
    # Configure logging
    logging.basicConfig(
        filename=rf'C:\git\backtest-baam\logs\{country}_observed_yields_AR_1.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    obs_dir = fr"{save_dir}\{country}\returns\observed_returns"
    os.makedirs(obs_dir, exist_ok=True)

    obs_model_dir = fr"{obs_dir}\AR_1"
    os.makedirs(obs_model_dir, exist_ok=True)

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
    #observed_yields_df_resampled = observed_yields_df_resampled.iloc[:, 1:]  # Drop the first column (e.g., 0.08333 years)
    observed_returns_df = observed_yields_df_resampled.pct_change(fill_method=None).dropna(how='all')
    
    # Run forecasts with a timer
    df_predictions, df_insample_residuals, df_insample_metrics = run_forecasts_parallel(
        country, 
        observed_returns_df,
        obs_model_dir,
        forecast_horizon=60,
        num_outer_workers=4  
    )
    
    if df_predictions is not None:
            predictions_file = os.path.join(obs_model_dir, f"forecasts.csv")
            df_predictions.to_csv(predictions_file, index=False)
            
            # Save residuals
            residuals_file = os.path.join(obs_model_dir, f"residuals.csv")
            df_insample_residuals.to_csv(residuals_file, index=False)

            # Save in sample metrics
            insample_metrics_file = os.path.join(obs_model_dir, f"insample_metrics.csv")
            df_insample_metrics.to_csv(insample_metrics_file, index=False)            
            
            outofsample_metrics_by_horizon = []
            outofsample_metrics_by_exec_date = []
            outofsample_metrics = []
            for maturity in df_predictions['maturity'].unique():
                temp = df_predictions[df_predictions['maturity']==maturity].copy()
                outofsample_metrics_temp = calculate_out_of_sample_metrics(temp)
                outofsample_metrics_temp["by_horizon"]['maturity'] = maturity
                outofsample_metrics_temp["by_execution_date"]['maturity'] = maturity
                outofsample_metrics_temp["by_row"]['maturity'] = maturity
                outofsample_metrics_by_horizon.append(outofsample_metrics_temp["by_horizon"])
                outofsample_metrics_by_exec_date.append(outofsample_metrics_temp["by_execution_date"])
                outofsample_metrics.append(outofsample_metrics_temp["by_row"])

            logging.info("Save results")
            # Save out of sample metrics metrics to CSV
            metrics_by_horizon_file = os.path.join(obs_model_dir, f"outofsample_metrics_by_horizon.csv")
            pd.concat(outofsample_metrics_by_horizon).to_csv(metrics_by_horizon_file, index=False)

            metrics_by_execution_date_file = os.path.join(obs_model_dir, f"outofsample_metrics_by_execution_date.csv")
            pd.concat(outofsample_metrics_by_exec_date).to_csv(metrics_by_execution_date_file, index=False)

            metrics_by_row_file = os.path.join(obs_model_dir, f"outofsample_metrics_by_row.csv")
            pd.concat(outofsample_metrics).to_csv(metrics_by_row_file, index=False)

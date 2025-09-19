
from sklearn.metrics import r2_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import logging
from datetime import datetime
import pathlib

import os
os.chdir(r'C:\git\backtest-baam\code')
from modeling.time_series_modeling import AR1Model
from data_preparation.data_loader import DataLoaderYC
from modeling.yield_curve_modeling import YieldCurveModel

save_dir = r'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data_joint'

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
    if not {"horizon", "actual", "prediction", "execution_date", "forecast_date"}.issubset(df_predictions.columns):
        raise ValueError("df_predictions must contain 'horizon', 'actual', 'prediction', 'execution_date', and 'forecast_date' columns.")

    # Drop rows with NaN in Actual or Prediction
    df_predictions = df_predictions.copy()
    df_predictions = df_predictions.dropna(subset=["actual", "prediction"])
    df_predictions = df_predictions.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual", "prediction"])
    # Calculate residuals and squared errors
    df_predictions["residual"] = df_predictions["actual"] - df_predictions["prediction"]
    df_predictions["squared_error"] = df_predictions["residual"] ** 2
    df_predictions["rmse_row"] = np.sqrt(df_predictions["squared_error"])  # RMSE for each row

    # Metrics by horizon
    metrics_by_horizon = (
        df_predictions.groupby("horizon", group_keys=False)
        .apply(lambda group: pd.Series({
            "mse": group["squared_error"].mean(),
            "r_squared": r2_score(group["actual"], group["prediction"]),
            "rmse": np.sqrt(group["squared_error"].mean())
        }), include_groups=False).reset_index()
    )

    # RMSE by execution date
    metrics_by_execution_date = (
        df_predictions.groupby("execution_date", group_keys=False)
        .apply(lambda group: pd.Series({
            "mse": group["squared_error"].mean(),
            "r_squared": r2_score(group["actual"], group["prediction"]),
            "rmse": np.sqrt(group["squared_error"].mean())
        }), include_groups=False)
        .reset_index()
    )

    return {
        "by_horizon": metrics_by_horizon,
        "by_execution_date": metrics_by_execution_date,
        "by_row": df_predictions[["execution_date", "forecast_date", "horizon", "rmse_row"]]  # RMSE for each row
    }

def calculate_and_save_metrics(df, output_dir):
    outofsample_metrics_by_horizon = []
    outofsample_metrics_by_exec_date = []
    outofsample_metrics = []
    for maturity in df['maturity'].unique():
        temp = df[df['maturity'] == maturity].copy()
        metrics = calculate_out_of_sample_metrics(temp)
        metrics["by_horizon"]['maturity'] = maturity
        metrics["by_execution_date"]['maturity'] = maturity
        metrics["by_row"]['maturity'] = maturity
        outofsample_metrics_by_horizon.append(metrics["by_horizon"])
        outofsample_metrics_by_exec_date.append(metrics["by_execution_date"])
        outofsample_metrics.append(metrics["by_row"])

    # Save metrics
    pd.concat(outofsample_metrics_by_horizon).to_csv(output_dir / "outofsample_metrics_by_horizon.csv", index=False)
    pd.concat(outofsample_metrics_by_exec_date).to_csv(output_dir / "outofsample_metrics_by_execution_date.csv", index=False)
    pd.concat(outofsample_metrics).to_csv(output_dir / "outofsample_metrics_by_row.csv", index=False)
    
def calculate_risk_metrics_long(sim_df, value_col, observed_col=None, expected_col=None, quantiles=[0.95, 0.975, 0.99]):
    """
    Calculate VaR, CVaR, volatility, and add observed/expected returns in long format.
    sim_df: DataFrame with columns ['execution_date', 'maturity', 'horizon', value_col, ...]
    observed_col: column name for observed returns (optional)
    expected_col: column name for expected returns (optional)
    Returns: long-format DataFrame with columns:
        ['execution_date', 'maturity', 'horizon', 'metric', 'value']
    """
    metrics = []
    grouped = sim_df.groupby(["execution_date", "maturity", "horizon"])
    for (execution_date, maturity, horizon), group in grouped:
        values = group[value_col].dropna().values
        if len(values) == 0:
            continue
        volatility = np.std(values)
        metrics.append({
            "execution_date": execution_date,
            "maturity": maturity,
            "horizon": horizon,
            "metric": "volatility",
            "value": volatility
        })
        for q in quantiles:
            var = np.quantile(values, 1 - q)
            cvar = values[values <= var].mean() if np.any(values <= var) else np.nan
            metrics.append({
                "execution_date": execution_date,
                "maturity": maturity,
                "horizon": horizon,
                "metric": f"VaR {int(q*100)}",
                "value": var
            })
            metrics.append({
                "execution_date": execution_date,
                "maturity": maturity,
                "horizon": horizon,
                "metric": f"CVaR {int(q*100)}",
                "value": cvar
            })
        # Observed and expected returns (if provided)
        if observed_col and observed_col in group.columns:
            obs_val = group[observed_col].mean()
            metrics.append({
                "execution_date": execution_date,
                "maturity": maturity,
                "horizon": horizon,
                "metric": f"Observed {value_col.replace('_', ' ').title()}",
                "value": obs_val
            })
        if expected_col and expected_col in group.columns:
            exp_val = group[expected_col].mean()
            metrics.append({
                "execution_date": execution_date,
                "maturity": maturity,
                "horizon": horizon,
                "metric": f"Expected {value_col.replace('_', ' ').title()}",
                "value": exp_val
            })
    return pd.DataFrame(metrics)

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
            forecast_value = model.params.iloc[0] + model.params.iloc[1] * current_value  # Intercept + AR(1) term

            # Add bootstrapped error
            forecast_value += bootstrapped_errors[horizon - 1]

            # Update for the next horizon
            current_value = forecast_value

            # Append simulation result
            simulation_results.append({
                "model": model_name,
                "execution_date": execution_date,
                "forecast_date": forecast_date,
                "horizon": horizon,
                "simulation_id": sim_id,
                "simulated_value": forecast_value
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
        forecast_results_annual = []
        
        # Ensure at least 3 years (36 months) of historical data is available
        min_data_points = 36
        train_data = pd.DataFrame(series[:execution_date], columns = [maturity])
        train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()
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
        
        # create simulations
        latest_observation = train_data[maturity].iloc[-1]  # Last observed value (starting point for simulations)
        simulations = parallel_generate_simulations(
                model=fitted_model,
                model_name="AR(1)",
                latest_obs=latest_observation,
                horizons=np.arange(1, forecast_horizon + 1),
                execution_date=execution_date,
                num_simulations=10  # Example: 1000 simulations
            )
        
        MONTHS_IN_YEAR = 12

        # --- Save monthly and annual returns in long format ---
        monthly_dir = obs_model_dir / "monthly" / "simulations" / f"{maturity.split()[0]}_years"
        annual_dir = obs_model_dir / "annual"  / "simulations" / f"{maturity.split()[0]}_years"
        monthly_dir.mkdir(parents=True, exist_ok=True)
        annual_dir.mkdir(parents=True, exist_ok=True)
                
        # transform simulations dict into dataframe
        simulations_df = pd.DataFrame(simulations)
        simulations_df['maturity'] = maturity

        # Pivot so each column is a simulation, each row is a forecast date
        monthly_returns = simulations_df.pivot(index="forecast_date", columns="simulation_id", values="simulated_value")
        monthly_returns.index.name = "forecast_date"

        # Save Monthly Returns in long format
        monthly_returns_long_format = monthly_returns.reset_index().melt(
            id_vars=["forecast_date"],
            var_name="simulation_id",
            value_name="monthly_returns"
        )
        monthly_returns_long_format["maturity"] = maturity
        monthly_returns_long_format["execution_date"] = execution_date
        monthly_returns_long_format["horizon"] = (
            (monthly_returns_long_format["forecast_date"].dt.to_period('M') -
            monthly_returns_long_format["execution_date"].dt.to_period('M'))
        ).apply(lambda x: x.n)
        monthly_file_path = monthly_dir / f"simulations_{execution_date.strftime('%d%m%Y')}.parquet"
        monthly_returns_long_format.to_parquet(monthly_file_path, index=False)

        # Calculate annual returns (arithmetic sum, not compounded)
        annual_returns = monthly_returns.groupby(np.arange(len(monthly_returns)) // MONTHS_IN_YEAR).sum()
        annual_returns.index.name = "index"

        # Save Annual Returns in long format
        annual_returns_long_format = annual_returns.reset_index().melt(
            id_vars=["index"],
            var_name="simulation_id",
            value_name="annual_returns"
        )
        annual_returns_long_format = annual_returns_long_format.rename(columns={"index": "horizon_years"})
        annual_returns_long_format["horizon_years"] = annual_returns_long_format["horizon_years"] + 1
        annual_returns_long_format["maturity"] = maturity
        annual_returns_long_format["execution_date"] = execution_date
        annual_returns_long_format["forecast_date"] = [
                pd.to_datetime(execution_date) + pd.DateOffset(years=int(h))
                for h in annual_returns_long_format["horizon_years"]
            ]
        annual_file_path = annual_dir / f"simulations_{execution_date.strftime('%d%m%Y')}.parquet"
        annual_returns_long_format.to_parquet(annual_file_path, index=False)

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
                "forecast_date": forecast_index[i],
                "horizon": horizons[i],
                "prediction": yld_forecast[i],
                "actual": actuals.iloc[i] if i < len(actuals) else np.nan
            }
            for i in range(len(forecast_index))
        ])
        
        forecast_df = pd.DataFrame(forecast_results).sort_values("forecast_date")
        monthly_returns_long_format = monthly_returns_long_format.merge(
            forecast_df[["forecast_date", "maturity", "execution_date", "horizon", "actual", "prediction"]],
            on=["forecast_date", "maturity", "execution_date", "horizon"],
            how="left"
        )
        monthly_metrics_long = calculate_risk_metrics_long(
            monthly_returns_long_format,
            value_col="monthly_returns",
            observed_col="actual",
            expected_col="prediction",
            quantiles=[0.95, 0.975, 0.99]
        )
        
        # Calculate annual returns (sequential blocks, not rolling)
        annual_returns_df = (
            forecast_df[['maturity', 'execution_date', 'forecast_date', 'prediction', 'actual']]
            .assign(block=lambda x: np.arange(len(x)) // MONTHS_IN_YEAR)
            .groupby(['maturity', 'execution_date', 'block'])
            .agg({
                'prediction': 'sum',
                'actual': 'sum',
                'forecast_date': 'first'  # Start date of each annual block
            })
            .reset_index()
            .rename(columns={'block': 'horizon_years'})
        )

        # Adjust horizon_years to start from 1
        annual_returns_df['horizon_years'] = annual_returns_df['horizon_years'] + 1
        annual_returns_df["forecast_date"] = [
                pd.to_datetime(execution_date) + pd.DateOffset(years=int(h))
                for h in annual_returns_df["horizon_years"]
            ]
        annual_returns_df = annual_returns_df.rename(columns={
                    "horizon_years": "horizon",
                    "annual_return": "prediction",
                    "annual_actual": "actual"
                })

        annual_returns_long_format = annual_returns_long_format.rename(columns={"horizon_years": "horizon"})
                
        annual_returns_long_format = annual_returns_long_format.merge(
            annual_returns_df[["forecast_date", "maturity", "execution_date", "horizon", "actual", "prediction"]],
            left_on=["forecast_date", "maturity", "execution_date", "horizon"],
            right_on=["forecast_date", "maturity", "execution_date", "horizon"],
            how="left"
        )
        
        annual_metrics_long = calculate_risk_metrics_long(
            annual_returns_long_format,
            value_col="annual_returns",
            observed_col="actual",
            expected_col="prediction",
            quantiles=[0.95, 0.975, 0.99]
        )
        
        annual_returns_dict = annual_returns_df.to_dict('records')
        forecast_results_annual.extend(annual_returns_dict)
        
        return forecast_results_annual, forecast_results, residuals, insample_metrics, monthly_metrics_long, annual_metrics_long
    
    except Exception as e:
        logging.error(f"Error in process_forecast_outer: {e}")
        return []


def run_forecasts_parallel(observed_df, obs_model_dir, forecast_horizon=60, num_outer_workers=4, subset_execution_dates=None):
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
    forecast_results_annual = []
    insample_residuals = []
    insample_metrics = []
    risk_metrics_monthly = []
    risk_metrics_annual = []
    
    # Minimum data points required (3 years of monthly data)
    min_data_points = 36

    # Prepare tasks for parallel processing
    for maturity in observed_df.columns:
        series = observed_df[maturity].dropna()
        execution_dates = subset_execution_dates if subset_execution_dates is not None else series.index
        for execution_date in execution_dates:
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
                (forecast_annual, forecasts, residuals, 
                 metrics, monthly_metrics_long, annual_metrics_long) = future.result()
                #forecast_results.extend(future.result())
                forecast_results_annual.extend(forecast_annual)
                forecast_results.extend(forecasts)
                insample_metrics.extend(metrics)
                insample_residuals.append(residuals)
                risk_metrics_monthly.append(monthly_metrics_long)
                risk_metrics_annual.append(annual_metrics_long)
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

    return pd.DataFrame(forecast_results_annual), pd.DataFrame(forecast_results), pd.concat(insample_residuals), pd.DataFrame(insample_metrics), pd.concat(risk_metrics_monthly), pd.concat(risk_metrics_annual)
    #return forecasts_df


if __name__ == "__main__":
    countries = ['US', 'EA', 'UK']
    
    for country in countries:
        print(country)
        # Configure logging
        logging.basicConfig(
            filename=rf'C:\git\backtest-baam\logs\{country}_observed_returns_AR_1.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Set base folder
        base_dir = pathlib.Path(save_dir) / country / "returns" / "observed_returns" / "AR_1"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Subfolders
        monthly_dir = base_dir / "monthly"
        annual_dir = base_dir / "annual"
        monthly_dir.mkdir(parents=True, exist_ok=True)
        annual_dir.mkdir(parents=True, exist_ok=True)

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
        
        # Example: Only use the first 5 dates for each maturity
        subset_execution_dates = {}
        for maturity in observed_returns_df.columns:
            custom_dates = pd.date_range(start="2002-01-01", end="2002-05-01", freq="MS")
        
        # Run forecasts with a timer
        (df_predictions_annual, df_predictions, df_insample_residuals, 
        df_insample_metrics, df_monthly_metrics_long, df_annual_metrics_long) = run_forecasts_parallel(
            observed_returns_df,
            base_dir,
            forecast_horizon=60,
            num_outer_workers=4, 
            subset_execution_dates=None
        )
        
        if df_predictions is not None:
                logging.info("Save results")
                # save average forecasts
                predictions_file = os.path.join(monthly_dir, f"forecasts.csv")
                df_predictions.to_csv(predictions_file, index=False)
                # save risk metrics
                risk_metrics_file_monthly = os.path.join(monthly_dir, f"risk_metrics.csv")
                df_monthly_metrics_long.to_csv(risk_metrics_file_monthly, index=False)
                # Save in sample metrics
                insample_metrics_file = os.path.join(monthly_dir, f"insample_metrics.csv")
                df_insample_metrics.to_csv(insample_metrics_file, index=False)  
                # save average forecast
                predictions_annual_file = os.path.join(annual_dir, f"forecasts.csv")
                df_predictions_annual.to_csv(predictions_annual_file, index=False)  
                # save risk metrics
                risk_metrics_file_annual = os.path.join(annual_dir, f"risk_metrics.csv")
                df_annual_metrics_long.to_csv(risk_metrics_file_annual, index=False)
                # Save residuals
                residuals_file = os.path.join(monthly_dir, f"residuals.csv")
                df_insample_residuals.to_csv(residuals_file, index=False)
                
                # save out of sample metrics
                calculate_and_save_metrics(df_predictions, monthly_dir)
                # save out of sample metrics
                calculate_and_save_metrics(df_predictions_annual, annual_dir)

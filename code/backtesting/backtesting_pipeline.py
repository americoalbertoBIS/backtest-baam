import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import logging
import os
import re
from sklearn.metrics import r2_score

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

os.chdir(r'C:\git\backtest-baam\code')

from backtesting.backtesting_logging import setup_mlflow, check_existing_results, log_backtest_results, clean_model_name
from data_preparation.conensus_forecast import ConsensusForecast
from data_preparation.data_transformations import convert_mom_to_yoy
from modeling.macro_modeling import output_gap, inflation_expectations
from modeling.time_series_modeling import fit_arx_model

from backtesting.config_models import models
from config_paths import QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH

def prepare_train_test_data(
    country, df, execution_date, consensus_df_gdp, consensus_df_inf,
    exogenous_variables, output_gap_method, inflation_method, horizons, macro_forecast
):
    """
    Prepares training and testing datasets for backtesting.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data.
        execution_date (datetime): Execution date for the backtest.
        consensus_df_gdp (pd.DataFrame): Consensus GDP forecast.
        consensus_df_inf (pd.DataFrame): Consensus inflation forecast.
        exogenous_variables (list): List of exogenous variable column names.
        output_gap_method (str): Method for calculating the output gap ("direct" or "HP").
        inflation_method (str): Method for calculating inflation expectations ("default" or "ucsv").
        horizons (list): Forecast horizons.
        macro_forecast (str): Macro forecast method ("consensus" or "ar_1").

    Returns:
        tuple: Training and testing datasets.
    """
    # Step 1: Split the data into train and test sets
    train_data = df.loc[:execution_date].copy()
    future_dates = pd.date_range(
        start=train_data.index[-1] + pd.DateOffset(months=1),
        periods=max(horizons),
        freq="MS",
    )
    extended_test_data = pd.DataFrame(index=future_dates)
    test_data = pd.concat([df.loc[execution_date + pd.DateOffset(months=1):], extended_test_data], axis = 1)
    combined_data = pd.concat([train_data, extended_test_data], axis=0)

    # Step 2: Add Output Gap
    if "output_gap" in exogenous_variables:
        #print("Calculate output gap")
        # Train set: Calculate output gap
        output_gap_train, output_gap_test = output_gap(
            country=country,
            data=combined_data,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method,
            macro_forecast=macro_forecast
        )

        # Add output gap to train and test datasets
        train_data["output_gap"] = output_gap_train
        test_data["output_gap"] = output_gap_test

    # Step 3: Add Inflation Expectations
    if "inflation" in exogenous_variables:
        # Train set: Calculate inflation expectations
        train_inflation, test_inflation = inflation_expectations(
            country=country,
            data=combined_data[f"{country}_CPI"],
            consensus_df=consensus_df_inf,  
            execution_date=execution_date,
            method=inflation_method,
            macro_forecast=macro_forecast
        )
        # Add inflation expectations to train and test datasets
        train_data["inflation"] = train_inflation
        test_data["inflation"] = test_inflation

    # Step 4: Add GDP YoY
    if "gdp_yoy" in exogenous_variables:
        # Train set: Calculate GDP YoY
        _, gdp = output_gap(
            country=country,
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )

        # Convert MoM to YoY for GDP
        gdp_train_yoy = convert_mom_to_yoy(gdp.values, col_name="gdp_yoy") * 100
        gdp_train_yoy.index = gdp.index
        gdp_test_yoy = convert_mom_to_yoy(gdp.values, col_name="gdp_yoy") * 100
        gdp_test_yoy.index = gdp.index

        # Add GDP YoY to train and test datasets
        train_data["gdp_yoy"] = gdp_train_yoy
        test_data["gdp_yoy"] = gdp_test_yoy

    return train_data, test_data

def forecast_values(model, test_data, lagged_beta1, horizons, exogenous_variables, execution_date, df, target_col):
    """
    Generates forecasts for the specified horizons.

    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted model.
        model_name (str): Name of the model.
        test_data (pd.DataFrame): Testing data.
        lagged_beta1 (float): Lagged beta value.
        horizons (list): Forecast horizons.
        exogenous_variables (list or set): List or set of exogenous variable column names.
        execution_date (datetime): Execution date for the backtest.
        df (pd.DataFrame): Input data.
        target_col (str): Target column.

    Returns:
        list: Forecast results.
    """
    results = []
    current_beta = lagged_beta1
    
    for horizon in range(1, max(horizons) + 1):
        forecast_date = execution_date + pd.DateOffset(months=horizon)

        # Initialize forecast value with the intercept and AR(1) term
        forecast_value = model.params[0] + model.params[1] * current_beta  # Intercept and AR(1) term

        # Add contributions from exogenous variables
        for variable in exogenous_variables:
            if variable in model.params.index:
                exogenous_value = test_data.loc[forecast_date, variable] if forecast_date in test_data.index else np.nan
                forecast_value += model.params[variable] * exogenous_value
                
        current_beta = forecast_value

        actual_value = df.loc[forecast_date, target_col] if forecast_date in df.index else np.nan
        
        results.append({
            "execution_date": execution_date,
            "forecast_date": forecast_date,
            "horizon": horizon,
            "prediction": forecast_value,
            "actual": actual_value,
        })

    return results

def generate_bootstrap_indices(
    available_dates, num_simulations, max_horizon, 
    bootstrap_type="iid", block_length=6, half_life=12
):
    """
    Generate bootstrapped indices for different bootstrapping types.
    """
    indices = []
    n = len(available_dates)
    if bootstrap_type == "iid":
        for _ in range(num_simulations):
            indices.append(np.random.choice(available_dates, size=max_horizon, replace=True))
    elif bootstrap_type == "block":
        for _ in range(num_simulations):
            sim_indices = []
            i = 0
            while i < max_horizon:
                start = np.random.randint(0, n - block_length + 1)
                block = available_dates[start:start+block_length]
                sim_indices.extend(block)
                i += block_length
            indices.append(sim_indices[:max_horizon])
    elif bootstrap_type == "half_life":
        weights = np.exp(-np.log(2) * np.arange(n)[::-1] / half_life)
        weights /= weights.sum()
        for _ in range(num_simulations):
            indices.append(np.random.choice(available_dates, size=max_horizon, replace=True, p=weights))
    else:
        raise ValueError("Unknown bootstrap_type")
    return np.array(indices)

def generate_and_save_bootstrap_indices(
    df, execution_dates, num_simulations, max_horizon, save_path,
    bootstrap_type="iid", block_length=6, half_life=12, save_csv = True
):
    """
    For each execution date, generate a long-format CSV of bootstrapped indices.
    Columns: ExecutionDate, SimulationID, Horizon, BootDate
    """
    records = []
    for exec_date in execution_dates:
        available_dates = df.loc[:exec_date].index
        boot_indices = generate_bootstrap_indices(
            available_dates, num_simulations, max_horizon, 
            bootstrap_type=bootstrap_type, block_length=block_length, half_life=half_life
        )
        for sim in range(num_simulations):
            for h, boot_date in enumerate(boot_indices[sim], 1):
                records.append({
                    "execution_date": exec_date,
                    "simulation_id": sim,
                    "horizon": h,
                    "bootstrap_residual_date": boot_date
                })
    df_boot = pd.DataFrame(records)
    if save_csv:
        df_boot.to_csv(save_path, index=False)
    return df_boot

def parallel_generate_simulations(
    model, model_name, test_data, lagged_beta1, horizons, 
    exogenous_variables, execution_date, df, target_col, 
    bootstrap_dates,
    num_simulations=1000
):
    """
    Parallelized generation of bootstrapped simulations for the specified horizons.

    Args:
        model: Fitted model.
        model_name: Name of the model.
        test_data: Testing data.
        lagged_beta1: Last observed beta value.
        horizons: Forecast horizons.
        output_gap_col, inflation_col, gdp_col: Exogenous variable columns.
        execution_date: Execution date.
        df: Input data.
        target_col: Target column.
        num_simulations: Number of simulations to generate.

    Returns:
        list: Simulation results for all horizons and simulation IDs.
    """
    #residuals = model.resid
    #bootstrapped_errors = np.random.choice(residuals, size=(num_simulations, max(horizons)), replace=True)
    bootstrap_dates_exec = bootstrap_dates[bootstrap_dates["ExecutionDate"] == execution_date].copy()
    
    simulations = []

    # Use ThreadPoolExecutor to parallelize simulations
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_single_simulation,
                model, model_name, test_data, lagged_beta1, horizons,
                exogenous_variables, execution_date, sim_id, bootstrap_dates_exec
            )
            for sim_id in range(num_simulations)
        ]

        for future in as_completed(futures):
            simulations.extend(future.result())

    return simulations

def process_single_simulation(
    model, model_name, test_data, lagged_beta1, horizons, 
    exogenous_variables, execution_date, sim_id, bootstrap_dates_exec
):
    """
    Process a single simulation for a specific execution date.

    Args:
        model: Fitted model.
        model_name: Name of the model.
        test_data: Testing data.
        lagged_beta1: Last observed beta value.
        horizons: Forecast horizons.
        output_gap_col, inflation_col, gdp_col: Exogenous variable columns.
        execution_date: Execution date.
        df: Input data.
        target_col: Target column.
        sim_id: Simulation ID.
        bootstrapped_errors: Bootstrapped errors for this simulation.

    Returns:
        list: Simulation results for all horizons.
    """
    residuals = model.resid
    current_beta = lagged_beta1
    simulation_results = []
    
    sim_boot_dates = bootstrap_dates_exec[bootstrap_dates_exec["SimulationID"] == sim_id].sort_values("Horizon")["BootDate"]
    bootstrapped_errors = residuals.reindex(sim_boot_dates.values).values

    for horizon in range(1, max(horizons) + 1):
        forecast_date = execution_date + pd.DateOffset(months=horizon)

        # Initialize forecast value with the intercept and AR(1) term
        forecast_value = model.params[0] + model.params[1] * current_beta  # Intercept and AR(1) term

        # Add contributions from exogenous variables
        for variable in exogenous_variables:
            if variable in model.params.index:
                exogenous_value = test_data.loc[forecast_date, variable] if forecast_date in test_data.index else np.nan
                forecast_value += model.params[variable] * exogenous_value

        # Add bootstrapped error
        forecast_value += bootstrapped_errors[horizon - 1]

        # Update for the next horizon
        current_beta = forecast_value

        # Append simulation result
        simulation_results.append({
            "model": model_name,
            "execution_date": execution_date,
            "forecast_date": forecast_date,
            "horizon": horizon,
            "simulation_id": sim_id,
            "simulated_value": forecast_value,
            "bootstrap_residual_date": pd.Timestamp(sim_boot_dates.values[horizon - 1]).strftime("%Y-%m-%d"),
            "bootstrap_residual": bootstrapped_errors[horizon - 1]
        })

    return simulation_results

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

def process_execution_date_parallel(
    execution_date, df, target_col, horizons, model_name, model_handler, model_params,
    bootstrap_dates,
    df_consensus_gdp, df_consensus_inf, num_simulations, country
):
    """
    Process a single execution date with parallelized simulations.

    Args:
        execution_date: The execution date.
        df: Input data.
        target_col: Target column.
        horizons: Forecast horizons.
        model_name: Name of the model.
        model_handler: Model handler object.
        model_params: Model parameters.
        df_consensus_gdp, df_consensus_inf: Consensus forecasts.
        num_simulations: Number of simulations to generate.
        country: Country name or code.

    Returns:
        tuple: Deterministic results (list) and simulations (list of dicts).
    """
    logging.info(f"Process execution date: {str(execution_date)}")
    # Provide a default empty list for exogenous variables
    exogenous_variables = model_params.get("exogenous_variables", [])

    # Prepare training and testing datasets
    logging.info("Prepare train and test data")
    #print("Prepare train and test data")
    train_data, test_data = prepare_train_test_data(
        country=country,
        df=df,
        execution_date=execution_date,
        consensus_df_gdp=df_consensus_gdp,
        consensus_df_inf=df_consensus_inf,
        exogenous_variables=exogenous_variables,
        output_gap_method=model_params.get("output_gap_method"),
        inflation_method=model_params.get("inflation_method"),
        macro_forecast=model_params.get("macro_forecast"),
        horizons=horizons
        #model_name=model_name
    )
    
    # Fit the model
    logging.info("Fit the model")
    #print("Fit the model")
    model = model_handler.fit(
        train_data=train_data,
        target_col=target_col,
        exogenous_vars=exogenous_variables
    )
    
    # Extract residuals (in-sample, based on train_data)
    logging.info("Extract residuals")
    residuals = extract_residuals(
        model=model,
        execution_date=execution_date
    )
    
    # Extract model metrics
    logging.info("Extract in sample metrics")
    insample_metrics = extract_insample_metrics(
        model=model,
        execution_date=execution_date,
        model_name=model_name,
        target_col=target_col
    )
    
    lagged_beta1 = train_data[target_col].iloc[-1]
    
    # Generate deterministic forecasts
    logging.info("Forecast values")
    results = forecast_values(
        model=model,
        test_data=test_data,
        lagged_beta1=lagged_beta1,
        horizons=horizons,
        exogenous_variables=exogenous_variables,
        execution_date=execution_date,
        df=df,
        target_col=target_col
    )

    logging.info("Generate simulations")
    # Generate simulations in parallel
    simulations = parallel_generate_simulations(
        model=model,
        model_name=model_name,
        test_data=test_data,
        lagged_beta1=lagged_beta1,
        horizons=horizons,
        exogenous_variables=exogenous_variables,
        execution_date=execution_date,
        df=df,
        target_col=target_col,
        num_simulations=num_simulations,
        bootstrap_dates=bootstrap_dates
    )

    return results, simulations, residuals, insample_metrics

def generate_execution_dates(data, consensus_df=None, execution_date_column="forecast_date", min_years=3, macro_forecast="consensus"):
    """
    Generates a list of valid execution dates for backtesting.

    Args:
        data (pd.DataFrame): Input dataset with a DateTime index.
        consensus_df (pd.DataFrame, optional): Consensus dataset with forecast dates.
        execution_date_column (str): Column in the consensus dataset that specifies execution dates.
        min_years (int): Minimum number of years of historical data required for training.
        macro_forecast (str): Macro forecast method ("consensus" or "ar_1").

    Returns:
        list: List of valid execution dates.
    """
    # Ensure the dataset has enough historical data
    min_start_index = min_years * 12  # Convert years to months
    first_valid_index = data.dropna().first_valid_index()  # First valid index after dropping NaNs

    if first_valid_index is None:
        raise ValueError("The dataset contains no valid data after dropping missing values.")

    # Ensure sufficient historical data is available for training
    if len(data.loc[first_valid_index:]) < min_start_index:
        raise ValueError(f"Not enough data. At least {min_years} years of valid data are required.")

    # Handle execution dates based on macro_forecast method
    if macro_forecast == "consensus":
        if consensus_df is None or execution_date_column not in consensus_df.columns:
            raise ValueError(f"Consensus dataset must be provided with a '{execution_date_column}' column for consensus forecasts.")

        # Use the forecast dates from the consensus dataset
        execution_dates = consensus_df[execution_date_column].sort_values().unique()

        # Filter execution dates to ensure sufficient training data
        execution_dates = [date for date in execution_dates if date >= data.index[min_start_index]]

    elif macro_forecast == "ar_1" or macro_forecast == None:
        # Generate execution dates starting after the first valid index and minimum training period
        execution_dates = data.loc[first_valid_index:].index[min_start_index:]

    else:
        raise ValueError(f"Unknown macro_forecast method: {macro_forecast}")

    return execution_dates

def expanding_window_backtest_double_parallel(
    country, df, target_col, horizons, model_name, model_handler, model_params,
    execution_dates, bootstrap_dates,
    df_consensus_gdp, df_consensus_inf, 
    min_years=3, num_simulations=1000, max_workers=min(os.cpu_count()//2, 12)
):
    """
    Performs an expanding window backtest with double parallelization.

    Args:
        country: Country name or code.
        df: Input data.
        target_col: Target column for the model.
        horizons: Forecast horizons.
        model_name: Name of the model.
        model_handler: Model handler object.
        model_params: Model parameters.
        df_consensus_gdp, df_consensus_inf: Consensus forecasts.
        min_years: Minimum years of data for training.
        num_simulations: Number of simulations to generate.
        max_workers: Maximum number of parallel workers.

    Returns:
        tuple: DataFrame of deterministic results and DataFrame of simulations.
    """
    results = []
    all_simulations = []
    insample_residuals = []
    insample_metrics = []
    
    #execution_dates = generate_execution_dates(
    #                    data=df,
    #                    consensus_df=df_consensus_gdp if model_params.get("macro_forecast") == "consensus" else None,
    #                    execution_date_column="forecast_date" if model_params.get("macro_forecast") == "consensus" else None,
    #                    min_years=min_years,
    #                    macro_forecast=model_params.get("macro_forecast")
    #                    )
    
    # Parallelize execution dates
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_execution_date_parallel,
                execution_date,
                df,
                target_col,
                horizons,
                model_name,
                model_handler,
                model_params,
                bootstrap_dates,
                df_consensus_gdp,
                df_consensus_inf,
                num_simulations,
                country
            ): execution_date
            for execution_date in execution_dates
        }

        for future in as_completed(futures):
            try:
                deterministic_results, simulations, residuals, metrics = future.result()
                results.extend(deterministic_results)
                all_simulations.extend(simulations)
                insample_residuals.append(residuals)
                insample_metrics.extend(metrics)
            except Exception as e:
                logging.info(f"Error processing execution date {futures[future]}: {e}")

    return pd.DataFrame(results), pd.DataFrame(all_simulations), pd.concat(insample_residuals), pd.DataFrame(insample_metrics)

def run_backtest_parallel_with_mlflow(
    country, df, horizons, target_col, model_name, model_handler, model_params,
    execution_dates, bootstrap_dates, df_consensus_gdp, df_consensus_inf, 
    save_dir="results", num_simulations=1000, max_workers=min(os.cpu_count()//2, 12)
):
    """
    Runs a backtest with MLflow tracking and logs the results.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data for backtesting.
        horizons (list): Forecast horizons.
        target_col (str): Target column for the model.
        model_name (str): Name of the model.
        model_handler (BaseModel): Model handler object (e.g., AR1Model, ARXModel).
        model_params (dict): Parameters for the model (e.g., methodologies for output gap and inflation).
        save_dir (str): Directory to save results.
        num_simulations (int): Number of simulations to generate for each execution date.
        max_workers (int): Maximum number of workers for parallel processing.

    Returns:
        tuple: DataFrame of predictions and DataFrame of simulations, if successful.
    """
    try:
        # Step 1: Setup MLflow
        setup_mlflow(f"{country}_{target_col}_shadow")

        backtest_type = "Expanding Window"

        # Step 2: Check for existing results
        #if check_existing_results(country, save_dir, target_col, model_name, method_name=backtest_type):
        #    logging.info(f"Model '{model_name}' with method '{backtest_type}' for '{country}' has already been backtested. Skipping...")
        #    return None, None

        # Step 3: Generate consensus forecasts
        #consensus_forecast = ConsensusForecast(QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH)
        #try:
        #    df_consensus_gdp, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} GDP")
        #    df_consensus_inf, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} INF")
        #except Exception as e:
        #    logging.error(f"Error retrieving consensus forecasts: {e}")
        #    raise RuntimeError(f"Error retrieving consensus forecasts: {e}")

        # Step 4: Perform parallelized backtest
        df_predictions, df_simulations, df_insample_residuals, df_insample_metrics = expanding_window_backtest_double_parallel(
            country=country,
            df=df,
            target_col=target_col,
            horizons=horizons,
            model_name=model_name,
            model_handler=model_handler,
            model_params=model_params,
            execution_dates=execution_dates,
            bootstrap_dates=bootstrap_dates,
            df_consensus_gdp=df_consensus_gdp,
            df_consensus_inf=df_consensus_inf,
            num_simulations=num_simulations,
            max_workers=max_workers
        )

        # Step 5: Start a unique MLflow run for this backtest
        unique_run_name = f"{model_name} - {target_col} - {country} - {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with mlflow.start_run(run_name=unique_run_name):
            # Step 6: Extract predictions and actuals
            predictions = {
                horizon: df_predictions[df_predictions["horizon"] == horizon]["prediction"].tolist()
                for horizon in horizons
            }
            actuals = {
                horizon: df_predictions[df_predictions["horizon"] == horizon]["actual"].tolist()
                for horizon in horizons
            }

            logging.info("Log results in MLFLOW")
            # Step 7: Log backtest results
            cleaned_model_name = clean_model_name(model_name)
            log_backtest_results(
                country, df, target_col, cleaned_model_name, backtest_type, horizons,
                predictions, actuals, df_predictions=df_predictions, save_dir=save_dir
            )

        logging.info(f"Backtest completed for {model_name} on {country}. Metrics logged to MLflow.")
        return df_predictions, df_simulations, df_insample_residuals, df_insample_metrics

    except Exception as e:
        logging.error(f"An error occurred during backtesting for model '{model_name}' on country '{country}': {e}")
        return None, None, None, None

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
    
def run_all_backtests_parallel(
    country, df, horizons, target_col, 
    execution_dates, bootstrap_dates, 
    df_consensus_gdp, df_consensus_inf,
    save_dir="results", model_config=None, 
    num_simulations=1000, max_workers=min(os.cpu_count()//2, 12)
):
    """
    Runs backtests for a single model and combines results for a specific target column.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data for backtesting.
        horizons (list): Forecast horizons.
        target_col (str): Target column for the model.
        save_dir (str): Directory to save results (default is "results").
        model_config (dict): Configuration for a single model.
        num_simulations (int): Number of simulations to generate for each execution date.
        max_workers (int): Maximum number of workers for parallel processing.

    Returns:
        None
    """
    if model_config is None:
        raise ValueError("No model configuration provided for backtesting.")

    model_name = model_config["name"]
    model_handler = model_config["handler"]
    model_params = model_config["params"]
    cleaned_model_name = clean_model_name(model_name)

    country_save_dir = os.path.join(save_dir, country)
    os.makedirs(country_save_dir, exist_ok=True)  # Ensure the folder exists

    factors_dir = os.path.join(country_save_dir, "factors", cleaned_model_name, target_col)
    os.makedirs(factors_dir, exist_ok=True)

    try:
        logging.info(f"Running backtest for model: {model_name} and target column: {target_col}...")

        # Run backtest for the current model and target column
        df_predictions, df_simulations, df_insample_residuals, df_insample_metrics = run_backtest_parallel_with_mlflow(
            country=country,
            df=df,
            horizons=horizons,
            target_col=target_col,
            model_name=model_name,
            model_handler=model_handler,
            model_params=model_params,
            execution_dates=execution_dates, 
            bootstrap_dates=bootstrap_dates, 
            df_consensus_gdp=df_consensus_gdp, 
            df_consensus_inf=df_consensus_inf,
            save_dir=save_dir,
            num_simulations=num_simulations,
            max_workers=max_workers
        )
        
        # Save results for the model and target column
        if df_predictions is not None and df_simulations is not None:
            predictions_file = os.path.join(factors_dir, f"forecasts.csv")
            df_predictions.to_csv(predictions_file, index=False)

            logging.info("Calculate out of sample metrics")
            outofsample_metrics = calculate_out_of_sample_metrics(df_predictions)

            logging.info("Save results")
            # Save out of sample metrics metrics to CSV
            metrics_by_horizon_file = os.path.join(factors_dir, f"outofsample_metrics_by_horizon.csv")
            outofsample_metrics["by_horizon"].to_csv(metrics_by_horizon_file, index=False)

            metrics_by_execution_date_file = os.path.join(factors_dir, f"outofsample_metrics_by_execution_date.csv")
            outofsample_metrics["by_execution_date"].to_csv(metrics_by_execution_date_file, index=False)

            metrics_by_row_file = os.path.join(factors_dir, f"outofsample_metrics_by_row.csv")
            outofsample_metrics["by_row"].to_csv(metrics_by_row_file, index=False)

            logging.info(f"Out-of-sample metrics saved for model: {cleaned_model_name} and target column: {target_col}")

            # Save deterministic forecasts
            #predictions_file = os.path.join(factors_dir, f"forecasts.csv")
            #df_predictions.to_csv(predictions_file, index=False)

            # Save simulations
            simulations_file = os.path.join(factors_dir, f"simulations.parquet")
            df_simulations.to_parquet(simulations_file, index=False)
            
            # Save residuals
            residuals_file = os.path.join(factors_dir, f"residuals.csv")
            df_insample_residuals.to_csv(residuals_file, index=False)

            # Save in sample metrics
            insample_metrics_file = os.path.join(factors_dir, f"insample_metrics.csv")
            df_insample_metrics.to_csv(insample_metrics_file, index=False)            
            
            logging.info(f"Results saved for model: {cleaned_model_name} and target column: {target_col}")

    except Exception as e:
        logging.info(f"Error occurred while running backtest for model {cleaned_model_name} and target column {target_col}: {e}")
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

import os
os.chdir(r'C:\git\backtest-baam\code')

from backtesting.backtesting_logging import setup_mlflow, check_existing_results, log_backtest_results
from data_preparation.conensus_forecast import ConsensusForecast
from data_preparation.data_transformations import convert_mom_to_yoy
from modeling.macro_modeling import output_gap, inflation_expectations
from modeling.time_series_modeling import fit_arx_model

from backtesting.config_models import models
from config_paths import QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH

def run_all_backtests(country, df, horizons, target_col, save_dir="results", models=None):
    """
    Runs backtests for multiple models and combines results.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data for backtesting.
        horizons (list): Forecast horizons.
        target_col (str): Target column for the model.
        save_dir (str): Directory to save results (default is "results").
        models (list): List of model configurations.

    Returns:
        None
    """
    if models is None:
        raise ValueError("No models provided for backtesting.")

    all_predictions = []
    all_metrics = []

    for model_config in models:
        model_name = model_config["name"]
        model_handler = model_config["handler"]
        model_params = model_config["params"]

        try:
            print(f"Running backtest for model: {model_name}...")

            # Run backtest for the current model
            df_predictions, model_metrics = run_backtest_with_mlflow(
                country=country,
                df=df,
                horizons=horizons,
                target_col=target_col,
                model_name=model_name,
                model_handler=model_handler,
                model_params=model_params,
                save_dir=save_dir
            )

            # Append results if the backtest was successful
            if df_predictions is not None:
                df_predictions["Model"] = model_name
                all_predictions.append(df_predictions)
                all_metrics.append(model_metrics)

        except Exception as e:
            print(f"Error occurred while running backtest for model {model_name}: {e}")
            continue

    # Combine all predictions and metrics into single DataFrames
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_predictions.to_csv(os.path.join(save_dir, f"combined_predictions_{country}_{target_col}.csv"), index=False)

    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_metrics.to_csv(os.path.join(save_dir, f"combined_metrics_{country}_{target_col}.csv"), index=False)

    print("Backtesting completed for all models.")

def run_backtest_with_mlflow(
    country, df, horizons, target_col, model_name, model_handler, model_params, save_dir="results"
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

    Returns:
        tuple: DataFrame of predictions and metrics, if successful.
    """
    try:
        # Step 1: Setup MLflow
        setup_mlflow(f"{country}_{target_col}_shadow")

        backtest_type="Expanding Window"
        # Step 2: Check for existing results
        if check_existing_results(country, save_dir, target_col, model_name, method_name=backtest_type):
            print(f"Model '{model_name}' with method '{backtest_type}' for '{country}' has already been backtested. Skipping...")
            return None, None

        # Step 3: Generate consensus forecasts
        consensus_forecast = ConsensusForecast(QUARTERLY_CF_FILE_PATH, MONTHLY_CF_FILE_PATH)
        try:
            df_consensus_gdp, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} GDP")
            df_consensus_inf, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} INF")
        except Exception as e:
            raise RuntimeError(f"Error retrieving consensus forecasts: {e}")

        main_run_name = f"{model_name} - Backtest Run - {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with mlflow.start_run(run_name=main_run_name):
            # Step 4: Perform backtest
            df_predictions = expanding_window_backtest(
                country=country,
                df=df,
                target_col=target_col,
                horizons=horizons,
                model_name=model_name,
                model_handler=model_handler,
                model_params=model_params,
                df_consensus_gdp=df_consensus_gdp,
                df_consensus_inf=df_consensus_inf
            )

            # Step 5: Extract predictions and actuals
            predictions = {
                horizon: df_predictions[df_predictions["Horizon"] == horizon]["Prediction"].tolist()
                for horizon in horizons
            }
            actuals = {
                horizon: df_predictions[df_predictions["Horizon"] == horizon]["Actual"].tolist()
                for horizon in horizons
            }

            # Step 6: Log backtest results
            metrics = log_backtest_results(
                df, target_col, model_name, "Expanding Window", horizons,
                predictions, actuals, df_predictions=df_predictions, save_dir=save_dir
            )

            print(f"Backtest completed for {model_name} on {country}. Metrics logged to MLflow.")
            return df_predictions, metrics

    except Exception as e:
        print(f"An error occurred during backtesting: {e}")
        return None, None
    
def expanding_window_backtest(
    country, df, target_col, horizons, model_name, model_handler, model_params,
    df_consensus_gdp, df_consensus_inf, min_years=3
):
    """
    Performs an expanding window backtest for the given model.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data for backtesting.
        target_col (str): Target column for the model.
        horizons (list): Forecast horizons.
        model_name (str): Name of the model.
        model_handler (BaseModel): Model handler object (e.g., AR1Model, ARXModel).
        model_params (dict): Parameters for the model (e.g., methodologies for output gap and inflation).
        df_consensus_gdp (pd.DataFrame): Consensus GDP forecast.
        df_consensus_inf (pd.DataFrame): Consensus inflation forecast.
        min_years (int): Minimum years of data for training.

    Returns:
        pd.DataFrame: Backtesting results.
    """
    min_start_index = min_years * 12  # Assuming monthly data
    min_consensus_date = df_consensus_gdp['forecast_date'].min()
    min_start_date_expanding = df.index[min_years * 12]
    actual_min_start_date = max(min_consensus_date, min_start_date_expanding)
    
    results = []

    for i in range(min_start_index, len(df)):
        execution_date = df.index[i]
        if execution_date < actual_min_start_date:
            continue
        # Step 1: Prepare training and testing datasets
        train_data, test_data = prepare_train_test_data(
            country=country,
            df=df,
            execution_date=execution_date,
            consensus_df_gdp=df_consensus_gdp,
            consensus_df_inf=df_consensus_inf,
            output_gap_col=f"{country}_Outputgap",
            inflation_col=f"{country}_Inflation",
            gdp_col=f"{country}_GDP_YoY",
            output_gap_method=model_params.get("output_gap_method"),
            inflation_method=model_params.get("inflation_method"),
            horizons=horizons,
            model_name=model_name
        )

        # Step 2: Dynamically determine exogenous variables
        exogenous_vars = []
        if "Output Gap" in model_name:
            exogenous_vars.append(f"{country}_Outputgap")
        if "Inflation" in model_name:
            exogenous_vars.append(f"{country}_Inflation")
        if "GDP" in model_name:
                exogenous_vars.append(f"{country}_GDP_YoY")

        # Step 3: Fit the model
        model = model_handler.fit(
            train_data=train_data,
            target_col=target_col,
            exogenous_vars=exogenous_vars
        )

        # Step 4: Generate forecasts
        lagged_beta1 = train_data[target_col].iloc[-1]  # Last observed value
        results.extend(
            forecast_values(
                model=model,
                model_name=model_name,
                test_data=test_data,
                lagged_beta1=lagged_beta1,
                horizons=horizons,
                output_gap_col=f"{country}_Outputgap",
                inflation_col=f"{country}_Inflation",
                gdp_col=f"{country}_GDP_YoY",
                execution_date=execution_date,
                df=df,
                target_col=target_col
            )
        )

    return pd.DataFrame(results)

def forecast_values(model, model_name, test_data, lagged_beta1, horizons, output_gap_col, inflation_col, gdp_col, execution_date, df, target_col):
    """
    Generates forecasts for the specified horizons.

    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted model.
        model_name (str): Name of the model.
        test_data (pd.DataFrame): Testing data.
        lagged_beta1 (float): Lagged beta value.
        horizons (list): Forecast horizons.
        output_gap_col (str): Column name for output gap.
        inflation_col (str): Column name for inflation.
        gdp_col (str): Column name for GDP.
        execution_date (datetime): Execution date for the backtest.
        df (pd.DataFrame): Input data.
        target_col (str): Target column.

    Returns:
        list: Forecast results.
    """
    results = []
    include_output_gap = "Output Gap" in model_name
    include_inflation = "Inflation" in model_name
    include_gdp = "GDP" in model_name

    for horizon in range(1, max(horizons) + 1):
        forecast_date = execution_date + pd.DateOffset(months=horizon)

        if forecast_date in test_data.index:
            output_gap_value = test_data.loc[forecast_date, output_gap_col] if include_output_gap else np.nan
            inflation_value = test_data.loc[forecast_date, inflation_col] if include_inflation else np.nan
            gdp_value = test_data.loc[forecast_date, gdp_col] if include_gdp else np.nan
        else:
            output_gap_value = np.nan
            inflation_value = np.nan
            gdp_value = np.nan

        forecast_value = model.params[0] + model.params[1] * lagged_beta1  # Intercept and AR(1) term

        # Dynamically map exogenous variables to their corresponding parameters
        if output_gap_col in model.params.index:
            forecast_value += model.params[output_gap_col] * output_gap_value
        if inflation_col in model.params.index:
            forecast_value += model.params[inflation_col] * inflation_value
        if gdp_col in model.params.index:
            forecast_value += model.params[gdp_col] * gdp_value

        lagged_beta1 = forecast_value

        actual_value = df.loc[forecast_date, target_col] if forecast_date in df.index else np.nan
        results.append({
            "ExecutionDate": execution_date,
            "ForecastDate": forecast_date,
            "Horizon": horizon,
            "Prediction": forecast_value,
            "Actual": actual_value,
        })

    return results

def prepare_train_test_data(
    country, df, execution_date, consensus_df_gdp, consensus_df_inf,
    output_gap_col, inflation_col, gdp_col,
    output_gap_method, inflation_method, horizons, model_name
):
    """
    Prepares training and testing datasets for backtesting.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data.
        execution_date (datetime): Execution date for the backtest.
        consensus_df_gdp (pd.DataFrame): Consensus GDP forecast.
        consensus_df_inf (pd.DataFrame): Consensus inflation forecast.
        output_gap_col (str): Column name for output gap.
        inflation_col (str): Column name for inflation expectations.
        gdp_col (str): Column name for GDP YoY.
        output_gap_method (str): Method for calculating the output gap ("direct" or "HP").
        inflation_method (str): Method for calculating inflation expectations ("default" or "ucsv").
        horizons (list): Forecast horizons.
        model_name (str): Name of the model.

    Returns:
        tuple: Training and testing datasets.
    """
    # Step 1: Split the data into train and test sets
    train_data = df.loc[:execution_date].copy()
    future_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=max(horizons),
        freq="MS",
    )
    extended_test_data = pd.DataFrame(index=future_dates)
    test_data = pd.concat([df.loc[execution_date:], extended_test_data])

    # Step 2: Add Output Gap
    if "Output Gap" in model_name:
        # Train set: Calculate output gap
        output_gap_train, gdp_train = output_gap(
            country=country,
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )
        # Test set: Calculate output gap
        output_gap_test, gdp_test = output_gap(
            country=country,
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )

        # Add output gap to train and test datasets
        train_data[output_gap_col] = output_gap_train
        test_data[output_gap_col] = output_gap_test.reindex(test_data.index)
        #print(f"Added {output_gap_col} to train_data and test_data")

    # Step 3: Add Inflation Expectations
    if "Inflation" in model_name:
        # Train set: Calculate inflation expectations
        inflation_yoy_train = inflation_expectations(
            data=df[f"{country}_CPI"],
            consensus_df=consensus_df_inf,
            execution_date=execution_date,
            method=inflation_method
        )
        # Test set: Calculate inflation expectations
        inflation_yoy_test = inflation_expectations(
            data=df[f"{country}_CPI"],
            consensus_df=consensus_df_inf,
            execution_date=execution_date,
            method=inflation_method
        )

        # Add inflation expectations to train and test datasets
        train_data[inflation_col] = inflation_yoy_train
        test_data[inflation_col] = inflation_yoy_test.reindex(test_data.index)
        #print(f"Added {inflation_col} to train_data and test_data")

    # Step 4: Add GDP YoY
    if "GDP" in model_name:
        # Train set: Calculate GDP YoY
        _, gdp_train = output_gap(
            country=country,
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )
        # Test set: Calculate GDP YoY
        _, gdp_test = output_gap(
            country=country,
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )

        # Convert MoM to YoY for GDP
        gdp_train_yoy = convert_mom_to_yoy(gdp_train.values, col_name="US_GDP_YoY") * 100
        gdp_train_yoy.index = gdp_train.index
        gdp_test_yoy = convert_mom_to_yoy(gdp_test.values, col_name="US_GDP_YoY") * 100
        gdp_test_yoy.index = gdp_test.index

        # Add GDP YoY to train and test datasets
        train_data[gdp_col] = gdp_train_yoy
        test_data[gdp_col] = gdp_test_yoy
        #print(f"Added {gdp_col} to train_data and test_data")

    return train_data, test_data
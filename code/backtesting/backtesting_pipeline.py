import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

import os
os.chdir(r'C:\git\backtest-baam\code')

from backtesting.backtesting_logging import setup_mlflow, check_existing_results, log_backtest_results
from data_preparation.conensus_forecast import ConsensusForecast
from data_preparation.data_transformations import convert_mom_to_yoy
from modeling.macro_modeling import gdp_with_consensus, output_gap, inflation_expectations
from modeling.time_series_modeling import fit_arx_model

def run_all_backtests(country, df, horizons, target_col, save_dir="results"):
    """
    Runs backtests for multiple models and combines results.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data for backtesting.
        horizons (list): Forecast horizons.
        target_col (str): Target column for the model.
        save_dir (str): Directory to save results (default is "results").

    Returns:
        None
    """
    # Define models and their corresponding functions
    models = {
        #"AR(1)": lambda data, **kwargs: fit_arx_model(data),
        "AR(1) + Output Gap": lambda data, **kwargs: fit_arx_model(df, output_gap=kwargs.get('output_gap')),
        #"AR(1) + GDP": lambda data, **kwargs: fit_arx_model(data, gdp=kwargs.get('gdp')),
        #"AR(1) + Inflation": lambda data, **kwargs: fit_arx_model(data, inflation=kwargs.get('inflation')),
        #"AR(1) + Inflation (UCSV)": lambda data, **kwargs: fit_arx_model(data, inflation=kwargs.get('inflation')),
        #"AR(1) + GDP + Inflation": lambda data, **kwargs: fit_arx_model(data, gdp=kwargs.get('gdp'), inflation=kwargs.get('inflation')),
        #"AR(1) + GDP + Inflation (UCSV)": lambda data, **kwargs: fit_arx_model(data, gdp=kwargs.get('gdp'), inflation=kwargs.get('inflation')),
        #"AR(1) + Output Gap + Inflation": lambda data, **kwargs: fit_arx_model(data, output_gap=kwargs.get('output_gap'), inflation=kwargs.get('inflation')),
        #"AR(1) + Output Gap + Inflation (UCSV)": lambda data, **kwargs: fit_arx_model(data, output_gap=kwargs.get('output_gap'), inflation=kwargs.get('inflation')),
        #"AR(1) + Output Gap (HP Filter)": lambda data, **kwargs: fit_arx_model(data, output_gap=kwargs.get('output_gap')),
        #"AR(1) + Output Gap (HP Filter) + Inflation": lambda data, **kwargs: fit_arx_model(data, output_gap=kwargs.get('output_gap'), inflation=kwargs.get('inflation')),
        #"AR(1) + Output Gap (HP Filter) + Inflation (UCSV)": lambda data, **kwargs: fit_arx_model(data, output_gap=kwargs.get('output_gap'), inflation=kwargs.get('inflation'))
    }

    # Initialize lists to store predictions and metrics
    all_predictions = []
    all_metrics = []

    # Iterate over models and run backtests
    for model_name, model_func in models.items():
        try:
            print(f"Running backtest for model: {model_name}...")

            # Determine methods based on model name
            output_gap_method = "hp_filter" if "HP Filter" in model_name else "direct"
            inflation_method = "ucsv" if "UCSV" in model_name else "default"

            # Run backtest for the current model
            df_predictions, model_metrics = run_backtest_with_mlflow(
                country, df, horizons, target_col, model_name, model_func,
                backtest_type="expanding", save_dir=save_dir,
                output_gap_method=output_gap_method, inflation_method=inflation_method
            )

            # Append results if the backtest was successful
            if df_predictions is not None:
                df_predictions['Model'] = model_name
                all_predictions.append(df_predictions)
                all_metrics.append(model_metrics)

        except Exception as e:
            print(f"Error occurred while running backtest for model {model_name}: {e}")
            continue

    # Combine all predictions and metrics into single DataFrames
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # Save combined predictions and metrics to CSV files
        combined_data_path = os.path.join(save_dir, f'combined_data_{country}_{target_col}_shadow.csv')
        combined_predictions_path = os.path.join(save_dir, f'full_predictions_{country}_{target_col}_shadow.csv')
        combined_metrics_path = os.path.join(save_dir, f'model_metrics_{country}_{target_col}_shadow.csv')

        combined_predictions.to_csv(combined_predictions_path, index=False)
        combined_metrics.to_csv(combined_metrics_path, index=False)

        print(f"Backtesting completed for all models. Results saved to {save_dir}.")
    else:
        print("No predictions or metrics were generated. All models may have been skipped.")

def run_backtest_with_mlflow(
    country, df, horizons, target_col, model_name, model_func, 
    backtest_type="expanding", save_dir="results", output_gap_method="direct", inflation_method="default"
):
    """
    Runs a backtest with MLflow tracking and logs the results.

    Args:
        country (str): Country name or code.
        df (pd.DataFrame): Input data for backtesting.
        horizons (list): Forecast horizons.
        target_col (str): Target column for the model.
        model_name (str): Name of the model.
        model_func (callable): Function to fit the model.
        backtest_type (str): Type of backtesting ("expanding" or others).
        save_dir (str): Directory to save results.
        output_gap_method (str): Method for calculating the output gap ("direct" or "HP").
        inflation_method (str): Method for calculating inflation expectations ("default" or "ucsv").

    Returns:
        tuple: DataFrame of predictions and metrics, if successful.
    """
    try:
        # Step 1: Setup MLflow
        setup_mlflow(f'{country}_{target_col}_shadow')

        # Step 2: Check for existing results
        if check_existing_results(country, save_dir, target_col, model_name):
            print(f"Model {model_name} for {country} has already been backtested. Skipping...")
            return None, None

        quarterly_file_path = r"\\msfsshared\\bnkg\RMAS\Resources\Databases\Consensus Economics Forecasts\Long Term"
        monthly_file_path = r"\\msfsshared\\bnkg\RMAS\Resources\Databases\Consensus Economics Forecasts\Monthly"
        # Step 3: Generate consensus forecasts
        consensus_forecast = ConsensusForecast(quarterly_file_path, monthly_file_path)
        try:
            df_consensus_gdp, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} GDP")
            df_consensus_inf, _ = consensus_forecast.get_consensus_forecast(country_var=f"{country} INF")
        except Exception as e:
            raise RuntimeError(f"Error retrieving consensus forecasts: {e}")

        # Step 4: Define run name
        main_run_name = f"{model_name} - {backtest_type.capitalize()} Backtest Run - {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # Step 5: Start MLflow run
        with mlflow.start_run(run_name=main_run_name):
            # Validate backtest type
            if backtest_type != "expanding":
                raise ValueError(f"Unsupported backtest type: {backtest_type}")

            # Define column names
            output_gap_col = f'{country}_Outputgap'
            inflation_col = f'{country}_Inflation'
            gdp_col = f'{country}_GDP_YoY'

            # Step 6: Perform backtest
            df_predictions = expanding_window_backtest(
                country, model_func, model_name, df, target_col, horizons, 
                df_consensus_gdp, df_consensus_inf, 
                output_gap_col=output_gap_col,
                inflation_col=inflation_col,
                gdp_col=gdp_col,
                output_gap_method=output_gap_method,
                inflation_method=inflation_method
            )

            # Step 7: Extract predictions and actuals
            predictions = {
                horizon: df_predictions[df_predictions['Horizon'] == horizon]['Prediction'].tolist()
                for horizon in horizons
            }
            actuals = {
                horizon: df_predictions[df_predictions['Horizon'] == horizon]['Actual'].tolist()
                for horizon in horizons
            }

            # Step 8: Log backtest results
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
    country, model_func, model_name, df, target_col, horizons, consensus_df_gdp, consensus_df_inf,
    output_gap_col="US_Outputgap", inflation_col="US_Inflation",
    gdp_col="US_GDP_YoY", min_years=3,
    output_gap_method="direct", inflation_method="default"
):
    """
    Performs an expanding window backtest for the given model.

    Args:
        model_func (callable): Function to fit the model.
        model_name (str): Name of the model.
        df (pd.DataFrame): Input data.
        target_col (str): Target column for the model.
        horizons (list): Forecast horizons.
        consensus_df_gdp (pd.DataFrame): Consensus GDP forecast.
        consensus_df_inf (pd.DataFrame): Consensus inflation forecast.
        output_gap_col (str): Column name for output gap.
        inflation_col (str): Column name for inflation expectations.
        gdp_col (str): Column name for GDP YoY.
        min_years (int): Minimum years of data for training.
        output_gap_method (str): Method for calculating the output gap ("direct" or "HP").
        inflation_method (str): Method for calculating inflation expectations ("default" or "ucsv").

    Returns:
        pd.DataFrame: Backtesting results.
    """

    min_consensus_date = consensus_df_gdp['forecast_date'].min()
    min_start_date_expanding = df.index[min_years * 12]
    actual_min_start_date = max(min_consensus_date, min_start_date_expanding)

    results = []

    for i in range(min_years * 12, len(df)):
        execution_date = df.index[i]

        if execution_date < actual_min_start_date:
            continue

        train_data, test_data = prepare_train_test_data(
            country, df, execution_date, consensus_df_gdp, consensus_df_inf,
            output_gap_col, inflation_col, gdp_col,
            output_gap_method, inflation_method, horizons, model_name
        )

        model = model_func(
            train_data[target_col],
            output_gap=train_data[output_gap_col] if output_gap_col in train_data else None,
            inflation=train_data[inflation_col] if inflation_col in train_data else None,
            gdp=train_data[gdp_col] if gdp_col in train_data else None
        )

        lagged_beta1 = train_data[target_col].iloc[-1]
        results.extend(forecast_values(model, model_name, test_data, lagged_beta1, horizons, output_gap_col, inflation_col, gdp_col, execution_date, df, target_col))

    results_df = pd.DataFrame(results)
    return results_df

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

        forecast_value = model.params[0] + model.params[1] * lagged_beta1
        param_index = 2
        if include_output_gap and output_gap_col in model.params.index:
            forecast_value += model.params[param_index] * output_gap_value
            param_index += 1
        if include_inflation and inflation_col in model.params.index:
            forecast_value += model.params[param_index] * inflation_value
            param_index += 1
        if include_gdp and gdp_col in model.params.index:
            forecast_value += model.params[param_index] * gdp_value

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

def prepare_train_test_data(country, df, execution_date, consensus_df_gdp, consensus_df_inf, 
                            output_gap_col, inflation_col, gdp_col, 
                            output_gap_method, inflation_method, horizons, model_name):
    """
    Prepares training and testing datasets for backtesting.

    Args:
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
            country="US",
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )
        # Test set: Calculate output gap
        output_gap_test, gdp_test = output_gap(
            country="US",
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
            data=df[f'{country}_CPI'],
            consensus_df=consensus_df_inf,
            execution_date=execution_date,
            method=inflation_method
        )
        # Test set: Calculate inflation expectations
        inflation_yoy_test = inflation_expectations(
            data=df[f'{country}_CPI'],
            consensus_df=consensus_df_inf,
            execution_date=execution_date,
            method=inflation_method
        )

        # Add inflation expectations to train and test datasets
        train_data[inflation_col] = inflation_yoy_train
        test_data[inflation_col] = inflation_yoy_test.reindex(test_data.index)
        print(f"Added {inflation_col} to train_data and test_data")

    # Step 4: Add GDP YoY
    if "GDP" in model_name:
        # Train set: Calculate GDP YoY
        _, gdp_train = output_gap(
            country="US",
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )
        # Test set: Calculate GDP YoY
        _, gdp_test = output_gap(
            country="US",
            data=df,
            consensus_df=consensus_df_gdp,
            execution_date=execution_date,
            method=output_gap_method
        )

        # Convert MoM to YoY for GDP
        gdp_train_yoy = convert_mom_to_yoy(gdp_train.values, col_name='US_GDP_YoY') * 100
        gdp_train_yoy.index = gdp_train.index
        gdp_test_yoy = convert_mom_to_yoy(gdp_test.values, col_name='US_GDP_YoY') * 100
        gdp_test_yoy.index = gdp_test.index

        # Add GDP YoY to train and test datasets
        train_data[gdp_col] = gdp_train_yoy
        test_data[gdp_col] = gdp_test_yoy
        print(f"Added {gdp_col} to train_data and test_data")

    return train_data, test_data
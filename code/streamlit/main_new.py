import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import gaussian_kde
import numpy as np
import plotly.graph_objects as go

benchmark_model = "AR_1"

@st.cache_data
def load_model_simulations(model, maturity_folder_path):
    """
    Load all simulation parquet files for a specific model and maturity.
    
    Args:
        model (str): The name of the model.
        maturity_folder_path (str): The folder path for the maturity.

    Returns:
        pd.DataFrame: Combined DataFrame of all simulations for the model.
    """
    all_parquet_files = [os.path.join(maturity_folder_path, f) for f in os.listdir(maturity_folder_path) if f.endswith(".parquet")]
    if all_parquet_files:
        # Concatenate all parquet files for the current model
        model_simulations = pd.concat([pd.read_parquet(file) for file in all_parquet_files], ignore_index=True)
        model_simulations["model"] = model  # Add a column to identify the model
        return model_simulations
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files are found
# Helper Functions
def load_data(folder_path, subfolder, model, file_name):
    """Load data from the specified folder path and rename inconsistent columns."""
    file_path = os.path.join(folder_path, subfolder, model, file_name)
    if os.path.exists(file_path):
        if file_name.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_name.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        else:
            st.warning(f"Unsupported file format: {file_name}")
            return None

        # Rename columns if necessary
        #rename_mapping = {
        #    "execution_date": "ExecutionDate",
        #    "forecasted_date": "ForecastDate",
        #    "prediction": "Prediction",
        #    "actual": "Actual",
        #    "horizon": "Horizon"
        #}
        #data = data.rename(columns=rename_mapping)
        return data
    else:
        st.warning(f"File not found: {file_path}")
        return None

# Helper function to calculate RMSE
def calculate_rmse(data, start_date, end_date=None):
    """Calculate RMSE from the forecasts data filtered by start and end dates."""
    filtered_data = data[
        (pd.to_datetime(data["execution_date"]) >= pd.to_datetime(start_date))
    ]
    if end_date:
        filtered_data = filtered_data[
            (pd.to_datetime(filtered_data["execution_date"]) <= pd.to_datetime(end_date))
        ]
    filtered_data = filtered_data.dropna(subset=["actual", "prediction"])
    rmse_data = (
        filtered_data
        .groupby("horizon")
        .apply(lambda x: np.sqrt(((x["prediction"] - x["actual"]) ** 2).mean()))
        .reset_index(name="rmse")
    )
    return rmse_data

# Cache loaded data to avoid redundant loading
@st.cache_data
def load_and_cache_forecasts(models, yields_folder_path, include_benchmark):
    cached_data = {}
    for model in models + ([benchmark_model] if include_benchmark else []):
        subfolder = "observed_yields" if model == f"{benchmark_model} (Benchmark)" else "estimated_yields"
        model_to_load = benchmark_model if model == f"{benchmark_model} (Benchmark)" else model
        data = load_data(yields_folder_path, subfolder, model_to_load, "forecasts.csv")
        if data is not None:
            data["model"] = model
            cached_data[model] = data
    return cached_data

def create_dual_axis_plot(data, x_col, y1_col, y2_col, title, y1_label, y2_label):
    """Create a dual-axis plot with coefficient on the primary axis and p-value on the secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add coefficient trace (primary y-axis)
    fig.add_trace(
        go.Scatter(x=data[x_col], y=data[y1_col], mode='lines', name=y1_label, line=dict(color='#3a6bac')),
        secondary_y=False
    )
    
    # Add p-value traces (secondary y-axis) with color coding
    fig.add_trace(
        go.Scatter(
            x=data[data[y2_col] < 0.05][x_col], 
            y=data[data[y2_col] < 0.05][y2_col], 
            mode='markers',  # Use markers only
            name=f"{y2_label} (< 0.05)", 
            marker=dict(color='#c28191', size=3, symbol='circle')
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=data[data[y2_col] >= 0.05][x_col], 
            y=data[data[y2_col] >= 0.05][y2_col], 
            mode='markers',  # Use markers only
            name=f"{y2_label} (>= 0.05)", 
            marker=dict(color='gray', size=3, symbol='circle')
        ),
        secondary_y=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Execution Date")
    fig.update_yaxes(title_text=y1_label, secondary_y=False)
    fig.update_yaxes(title_text=y2_label, secondary_y=True, showgrid=False)  # Remove grid lines for secondary axis
    
    # Update layout
    fig.update_layout(title=title, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    
    return fig


@st.cache_data
def load_and_cache_data(models, yields_folder_path, include_benchmark, file_name):
    cached_data = {}
    for model in models + ([benchmark_model] if include_benchmark else []):
        subfolder = "observed_yields" if model == f"{benchmark_model} (Benchmark)" else "estimated_yields"
        model_to_load = benchmark_model if model == f"{benchmark_model} (Benchmark)" else model
        data = load_data(yields_folder_path, subfolder, model_to_load, file_name)
        if subfolder == "observed_yields":
            data['maturity'] = data['maturity'].str.split(' ', expand = True)[0]
        else:
            data['maturity'] = data['maturity'].astype(str)
            data = data.rename(columns={"mean_simulated": "prediction"})
            
        if data is not None:
            data["model"] = model
            cached_data[model] = data
    return cached_data

# App Configuration
st.set_page_config(layout="wide", page_title="Backtesting Results Dashboard")

# Sidebar: Country Selection
st.sidebar.header("Backtesting results dashboard")
countries = ["US", "EA", "UK"]
selected_country = st.sidebar.selectbox("Select Country", countries, index=0)

# Dynamically update the folder path based on the selected country
base_folder = r"\\msfsshared\BNKG\\RMAS\Users\Alberto\backtest-baam\data_test"
folder_path = os.path.join(base_folder, selected_country, "factors")

# Get available models and target variables
models = [f.name for f in os.scandir(folder_path) if f.is_dir() and f.name != "archive"]  # Exclude "archive"
target_variables = ["beta1", "beta2", "beta3"]

macro_categories = ["Factors", "Yields", "Returns", "Consensus Economics"]
selected_macro = st.sidebar.radio("Select Category", macro_categories)

if selected_macro == "Factors":

    tab_names = [
        "Backtesting overview",
        "Out-of-sample metrics",
        "In-sample metrics",
        "Simulations analysis"
    ]
    tabs = st.tabs(tab_names)
    tab_factors, tab_out_of_sample, tab_in_sample, tab_simulations = tabs

    with tab_factors:
        st.title("Actuals vs predictions for factors")

        # Create a 3-column layout for the plots
        #st.subheader("Plots: Actuals vs Predictions")
        cols = st.columns(3)

        # Loop through all factors
        for idx, (col, target_variable) in enumerate(zip(cols, target_variables)):  # target_variables = ["beta1", "beta2", "beta3"]
            with col:
                st.subheader(f"{target_variable}")
                combined_data = []

                # Dropdown for selecting models (specific to this factor)
                #st.subheader("Model Selection")
                selected_models = st.multiselect(
                    f"Select models (Max 3)",
                    models,  # Use the list of available models
                    default=models[:1],  # Default to the first model
                    key=f"model_selector_{target_variable}_{idx}"  # Unique key for each factor
                )

                # Restrict to a maximum of 3 models
                if len(selected_models) > 3:
                    st.warning(f"Please select up to 3 models for {target_variable}. Only the first 3 models will be used.")
                    selected_models = selected_models[:3]  # Limit to the first 3 models

                # Date selectors for this specific panel
                #st.subheader("Filter by Date Range")
                min_date = datetime.strptime("1950-01-01", "%Y-%m-%d").date()
                max_date = datetime.strptime("2030-12-31", "%Y-%m-%d").date()

                date_cols = st.columns(2)
                with date_cols[0]:
                    start_date = st.date_input(
                        f"Start date for",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"start_date_input_{target_variable}_{idx}"  # Unique key for each factor
                    )
                with date_cols[1]:
                    end_date = st.date_input(
                        f"End date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"end_date_input_{target_variable}_{idx}"  # Unique key for each factor
                    )

                # Ensure the start date is not after the end date
                if start_date > end_date:
                    st.warning(f"Start date cannot be after the end date for {target_variable}. Please adjust the dates.")
                else:
                    # Loop through selected models and load data
                    for model in selected_models:
                        # Construct the path to the forecasts.csv file
                        forecasts_file_path = os.path.join(folder_path, model, target_variable, "forecasts.csv")
                        if os.path.exists(forecasts_file_path):
                            # Load the forecasts.csv file
                            data = load_data(folder_path, os.path.join(model, target_variable), "", "forecasts.csv")

                            # Ensure the required columns exist
                            required_columns = ["execution_date", "forecast_date", "prediction", "actual"]
                            if all(col in data.columns for col in required_columns):
                                # Convert ExecutionDate to datetime if not already in datetime format
                                if not pd.api.types.is_datetime64_any_dtype(data["execution_date"]):
                                    data["execution_date"] = pd.to_datetime(data["execution_date"])

                                # Filter data using the selected date range
                                filtered_data = data[
                                    (data["execution_date"].dt.date >= start_date) &
                                    (data["execution_date"].dt.date <= end_date)
                                ]

                                if not filtered_data.empty:
                                    filtered_data["model"] = model
                                    combined_data.append(filtered_data)
                            else:
                                st.warning(f"Required columns are missing in the forecasts data for model: {model}, factor: {target_variable}")
                        else:
                            st.warning(f"File not found: {forecasts_file_path}")

                    # Combine data from all models
                    if combined_data:
                        combined_data = pd.concat(combined_data, ignore_index=True)

                        # Prepare the actuals data
                        realized_beta = combined_data.groupby("forecast_date")["actual"].mean()

                        # Create the Plotly figure
                        fig = go.Figure()

                        # Define colors based on the number of selected models
                        if len(selected_models) == 1:
                            model_colors = ["#c28191"]  # Slightly darker reddish-pink
                            #model_colors = ["dimgrey"]  # Slightly darker grey
                        elif len(selected_models) == 2:
                            model_colors = ["#c28191", "#3a6bac"]  # Reddish-pink and blue
                        else:  # len(selected_models) == 3
                            model_colors = ["#c28191", "#3a6bac", "#eaa121"]  # Reddish-pink, blue, and yellow-orange

                        model_color_mapping = {model: model_colors[i] for i, model in enumerate(selected_models)}

                        for model in selected_models:
                            model_data = combined_data[combined_data["model"] == model]
                            unique_execution_dates = model_data["execution_date"].unique()
                            
                            # Add traces for each execution date, but hide them from the legend
                            for execution_date in unique_execution_dates:
                                subset = model_data[model_data["execution_date"] == execution_date]
                                fig.add_trace(go.Scatter(
                                    x=subset["forecast_date"],
                                    y=subset["prediction"],
                                    mode="lines",
                                    line=dict(color=model_color_mapping[model], width=1),
                                    opacity=0.5,
                                    name=None,  # Do not show individual traces in the legend
                                    showlegend=False  # Ensure these traces are not in the legend
                                ))

                            # Add a single legend entry for the model
                            fig.add_trace(go.Scatter(
                                x=[None],  # Dummy data
                                y=[None],  # Dummy data
                                mode="lines",
                                line=dict(color=model_color_mapping[model], width=1),
                                name=model,  # Legend entry for the model
                                showlegend=True  # Ensure this trace appears in the legend
                            ))

                        # Add the actuals line (black)
                        fig.add_trace(go.Scatter(
                            x=realized_beta.index,
                            y=realized_beta.values,
                            mode="lines",
                            line=dict(color="black", width=2),
                            name="Actual"
                        ))

                        # Update layout
                        fig.update_layout(
                            #title=f"{target_variable}: Forecasted vs Actual",
                            #xaxis_title="Forecast Date",
                            #yaxis_title=f"{target_variable}",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.6, xanchor="center", x=0.5),
                            template="plotly_white"
                        )

                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data available for {target_variable}.")

    # Out-of-Sample Model Comparison Tab
    with tab_out_of_sample:
        st.title("Out-of-sample model comparison")

        # Determine the date range for the data
        all_dates = []
        for model in models:
            for target_variable in target_variables:
                # Load data for out-of-sample metrics
                data_horizon = load_data(folder_path, model, target_variable, "outofsample_metrics_by_horizon.csv")
                data_execution = load_data(folder_path, model, target_variable, "outofsample_metrics_by_execution_date.csv")
                if data_horizon is not None and "execution_date" in data_horizon.columns:
                    all_dates.extend(pd.to_datetime(data_horizon["execution_date"]).tolist())
                if data_execution is not None and "execution_date" in data_execution.columns:
                    all_dates.extend(pd.to_datetime(data_execution["execution_date"]).tolist())

        # Calculate the minimum and maximum dates
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
        else:
            min_date = pd.to_datetime("1950-01-01")  # Default fallback
            max_date = pd.to_datetime("2025-12-31")  # Default fallback

        # Create columns for Default RMSE by Horizon (First Row)
        st.subheader("RMSE by forecasting horizon (full sample)")
        cols = st.columns(3)
        selected_models_per_beta = {}  # To store selected models for each beta
        for col, target_variable in zip(cols, target_variables):
            with col:
                st.subheader(f"{target_variable}")
                
                # Dropdown for selecting models specific to this beta (factor)
                selected_models = st.multiselect(
                    f"Select models", 
                    models, 
                    default=models[:1],
                    key=f"model_selector_{target_variable}"  # Unique key for each beta
                )
                selected_models_per_beta[target_variable] = selected_models
                
                # Default Plot: RMSE by Horizon (Pre-calculated)
                combined_data = []
                for model in selected_models:
                    data = load_data(folder_path, model, target_variable, "outofsample_metrics_by_horizon.csv")
                    if data is not None:
                        data["model"] = model
                        combined_data.append(data)
                
                if combined_data:
                    combined_data = pd.concat(combined_data, ignore_index=True)
                    fig = px.line(
                        combined_data,
                        x="horizon",
                        y="rmse",
                        color="model",
                        markers=True,
                        title=f"{target_variable}",
                        labels={"rmse": "RMSE", "horizon": "Horizon"},
                    )
                    fig.update_layout(
                        legend=dict(
                            orientation="h",  # Horizontal legend
                            yanchor="bottom",
                            y=-0.5,  # Adjust legend position below the plot
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for Default RMSE ({target_variable})")

        # Date pickers for Dynamic RMSE by Horizon (Placed Under the First Row)
        st.subheader("RMSE by forecasting horizon (from selected date range)")
        cols = st.columns(2)
        with cols[0]:
            dynamic_start_date = st.date_input(
                "Select start date",
                value=min_date,  # Default to the earliest date
                min_value=min_date,
                max_value=max_date,
                key="dynamic_start_date_horizon"
            )
        with cols[1]:
            dynamic_end_date = st.date_input(
                "Select end date",
                value=max_date,  # Default to the latest date
                min_value=min_date,
                max_value=max_date,
                key="dynamic_end_date_horizon"
            )

        # Create columns for Dynamic RMSE by Horizon (Second Row)
        cols = st.columns(3)
        for col, target_variable in zip(cols, target_variables):
            with col:
                combined_data = []
                for model in selected_models_per_beta[target_variable]:  # Use models selected in the first row
                    data = load_data(folder_path, model, target_variable, "forecasts.csv")
                    if data is not None:
                        filtered_data = data[
                            (pd.to_datetime(data['execution_date']) >= pd.to_datetime(dynamic_start_date)) &
                            (pd.to_datetime(data['execution_date']) <= pd.to_datetime(dynamic_end_date))
                        ]
                        rmse_data = calculate_rmse(filtered_data, dynamic_start_date)
                        rmse_data["model"] = model
                        combined_data.append(rmse_data)
                
                if combined_data:
                    combined_data = pd.concat(combined_data, ignore_index=True)
                    fig = px.line(
                        combined_data,
                        x="horizon",
                        y="rmse",
                        color="model",
                        markers=True,
                        title=f"{target_variable}",
                        labels={"horizon": "Horizon", "rmse": "RMSE"},
                    )
                    fig.update_layout(
                        legend=dict(
                            orientation="h",  # Horizontal legend
                            yanchor="bottom",
                            y=-0.5,  # Adjust legend position below the plot
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for Dynamic RMSE ({target_variable})")

        # Global Date Pickers for RMSE by Execution Date (Placed Above the Last Row)
        st.subheader("RMSE by execution date (from selected date range)")
        cols = st.columns(2)
        with cols[0]:
            execution_start_date = st.date_input(
                "Select start date",
                value=min_date,  # Default to the earliest date
                min_value=min_date,
                max_value=max_date,
                key="execution_start_date"
            )
        with cols[1]:
            execution_end_date = st.date_input(
                "Select end date",
                value=max_date,  # Default to the latest date
                min_value=min_date,
                max_value=max_date,
                key="execution_end_date"
            )

        # Create columns for RMSE by Execution Date (Last Row)
        cols = st.columns(3)
        for col, target_variable in zip(cols, target_variables):
            with col:
                combined_data = []
                for model in selected_models_per_beta[target_variable]:  # Use models selected in the first row
                    data = load_data(folder_path, model, target_variable, "outofsample_metrics_by_execution_date.csv")
                    if data is not None:
                        filtered_data = data[
                            (pd.to_datetime(data['execution_date']) >= pd.to_datetime(execution_start_date)) &
                            (pd.to_datetime(data['execution_date']) <= pd.to_datetime(execution_end_date))
                        ]
                        filtered_data["model"] = model
                        combined_data.append(filtered_data)
                
                if combined_data:
                    combined_data = pd.concat(combined_data, ignore_index=True)
                    fig = px.line(
                        combined_data,
                        x="execution_date",
                        y="rmse",
                        color="model",
                        title=f"{target_variable}",
                        labels={"rmse": "RMSE", "execution_date": "Execution Date"},
                    )
                    fig.update_layout(
                        legend=dict(
                            orientation="h",  # Horizontal legend
                            yanchor="bottom",
                            y=-0.5,  # Adjust legend position below the plot
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for RMSE by Execution Date ({target_variable})")
           
    # In-Sample Model Specifics Tab
    with tab_in_sample:
        st.title("Factors in-sample model metrics")

        # Load all in-sample metrics data to determine the date range
        all_data = []
        for model in models:
            for target_variable in target_variables:
                data = load_data(folder_path, model, target_variable, "insample_metrics.csv")
                #data = data.drop_duplicates(subset=["execution_date", "metric"])
                if data is not None:
                    data["execution_date"] = pd.to_datetime(data["execution_date"])
                    all_data.append(data)
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            min_date = combined_data["execution_date"].min()
            max_date = combined_data["execution_date"].max()
        else:
            min_date = pd.to_datetime("2000-01-01")
            max_date = pd.to_datetime("2023-12-31")

        # Start Date Filter
        start_date = st.date_input(
            "Select start date",
            value=min_date,  # Default to the earliest date in the data
            min_value=min_date,
            max_value=max_date,
            key="start_date_filter"
        )

        # Dropdown to select a specific model for each beta
        cols = st.columns(3)
        selected_models = {}
        for col, target_variable in zip(cols, target_variables):
            with col:
                st.subheader(f"{target_variable}")
                selected_models[target_variable] = st.selectbox(
                    f"Select model",
                    models,
                    index=0,
                    key=f"model_selector_insample_{target_variable}"
                )

        # Load the in-sample metrics data for each beta
        insample_data = {}
        for target_variable, model in selected_models.items():
            insample_data[target_variable] = load_data(folder_path, model, target_variable, "insample_metrics.csv")
            insample_data[target_variable] = insample_data[target_variable].drop_duplicates(subset=["execution_date", "indicator", "target_col", "metric"], keep='first')
        # Replace 'beta1', 'beta2', 'beta3' in 'indicator' with 'lagged beta'
        for target_variable, data in insample_data.items():
            if data is not None:
                data["indicator"] = data["indicator"].replace(
                    {"beta1": "lagged beta", "beta2": "lagged beta", "beta3": "lagged beta"}
                )
                # Convert execution_date to datetime and sort
                data["execution_date"] = pd.to_datetime(data["execution_date"])
                data.sort_values("execution_date", inplace=True)
                # Filter data based on the selected start date
                insample_data[target_variable] = data[data["execution_date"] >= pd.to_datetime(start_date)]

        # Define the desired row order
        row_order = ["adjusted_r_squared", "lagged beta", "output_gap", "inflation", "const"]

        # Dynamically generate rows of graphs based on the ordered indicators
        for row in row_order:
            if row == "adjusted_r_squared":
                # Plot Adjusted R-Squared in its own row
                st.subheader("Adjusted R-Squared")
                cols = st.columns(3)
                for col, target_variable in zip(cols, target_variables):
                    with col:
                        if target_variable in insample_data and insample_data[target_variable] is not None:
                            filtered_data = insample_data[target_variable][
                                insample_data[target_variable]["metric"] == "adjusted_r_squared"
                            ]
                            
                            fig = px.line(
                                filtered_data,
                                x="execution_date",
                                y="value",
                                title=f"{target_variable}",
                                labels={"execution_date": "Execution Date", "value": "Adjusted R-Squared"},
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                # Plot coefficients and overlay p-values for the other indicators
                st.subheader(f"Time varying coefficient for {row}")
                cols = st.columns(3)
                for col, target_variable in zip(cols, target_variables):
                    with col:
                        if target_variable in insample_data and insample_data[target_variable] is not None:
                            filtered_data = insample_data[target_variable][
                                (insample_data[target_variable]["indicator"] == row) &
                                (insample_data[target_variable]["metric"].isin(["coefficient", "p_value"]))
                            ]
                            if not filtered_data.empty:
                                # Pivot data to get coefficient and p_value as separate columns
                                pivot_data = filtered_data.pivot(
                                    index="execution_date", columns="metric", values="value"
                                ).reset_index()
                                pivot_data["execution_date"] = pd.to_datetime(pivot_data["execution_date"])
                                if "coefficient" in pivot_data.columns and "p_value" in pivot_data.columns:
                                    fig = create_dual_axis_plot(
                                        data=pivot_data,
                                        x_col="execution_date",
                                        y1_col="coefficient",
                                        y2_col="p_value",
                                        title=f"",
                                        y1_label="Coefficient",
                                        y2_label="P-Value"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

    #elif selected_tab == "Factors simulations analysis":
    with tab_simulations:
        st.title("Factors simulations analysis: heatmaps, fan chart, and distributions")

        # Dropdown to select a specific model and beta
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox(
                "Select model",
                models,
                index=0,
                key="simulation_model_selector"
            )
        with col2:
            selected_beta = st.selectbox(
                "Select factor",
                target_variables,
                index=0,
                key="simulation_beta_selector"
            )

        # Load the simulations data and forecasts data for the selected model and beta
        simulations_data = load_data(folder_path, selected_model, selected_beta, "simulations.parquet")
        forecasts_data = load_data(folder_path, selected_model, selected_beta, "forecasts.csv")  # Load forecasts data

        if simulations_data is not None and forecasts_data is not None:
            # Convert ExecutionDate and ForecastDate to datetime
            simulations_data["execution_date"] = pd.to_datetime(simulations_data["execution_date"])
            simulations_data["forecast_date"] = pd.to_datetime(simulations_data["forecast_date"])
            forecasts_data["execution_date"] = pd.to_datetime(forecasts_data["execution_date"])
            forecasts_data["forecast_date"] = pd.to_datetime(forecasts_data["forecast_date"])

            # Group actuals by ForecastDate
            actuals = forecasts_data[["forecast_date", "actual"]].groupby("forecast_date").last().dropna()

            # Determine the date range for the data
            min_date = simulations_data["execution_date"].min()
            max_date = simulations_data["execution_date"].max()

            # Add start and end date pickers
            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input(
                    "Select start date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="simulation_start_date"
                )
            with col4:
                end_date = st.date_input(
                    "Select end date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="simulation_end_date"
                )

            # Filter data based on the selected date range
            filtered_simulations = simulations_data[
                (simulations_data["execution_date"] >= pd.to_datetime(start_date)) &
                (simulations_data["execution_date"] <= pd.to_datetime(end_date))
            ]

            # --- Visualization 1: Horizon–Origin Heatmaps ---
            st.subheader("Horizon–Origin heatmap (HOP)")

            # Median Forecasts
            median_forecasts = filtered_simulations.groupby(["execution_date", "horizon"])["simulated_value"].quantile(0.5).unstack()
            fig = px.imshow(
                median_forecasts.T,#.sort_index(ascending=False),  # Transpose to align axes
                labels={"x": "Execution Date", "y": "Horizon", "color": "Median forecast"},
                title="Horizon–Origin Heatmap: median factor forecasts",
                color_continuous_scale="RdBu",
                origin='lower'
            )
            #fig.update_layout(yaxis_autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            # Prediction Interval Width (90%)
            pi_width = filtered_simulations.groupby(["execution_date", "horizon"])["simulated_value"].quantile(0.95).unstack() - \
                    filtered_simulations.groupby(["execution_date", "horizon"])["simulated_value"].quantile(0.05).unstack()
            fig = px.imshow(
                pi_width.T.sort_index(ascending=False),  # Transpose to align axes
                labels={"x": "Execution Date", "y": "Horizon", "color": "90% PI Width"},
                title="Horizon–Origin heatmap: 90% prediction interval (PI) width",
                color_continuous_scale="RdBu",
                origin='lower'
            )
            st.plotly_chart(fig, use_container_width=True)

            # PIT (Probability Integral Transform)
            simulated_cdf = filtered_simulations.groupby(["execution_date", "horizon"]).apply(
                lambda group: (group["simulated_value"] <= group["forecast_date"].map(actuals["actual"])).mean()
            ).unstack()
            fig = px.imshow(
                simulated_cdf.T.sort_index(ascending=False),  # Transpose to align axes
                labels={"x": "Execution Date", "y": "Horizon", "color": "PIT"},
                title="Horizon–Origin heatmap: probability integral transform (PIT)",
                color_continuous_scale="RdBu",
                origin='lower'
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Visualization 2: Actuals with Fan Chart ---
            st.subheader("Actuals and simulations")

            # Calculate percentiles (5th, 50th, 95th) for the fan chart
            percentiles = filtered_simulations.groupby("execution_date")["simulated_value"].quantile([0.05, 0.5, 0.95]).unstack(level=-1)
            percentiles.columns = ["5th percentile", "median", "95th percentile"]

            # Merge with actuals for the time series plot
            actuals_for_plot = forecasts_data.groupby("forecast_date")["actual"].last().dropna()
            combined_data = percentiles.join(actuals_for_plot, how="inner")

            # Create the fan chart with actuals
            fig = go.Figure()

            # Add fan chart (shaded area for 5th to 95th percentiles)
            fig.add_trace(go.Scatter(
                x=combined_data.index,
                y=combined_data["95th percentile"],
                mode="lines",
                line=dict(width=0),
                name="95th Percentile",
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=combined_data.index,
                y=combined_data["5th percentile"],
                mode="lines",
                fill="tonexty",  # Fill between 5th and 95th percentiles
                fillcolor="rgba(0,100,200,0.2)",
                line=dict(width=0),
                name="5th Percentile",
                showlegend=False
            ))

            # Add median line
            fig.add_trace(go.Scatter(
                x=combined_data.index,
                y=combined_data["median"],
                mode="lines",
                line=dict(color="#3a6bac", width=2),
                name="Median Simulation"
            ))

            # Add actuals line
            fig.add_trace(go.Scatter(
                x=combined_data.index,
                y=combined_data["actual"],
                mode="lines",
                line=dict(color="#c28191", width=2),
                name="Actual"
            ))

            # Update layout
            fig.update_layout(
                #title="Actuals and Simulations with Fan Chart",
                xaxis_title="Execution Date",
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            # --- Visualization: Historical Actuals and Fan Chart ---
            st.subheader("Historical actuals and projections")

            # Determine the available execution dates
            available_execution_dates = sorted(simulations_data["execution_date"].dropna().unique())  # Drop NaN values if any
            available_execution_dates_str = [date.strftime("%Y-%m") for date in available_execution_dates]  # Convert to "YYYY-MM" strings

            if available_execution_dates_str:
                # Add a text input for selecting an ExecutionDate as a string
                default_date = available_execution_dates_str[-1]  # Default to the latest available date
                selected_execution_date_str = st.text_input(
                    "Enter execution date (format: YYYY-MM)",
                    value=default_date  # Default value
                )

                # Validate the input
                try:
                    # Convert the input string back to a pandas Timestamp
                    selected_execution_date = pd.to_datetime(selected_execution_date_str, format="%Y-%m")

                    # Check if the entered date is within the available range
                    if selected_execution_date_str not in available_execution_dates_str:
                        st.warning(f"The entered date {selected_execution_date_str} is not in the available range.")
                    else:
                        # --- Data Preparation ---
                        # Historical actuals
                        historical_actuals = actuals[actuals.index <= selected_execution_date]

                        # Projections
                        simulations_for_execution_date = simulations_data[simulations_data["execution_date"] == selected_execution_date]
                        projections_data = None

                        if not simulations_for_execution_date.empty:
                            # Calculate percentiles (5th, 50th, 95th) for the fan chart
                            percentiles = simulations_for_execution_date.groupby("horizon")["simulated_value"].quantile([0.05, 0.5, 0.95]).unstack(level=-1)

                            if not percentiles.empty:
                                percentiles.columns = ["5th percentile", "median", "95th percentile"]
                                percentiles["forecast_date"] = [selected_execution_date + pd.DateOffset(months=int(h)) for h in percentiles.index]
                                percentiles = percentiles.set_index("forecast_date")
                                projections_data = percentiles.join(actuals, how="left")

                        # --- Create Two Columns for Side-by-Side Graphs ---
                        col1, col2 = st.columns(2)

                        # --- Left Column: Historical Actuals ---
                        with col1:
                            st.subheader("Actual")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=historical_actuals.index,
                                y=historical_actuals["actual"],
                                mode="lines",
                                line=dict(color="#c28191", width=2),
                                name="Actual"
                            ))

                            # Add zero reference line if data includes negative values
                            if historical_actuals["actual"].min() < 0:
                                fig.add_shape(
                                    type="line",
                                    x0=historical_actuals.index.min(),
                                    x1=historical_actuals.index.max(),
                                    y0=0,
                                    y1=0,
                                    line=dict(color="black", width=1),  # Thin black solid line
                                    xref="x",
                                    yref="y"
                                )

                            fig.update_layout(
                                #title="Actual",
                                xaxis_title="Forecast Date",
                                yaxis_title="Value",
                                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # --- Right Column: Projections and Prediction Intervals ---
                        with col2:
                            st.subheader("Projections and prediction intervals vs actual")

                            if projections_data is not None and not projections_data.empty:
                                fig = go.Figure()

                                # Add fan chart (shaded area for 5th to 95th percentiles)
                                fig.add_trace(go.Scatter(
                                    x=projections_data.index,
                                    y=projections_data["95th percentile"],
                                    mode="lines",
                                    line=dict(width=0),
                                    name="95th Percentile",
                                    showlegend=False
                                ))
                                fig.add_trace(go.Scatter(
                                    x=projections_data.index,
                                    y=projections_data["5th percentile"],
                                    mode="lines",
                                    fill="tonexty",  # Fill between 5th and 95th percentiles
                                    fillcolor="rgba(0,100,200,0.2)",
                                    line=dict(width=0),
                                    name="5th Percentile",
                                    showlegend=False
                                ))

                                # Add median line
                                fig.add_trace(go.Scatter(
                                    x=projections_data.index,
                                    y=projections_data["median"],
                                    mode="lines",
                                    line=dict(color="#3a6bac", width=2),
                                    name="Median Simulation"
                                ))

                                # Add actuals line
                                fig.add_trace(go.Scatter(
                                    x=projections_data.index,
                                    y=projections_data["actual"],
                                    mode="lines+markers",
                                    line=dict(color="#c28191", width=2),
                                    name="Actual"
                                ))

                                # Add zero reference line if data includes negative values
                                if projections_data[["5th percentile", "median", "95th percentile"]].min().min() < 0:
                                    fig.add_shape(
                                        type="line",
                                        x0=projections_data.index.min(),
                                        x1=projections_data.index.max(),
                                        y0=0,
                                        y1=0,
                                        line=dict(color="black", width=1),  # Thin black solid line
                                        xref="x",
                                        yref="y"
                                    )

                                fig.update_layout(
                                    #title=f"Projections and Prediction Intervals from Execution Date: {selected_execution_date_str}",
                                    xaxis_title="Forecast Date",
                                    yaxis_title="Value",
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No projections data available for the selected Execution Date.")
                except ValueError:
                    st.error("Invalid date format. Please enter the date in the format YYYY-MM.")
            else:
                st.warning("No execution dates available for the selected filters.")
                
            # --- Visualization 3: Horizon-Specific Simulated Distributions ---
            st.subheader("Horizon-specific simulated distributions vs actual")

            # Use all available forecast dates for the calendar date picker
            available_forecast_dates = sorted(filtered_simulations["forecast_date"].unique())
            if available_forecast_dates:
                # Set default to January 2025 if available, otherwise use the first available date
                default_date = pd.Timestamp("2025-01-01") if pd.Timestamp("2025-01-01") in available_forecast_dates else available_forecast_dates[0]

                # Calendar-based date picker for ForecastDate
                selected_forecast_date = st.date_input(
                    "Select forecast date",
                    value=default_date,  # Default to January 2025 or the first available date
                    min_value=available_forecast_dates[0],
                    max_value=available_forecast_dates[-1],
                    key="simulation_forecast_date_selector"
                )

                # Filter simulations for the selected ForecastDate
                simulations_for_date = filtered_simulations[filtered_simulations["forecast_date"] == pd.to_datetime(selected_forecast_date)]

                # Extract the realized value for the selected ForecastDate
                realized_values_for_date = actuals.loc[actuals.index.get_level_values("forecast_date") == pd.to_datetime(selected_forecast_date)]
                if not realized_values_for_date.empty:
                    realized_value = realized_values_for_date["actual"].iloc[0]  # Take the first realized value (it's the same for all horizons)
                else:
                    realized_value = None  # Handle missing realized values

                if not simulations_for_date.empty:
                    # Allow users to select up to 5 horizons
                    unique_horizons = sorted(simulations_for_date["horizon"].unique())
                    default_horizons = [h for h in [6, 12, 24, 48, 60] if h in unique_horizons]  # Default to specific horizons if available
                    selected_horizons = st.multiselect(
                        "Select horizons to plot (Max 5)",
                        unique_horizons,
                        default=default_horizons,  # Use the default horizons if available
                        key="simulation_horizon_selector"
                    )
                    if len(selected_horizons) > 5:
                        st.warning("Please select up to 5 horizons.")
                    else:
                        # Group simulations by Horizon
                        fig = go.Figure()
                        for horizon in selected_horizons:
                            horizon_data = simulations_for_date[simulations_for_date["horizon"] == horizon]
                            fig.add_trace(go.Histogram(
                                x=horizon_data["simulated_value"],
                                histnorm="probability density",
                                name=f"Horizon {horizon}",
                                opacity=0.7
                            ))
                        # Add realized value as a vertical dashed line
                        if realized_value is not None:
                            fig.add_vline(
                                x=realized_value,
                                line_dash="dash",
                                line_color="black",
                                annotation_text="Realized Value",
                                annotation_position="top right"
                            )
                        # Update layout
                        fig.update_layout(
                            title=f"Simulated Distributions vs Actuals for {selected_forecast_date}",
                            xaxis_title="Simulated Value",
                            yaxis_title="Density",
                            barmode="overlay",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for the selected Forecast Date.")
            else:
                st.warning("No forecast dates available for the selected filters.")
        else:
            st.warning("No data available for the selected model and beta.")

elif selected_macro == "Yields":
    tab_names = [
        "Backtesting overview",
        "Out-of-sample metrics",
        "Simulations analysis"
    ]
    tabs = st.tabs(tab_names)
    tab_yields_1, tab_yields, tab_sim_yields = tabs

    with tab_yields_1:
        st.title("Actuals vs predictions for yields")

        # Dynamically update the folder path for yields based on the selected country
        yields_folder_path = os.path.join(base_folder, selected_country, "yields")

        # Define paths for estimated and observed yields
        estimated_yields_folder = os.path.join(yields_folder_path, "estimated_yields")
        observed_yields_folder = os.path.join(yields_folder_path, "observed_yields")

        # Get available models for estimated yields
        estimated_models = [f.name for f in os.scandir(estimated_yields_folder) if f.is_dir()]
        benchmark_model = "AR_1"  # Observed yields benchmark model
        # --- TEMPORARY UK BENCHMARK FIX ---
        if selected_country == "UK":
            # Use estimated AR_1 as benchmark for UK
            sample_file_path = os.path.join(estimated_yields_folder, benchmark_model, "forecasts.csv")
            benchmark_is_estimated = True
        else:
            sample_file_path = os.path.join(observed_yields_folder, benchmark_model, "forecasts.csv")
            benchmark_is_estimated = False

        if os.path.exists(sample_file_path):
            sample_data = pd.read_csv(sample_file_path)
            #sample_data = sample_data.rename(columns={"execution_date": "ExecutionDate", "forecasted_date": "ForecastDate"})
            sample_data["maturity"] = sample_data["maturity"].astype(str)
            available_maturities = sample_data["maturity"].str.split(' ', expand = True)[0].unique()
        else:
            st.warning("Sample file not found. Cannot determine available maturities.")
            available_maturities = []

        # Dropdown for selecting models
        #st.subheader("Model Selection")
        
        col_1, col_2 = st.columns(2)
        with col_1:
            estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]
            selected_models_yields = st.multiselect(
                "Select models (Max 3)",
                estimated_models_with_labels,
                default=estimated_models[:1],  # Default to the first estimated model
                key="model_selector_yields"
            )

            # Restrict to a maximum of 3 models
            if len(selected_models_yields) > 3:
                st.warning("Please select up to 3 models for Yields. Only the first 3 models will be used.")
                selected_models_yields = selected_models_yields[:3]  # Limit to the first 3 models

        with col_2:
            # Dropdown for selecting a single maturity
            #st.subheader("Maturity Selection")
            if available_maturities.size > 0:
                selected_maturity = st.selectbox(
                    "Select a Maturity to Display",
                    available_maturities.tolist(),
                    key="selected_maturity_yields"
                )
            else:
                st.warning("No maturities available. Please check the data.")
                selected_maturity = None

        # Date selectors for yields (side by side)
        #st.subheader("Filter by Date Range")
        min_date_yields = datetime.strptime("1950-01-01", "%Y-%m-%d").date()
        max_date_yields = datetime.strptime("2030-12-31", "%Y-%m-%d").date()

        # Place start and end date inputs side by side
        date_cols_yields = st.columns(2)
        with date_cols_yields[0]:
            start_date_yields = st.date_input(
                "Start date",
                value=min_date_yields,
                min_value=min_date_yields,
                max_value=max_date_yields,
                key="start_date_input_yields"
            )
        with date_cols_yields[1]:
            end_date_yields = st.date_input(
                "End date",
                value=max_date_yields,
                min_value=min_date_yields,
                max_value=max_date_yields,
                key="end_date_input_yields"
            )

        # Ensure the start date is not after the end date
        if start_date_yields > end_date_yields:
            st.warning("Start date cannot be after the end date for Yields. Please adjust the dates.")
        elif selected_maturity is None:
            st.warning("Please select a maturity to proceed.")
        else:
            combined_data_yields = []

            # Loop through selected models and load data
            for model in selected_models_yields:
                # Determine the folder path (estimated or observed)
                if model == f"{benchmark_model} (Benchmark)":
                    subfolder = observed_yields_folder
                    model_name = benchmark_model
                else:
                    subfolder = estimated_yields_folder
                    model_name = model

                # Construct the path to the forecasts.csv file
                forecasts_file_path_yields = os.path.join(subfolder, model_name, "forecasts.csv")
                if os.path.exists(forecasts_file_path_yields):
                    # Load the forecasts.csv file
                    data_yields = load_data(yields_folder_path, subfolder, model_name, "forecasts.csv")
                    if 'mean_simulated' in data_yields.columns:
                        data_yields = data_yields.rename(columns={"mean_simulated": "prediction"})
                        data_yields['maturity'] = data_yields['maturity'].astype(str)
                        
                    if subfolder == observed_yields_folder:
                        data_yields['maturity'] = data_yields['maturity'].str.split(' ', expand = True)[0]  # Keep only the numeric part of the maturity
                        data_yields = data_yields.rename(columns={"forecasted_date": "forecast_date"})
                        
                    # Ensure the required columns exist
                    required_columns_yields = ["execution_date", "forecast_date", "prediction", "actual", "maturity"]
                    if all(col in data_yields.columns for col in required_columns_yields):
                        # Convert ExecutionDate to datetime if not already in datetime format
                        if not pd.api.types.is_datetime64_any_dtype(data_yields["execution_date"]):
                            data_yields["execution_date"] = pd.to_datetime(data_yields["execution_date"])

                        # Filter data using the selected date range and maturity
                        filtered_data_yields = data_yields[
                            (data_yields["execution_date"].dt.date >= start_date_yields) &
                            (data_yields["execution_date"].dt.date <= end_date_yields) &
                            (data_yields["maturity"] == selected_maturity)
                        ]

                        if not filtered_data_yields.empty:
                            # Sort data by ExecutionDate and ForecastDate
                            filtered_data_yields = filtered_data_yields.sort_values(by=["execution_date", "forecast_date"])
                            filtered_data_yields["model"] = model
                            if model == f"{benchmark_model} (Benchmark)":
                                filtered_data_yields['prediction'] = filtered_data_yields['prediction']/100
                                filtered_data_yields['actual'] = filtered_data_yields['actual']/100
                            combined_data_yields.append(filtered_data_yields)
                    else:
                        st.warning(f"Required columns are missing in the forecasts data for model: {model}, yields.")
                else:
                    st.warning(f"File not found: {forecasts_file_path_yields}")

            # Combine data from all models
            if combined_data_yields:
                combined_data_yields = pd.concat(combined_data_yields, ignore_index=True)

                # Prepare the actuals data
                realized_yields = combined_data_yields.groupby("forecast_date")["actual"].mean()

                # Create the Plotly figure
                fig_yields = go.Figure()

                # Define colors based on the number of selected models
                if len(selected_models_yields) == 1:
                    model_colors_yields = ["#c28191"]  # Slightly darker reddish-pink
                elif len(selected_models_yields) == 2:
                    model_colors_yields = ["#c28191", "#3a6bac"]  # Reddish-pink and blue
                else:  # len(selected_models_yields) == 3
                    model_colors_yields = ["#c28191", "#3a6bac", "#eaa121"]  # Reddish-pink, blue, and yellow-orange

                model_color_mapping_yields = {model: model_colors_yields[i] for i, model in enumerate(selected_models_yields)}

                for model in selected_models_yields:
                    model_data_yields = combined_data_yields[combined_data_yields["model"] == model]
                    unique_execution_dates_yields = model_data_yields["execution_date"].unique()
                    
                    # Add traces for each execution date, but hide them from the legend
                    for execution_date in unique_execution_dates_yields:
                        subset_yields = model_data_yields[model_data_yields["execution_date"] == execution_date]
                        fig_yields.add_trace(go.Scatter(
                            x=subset_yields["forecast_date"],
                            y=subset_yields["prediction"],
                            mode="lines",
                            line=dict(color=model_color_mapping_yields[model], width=1),
                            opacity=0.5,
                            name=None,  # Do not show individual traces in the legend
                            showlegend=False  # Ensure these traces are not in the legend
                        ))

                    # Add a single legend entry for the model
                    fig_yields.add_trace(go.Scatter(
                        x=[None],  # Dummy data
                        y=[None],  # Dummy data
                        mode="lines",
                        line=dict(color=model_color_mapping_yields[model], width=1),
                        name=model,  # Legend entry for the model
                        showlegend=True  # Ensure this trace appears in the legend
                    ))

                # Add the actuals line (black)
                fig_yields.add_trace(go.Scatter(
                    x=realized_yields.index,
                    y=realized_yields.values,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="Actual"
                ))

                # Update layout
                fig_yields.update_layout(
                    #title=f"Yields: Forecasted vs Actual (Maturity: {selected_maturity})",
                    xaxis_title="Forecast Date",
                    yaxis_title="Yields",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    template="plotly_white"
                )

                # Display the plot
                st.plotly_chart(fig_yields, use_container_width=True)
            else:
                st.warning(f"No data available for Yields (Maturity: {selected_maturity}).")

    #elif selected_tab == "Yields out-of-sample metrics":
    with tab_yields:
        st.title("Yields Out-of-Sample Metrics")

        # Model and maturity filters
        filter_cols = st.columns(2)
        with filter_cols[0]:
            estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]
            selected_models = st.multiselect(
                "Select models",
                estimated_models_with_labels,
                default=estimated_models[:1],
                key="yields_oos_models_selector"
            )
            include_benchmark = f"{benchmark_model} (Benchmark)" in selected_models

        with filter_cols[1]:
            # Convert available_maturities to float and filter out invalids
            available_maturities_float = [float(m) for m in available_maturities if str(m) not in ["nan", "None"]]
            default_maturities = [0.25, 2.0, 5.0, 10.0]
            default_selected = [m for m in default_maturities if m in available_maturities_float]
            selected_maturities = st.multiselect(
                "Select maturities to plot (Max 4)",
                available_maturities_float,
                default=default_selected,
                key="yields_oos_maturities_selector"
            )
            if len(selected_maturities) > 4:
                st.warning("Please select up to 4 maturities. Only the first 4 will be used.")
                selected_maturities = selected_maturities[:4]

        if not selected_maturities:
            st.warning("No valid maturities available for plotting.")
        else:
            # --- First Row: RMSE by Horizon (Full Sample) ---
            st.subheader("RMSE by horizon (full sample)")
            cols = st.columns(4)
            for idx, maturity in enumerate(selected_maturities):
                with cols[idx]:
                    horizon_data = load_and_cache_data(
                        selected_models, yields_folder_path, include_benchmark, "outofsample_metrics_by_horizon.csv"
                    )
                    combined_data = []
                    for model, data in horizon_data.items():
                        filtered_data = data[data["maturity"].astype(float) == maturity]
                        filtered_data = filtered_data[filtered_data["horizon"] != 0]
                        if not filtered_data.empty:
                            if model == f"{benchmark_model} (Benchmark)":
                                filtered_data["rmse"] = filtered_data["rmse"] / 100
                            filtered_data["model"] = model
                            combined_data.append(filtered_data)
                    if combined_data:
                        combined_data = pd.concat(combined_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="horizon",
                            y="rmse",
                            color="model",
                            title=f"{maturity}y",
                            labels={"rmse": "RMSE", "horizon": "Horizon"},
                            markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data for {maturity}y.")

            # --- Second Row: RMSE by Execution Date (Full Sample) ---
            st.subheader("RMSE by execution date (full sample)")
            cols = st.columns(4)
            for idx, maturity in enumerate(selected_maturities):
                with cols[idx]:
                    execution_data = load_and_cache_data(
                        selected_models, yields_folder_path, include_benchmark, "outofsample_metrics_by_execution_date.csv"
                    )
                    combined_data = []
                    for model, data in execution_data.items():
                        filtered_data = data[data["maturity"].astype(float) == maturity]
                        if not filtered_data.empty:
                            if model == f"{benchmark_model} (Benchmark)":
                                filtered_data["rmse"] = filtered_data["rmse"] / 100
                            filtered_data["model"] = model
                            combined_data.append(filtered_data)
                    if combined_data:
                        combined_data = pd.concat(combined_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="execution_date",
                            y="rmse",
                            color="model",
                            title=f"{maturity}y",
                            labels={"rmse": "RMSE", "execution_date": "Execution Date"},
                            #markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data for {maturity}y.")

            # --- Third Row: Dynamic RMSE by Horizon (Date Range) ---
            st.subheader("Dynamic RMSE by horizon (select date range)")
            # Get all dates from observed yields for date pickers
            if os.path.exists(sample_file_path):
                sample_data = pd.read_csv(sample_file_path)
                all_dates = pd.to_datetime(sample_data["execution_date"]).tolist()
            else:
                all_dates = []
            min_date = min(all_dates) if all_dates else pd.to_datetime("1950-01-01")
            max_date = max(all_dates) if all_dates else pd.to_datetime("2025-12-31")
            date_cols = st.columns(2)
            with date_cols[0]:
                dynamic_start_date = st.date_input(
                    "Start date (dynamic horizon)",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="dynamic_start_date_horizon_yields"
                )
            with date_cols[1]:
                dynamic_end_date = st.date_input(
                    "End date (dynamic horizon)",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="dynamic_end_date_horizon_yields"
                )
            cols = st.columns(4)
            for idx, maturity in enumerate(selected_maturities):
                with cols[idx]:
                    forecasts_data = load_and_cache_data(
                        selected_models, yields_folder_path, include_benchmark, "forecasts.csv"
                    )
                    combined_data = []
                    for model, data in forecasts_data.items():
                        filtered_data = data[
                            (data["maturity"].astype(float) == maturity) &
                            (pd.to_datetime(data["execution_date"]) >= pd.to_datetime(dynamic_start_date)) &
                            (pd.to_datetime(data["execution_date"]) <= pd.to_datetime(dynamic_end_date))
                        ]
                        if not filtered_data.empty and "prediction" in filtered_data.columns and "actual" in filtered_data.columns:
                            rmse_data = calculate_rmse(filtered_data, dynamic_start_date, dynamic_end_date)
                            if model == f"{benchmark_model} (Benchmark)":
                                rmse_data["rmse"] = rmse_data["rmse"] / 100
                            rmse_data["model"] = model
                            combined_data.append(rmse_data)
                    if combined_data:
                        combined_data = pd.concat(combined_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="horizon",
                            y="rmse",
                            color="model",
                            title=f"{maturity}y",
                            labels={"horizon": "Horizon", "rmse": "RMSE"},
                            markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No dynamic data for {maturity}y.")

            # --- Fourth Row: Dynamic RMSE by Execution Date (Date Range) ---
            st.subheader("Dynamic RMSE by execution date (select date range)")
            cols = st.columns(4)
            for idx, maturity in enumerate(selected_maturities):
                with cols[idx]:
                    forecasts_data = load_and_cache_data(
                        selected_models, yields_folder_path, include_benchmark, "forecasts.csv"
                    )
                    combined_data = []
                    for model, data in forecasts_data.items():
                        filtered_data = data[
                            (data["maturity"].astype(float) == maturity) &
                            (pd.to_datetime(data["execution_date"]) >= pd.to_datetime(dynamic_start_date)) &
                            (pd.to_datetime(data["execution_date"]) <= pd.to_datetime(dynamic_end_date))
                        ]
                        if not filtered_data.empty and "prediction" in filtered_data.columns and "actual" in filtered_data.columns:
                            rmse_data = (
                                filtered_data.groupby("execution_date")
                                .apply(lambda x: np.sqrt(((x["prediction"] - x["actual"]) ** 2).mean()))
                                .reset_index(name="rmse")
                            )
                            if model == f"{benchmark_model} (Benchmark)":
                                rmse_data["rmse"] = rmse_data["rmse"] / 100
                            rmse_data["model"] = model
                            combined_data.append(rmse_data)
                    if combined_data:
                        combined_data = pd.concat(combined_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="execution_date",
                            y="rmse",
                            color="model",
                            title=f"{maturity}y",
                            labels={"execution_date": "Execution Date", "rmse": "RMSE"},
                            #markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No dynamic data for {maturity}y.")

    #elif selected_tab == "Yields simulations analysis":
    with tab_sim_yields:
        st.title("Yields simulations analysis: heatmaps and fan chart")
        # Dynamically update the folder path for yields based on the selected country
        yields_folder_path = os.path.join(base_folder, selected_country, "yields")

        # Define paths for estimated and observed yields
        estimated_yields_folder = os.path.join(yields_folder_path, "estimated_yields")
        observed_yields_folder = os.path.join(yields_folder_path, "observed_yields")

        # Get available models for estimated yields
        estimated_models = [f.name for f in os.scandir(estimated_yields_folder) if f.is_dir()]
        benchmark_model = "AR_1"  # Observed yields benchmark model
        # --- TEMPORARY UK BENCHMARK FIX ---
        if selected_country == "UK":
            # Use estimated AR_1 as benchmark for UK
            sample_file_path_yields = os.path.join(estimated_yields_folder, benchmark_model, "forecasts.csv")
            benchmark_is_estimated = True
        else:
            sample_file_path_yields = os.path.join(observed_yields_folder, benchmark_model, "forecasts.csv")
            benchmark_is_estimated = False

        if os.path.exists(sample_file_path_yields):
            sample_data_yields = pd.read_csv(sample_file_path_yields)
            sample_data_yields["maturity"] = sample_data_yields["maturity"].astype(str)
            available_maturities = sample_data_yields["maturity"].str.split(' ', expand=True)[0].unique()
        else:
            available_maturities = np.array([])

        filter_cols1, filter_cols2 = st.columns(2)
        with filter_cols1:
            # Dropdown for selecting model
            #st.subheader("Model Selection")
            estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]
            selected_model = st.selectbox(
                "Select model",
                estimated_models_with_labels,
                key="sim_yields_model_selector"  # Unique key for this dropdown
            )
        with filter_cols2:
            # Dropdown for selecting maturity
            #st.subheader("Maturity Selection")
            if available_maturities.size > 0:
                selected_maturity = st.selectbox(
                    "Select maturity",
                    available_maturities.tolist(),
                    key="sim_yields_maturity_selector"  # Unique key for this dropdown
                )
            else:
                st.warning("No maturities available for simulations.")
                selected_maturity = None

        # Load and concatenate all parquet files for the selected maturity
        if selected_model and selected_maturity:

            yields_folder_path = os.path.join(base_folder, selected_country, "yields")

            # Adjust folder for benchmark model
            model_folder = benchmark_model if selected_model == f"{benchmark_model} (Benchmark)" else selected_model
            maturity_folder = selected_maturity + "_years"  # Convert maturity to folder name format
            folder_path = os.path.join(yields_folder_path, "estimated_yields", model_folder, "simulations", maturity_folder)

            if os.path.exists(folder_path):
                # Concatenate all parquet files in the folder
                all_parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
                if all_parquet_files:
                    simulations_data = pd.concat([pd.read_parquet(file) for file in all_parquet_files], ignore_index=True)

                    # Ensure the data contains the expected columns
                    required_columns = ["forecast_date", "simulated_value", "maturity", "simulation_id", "execution_date", "model", "horizon"]
                    if all(col in simulations_data.columns for col in required_columns):
                        # --- Visualization 1: Full-Sample Horizon-Origin Heatmap (Median Forecasts) ---
                        st.subheader("Horizon-Origin heatmap for median forecast")
                        median_forecasts = simulations_data.groupby(["execution_date", "horizon"])["simulated_value"].quantile(0.5).unstack()
                        fig = px.imshow(
                            median_forecasts.T,  # Transpose to align axes
                            labels={"x": "Execution Date", "y": "Horizon", "color": "Median Simulated Yield"},
                            #title="Full-Sample Horizon-Origin Heatmap: Median Simulated Yields",
                            color_continuous_scale="RdBu",
                            origin="lower"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- Visualization 2: Full-Sample 90% Prediction Interval Width Heatmap ---
                        st.subheader("Horizon-Origin heatmap for 90% prediction interval width")
                        pi_width = simulations_data.groupby(["execution_date", "horizon"])["simulated_value"].quantile(0.95).unstack() - \
                                simulations_data.groupby(["execution_date", "horizon"])["simulated_value"].quantile(0.05).unstack()
                        fig = px.imshow(
                            pi_width.T,  # Transpose to align axes
                            labels={"x": "Execution Date", "y": "Horizon", "color": "90% PI Width"},
                            #title="Full-Sample Horizon-Origin Heatmap: 90% Prediction Interval Width",
                            color_continuous_scale="RdBu",
                            origin="lower"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- Visualization 3: Full-Sample PIT Heatmap ---
                        st.subheader("Horizon-Origin heatmap for probability integral transform (PIT)")
                        # Mock actuals for PIT (replace this with real actuals if available)
                        actuals = simulations_data.groupby("forecast_date")["simulated_value"].mean()
                        simulated_cdf = simulations_data.groupby(["execution_date", "horizon"]).apply(
                            lambda group: (group["simulated_value"] <= actuals.get(group["forecast_date"].iloc[0], 0)).mean()
                        ).unstack()
                        fig = px.imshow(
                            simulated_cdf.T,  # Transpose to align axes
                            labels={"x": "Execution Date", "y": "Horizon", "color": "PIT"},
                            #title="Full-Sample Horizon-Origin Heatmap: PIT",
                            color_continuous_scale="RdBu",
                            origin="lower"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- Visualization 4: Actuals and Simulations with Fan Chart ---
                        st.subheader("Actuals and simulations with fan chart")

                        # Mock actuals data (replace with real actuals if available)
                        actuals = simulations_data.groupby("forecast_date")["simulated_value"].mean()  # Mocked actuals
                        forecasts_data = simulations_data.copy()  # Replace with actual forecast data if available

                        # Calculate percentiles (5th, 50th, 95th) for the fan chart
                        percentiles = simulations_data.groupby("execution_date")["simulated_value"].quantile([0.05, 0.5, 0.95]).unstack(level=-1)
                        percentiles.columns = ["5th percentile", "median", "95th percentile"]

                        # Merge with actuals for the time series plot
                        actuals_for_plot = forecasts_data.groupby("forecast_date")["simulated_value"].last().dropna()
                        combined_data = percentiles.join(actuals_for_plot, how="inner")

                        # Create the fan chart with actuals
                        fig = go.Figure()

                        # Add fan chart (shaded area for 5th to 95th percentiles)
                        fig.add_trace(go.Scatter(
                            x=combined_data.index,
                            y=combined_data["95th percentile"],
                            mode="lines",
                            line=dict(width=0),
                            name="95th Percentile",
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=combined_data.index,
                            y=combined_data["5th percentile"],
                            mode="lines",
                            fill="tonexty",  # Fill between 5th and 95th percentiles
                            fillcolor="rgba(0,100,200,0.2)",
                            line=dict(width=0),
                            name="5th Percentile",
                            showlegend=False
                        ))

                        # Add median line
                        fig.add_trace(go.Scatter(
                            x=combined_data.index,
                            y=combined_data["median"],
                            mode="lines",
                            line=dict(color="#3a6bac", width=2),
                            name="Median Simulation"
                        ))

                        # Add actuals line
                        fig.add_trace(go.Scatter(
                            x=combined_data.index,
                            y=combined_data["simulated_value"],
                            mode="lines",
                            line=dict(color="#c28191", width=2),
                            name="Actual"
                        ))

                        # Update layout
                        fig.update_layout(
                            #title="Actuals and Simulations with Fan Chart",
                            xaxis_title="Execution Date",
                            yaxis_title="Value",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("The simulations data does not contain the expected columns.")
                else:
                    st.warning(f"No parquet files found in the folder: {folder_path}")
            else:
                st.warning(f"Folder does not exist: {folder_path}")
        else:
            st.warning("Please select a model and maturity to proceed.")

elif selected_macro == "Returns":
    tab_names = [
        "Backtesting overview",
        "Out-of-sample metrics",
        "Forward looking distributions",
        "CRPS analysis"  
    ]
    tabs = st.tabs(tab_names)
    tab_returns, tab_returns_out_of_sample, tab_simulation_comparison, tab_crps = tabs

    with tab_returns:
        st.title("Returns analysis: VaR, CVaR, observed and expected returns")
        
        # Get available models for estimated returns
        estimated_returns_folder = os.path.join(base_folder, selected_country, "returns", "estimated_returns")
        estimated_models = [f.name for f in os.scandir(estimated_returns_folder) if f.is_dir()]
        benchmark_model = "AR_1"
        estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]

        # For returns tabs
        returns_observed_folder = os.path.join(base_folder, selected_country, "returns", "observed_returns", benchmark_model, "annual")
        sample_file_path = os.path.join(returns_observed_folder, "forecasts.csv")
        if os.path.exists(sample_file_path):
            sample_data = pd.read_csv(sample_file_path)
            available_maturities = sample_data["maturity"].unique()
        else:
            available_maturities = np.array([])  # or []

        returns_filters = st.columns(4)
        with returns_filters[0]:
            selected_model = st.selectbox(
                "Select Model",
                estimated_models_with_labels,
                key="returns_model_selector"
            )
        with returns_filters[1]:
            selected_frequency = st.selectbox(
                "Select Frequency",
                ["monthly", "annual"],
                key="returns_frequency_selector"
            )

        # Adjust folder path for benchmark
        if selected_model == f"{benchmark_model} (Benchmark)":
            metrics_folder = os.path.join(base_folder, selected_country, "returns", "observed_returns", benchmark_model, selected_frequency)
            is_benchmark = True
        else:
            metrics_folder = os.path.join(base_folder, selected_country, "returns", "estimated_returns", selected_model, selected_frequency)
            is_benchmark = False

        metrics_file = os.path.join(metrics_folder, "risk_metrics.csv")
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            metrics_df = metrics_df[metrics_df["metric"] != "metric"]
            metrics_df['execution_date'] = pd.to_datetime(metrics_df['execution_date'])
            metrics_df['value']  = pd.to_numeric(metrics_df['value'])
            metrics_df["horizon"] = metrics_df["horizon"].astype(float)
            
            if 'maturity_years' in metrics_df.columns:
                metrics_df["maturity_years"] = metrics_df["maturity_years"].astype(float)
                metrics_df = metrics_df.rename(columns={'maturity_years':'maturity'})
            with returns_filters[2]:
                available_maturities = metrics_df["maturity"].unique()
                available_maturities.sort()
                selected_maturity = st.selectbox(
                    "Select maturity (years)",
                    sorted(available_maturities),
                    key="returns_maturity_selector"
                )
            with returns_filters[3]:
                available_horizons = metrics_df["horizon"].unique()
                available_horizons.sort()
                selected_horizon = st.selectbox(
                    "Select horizon",
                    sorted(available_horizons),
                    key="returns_horizon_selector"
                )
                var_options = ["VaR 95", "VaR 97", "VaR 99"]
                selected_var = st.selectbox(
                    "Select VaR metric",
                    var_options,
                    index=1,
                    key="returns_var_selector"
                )

            observed_metric = "observed_return"
            expected_metric = "expected_return"

            filtered_df = metrics_df[
                (metrics_df["maturity"] == selected_maturity) &
                (metrics_df["horizon"] == selected_horizon) &
                (metrics_df["metric"].isin([
                    selected_var,
                    "CVaR " + selected_var.split(" ")[1],
                    observed_metric, expected_metric, "volatility"
                ]))
            ]

            filtered_df = (
                filtered_df.groupby(["execution_date", "metric"], as_index=False)
                .agg({"value": "last"})
            )
            pivot_df = filtered_df.pivot(index="execution_date", columns="metric", values="value").sort_index()
            # Pivot the data so each metric is a column
            #pivot_df = filtered_df.pivot(index="execution_date", columns="metric", values="value").sort_index()

            # Plot the metrics
            fig = go.Figure()

            # Observed
            if observed_metric in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[observed_metric],
                    mode="lines",
                    name=observed_metric,
                    line=dict(color="#3a6bac"),
                    marker=dict(size=6)
                ))

            # Expected
            if expected_metric in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[expected_metric],
                    mode="lines",
                    name=expected_metric,
                    line=dict(color="#eaa121"),
                    marker=dict(size=6)
                ))

            # VaR
            if selected_var in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[selected_var],
                    mode="lines",
                    name=selected_var,
                    line=dict(color="#c28191")
                ))

            # CVaR
            cvar_metric = "CVaR " + selected_var.split(" ")[1]
            if cvar_metric in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[cvar_metric],
                    mode="lines",
                    name=cvar_metric,
                    line=dict(color="#aa322f")
                ))

            # Highlight breaches where Observed < VaR
            if observed_metric in pivot_df.columns and selected_var in pivot_df.columns:
                breaches = pivot_df[observed_metric] < pivot_df[selected_var]
                fig.add_trace(go.Scatter(
                    x=pivot_df.index[breaches],
                    y=pivot_df[observed_metric][breaches],
                    mode="markers",
                    name="VaR Breaches",
                    marker=dict(color="black", size=8, symbol="x")
                ))

            fig.update_layout(
                xaxis_title="Execution Date",
                yaxis_title="Returns",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Volatility Plot ---
            if "volatility" in pivot_df.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df["volatility"],
                    mode="lines",
                    name="Volatility",
                    line=dict(color="gray", dash="dot")
                ))
                fig_vol.update_layout(
                    title="Volatility",
                    xaxis_title="Execution Date",
                    yaxis_title="Volatility",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    template="plotly_white"
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        else:
            st.warning(f"File not found: {metrics_file}")


    with tab_returns_out_of_sample:
        st.title("Returns Out-of-Sample Metrics")

        # Frequency filter
        selected_frequency = st.radio(
            "Select frequency",
            ["annual", "monthly"],
            index=0,
            horizontal=True,
            key="returns_oos_frequency_selector"
        )

        # Get available models for estimated returns
        estimated_returns_folder = os.path.join(base_folder, selected_country, "returns", "estimated_returns")
        estimated_models = [f.name for f in os.scandir(estimated_returns_folder) if f.is_dir()]
        benchmark_model = "AR_1"
        estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]

        # Get available maturities from observed_returns AR_1
        returns_observed_folder = os.path.join(base_folder, selected_country, "returns", "observed_returns", benchmark_model, selected_frequency)
        sample_file_path = os.path.join(returns_observed_folder, "forecasts.csv")
        if os.path.exists(sample_file_path):
            sample_data = pd.read_csv(sample_file_path)
            # Extract numeric part and convert to float for selection/filtering
            available_maturities = sample_data["maturity"].unique()
            available_maturities_float = []
            for m in available_maturities:
                try:
                    available_maturities_float.append(float(str(m).split()[0]))
                except Exception:
                    pass
        else:
            available_maturities = np.array([])
            available_maturities_float = []

        # Filters
        filter_cols = st.columns(2)
        with filter_cols[0]:
            selected_models = st.multiselect(
                "Select models",
                estimated_models_with_labels,
                default=estimated_models[:1],
                key="returns_oos_models_selector"
            )
            include_benchmark = f"{benchmark_model} (Benchmark)" in selected_models

        with filter_cols[1]:
            default_maturities = [0.25, 2.0, 5.0, 10.0]
            default_selected = [m for m in default_maturities if m in available_maturities_float]
            selected_maturities = st.multiselect(
                "Select maturities to plot (Max 4)",
                available_maturities_float,
                default=default_selected,
                key="returns_oos_maturities_selector"
            )
            if len(selected_maturities) > 4:
                st.warning("Please select up to 4 maturities. Only the first 4 will be used.")
                selected_maturities = selected_maturities[:4]


        if not selected_maturities:
            st.warning("No valid maturities available for plotting.")
        else:
            # --- First Row: RMSE by Execution Date (left) and RMSE by Horizon (right) ---
            st.subheader("RMSE by execution date and by horizon (full sample)")
            cols = st.columns(4)
            for idx, maturity in enumerate(selected_maturities):
                with cols[idx]:
                    st.markdown(f"**{maturity}y**")
                    # RMSE by Horizon
                    horizon_data = []
                    for model in selected_models:
                        if model == f"{benchmark_model} (Benchmark)":
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "observed_returns", benchmark_model, selected_frequency
                            )
                        else:
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "estimated_returns", model, selected_frequency
                            )
                        horizon_file = os.path.join(metrics_folder, "outofsample_metrics_by_horizon.csv")
                        if os.path.exists(horizon_file):
                            horizon_df = pd.read_csv(horizon_file)
                            filtered_df = horizon_df[horizon_df["maturity"].apply(lambda x: float(str(x).split()[0])) == maturity].copy()
                            filtered_df["model"] = model
                            horizon_data.append(filtered_df)
                    if horizon_data:
                        combined_data = pd.concat(horizon_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="horizon",
                            y="rmse",
                            color="model",
                            title="RMSE by horizon",
                            labels={"rmse": "RMSE", "horizon": "Horizon"},
                            markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data for RMSE by Horizon for {maturity}y.")

                    # RMSE by Execution Date
                    execution_data = []
                    for model in selected_models:
                        if model == f"{benchmark_model} (Benchmark)":
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "observed_returns", benchmark_model, selected_frequency
                            )
                        else:
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "estimated_returns", model, selected_frequency
                            )
                        exec_file = os.path.join(metrics_folder, "outofsample_metrics_by_execution_date.csv")
                        if os.path.exists(exec_file):
                            exec_df = pd.read_csv(exec_file)
                            filtered_df = exec_df[exec_df["maturity"].apply(lambda x: float(str(x).split()[0])) == maturity].copy()
                            filtered_df["model"] = model
                            execution_data.append(filtered_df)
                    if execution_data:
                        combined_data = pd.concat(execution_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="execution_date",
                            y="rmse",
                            color="model",
                            title="RMSE by execution date",
                            labels={"rmse": "RMSE", "execution_date": "Execution Date"},
                            #markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data for RMSE by Execution Date for {maturity}y.")

            # --- Second Row: Dynamic RMSE by Execution Date and Horizon (Custom Date Range) ---
            st.subheader("Dynamic RMSE by execution date and by horizon (select date range)")
            # Find min/max date from available data
            min_date = None
            max_date = None
            for model in selected_models:
                if model == f"{benchmark_model} (Benchmark)":
                    metrics_folder = os.path.join(
                        base_folder, selected_country, "returns", "observed_returns", benchmark_model, selected_frequency
                    )
                else:
                    metrics_folder = os.path.join(
                        base_folder, selected_country, "returns", "estimated_returns", model, selected_frequency
                    )
                forecasts_file = os.path.join(metrics_folder, "forecasts.csv")
                if os.path.exists(forecasts_file):
                    df = pd.read_csv(forecasts_file)
                    if min_date is None or pd.to_datetime(df["execution_date"]).min() < min_date:
                        min_date = pd.to_datetime(df["execution_date"]).min()
                    if max_date is None or pd.to_datetime(df["execution_date"]).max() > max_date:
                        max_date = pd.to_datetime(df["execution_date"]).max()
            if min_date is None:
                min_date = pd.to_datetime("2000-01-01")
            if max_date is None:
                max_date = pd.to_datetime("2025-12-31")

            date_cols = st.columns(2)
            with date_cols[0]:
                dynamic_start_date = st.date_input(
                    "Select start date (dynamic)",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="returns_dynamic_start_date"
                )
            with date_cols[1]:
                dynamic_end_date = st.date_input(
                    "Select end date (dynamic)",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="returns_dynamic_end_date"
                )

            cols = st.columns(4)
            for idx, maturity in enumerate(selected_maturities):
                with cols[idx]:
                    st.markdown(f"**{maturity}y**")
                    # Dynamic RMSE by Horizon
                    combined_data = []
                    for model in selected_models:
                        if model == f"{benchmark_model} (Benchmark)":
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "observed_returns", benchmark_model, selected_frequency
                            )
                        else:
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "estimated_returns", model, selected_frequency
                            )
                        forecasts_file = os.path.join(metrics_folder, "forecasts.csv")
                        if os.path.exists(forecasts_file):
                            df = pd.read_csv(forecasts_file)
                            filtered = df[
                                (df["maturity"].apply(lambda x: float(str(x).split()[0])) == maturity) &
                                (pd.to_datetime(df["execution_date"]) >= pd.to_datetime(dynamic_start_date)) &
                                (pd.to_datetime(df["execution_date"]) <= pd.to_datetime(dynamic_end_date))
                            ]
                            if not filtered.empty and "prediction" in filtered.columns and "actual" in filtered.columns:
                                # Use your calculate_rmse function if available
                                rmse_data = (
                                    filtered.groupby("horizon")
                                    .apply(lambda x: np.sqrt(((x["prediction"] - x["actual"]) ** 2).mean()))
                                    .reset_index(name="rmse")
                                )
                                rmse_data["model"] = model
                                combined_data.append(rmse_data)
                    if combined_data:
                        combined_data = pd.concat(combined_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="horizon",
                            y="rmse",
                            color="model",
                            title="Dynamic RMSE by horizon",
                            labels={"horizon": "Horizon", "rmse": "RMSE"},
                            markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data for dynamic RMSE by horizon for {maturity}y.")

                    # Dynamic RMSE by Execution Date
                    combined_data = []
                    for model in selected_models:
                        if model == f"{benchmark_model} (Benchmark)":
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "observed_returns", benchmark_model, selected_frequency
                            )
                        else:
                            metrics_folder = os.path.join(
                                base_folder, selected_country, "returns", "estimated_returns", model, selected_frequency
                            )
                        forecasts_file = os.path.join(metrics_folder, "forecasts.csv")
                        if os.path.exists(forecasts_file):
                            df = pd.read_csv(forecasts_file)
                            filtered = df[
                                (df["maturity"].apply(lambda x: float(str(x).split()[0])) == maturity) &
                                (pd.to_datetime(df["execution_date"]) >= pd.to_datetime(dynamic_start_date)) &
                                (pd.to_datetime(df["execution_date"]) <= pd.to_datetime(dynamic_end_date))
                            ]
                            if not filtered.empty and "prediction" in filtered.columns and "actual" in filtered.columns:
                                rmse_data = (
                                    filtered.groupby("execution_date")
                                    .apply(lambda x: np.sqrt(((x["prediction"] - x["actual"]) ** 2).mean()))
                                    .reset_index(name="rmse")
                                )
                                rmse_data["model"] = model
                                combined_data.append(rmse_data)
                    if combined_data:
                        combined_data = pd.concat(combined_data, ignore_index=True)
                        fig = px.line(
                            combined_data,
                            x="execution_date",
                            y="rmse",
                            color="model",
                            title="Dynamic RMSE by execution date",
                            labels={"execution_date": "Execution Date", "rmse": "RMSE"},
                            #markers=True
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data for dynamic RMSE by execution date for {maturity}y.")

    #elif selected_tab == "Returns forward looking distributions":
    with tab_simulation_comparison:
        if selected_country == "UK":
            st.warning("Simulations are not available for the UK at the moment.")
        else:
            st.title("Returns Forward Looking Distributions")

            # Get available models for estimated returns
            estimated_returns_folder = os.path.join(base_folder, selected_country, "returns", "estimated_returns")
            estimated_models = [f.name for f in os.scandir(estimated_returns_folder) if f.is_dir()]
            benchmark_model = "AR_1"
            estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]

            # For returns tabs
            returns_observed_folder = os.path.join(base_folder, selected_country, "returns", "observed_returns", benchmark_model, "annual")
            sample_file_path = os.path.join(returns_observed_folder, "forecasts.csv")
            if os.path.exists(sample_file_path):
                sample_data = pd.read_csv(sample_file_path)
                available_maturities = sample_data["maturity"].unique()
            else:
                available_maturities = np.array([])  # or []

            # Filters: Model and Maturity Selection
            tab_simulation_comparison_filters = st.columns(2)
            with tab_simulation_comparison_filters[0]:
                # Dropdown for selecting models
                selected_models = st.multiselect(
                    "Select models",
                    estimated_models,  # Use the list of available models
                    default=estimated_models[:2],  # Default to the first two models
                    key="simulation_comparison_model_selector"  # Unique key for this dropdown
                )

            with tab_simulation_comparison_filters[1]:
                # Dropdown for selecting maturity
                if available_maturities.size > 0:
                    selected_maturity = st.selectbox(
                        "Select maturity",
                        available_maturities.tolist(),
                        key="simulation_comparison_maturity_selector"  # Unique key for this dropdown
                    )
                else:
                    st.warning("No maturities available for simulations.")
                    selected_maturity = None

            # Check if at least one model and a maturity are selected
            if selected_models and selected_maturity:
                combined_data = []

                # Loop through each selected model
                for model in selected_models:
                    # Adjust folder paths for the model and maturity
                    maturity_folder = f"{selected_maturity.replace(' ', '_')}"  # Format maturity as "X.XX_years"
                    folder_path = os.path.join(base_folder, selected_country, "returns", "estimated_returns", model, "annual", "simulations", 
                                            maturity_folder)

                    if os.path.exists(folder_path):
                        # Load the simulations for the current model using caching
                        model_simulations = load_model_simulations(model, folder_path)

                        if not model_simulations.empty:
                            # Append to the combined data
                            combined_data.append(model_simulations)
                        else:
                            st.warning(f"No parquet files found for model: {model}")
                    else:
                        st.warning(f"Folder does not exist: {folder_path}")

                # Combine all data across models
                if combined_data:
                    combined_data = pd.concat(combined_data, ignore_index=True)

                    # Time Series Plot
                    st.subheader("Time Series Simulations Distributions")

                    # Define a color palette for models
                    color_palette = ["#3a6bac", "#aa322f", "#eaa121", "#633d83", "#d55b20", "#427f6d", "#784722"]

                    # Calculate percentiles and summary statistics for each model
                    summary_data = combined_data.groupby(["execution_date", "model"])["simulated_value"].agg(
                        mean="mean",
                        p5=lambda x: np.percentile(x, 5),
                        p95=lambda x: np.percentile(x, 95)
                    ).reset_index()

                    # Plot the results using Plotly
                    fig_time_series = go.Figure()

                    # Loop through each selected model and add traces
                    for i, model in enumerate(selected_models):
                        model_data = summary_data[summary_data["model"] == model]

                        # Define the color for this model
                        model_color = color_palette[i % len(color_palette)]

                        # Add mean line
                        fig_time_series.add_trace(go.Scatter(
                            x=model_data["execution_date"],
                            y=model_data["mean"],
                            mode="lines",
                            name=f"{model} Mean",
                            line=dict(width=2, color=model_color)
                        ))

                        # Add shaded area for 5th to 95th percentiles
                        fig_time_series.add_trace(go.Scatter(
                            x=model_data["execution_date"].tolist() + model_data["execution_date"].tolist()[::-1],
                            y=model_data["p95"].tolist() + model_data["p5"].tolist()[::-1],
                            fill="toself",
                            fillcolor=f"rgba({int(255 * (i / len(color_palette)))}, 100, 200, 0.2)",  # Adjust transparency of the band
                            line=dict(width=0),
                            name=f"{model} 5th-95th Percentile"
                        ))

                    # Update layout
                    fig_time_series.update_layout(
                        xaxis_title="Execution Date",
                        yaxis_title="Annual Return",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                        template="plotly_white"
                    )

                    # Display the plot
                    st.plotly_chart(fig_time_series, use_container_width=True)

                    # Date Filter for KDE
                    st.subheader("Filter by Execution Date for KDE")
                    available_execution_dates = sorted(combined_data["execution_date"].unique())
                    selected_execution_date = st.selectbox(
                        "Select Execution Date",
                        available_execution_dates,
                        key="simulation_comparison_execution_date_selector"
                    )

                    # Filter data for the selected execution date
                    execution_date_data = combined_data[combined_data["execution_date"] == selected_execution_date]

                    # KDE Subplots for Each Horizon
                    st.subheader(f"KDE Distributions for Execution Date: {selected_execution_date}")

                    # Get unique horizons
                    unique_horizons = sorted(execution_date_data["horizon"].unique())

                    # Create a 2x3 grid using Streamlit columns
                    rows = []
                    n_cols = 3  # Number of columns
                    for i in range(0, len(unique_horizons) + 1, n_cols):
                        rows.append(st.columns(n_cols))

                    # Loop through each horizon and add KDE plots
                    for idx, horizon in enumerate(unique_horizons):
                        row_idx = idx // n_cols
                        col_idx = idx % n_cols

                        with rows[row_idx][col_idx]:
                            # Filter data for the current horizon
                            horizon_data = execution_date_data[execution_date_data["horizon"] == horizon]

                            # Create a Plotly figure for the current horizon
                            fig_kde = go.Figure()

                            # Loop through selected models
                            for i, model in enumerate(selected_models):
                                model_data = horizon_data[horizon_data["model"] == model]

                                if not model_data.empty:
                                    # Calculate KDE
                                    kde = gaussian_kde(model_data["simulated_value"])
                                    x_range = np.linspace(model_data["simulated_value"].min(), model_data["simulated_value"].max(), 500)
                                    y_kde = kde(x_range)

                                    # Define the color for this model
                                    model_color = color_palette[i % len(color_palette)]

                                    # Add KDE trace to the figure
                                    fig_kde.add_trace(go.Scatter(
                                        x=x_range,
                                        y=y_kde,
                                        mode="lines",
                                        name=f"{model}",
                                        line=dict(width=2, color=model_color)
                                    ))

                                    # Calculate VaR (5th percentile)
                                    var_5 = np.percentile(model_data["simulated_value"], 5)

                                    # Add vertical line for VaR
                                    fig_kde.add_shape(
                                        type="line",
                                        x0=var_5, x1=var_5,
                                        y0=0, y1=1,
                                        xref="x", yref="paper",
                                        line=dict(color=model_color, width=2, dash="dot"),
                                        name=f"{model} VaR"
                                    )

                            # Add vertical line at zero
                            fig_kde.add_shape(
                                type="line",
                                x0=0, x1=0,
                                y0=0, y1=1,
                                xref="x", yref="paper",
                                line=dict(color="black", width=2, dash="dash"),
                                name="Zero Line"
                            )

                            # Update layout
                            fig_kde.update_layout(
                                title=f"{horizon} years horizon",
                                xaxis_title="Annual Return",
                                yaxis_title="Density",
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                                template="plotly_white"
                            )

                            # Display the KDE plot
                            st.plotly_chart(fig_kde, use_container_width=True)

                    # Add the "All Horizons" plot in the last available slot
                    row_idx = len(unique_horizons) // n_cols
                    col_idx = len(unique_horizons) % n_cols

                    with rows[row_idx][col_idx]:
                        # Create a Plotly figure for "All Horizons"
                        fig_all_horizons = go.Figure()

                        # Add simulated returns for all models
                        for i, model in enumerate(selected_models):
                            model_data = execution_date_data[execution_date_data["model"] == model]

                            if not model_data.empty:
                                kde = gaussian_kde(model_data["simulated_value"])
                                x_range = np.linspace(model_data["simulated_value"].min(), model_data["simulated_value"].max(), 500)
                                y_kde = kde(x_range)

                                model_color = color_palette[i % len(color_palette)]
                                fig_all_horizons.add_trace(go.Scatter(
                                    x=x_range,
                                    y=y_kde,
                                    mode="lines",
                                    name=f"{model}",
                                    line=dict(width=2, color=model_color)
                                ))

                        # Add vertical line at zero
                        fig_all_horizons.add_shape(
                            type="line",
                            x0=0, x1=0,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            line=dict(color="black", width=2, dash="dash"),
                            name="Zero Line"
                        )

                        # Update layout
                        fig_all_horizons.update_layout(
                            title="All Horizons (Simulated vs Observed Returns)",
                            xaxis_title="Annual Return",
                            yaxis_title="Density",
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                            template="plotly_white"
                        )

                        # Display the "All Horizons" plot
                        st.plotly_chart(fig_all_horizons, use_container_width=True)

    with tab_crps:
        if selected_country != "US":
            st.warning("CRPS analysis is currently available only for the US.")
        else:
            st.title("CRPS (Continuous Ranked Probability Score) Analysis")

            # --- Model and Maturity Filters ---
            crps_models_all = ["AR_1", "Mixed_Model", "Mixed_Model_curvMacro"]

            # Load available maturities from any CRPS file (prefer AR_1)
            crps_sample_file = os.path.join(
                base_folder, "US", "returns", "estimated_returns", "AR_1", "annual", "crps_by_horizon.csv"
            )
            if os.path.exists(crps_sample_file):
                crps_sample_data = pd.read_csv(crps_sample_file)
                available_maturities = sorted(crps_sample_data["maturity"].unique())
            else:
                available_maturities = [0.25, 2.0, 5.0, 10.0]

            filter_cols = st.columns(2)
            with filter_cols[0]:
                selected_crps_models = st.multiselect(
                    "Select models",
                    crps_models_all,
                    default=crps_models_all[:2],
                    key="crps_models_selector"
                )
            with filter_cols[1]:
                default_crps_maturities = [m for m in [0.25, 2.0, 5.0, 10.0] if m in available_maturities]
                selected_crps_maturities = st.multiselect(
                    "Select maturities to plot (Max 4)",
                    available_maturities,
                    default=default_crps_maturities,
                    key="crps_maturities_selector"
                )
                if len(selected_crps_maturities) > 4:
                    st.warning("Please select up to 4 maturities. Only the first 4 will be used.")
                    selected_crps_maturities = selected_crps_maturities[:4]

            if not selected_crps_models or not selected_crps_maturities:
                st.info("Please select at least one model and one maturity.")
            else:
                # --- CRPS by Horizon ---
                st.subheader("Average CRPS by Horizon (Selected Maturities)")
                crps_horizon_data = []
                for model in selected_crps_models:
                    crps_file = os.path.join(
                        base_folder, "US", "returns", "estimated_returns", model, "annual", "crps_by_horizon.csv"
                    )
                    if os.path.exists(crps_file):
                        df = pd.read_csv(crps_file)
                        df = df[df["maturity"].isin(selected_crps_maturities)]
                        df["model"] = model
                        df["horizon_years"] = df["horizon"] / 12
                        crps_horizon_data.append(df)
                if crps_horizon_data:
                    crps_horizon_df = pd.concat(crps_horizon_data, ignore_index=True)
                    cols = st.columns(len(selected_crps_maturities))
                    for idx, maturity in enumerate(selected_crps_maturities):
                        with cols[idx]:
                            fig = px.line(
                                crps_horizon_df[crps_horizon_df["maturity"] == maturity],
                                x="horizon_years",
                                y="crps",
                                color="model",
                                markers=True,
                                title=f"Maturity: {maturity}y",
                                labels={"horizon_years": "Horizon (years)", "crps": "Average CRPS"}
                            )
                            fig.update_layout(
                                legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No CRPS by horizon data available for selected models.")

                # --- CRPS by Execution Date ---
                st.subheader("Average CRPS by Execution Date (Selected Maturities)")
                crps_exec_data = []
                for model in selected_crps_models:
                    crps_file = os.path.join(
                        base_folder, "US", "returns", "estimated_returns", model, "annual", "crps_by_execution_date.csv"
                    )
                    if os.path.exists(crps_file):
                        df = pd.read_csv(crps_file)
                        df = df[df["maturity"].isin(selected_crps_maturities)]
                        df["model"] = model
                        df["execution_date"] = pd.to_datetime(df["execution_date"])
                        crps_exec_data.append(df)
                if crps_exec_data:
                    crps_exec_df = pd.concat(crps_exec_data, ignore_index=True)
                    cols = st.columns(len(selected_crps_maturities))
                    for idx, maturity in enumerate(selected_crps_maturities):
                        with cols[idx]:
                            fig = px.line(
                                crps_exec_df[crps_exec_df["maturity"] == maturity],
                                x="execution_date",
                                y="crps",
                                color="model",
                                #markers=True,
                                title=f"Maturity: {maturity}y",
                                labels={"execution_date": "Execution Date", "crps": "Average CRPS"}
                            )
                            fig.update_layout(
                                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No CRPS by execution date data available for selected models.")

            # --- CRPS Heatmaps for Selected Model ---
            st.subheader("CRPS Heatmaps")

            # Let user pick a model for the heatmaps (default to Mixed_Model if available)
            heatmap_models = [m for m in crps_models_all if m in selected_crps_models]
            default_heatmap_model = "Mixed_Model" if "Mixed_Model" in heatmap_models else heatmap_models[0] if heatmap_models else None

            if heatmap_models and default_heatmap_model:
                selected_heatmap_model = st.selectbox(
                    "Select model for heatmaps",
                    heatmap_models,
                    index=heatmap_models.index(default_heatmap_model),
                    key="crps_heatmap_model_selector"
                )

                crps_file_horizon = os.path.join(
                    base_folder, "US", "returns", "estimated_returns", selected_heatmap_model, "annual", "crps_by_horizon.csv"
                )
                crps_file_exec = os.path.join(
                    base_folder, "US", "returns", "estimated_returns", selected_heatmap_model, "annual", "crps_by_execution_date.csv"
                )
                if os.path.exists(crps_file_exec):

                    df_exec = pd.read_csv(crps_file_exec)
                    df_exec["execution_date"] = pd.to_datetime(df_exec["execution_date"])
                    df_exec["execution_date_str"] = df_exec["execution_date"].dt.strftime('%Y-%m')
                    heatmap_data_exec = df_exec.groupby(["execution_date_str", "maturity"])["crps"].mean().unstack()
                    fig2 = px.imshow(
                        heatmap_data_exec.T,
                        labels={"x": "Execution Date", "y": "Maturity", "color": "CRPS"},
                        color_continuous_scale="RdBu_r",
                        origin="lower"
                    )
                    fig2.update_layout(
                        title="CRPS by Execution Date and Maturity",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info(f"No CRPS heatmap data available for {selected_heatmap_model}.")
            else:
                st.info("No model available for heatmaps in your selection.")

elif selected_macro == "Consensus Economics":
    st.title("Consensus Economics: RMSE by Horizon")

    # Load data
    data_folder = r'L:\RMAS\Users\Alberto\backtest-baam\data_test\consensus_backtest'
    rmse_file = os.path.join(data_folder, "rmse_horizon_all_countries_indicators.csv")
    master_rmse_horizon = pd.read_csv(rmse_file)

    countries = master_rmse_horizon['country'].unique()
    indicators = ["GDP", "CPI", "STR", "LTR"]
    model_colors = {"AR(1)": "#aa322f", "Consensus": "#3a6bac"}

    selected_country = st.selectbox("Select country", countries, key="consensus_country_selector")

    # 4 panels for the 4 indicators
    cols = st.columns(4)
    for i, indicator in enumerate(indicators):
        with cols[i]:
            df = master_rmse_horizon[
                (master_rmse_horizon['country'] == selected_country) &
                (master_rmse_horizon['indicator'] == indicator)
            ]
            if df.empty:
                st.write(f"**{indicator}**\n(no data)")
                continue

            # Prepare data for dots at integer horizons
            horizons = df['horizon']
            mask_int = horizons % 1 == 0
            dot_horizons = horizons[mask_int]

            fig = go.Figure()
            # AR(1) line and dots
            fig.add_trace(go.Scatter(
                x=horizons,
                y=df['RMSE_AR1'],
                mode="lines+markers",
                name="AR(1)",
                line=dict(color=model_colors["AR(1)"], width=2),
                marker=dict(size=7, color=model_colors["AR(1)"]),
                showlegend=True
            ))
            # Consensus line and dots
            fig.add_trace(go.Scatter(
                x=horizons,
                y=df['RMSE_consensus'],
                mode="lines+markers",
                name="Consensus",
                line=dict(color=model_colors["Consensus"], width=2),
                marker=dict(size=7, color=model_colors["Consensus"]),
                showlegend=True
            ))

            fig.update_layout(
                title=indicator,
                xaxis_title="Horizon (years)",
                yaxis_title="RMSE",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                ),
                template="plotly_white",
                margin=dict(l=10, r=10, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
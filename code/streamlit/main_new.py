import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

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
        rename_mapping = {
            "execution_date": "ExecutionDate",
            "forecasted_date": "ForecastDate",
            "prediction": "Prediction",
            "actual": "Actual",
            "horizon": "Horizon"
        }
        data = data.rename(columns=rename_mapping)
        return data
    else:
        st.warning(f"File not found: {file_path}")
        return None

# Helper function to calculate RMSE
def calculate_rmse(data, start_date, end_date=None):
    """Calculate RMSE from the forecasts data filtered by start and end dates."""
    filtered_data = data[
        (pd.to_datetime(data["ExecutionDate"]) >= pd.to_datetime(start_date))
    ]
    if end_date:
        filtered_data = filtered_data[
            (pd.to_datetime(filtered_data["ExecutionDate"]) <= pd.to_datetime(end_date))
        ]
    filtered_data = filtered_data.dropna(subset=["Actual", "Prediction"])
    rmse_data = (
        filtered_data
        .groupby("Horizon")
        .apply(lambda x: np.sqrt(((x["Prediction"] - x["Actual"]) ** 2).mean()))
        .reset_index(name="RMSE")
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
            data["Model"] = model
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
        if data is not None:
            data["Model"] = model
            cached_data[model] = data
    return cached_data

# App Configuration
st.set_page_config(layout="wide", page_title="Backtesting Results Dashboard")

# Sidebar: Country Selection
st.sidebar.header("Configuration")
countries = ["US", "EA", "UK"]
selected_country = st.sidebar.selectbox("Select Country", countries, index=0)

# Dynamically update the folder path based on the selected country
base_folder = r"\\msfsshared\BNKG\\RMAS\Users\Alberto\backtest-baam\data"
folder_path = os.path.join(base_folder, selected_country, "factors")

# Get available models and target variables
models = [f.name for f in os.scandir(folder_path) if f.is_dir() and f.name != "archive"]  # Exclude "archive"
target_variables = ["beta1", "beta2", "beta3"]

# Tabs for Out-of-Sample and In-Sample Metrics
tab_factors, tab_out_of_sample, tab_in_sample, tab_simulations, tab_yields_1, tab_yields, tab_sim_yields, tab_returns, tab_simulation_comparison = st.tabs(
    ["Factors overview","Out-of-Sample Model Comparison", "In-Sample Model Specifics", "Simulations", 
     "Yields overview", "Yields Backtesting", "Yields Simulations", "Returns", "Returns forward looking distributions"]
)

with tab_factors:
    st.title("Actuals vs Predictions for Factors")

    # Create a 3-column layout for the plots
    st.subheader("Plots: Actuals vs Predictions")
    cols = st.columns(3)

    # Loop through all factors
    for idx, (col, target_variable) in enumerate(zip(cols, target_variables)):  # target_variables = ["beta1", "beta2", "beta3"]
        with col:
            st.subheader(f"{target_variable}: Actuals vs Predictions")
            combined_data = []

            # Dropdown for selecting models (specific to this factor)
            st.subheader("Model Selection")
            selected_models = st.multiselect(
                f"Select Models for {target_variable} (Max 3)",
                models,  # Use the list of available models
                default=models[:1],  # Default to the first model
                key=f"model_selector_{target_variable}_{idx}"  # Unique key for each factor
            )

            # Restrict to a maximum of 3 models
            if len(selected_models) > 3:
                st.warning(f"Please select up to 3 models for {target_variable}. Only the first 3 models will be used.")
                selected_models = selected_models[:3]  # Limit to the first 3 models

            # Date selectors for this specific panel
            st.subheader("Filter by Date Range")
            min_date = datetime.strptime("1950-01-01", "%Y-%m-%d").date()
            max_date = datetime.strptime("2030-12-31", "%Y-%m-%d").date()

            date_cols = st.columns(2)
            with date_cols[0]:
                start_date = st.date_input(
                    f"Start Date for {target_variable}",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f"start_date_input_{target_variable}_{idx}"  # Unique key for each factor
                )
            with date_cols[1]:
                end_date = st.date_input(
                    f"End Date for {target_variable}",
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
                        required_columns = ["ExecutionDate", "ForecastDate", "Prediction", "Actual"]
                        if all(col in data.columns for col in required_columns):
                            # Convert ExecutionDate to datetime if not already in datetime format
                            if not pd.api.types.is_datetime64_any_dtype(data["ExecutionDate"]):
                                data["ExecutionDate"] = pd.to_datetime(data["ExecutionDate"])

                            # Filter data using the selected date range
                            filtered_data = data[
                                (data["ExecutionDate"].dt.date >= start_date) &
                                (data["ExecutionDate"].dt.date <= end_date)
                            ]

                            if not filtered_data.empty:
                                filtered_data["Model"] = model
                                combined_data.append(filtered_data)
                        else:
                            st.warning(f"Required columns are missing in the forecasts data for model: {model}, factor: {target_variable}")
                    else:
                        st.warning(f"File not found: {forecasts_file_path}")

                # Combine data from all models
                if combined_data:
                    combined_data = pd.concat(combined_data, ignore_index=True)

                    # Prepare the actuals data
                    realized_beta = combined_data.groupby("ForecastDate")["Actual"].mean()

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
                        model_data = combined_data[combined_data["Model"] == model]
                        unique_execution_dates = model_data["ExecutionDate"].unique()
                        
                        # Add traces for each execution date, but hide them from the legend
                        for execution_date in unique_execution_dates:
                            subset = model_data[model_data["ExecutionDate"] == execution_date]
                            fig.add_trace(go.Scatter(
                                x=subset["ForecastDate"],
                                y=subset["Prediction"],
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
                        title=f"{target_variable}: Forecasted vs Actual",
                        xaxis_title="Forecast Date",
                        yaxis_title=f"{target_variable}",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                        template="plotly_white"
                    )

                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {target_variable}.")

# Out-of-Sample Model Comparison Tab
with tab_out_of_sample:
    st.title("Out-of-Sample Model Comparison")

    # Determine the date range for the data
    all_dates = []
    for model in models:
        for target_variable in target_variables:
            # Load data for out-of-sample metrics
            data_horizon = load_data(folder_path, model, target_variable, "outofsample_metrics_by_horizon.csv")
            data_execution = load_data(folder_path, model, target_variable, "outofsample_metrics_by_execution_date.csv")
            if data_horizon is not None and "ExecutionDate" in data_horizon.columns:
                all_dates.extend(pd.to_datetime(data_horizon["ExecutionDate"]).tolist())
            if data_execution is not None and "ExecutionDate" in data_execution.columns:
                all_dates.extend(pd.to_datetime(data_execution["ExecutionDate"]).tolist())

    # Calculate the minimum and maximum dates
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
    else:
        min_date = pd.to_datetime("1950-01-01")  # Default fallback
        max_date = pd.to_datetime("2025-12-31")  # Default fallback

    # Create columns for Default RMSE by Horizon (First Row)
    st.subheader("Default RMSE by Horizon (Full Sample)")
    cols = st.columns(3)
    selected_models_per_beta = {}  # To store selected models for each beta
    for col, target_variable in zip(cols, target_variables):
        with col:
            st.write(f"Results for {target_variable}")
            
            # Dropdown for selecting models specific to this beta (factor)
            selected_models = st.multiselect(
                f"Select Models for {target_variable}", 
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
                    data["Model"] = model
                    combined_data.append(data)
            
            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                fig = px.line(
                    combined_data,
                    x="Horizon",
                    y="rmse",
                    color="Model",
                    markers=True,
                    title=f"Default RMSE by Horizon ({target_variable})",
                    labels={"rmse": "RMSE", "Horizon": "Horizon"},
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
    st.subheader("Dynamic RMSE by Horizon (From Selected Date Range)")
    cols = st.columns(2)
    with cols[0]:
        dynamic_start_date = st.date_input(
            "Select Start Date for Dynamic RMSE by Horizon",
            value=min_date,  # Default to the earliest date
            min_value=min_date,
            max_value=max_date,
            key="dynamic_start_date_horizon"
        )
    with cols[1]:
        dynamic_end_date = st.date_input(
            "Select End Date for Dynamic RMSE by Horizon",
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
                        (pd.to_datetime(data['ExecutionDate']) >= pd.to_datetime(dynamic_start_date)) &
                        (pd.to_datetime(data['ExecutionDate']) <= pd.to_datetime(dynamic_end_date))
                    ]
                    rmse_data = calculate_rmse(filtered_data, dynamic_start_date)
                    rmse_data["Model"] = model
                    combined_data.append(rmse_data)
            
            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                fig = px.line(
                    combined_data,
                    x="Horizon",
                    y="RMSE",
                    color="Model",
                    markers=True,
                    title=f"Dynamic RMSE by Horizon ({target_variable})",
                    labels={"Horizon": "Horizon", "RMSE": "RMSE"},
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
    st.subheader("RMSE by Execution Date (From Selected Date Range)")
    cols = st.columns(2)
    with cols[0]:
        execution_start_date = st.date_input(
            "Select Start Date for RMSE by Execution Date",
            value=min_date,  # Default to the earliest date
            min_value=min_date,
            max_value=max_date,
            key="execution_start_date"
        )
    with cols[1]:
        execution_end_date = st.date_input(
            "Select End Date for RMSE by Execution Date",
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
                        (pd.to_datetime(data['ExecutionDate']) >= pd.to_datetime(execution_start_date)) &
                        (pd.to_datetime(data['ExecutionDate']) <= pd.to_datetime(execution_end_date))
                    ]
                    filtered_data["Model"] = model
                    combined_data.append(filtered_data)
            
            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                fig = px.line(
                    combined_data,
                    x="ExecutionDate",
                    y="rmse",
                    color="Model",
                    title=f"RMSE by Execution Date ({target_variable})",
                    labels={"rmse": "RMSE", "ExecutionDate": "Execution Date"},
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
    st.title("In-Sample Model Specifics")

    # Load all in-sample metrics data to determine the date range
    all_data = []
    for model in models:
        for target_variable in target_variables:
            data = load_data(folder_path, model, target_variable, "insample_metrics.csv")
            if data is not None:
                data["ExecutionDate"] = pd.to_datetime(data["ExecutionDate"])
                all_data.append(data)
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        min_date = combined_data["ExecutionDate"].min()
        max_date = combined_data["ExecutionDate"].max()
    else:
        min_date = pd.to_datetime("2000-01-01")
        max_date = pd.to_datetime("2023-12-31")

    # Start Date Filter
    start_date = st.date_input(
        "Select Start Date to Filter Data",
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
            selected_models[target_variable] = st.selectbox(
                f"Select Model for {target_variable}",
                models,
                index=0,
                key=f"model_selector_insample_{target_variable}"
            )

    # Load the in-sample metrics data for each beta
    insample_data = {}
    for target_variable, model in selected_models.items():
        insample_data[target_variable] = load_data(folder_path, model, target_variable, "insample_metrics.csv")

    # Replace 'beta1', 'beta2', 'beta3' in 'indicator' with 'lagged beta'
    for target_variable, data in insample_data.items():
        if data is not None:
            data["indicator"] = data["indicator"].replace(
                {"beta1": "lagged beta", "beta2": "lagged beta", "beta3": "lagged beta"}
            )
            # Convert execution_date to datetime and sort
            data["ExecutionDate"] = pd.to_datetime(data["ExecutionDate"])
            data.sort_values("ExecutionDate", inplace=True)
            # Filter data based on the selected start date
            insample_data[target_variable] = data[data["ExecutionDate"] >= pd.to_datetime(start_date)]

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
                            x="ExecutionDate",
                            y="value",
                            title=f"Adjusted R-Squared ({target_variable})",
                            labels={"ExecutionDate": "Execution Date", "value": "Adjusted R-Squared"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            # Plot coefficients and overlay p-values for the other indicators
            st.subheader(f"Indicator: {row}")
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
                                index="ExecutionDate", columns="metric", values="value"
                            ).reset_index()
                            if "coefficient" in pivot_data.columns and "p_value" in pivot_data.columns:
                                fig = create_dual_axis_plot(
                                    data=pivot_data,
                                    x_col="ExecutionDate",
                                    y1_col="coefficient",
                                    y2_col="p_value",
                                    title=f"{row} Coefficient and P-Value ({target_variable})",
                                    y1_label="Coefficient",
                                    y2_label="P-Value"
                                )
                                st.plotly_chart(fig, use_container_width=True)

with tab_simulations:
    st.title("Simulations Analysis: Heatmaps, Fan Chart, and Distributions")

    # Dropdown to select a specific model and beta
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            models,
            index=0,
            key="simulation_model_selector"
        )
    with col2:
        selected_beta = st.selectbox(
            "Select Beta",
            target_variables,
            index=0,
            key="simulation_beta_selector"
        )

    # Load the simulations data and forecasts data for the selected model and beta
    simulations_data = load_data(folder_path, selected_model, selected_beta, "simulations.parquet")
    forecasts_data = load_data(folder_path, selected_model, selected_beta, "forecasts.csv")  # Load forecasts data

    if simulations_data is not None and forecasts_data is not None:
        # Convert ExecutionDate and ForecastDate to datetime
        simulations_data["ExecutionDate"] = pd.to_datetime(simulations_data["ExecutionDate"])
        simulations_data["ForecastDate"] = pd.to_datetime(simulations_data["ForecastDate"])
        forecasts_data["ExecutionDate"] = pd.to_datetime(forecasts_data["ExecutionDate"])
        forecasts_data["ForecastDate"] = pd.to_datetime(forecasts_data["ForecastDate"])

        # Group actuals by ForecastDate
        actuals = forecasts_data[["ForecastDate", "Actual"]].groupby("ForecastDate").last().dropna()

        # Determine the date range for the data
        min_date = simulations_data["ExecutionDate"].min()
        max_date = simulations_data["ExecutionDate"].max()

        # Add start and end date pickers
        col3, col4 = st.columns(2)
        with col3:
            start_date = st.date_input(
                "Select Start Date for Execution Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="simulation_start_date"
            )
        with col4:
            end_date = st.date_input(
                "Select End Date for Execution Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="simulation_end_date"
            )

        # Filter data based on the selected date range
        filtered_simulations = simulations_data[
            (simulations_data["ExecutionDate"] >= pd.to_datetime(start_date)) &
            (simulations_data["ExecutionDate"] <= pd.to_datetime(end_date))
        ]

        # --- Visualization 1: Horizon–Origin Heatmaps ---
        st.subheader("Horizon–Origin Heatmaps (HOP)")

        # Median Forecasts
        median_forecasts = filtered_simulations.groupby(["ExecutionDate", "Horizon"])["SimulatedValue"].quantile(0.5).unstack()
        fig = px.imshow(
            median_forecasts.T,#.sort_index(ascending=False),  # Transpose to align axes
            labels={"x": "Execution Date", "y": "Horizon", "color": "Median Forecast"},
            title="Horizon–Origin Heatmap: Median Factor Forecasts",
            color_continuous_scale="RdBu",
            origin='lower'
        )
        #fig.update_layout(yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        # Prediction Interval Width (90%)
        pi_width = filtered_simulations.groupby(["ExecutionDate", "Horizon"])["SimulatedValue"].quantile(0.95).unstack() - \
                   filtered_simulations.groupby(["ExecutionDate", "Horizon"])["SimulatedValue"].quantile(0.05).unstack()
        fig = px.imshow(
            pi_width.T.sort_index(ascending=False),  # Transpose to align axes
            labels={"x": "Execution Date", "y": "Horizon", "color": "90% PI Width"},
            title="Horizon–Origin Heatmap: 90% Prediction Interval Width",
            color_continuous_scale="RdBu",
            origin='lower'
        )
        st.plotly_chart(fig, use_container_width=True)

        # PIT (Probability Integral Transform)
        simulated_cdf = filtered_simulations.groupby(["ExecutionDate", "Horizon"]).apply(
            lambda group: (group["SimulatedValue"] <= group["ForecastDate"].map(actuals["Actual"])).mean()
        ).unstack()
        fig = px.imshow(
            simulated_cdf.T.sort_index(ascending=False),  # Transpose to align axes
            labels={"x": "Execution Date", "y": "Horizon", "color": "PIT"},
            title="Horizon–Origin Heatmap: PIT",
            color_continuous_scale="RdBu",
            origin='lower'
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Visualization 2: Actuals with Fan Chart ---
        st.subheader("Actuals and Simulations with Fan Chart")

        # Calculate percentiles (5th, 50th, 95th) for the fan chart
        percentiles = filtered_simulations.groupby("ExecutionDate")["SimulatedValue"].quantile([0.05, 0.5, 0.95]).unstack(level=-1)
        percentiles.columns = ["5th Percentile", "Median", "95th Percentile"]

        # Merge with actuals for the time series plot
        actuals_for_plot = forecasts_data.groupby("ForecastDate")["Actual"].last().dropna()
        combined_data = percentiles.join(actuals_for_plot, how="inner")

        # Create the fan chart with actuals
        fig = go.Figure()

        # Add fan chart (shaded area for 5th to 95th percentiles)
        fig.add_trace(go.Scatter(
            x=combined_data.index,
            y=combined_data["95th Percentile"],
            mode="lines",
            line=dict(width=0),
            name="95th Percentile",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=combined_data.index,
            y=combined_data["5th Percentile"],
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
            y=combined_data["Median"],
            mode="lines",
            line=dict(color="#3a6bac", width=2),
            name="Median Simulation"
        ))

        # Add actuals line
        fig.add_trace(go.Scatter(
            x=combined_data.index,
            y=combined_data["Actual"],
            mode="lines",
            line=dict(color="#c28191", width=2),
            name="Actual"
        ))

        # Update layout
        fig.update_layout(
            title="Actuals and Simulations with Fan Chart",
            xaxis_title="Execution Date",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        # --- Visualization: Historical Actuals and Fan Chart ---
        st.subheader("Historical Actuals and Projections")

        # Determine the available execution dates
        available_execution_dates = sorted(simulations_data["ExecutionDate"].dropna().unique())  # Drop NaN values if any
        available_execution_dates_str = [date.strftime("%Y-%m") for date in available_execution_dates]  # Convert to "YYYY-MM" strings

        if available_execution_dates_str:
            # Add a text input for selecting an ExecutionDate as a string
            default_date = available_execution_dates_str[-1]  # Default to the latest available date
            selected_execution_date_str = st.text_input(
                "Enter Execution Date (format: YYYY-MM)",
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
                    simulations_for_execution_date = simulations_data[simulations_data["ExecutionDate"] == selected_execution_date]
                    projections_data = None

                    if not simulations_for_execution_date.empty:
                        # Calculate percentiles (5th, 50th, 95th) for the fan chart
                        percentiles = simulations_for_execution_date.groupby("Horizon")["SimulatedValue"].quantile([0.05, 0.5, 0.95]).unstack(level=-1)

                        if not percentiles.empty:
                            percentiles.columns = ["5th Percentile", "Median", "95th Percentile"]
                            percentiles["ForecastDate"] = [selected_execution_date + pd.DateOffset(months=int(h)) for h in percentiles.index]
                            percentiles = percentiles.set_index("ForecastDate")
                            projections_data = percentiles.join(actuals, how="left")

                    # --- Create Two Columns for Side-by-Side Graphs ---
                    col1, col2 = st.columns(2)

                    # --- Left Column: Historical Actuals ---
                    with col1:
                        st.subheader("Historical Actuals")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=historical_actuals.index,
                            y=historical_actuals["Actual"],
                            mode="lines",
                            line=dict(color="#c28191", width=2),
                            name="Actual"
                        ))

                        # Add zero reference line if data includes negative values
                        if historical_actuals["Actual"].min() < 0:
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
                            title="Historical Actuals",
                            xaxis_title="Forecast Date",
                            yaxis_title="Value",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # --- Right Column: Projections and Prediction Intervals ---
                    with col2:
                        st.subheader("Projections and Prediction Intervals")

                        if projections_data is not None and not projections_data.empty:
                            fig = go.Figure()

                            # Add fan chart (shaded area for 5th to 95th percentiles)
                            fig.add_trace(go.Scatter(
                                x=projections_data.index,
                                y=projections_data["95th Percentile"],
                                mode="lines",
                                line=dict(width=0),
                                name="95th Percentile",
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=projections_data.index,
                                y=projections_data["5th Percentile"],
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
                                y=projections_data["Median"],
                                mode="lines",
                                line=dict(color="#3a6bac", width=2),
                                name="Median Simulation"
                            ))

                            # Add actuals line
                            fig.add_trace(go.Scatter(
                                x=projections_data.index,
                                y=projections_data["Actual"],
                                mode="lines+markers",
                                line=dict(color="#c28191", width=2),
                                name="Actual"
                            ))

                            # Add zero reference line if data includes negative values
                            if projections_data[["5th Percentile", "Median", "95th Percentile"]].min().min() < 0:
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
                                title=f"Projections and Prediction Intervals from Execution Date: {selected_execution_date_str}",
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
        st.subheader("Horizon-Specific Simulated Distributions vs Actuals")

        # Use all available forecast dates for the calendar date picker
        available_forecast_dates = sorted(filtered_simulations["ForecastDate"].unique())
        if available_forecast_dates:
            # Set default to January 2025 if available, otherwise use the first available date
            default_date = pd.Timestamp("2025-01-01") if pd.Timestamp("2025-01-01") in available_forecast_dates else available_forecast_dates[0]

            # Calendar-based date picker for ForecastDate
            selected_forecast_date = st.date_input(
                "Select Forecast Date to Analyze",
                value=default_date,  # Default to January 2025 or the first available date
                min_value=available_forecast_dates[0],
                max_value=available_forecast_dates[-1],
                key="simulation_forecast_date_selector"
            )

            # Filter simulations for the selected ForecastDate
            simulations_for_date = filtered_simulations[filtered_simulations["ForecastDate"] == pd.to_datetime(selected_forecast_date)]

            # Extract the realized value for the selected ForecastDate
            realized_values_for_date = actuals.loc[actuals.index.get_level_values("ForecastDate") == pd.to_datetime(selected_forecast_date)]
            if not realized_values_for_date.empty:
                realized_value = realized_values_for_date["Actual"].iloc[0]  # Take the first realized value (it's the same for all horizons)
            else:
                realized_value = None  # Handle missing realized values

            if not simulations_for_date.empty:
                # Allow users to select up to 5 horizons
                unique_horizons = sorted(simulations_for_date["Horizon"].unique())
                default_horizons = [h for h in [6, 12, 24, 48, 60] if h in unique_horizons]  # Default to specific horizons if available
                selected_horizons = st.multiselect(
                    "Select Horizons to Plot (Max 5)",
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
                        horizon_data = simulations_for_date[simulations_for_date["Horizon"] == horizon]
                        fig.add_trace(go.Histogram(
                            x=horizon_data["SimulatedValue"],
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

with tab_yields_1:
    st.title("Yields Backtesting")

    # Dynamically update the folder path for yields based on the selected country
    yields_folder_path = os.path.join(base_folder, selected_country, "yields")

    # Define paths for estimated and observed yields
    estimated_yields_folder = os.path.join(yields_folder_path, "estimated_yields")
    observed_yields_folder = os.path.join(yields_folder_path, "observed_yields")

    # Get available models for estimated yields
    estimated_models = [f.name for f in os.scandir(estimated_yields_folder) if f.is_dir()]
    benchmark_model = "AR_1"  # Observed yields benchmark model

    # Load a sample file to extract available maturities
    sample_file_path = os.path.join(observed_yields_folder, benchmark_model, "forecasts.csv")
    if os.path.exists(sample_file_path):
        sample_data = pd.read_csv(sample_file_path)
        sample_data = sample_data.rename(columns={"execution_date": "ExecutionDate", "forecasted_date": "ForecastDate"})
        available_maturities = sample_data["maturity"].unique()
    else:
        st.warning("Sample file not found. Cannot determine available maturities.")
        available_maturities = []

    # Dropdown for selecting models
    st.subheader("Model Selection")
    estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]
    selected_models_yields = st.multiselect(
        "Select Models for Yields (Max 3)",
        estimated_models_with_labels,
        default=estimated_models[:1],  # Default to the first estimated model
        key="model_selector_yields"
    )

    # Restrict to a maximum of 3 models
    if len(selected_models_yields) > 3:
        st.warning("Please select up to 3 models for Yields. Only the first 3 models will be used.")
        selected_models_yields = selected_models_yields[:3]  # Limit to the first 3 models

    # Dropdown for selecting a single maturity
    st.subheader("Maturity Selection")
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
    st.subheader("Filter by Date Range")
    min_date_yields = datetime.strptime("1950-01-01", "%Y-%m-%d").date()
    max_date_yields = datetime.strptime("2030-12-31", "%Y-%m-%d").date()

    # Place start and end date inputs side by side
    date_cols_yields = st.columns(2)
    with date_cols_yields[0]:
        start_date_yields = st.date_input(
            "Start Date for Yields",
            value=min_date_yields,
            min_value=min_date_yields,
            max_value=max_date_yields,
            key="start_date_input_yields"
        )
    with date_cols_yields[1]:
        end_date_yields = st.date_input(
            "End Date for Yields",
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

                # Ensure the required columns exist
                required_columns_yields = ["ExecutionDate", "ForecastDate", "Prediction", "Actual", "maturity"]
                if all(col in data_yields.columns for col in required_columns_yields):
                    # Convert ExecutionDate to datetime if not already in datetime format
                    if not pd.api.types.is_datetime64_any_dtype(data_yields["ExecutionDate"]):
                        data_yields["ExecutionDate"] = pd.to_datetime(data_yields["ExecutionDate"])

                    # Filter data using the selected date range and maturity
                    filtered_data_yields = data_yields[
                        (data_yields["ExecutionDate"].dt.date >= start_date_yields) &
                        (data_yields["ExecutionDate"].dt.date <= end_date_yields) &
                        (data_yields["maturity"] == selected_maturity)
                    ]

                    if not filtered_data_yields.empty:
                        # Sort data by ExecutionDate and ForecastDate
                        filtered_data_yields = filtered_data_yields.sort_values(by=["ExecutionDate", "ForecastDate"])
                        filtered_data_yields["Model"] = model
                        combined_data_yields.append(filtered_data_yields)
                else:
                    st.warning(f"Required columns are missing in the forecasts data for model: {model}, yields.")
            else:
                st.warning(f"File not found: {forecasts_file_path_yields}")

        # Combine data from all models
        if combined_data_yields:
            combined_data_yields = pd.concat(combined_data_yields, ignore_index=True)

            # Prepare the actuals data
            realized_yields = combined_data_yields.groupby("ForecastDate")["Actual"].mean()

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
                model_data_yields = combined_data_yields[combined_data_yields["Model"] == model]
                unique_execution_dates_yields = model_data_yields["ExecutionDate"].unique()
                
                # Add traces for each execution date, but hide them from the legend
                for execution_date in unique_execution_dates_yields:
                    subset_yields = model_data_yields[model_data_yields["ExecutionDate"] == execution_date]
                    fig_yields.add_trace(go.Scatter(
                        x=subset_yields["ForecastDate"],
                        y=subset_yields["Prediction"],
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
                title=f"Yields: Forecasted vs Actual (Maturity: {selected_maturity})",
                xaxis_title="Forecast Date",
                yaxis_title="Yields",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                template="plotly_white"
            )

            # Display the plot
            st.plotly_chart(fig_yields, use_container_width=True)
        else:
            st.warning(f"No data available for Yields (Maturity: {selected_maturity}).")

with tab_yields:
    st.title("Yields Backtesting: Out-of-Sample Metrics")

    # Dynamically update the folder path for yields based on the selected country
    yields_folder_path = os.path.join(base_folder, selected_country, "yields")

    # Define paths for estimated and observed yields
    estimated_yields_folder = os.path.join(yields_folder_path, "estimated_yields")
    observed_yields_folder = os.path.join(yields_folder_path, "observed_yields")

    # Get available models for estimated yields
    estimated_models = [f.name for f in os.scandir(estimated_yields_folder) if f.is_dir()]
    benchmark_model = "AR_1"  # Observed yields benchmark model

    # Load a sample file to extract available maturities
    sample_file_path = os.path.join(observed_yields_folder, benchmark_model, "forecasts.csv")
    if os.path.exists(sample_file_path):
        sample_data = pd.read_csv(sample_file_path)
        sample_data = sample_data.rename(columns={"execution_date": "ExecutionDate", "forecasted_date": "ForecastDate"})
        available_maturities = sample_data["maturity"].unique()
    else:
        st.warning("Sample file not found. Cannot determine available maturities.")
        available_maturities = []

    # Dropdown for selecting models
    st.subheader("Model Selection")
    estimated_models_with_labels = estimated_models + [f"{benchmark_model} (Benchmark)"]
    selected_models = st.multiselect(
        "Select Models to Compare",
        estimated_models_with_labels,
        default=estimated_models[:1],  # Default to the first estimated model
        key="selected_models_yields"
    )

    # Check if the benchmark model is selected
    include_benchmark = f"{benchmark_model} (Benchmark)" in selected_models

    # Dropdown for selecting maturities
    st.subheader("Maturity Selection")
    if available_maturities.size > 0:
        selected_maturities = st.multiselect(
            "Select Maturities to Include",
            available_maturities.tolist(),
            default=available_maturities[:1].tolist(),
            key="selected_maturities"
        )
    else:
        st.warning("No maturities available. Please check the data.")
        selected_maturities = []

    # Cache and load data for RMSE by Horizon
    #rmse_horizon_data = load_and_cache_data(selected_models, yields_folder_path, include_benchmark, "outofsample_metrics_by_horizon.csv")

    # Default date range for RMSE by Horizon
    #all_dates = []
    #for model, data in rmse_horizon_data.items():
    #    if "ExecutionDate" in data.columns:
    #        all_dates.extend(pd.to_datetime(data["ExecutionDate"]).tolist())
    min_date = min(all_dates) if all_dates else pd.to_datetime("1950-01-01")
    max_date = max(all_dates) if all_dates else pd.to_datetime("2025-12-31")

    # Loop over maturities and create plots
    # Loop over maturities and create two rows of graphs for each maturity
    # Loop over maturities and create two rows of graphs for each maturity
    for maturity in selected_maturities:
        st.subheader(f"Maturity: {maturity}")

        # --- First Row: Default RMSE by Execution Date and Horizon ---
        col1, col2 = st.columns(2)

        # Left Column: RMSE by Execution Date (Full History)
        with col1:
            st.write(f"RMSE by Execution Date for {maturity} (Full History)")
            execution_data = load_and_cache_data(
                selected_models, yields_folder_path, include_benchmark, "outofsample_metrics_by_execution_date.csv"
            )
            combined_data = []
            for model, data in execution_data.items():
                filtered_data = data[data["maturity"] == maturity]
                if not filtered_data.empty:
                    filtered_data["ExecutionDate"] = pd.to_datetime(filtered_data["ExecutionDate"])
                    
                    # Divide RMSE by 100 for the benchmark model
                    if model == f"{benchmark_model} (Benchmark)":
                        filtered_data["rmse"] = filtered_data["rmse"] / 100
                    
                    filtered_data["Model"] = model
                    combined_data.append(filtered_data)
            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                fig = px.line(
                    combined_data,
                    x="ExecutionDate",
                    y="rmse",
                    color="Model",
                    title=f"RMSE by Execution Date ({maturity})",
                    labels={"rmse": "RMSE", "ExecutionDate": "Execution Date"}
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for RMSE by Execution Date for {maturity}.")

        # Right Column: RMSE by Horizon (Full History)
        with col2:
            st.write(f"RMSE by Horizon for {maturity} (Full History)")
            horizon_data = load_and_cache_data(
                selected_models, yields_folder_path, include_benchmark, "outofsample_metrics_by_horizon.csv"
            )
            combined_data = []
            for model, data in horizon_data.items():
                filtered_data = data[data["maturity"] == maturity]
                filtered_data = filtered_data[filtered_data["Horizon"] != 0]
                if not filtered_data.empty:
                    # Divide RMSE by 100 for the benchmark model
                    if model == f"{benchmark_model} (Benchmark)":
                        filtered_data["rmse"] = filtered_data["rmse"] / 100
                    
                    filtered_data["Model"] = model
                    combined_data.append(filtered_data)
            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                fig = px.line(
                    combined_data,
                    x="Horizon",
                    y="rmse",
                    color="Model",
                    title=f"RMSE by Horizon ({maturity})",
                    labels={"rmse": "RMSE", "Horizon": "Horizon"}
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for RMSE by Horizon for {maturity}.")

        # --- Second Row: Dynamic RMSE by Execution Date and Horizon (Custom Date Range) ---
        col1, col2 = st.columns(2)
        with col1:
            dynamic_start_date = st.date_input(
                f"Select Start Date for {maturity}",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"dynamic_start_date_{maturity}"  # Unique key for start date picker
            )
        with col2:
            dynamic_end_date = st.date_input(
                f"Select End Date for {maturity}",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"dynamic_end_date_{maturity}"  # Unique key for end date picker
            )

        # --- Second Row: Dynamic RMSE by Execution Date and Horizon (Custom Date Range) ---
        col1, col2 = st.columns(2)

        # Dynamic RMSE by Execution Date
        with col1:
            st.write(f"Dynamic RMSE by Execution Date for {maturity} (Custom Date Range)")

            forecasts_data = load_and_cache_data(
                selected_models, yields_folder_path, include_benchmark, "forecasts.csv"
            )

            combined_data = []
            for model, data in forecasts_data.items():
                if "ExecutionDate" not in data.columns or "Actual" not in data.columns or "Prediction" not in data.columns:
                    st.warning(f"Required columns are missing in the forecasts data for model: {model}")
                    continue

                filtered_data = data[
                    (data["maturity"] == maturity) &
                    (pd.to_datetime(data["ExecutionDate"]) >= pd.to_datetime(dynamic_start_date)) &
                    (pd.to_datetime(data["ExecutionDate"]) <= pd.to_datetime(dynamic_end_date))
                ]
                if not filtered_data.empty:
                    # Calculate RMSE by Execution Date
                    rmse_data = (
                        filtered_data.groupby("ExecutionDate")
                        .apply(lambda x: np.sqrt(((x["Prediction"] - x["Actual"]) ** 2).mean()))
                        .reset_index(name="RMSE")
                    )
                    
                    # Divide RMSE by 100 for the benchmark model
                    if model == f"{benchmark_model} (Benchmark)":
                        rmse_data["RMSE"] = rmse_data["RMSE"] / 100
                    
                    rmse_data["Model"] = model
                    combined_data.append(rmse_data)

            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                fig = px.line(
                    combined_data,
                    x="ExecutionDate",
                    y="RMSE",
                    color="Model",
                    title=f"Dynamic RMSE by Execution Date ({maturity})",
                    labels={"ExecutionDate": "Execution Date", "RMSE": "RMSE"}
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for Dynamic RMSE by Execution Date for {maturity}.")

        # Dynamic RMSE by Horizon
        with col2:
            st.write(f"Dynamic RMSE by Horizon for {maturity} (Custom Date Range)")

            combined_data = []
            for model, data in forecasts_data.items():
                filtered_data = data[
                    (data["maturity"] == maturity) &
                    (pd.to_datetime(data["ExecutionDate"]) >= pd.to_datetime(dynamic_start_date)) &
                    (pd.to_datetime(data["ExecutionDate"]) <= pd.to_datetime(dynamic_end_date))
                ]
                if not filtered_data.empty:
                    # Calculate RMSE by Horizon
                    rmse_data = calculate_rmse(filtered_data, dynamic_start_date, dynamic_end_date)

                    # Divide RMSE by 100 for the benchmark model
                    if model == f"{benchmark_model} (Benchmark)":
                        rmse_data["RMSE"] = rmse_data["RMSE"] / 100
                    
                    rmse_data["Model"] = model
                    combined_data.append(rmse_data)

            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                fig = px.line(
                    combined_data,
                    x="Horizon",
                    y="RMSE",
                    color="Model",
                    title=f"Dynamic RMSE by Horizon ({maturity})",
                    labels={"Horizon": "Horizon", "RMSE": "RMSE"}
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for Dynamic RMSE by Horizon for {maturity}.")

with tab_sim_yields:
    st.title("Simulated Yields Analysis: Full-Sample Heatmaps and Fan Chart")

    # Dropdown for selecting model
    st.subheader("Model Selection")
    selected_model = st.selectbox(
        "Select Model for Simulated Yields",
        estimated_models_with_labels,
        key="sim_yields_model_selector"  # Unique key for this dropdown
    )

    # Dropdown for selecting maturity
    st.subheader("Maturity Selection")
    if available_maturities.size > 0:
        selected_maturity = st.selectbox(
            "Select Maturity for Simulated Yields",
            available_maturities.tolist(),
            key="sim_yields_maturity_selector"  # Unique key for this dropdown
        )
    else:
        st.warning("No maturities available for simulations.")
        selected_maturity = None

    # Load and concatenate all parquet files for the selected maturity
    if selected_model and selected_maturity:
        # Adjust folder for benchmark model
        model_folder = benchmark_model if selected_model == f"{benchmark_model} (Benchmark)" else selected_model
        maturity_folder = selected_maturity.replace(" ", "_")  # Convert maturity to folder name format
        folder_path = os.path.join(yields_folder_path, "estimated_yields", model_folder, "simulations", maturity_folder)

        if os.path.exists(folder_path):
            # Concatenate all parquet files in the folder
            all_parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
            if all_parquet_files:
                simulations_data = pd.concat([pd.read_parquet(file) for file in all_parquet_files], ignore_index=True)

                # Ensure the data contains the expected columns
                required_columns = ["ForecastDate", "SimulatedValue", "Maturity", "SimulationID", "ExecutionDate", "Model", "Horizon"]
                if all(col in simulations_data.columns for col in required_columns):
                    # --- Visualization 1: Full-Sample Horizon-Origin Heatmap (Median Forecasts) ---
                    st.subheader("Full-Sample Horizon-Origin Heatmap (Median Forecasts)")
                    median_forecasts = simulations_data.groupby(["ExecutionDate", "Horizon"])["SimulatedValue"].quantile(0.5).unstack()
                    fig = px.imshow(
                        median_forecasts.T,  # Transpose to align axes
                        labels={"x": "Execution Date", "y": "Horizon", "color": "Median Simulated Yield"},
                        title="Full-Sample Horizon-Origin Heatmap: Median Simulated Yields",
                        color_continuous_scale="RdBu",
                        origin="lower"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Visualization 2: Full-Sample 90% Prediction Interval Width Heatmap ---
                    st.subheader("Full-Sample Horizon-Origin Heatmap (90% Prediction Interval Width)")
                    pi_width = simulations_data.groupby(["ExecutionDate", "Horizon"])["SimulatedValue"].quantile(0.95).unstack() - \
                               simulations_data.groupby(["ExecutionDate", "Horizon"])["SimulatedValue"].quantile(0.05).unstack()
                    fig = px.imshow(
                        pi_width.T,  # Transpose to align axes
                        labels={"x": "Execution Date", "y": "Horizon", "color": "90% PI Width"},
                        title="Full-Sample Horizon-Origin Heatmap: 90% Prediction Interval Width",
                        color_continuous_scale="RdBu",
                        origin="lower"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Visualization 3: Full-Sample PIT Heatmap ---
                    st.subheader("Full-Sample Horizon-Origin Heatmap (PIT)")
                    # Mock actuals for PIT (replace this with real actuals if available)
                    actuals = simulations_data.groupby("ForecastDate")["SimulatedValue"].mean()
                    simulated_cdf = simulations_data.groupby(["ExecutionDate", "Horizon"]).apply(
                        lambda group: (group["SimulatedValue"] <= actuals.get(group["ForecastDate"].iloc[0], 0)).mean()
                    ).unstack()
                    fig = px.imshow(
                        simulated_cdf.T,  # Transpose to align axes
                        labels={"x": "Execution Date", "y": "Horizon", "color": "PIT"},
                        title="Full-Sample Horizon-Origin Heatmap: PIT",
                        color_continuous_scale="RdBu",
                        origin="lower"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Visualization 4: Actuals and Simulations with Fan Chart ---
                    st.subheader("Actuals and Simulations with Fan Chart")

                    # Mock actuals data (replace with real actuals if available)
                    actuals = simulations_data.groupby("ForecastDate")["SimulatedValue"].mean()  # Mocked actuals
                    forecasts_data = simulations_data.copy()  # Replace with actual forecast data if available

                    # Calculate percentiles (5th, 50th, 95th) for the fan chart
                    percentiles = simulations_data.groupby("ExecutionDate")["SimulatedValue"].quantile([0.05, 0.5, 0.95]).unstack(level=-1)
                    percentiles.columns = ["5th Percentile", "Median", "95th Percentile"]

                    # Merge with actuals for the time series plot
                    actuals_for_plot = forecasts_data.groupby("ForecastDate")["SimulatedValue"].last().dropna()
                    combined_data = percentiles.join(actuals_for_plot, how="inner")

                    # Create the fan chart with actuals
                    fig = go.Figure()

                    # Add fan chart (shaded area for 5th to 95th percentiles)
                    fig.add_trace(go.Scatter(
                        x=combined_data.index,
                        y=combined_data["95th Percentile"],
                        mode="lines",
                        line=dict(width=0),
                        name="95th Percentile",
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=combined_data.index,
                        y=combined_data["5th Percentile"],
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
                        y=combined_data["Median"],
                        mode="lines",
                        line=dict(color="#3a6bac", width=2),
                        name="Median Simulation"
                    ))

                    # Add actuals line
                    fig.add_trace(go.Scatter(
                        x=combined_data.index,
                        y=combined_data["SimulatedValue"],
                        mode="lines",
                        line=dict(color="#c28191", width=2),
                        name="Actual"
                    ))

                    # Update layout
                    fig.update_layout(
                        title="Actuals and Simulations with Fan Chart",
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


with tab_returns:
    st.title("Returns Analysis: VaR, CVaR, Observed, and Expected Returns")

    # Dropdown for selecting model
    st.subheader("Model Selection")
    selected_model = st.selectbox(
        "Select Model for Returns Analysis",
        estimated_models,  # Use the list of available models
        key="returns_model_selector"  # Unique key for this dropdown
    )

    # Load the annual metrics CSV for the selected model
    returns_folder_path = os.path.join(base_folder, selected_country, "returns", "estimated_returns", selected_model)
    annual_metrics_file = os.path.join(returns_folder_path, "annual_metrics.csv")

    if os.path.exists(annual_metrics_file):
        metrics_df = pd.read_csv(annual_metrics_file)

        # Dropdown for selecting maturity
        st.subheader("Maturity Selection")
        available_maturities = metrics_df["Maturity (Years)"].unique()
        selected_maturity = st.selectbox(
            "Select Maturity (Years)",
            sorted(available_maturities),
            key="returns_maturity_selector"  # Unique key for this dropdown
        )

        # Dropdown for selecting horizon
        st.subheader("Horizon Selection")
        available_horizons = metrics_df["Horizon (Years)"].unique()
        selected_horizon = st.selectbox(
            "Select Horizon (Years)",
            sorted(available_horizons),
            key="returns_horizon_selector"  # Unique key for this dropdown
        )

        # Filter the DataFrame for the selected maturity and horizon
        filtered_df = metrics_df[
            (metrics_df["Maturity (Years)"] == selected_maturity) &
            (metrics_df["Horizon (Years)"] == selected_horizon) &
            (metrics_df["Metric"].isin(["VaR", "CVaR", "Observed Annual Return", "Expected Returns"]))
        ]

        # Check for duplicates
        duplicate_rows = filtered_df[filtered_df.duplicated(subset=["Execution Date", "Metric"], keep=False)]
        if not duplicate_rows.empty:
            st.warning("Duplicate rows detected. Aggregating duplicates by taking the mean.")
            filtered_df = filtered_df.groupby(["Execution Date", "Metric"], as_index=False)["Value"].mean()

        # Pivot the data to align metrics by execution date
        pivot_df = filtered_df.pivot(index="Execution Date", columns="Metric", values="Value")

        # Ensure all required metrics are present
        required_metrics = ["VaR", "CVaR", "Observed Annual Return", "Expected Returns"]
        missing_metrics = [metric for metric in required_metrics if metric not in pivot_df.columns]
        if missing_metrics:
            st.warning(f"Missing the following metrics: {', '.join(missing_metrics)}")
        else:
            # Plot using Plotly
            fig = go.Figure()

            # Add Observed Annual Return
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df["Observed Annual Return"],
                mode="lines",
                name="Observed Annual Return",
                line=dict(color="#3a6bac"),
                marker=dict(size=6)
            ))

            # Add Expected Returns
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df["Expected Returns"],
                mode="lines",
                name="Expected Returns",
                line=dict(color="#eaa121"),
                marker=dict(size=6)
            ))

            # Add VaR
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df["VaR"],
                mode="lines",
                name="VaR (Threshold)",
                line=dict(color="#c28191") #, dash="dash"
            ))

            # Add CVaR
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df["CVaR"],
                mode="lines",
                name="CVaR (Tail Average)",
                line=dict(color="orange") # , dash="dot"
            ))

            # Highlight breaches where Observed Annual Return < VaR
            breaches = pivot_df["Observed Annual Return"] < pivot_df["VaR"]
            fig.add_trace(go.Scatter(
                x=pivot_df.index[breaches],
                y=pivot_df["Observed Annual Return"][breaches],
                mode="markers",
                name="VaR Breaches",
                marker=dict(color="black", size=8, symbol="x")
            ))

            # Update layout
            fig.update_layout(
                title=f"VaR, CVaR, Observed, and Expected Returns Across Execution Dates (Horizon={selected_horizon}y, Maturity={selected_maturity}y)",
                xaxis_title="Execution Date",
                yaxis_title="Returns",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                template="plotly_white"
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"File not found: {annual_metrics_file}")

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
        model_simulations["Model"] = model  # Add a column to identify the model
        return model_simulations
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files are found


from scipy.stats import gaussian_kde
import numpy as np
import plotly.graph_objects as go

with tab_simulation_comparison:
    st.title("Simulation Comparison: Distribution Analysis Across Models")

    # Dropdown for selecting models
    st.subheader("Model Selection")
    selected_models = st.multiselect(
        "Select Models to Compare",
        estimated_models,  # Use the list of available models
        default=estimated_models[:2],  # Default to the first two models
        key="simulation_comparison_model_selector"  # Unique key for this dropdown
    )

    # Dropdown for selecting maturity
    st.subheader("Maturity Selection")
    if available_maturities.size > 0:
        selected_maturity = st.selectbox(
            "Select Maturity for Simulations",
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
            maturity_folder = f"{selected_maturity}_years"  # Format maturity as "X.XX_years"
            folder_path = os.path.join(base_folder, selected_country, "returns", "estimated_returns", model, "annual", maturity_folder)

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

            # Row 1: Time-Series Summary Statistics
            st.subheader("Time-Series Summary Statistics Comparison")

            # Define a color palette for models
            color_palette = ["#3a6bac", "#aa322f", "#eaa121", "#633d83", "#d55b20", "#427f6d", "#784722"]

            # Calculate percentiles and summary statistics for each model
            summary_data = combined_data.groupby(["ExecutionDate", "Model"])["AnnualReturn"].agg(
                mean="mean",
                p5=lambda x: np.percentile(x, 5),
                p95=lambda x: np.percentile(x, 95)
            ).reset_index()

            # Plot the results using Plotly
            fig_time_series = go.Figure()

            # Loop through each selected model and add traces
            for i, model in enumerate(selected_models):
                model_data = summary_data[summary_data["Model"] == model]

                # Define the color for this model
                model_color = color_palette[i % len(color_palette)]

                # Add mean line
                fig_time_series.add_trace(go.Scatter(
                    x=model_data["ExecutionDate"],
                    y=model_data["mean"],
                    mode="lines",
                    name=f"{model} Mean",
                    line=dict(width=2, color=model_color)
                ))

                # Add shaded area for 5th to 95th percentiles
                fig_time_series.add_trace(go.Scatter(
                    x=model_data["ExecutionDate"].tolist() + model_data["ExecutionDate"].tolist()[::-1],
                    y=model_data["p95"].tolist() + model_data["p5"].tolist()[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(255 * (i / len(color_palette)))}, 100, 200, 0.2)",  # Adjust transparency of the band
                    line=dict(width=0),
                    name=f"{model} 5th-95th Percentile"
                ))

            # Update layout
            fig_time_series.update_layout(
                title="Simulation Distributions Comparison Across Models (Time-Series)",
                xaxis_title="Execution Date",
                yaxis_title="Annual Return",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                template="plotly_white"
            )

            # Display the plot
            st.plotly_chart(fig_time_series, use_container_width=True)

            # Row 2: KDE Subplots for Each Horizon (2x3 layout with an additional "All Horizons" plot)
            st.subheader("KDE Distribution by Horizon and Overall Across Models")

            # Get unique horizons
            unique_horizons = sorted(combined_data["Horizon (Years)"].unique())

            # Create a 2x3 grid using Streamlit columns
            rows = []
            n_cols = 3  # Number of columns
            for i in range(0, len(unique_horizons) + 1, n_cols):
                # Append a new row of columns
                rows.append(st.columns(n_cols))

            # Loop through each horizon and add KDE plots
            for idx, horizon in enumerate(unique_horizons):
                row_idx = idx // n_cols
                col_idx = idx % n_cols

                with rows[row_idx][col_idx]:
                    # Filter data for the current horizon
                    horizon_data = combined_data[combined_data["Horizon (Years)"] == horizon]

                    # Create a Plotly figure for the current horizon
                    fig_kde = go.Figure()

                    # Loop through selected models
                    for i, model in enumerate(selected_models):
                        model_data = horizon_data[horizon_data["Model"] == model]

                        if not model_data.empty:
                            # Calculate KDE
                            kde = gaussian_kde(model_data["AnnualReturn"])
                            x_range = np.linspace(model_data["AnnualReturn"].min(), model_data["AnnualReturn"].max(), 500)
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
                            var_5 = np.percentile(model_data["AnnualReturn"], 5)

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
                        title=f"Horizon {horizon} Years",
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

                # Loop through selected models
                for i, model in enumerate(selected_models):
                    model_data = combined_data[combined_data["Model"] == model]

                    if not model_data.empty:
                        # Calculate KDE for all horizons combined
                        kde = gaussian_kde(model_data["AnnualReturn"])
                        x_range = np.linspace(model_data["AnnualReturn"].min(), model_data["AnnualReturn"].max(), 500)
                        y_kde = kde(x_range)

                        # Define the color for this model
                        model_color = color_palette[i % len(color_palette)]

                        # Add KDE trace to the figure
                        fig_all_horizons.add_trace(go.Scatter(
                            x=x_range,
                            y=y_kde,
                            mode="lines",
                            name=f"{model}",
                            line=dict(width=2, color=model_color)
                        ))

                        # Calculate VaR (5th percentile)
                        var_5 = np.percentile(model_data["AnnualReturn"], 5)

                        # Add vertical line for VaR
                        fig_all_horizons.add_shape(
                            type="line",
                            x0=var_5, x1=var_5,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            line=dict(color=model_color, width=2, dash="dot"),
                            name=f"{model} VaR"
                        )

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
                    title="All Horizons",
                    xaxis_title="Annual Return",
                    yaxis_title="Density",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    template="plotly_white"
                )

                # Display the "All Horizons" plot
                st.plotly_chart(fig_all_horizons, use_container_width=True)
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Define the base path for data
BASE_PATH = r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data"
COLORS = {
    'Red': '#aa322f',
    'Blue': '#3a6bac',
    'Yellow': '#eaa121',
    'Purple': '#633d83',
    'Orange': '#d55b20',
    'Black': '#000000',
    'Brown': '#784722',
    'Green': '#427f6d'
}

def plot_forecasts_with_actuals_dots_and_line(df_predictions, model='AR(1)', selected_target='beta1'):
    """
    Plot forecasted paths as dots and actual beta values as a line for a specific model and target variable.

    Args:
        df_predictions (pd.DataFrame): DataFrame containing forecasted paths with columns:
            - ExecutionDate
            - ForecastDate
            - Prediction
            - Actual
            - Model
        model (str): The model to filter and plot (default: 'AR(1)').
        selected_target (str): The target variable to filter and plot (default: 'beta1').

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure object.
    """
    fig = go.Figure()

    # Filter predictions for the selected model and target variable
    df_predictions = df_predictions[
        (df_predictions['Model'] == model) & 
        (df_predictions['TargetVariable'] == selected_target)
    ]

    # Ensure ExecutionDate and ForecastDate are datetime objects
    df_predictions['ExecutionDate'] = pd.to_datetime(df_predictions['ExecutionDate'])
    df_predictions['ForecastDate'] = pd.to_datetime(df_predictions['ForecastDate'])
    df_predictions = df_predictions.sort_values(by=["ExecutionDate", "ForecastDate"])
    # Extract the actual beta series (aligned with ForecastDate)
    actual_series = df_predictions.drop_duplicates(subset=['ForecastDate']).set_index('ForecastDate')['Actual']

    # Loop through execution dates and plot forecasts as dots
    unique_execution_dates = df_predictions['ExecutionDate'].unique()
    for i, execution_date in enumerate(unique_execution_dates):
        subset = df_predictions[df_predictions['ExecutionDate'] == execution_date]

        if not subset.empty:
            # Highlight every 60th execution date with larger blue dots
            #if i % 60 == 0:
            #    fig.add_trace(go.Scatter(
            #        x=subset['ForecastDate'], 
            #        y=subset['Prediction'], 
            #        mode='markers',  # Use markers only
            #        name=f"Forecast {execution_date.date()}",
            #        marker=dict(color='blue', size=4)  # Larger blue dots
            #    ))
            # Plot other execution dates as small gray dots
            #else:
                fig.add_trace(go.Scatter(
                    x=subset['ForecastDate'], 
                    y=subset['Prediction'], 
                    mode='markers',  # Use markers only
                    marker=dict(color='gray', size=2),  # Small gray dots
                    showlegend=False,
                    opacity=0.4
                ))

    # Highlight the last execution date with larger red dots
    #last_execution_date = unique_execution_dates[-1]
    #subset = df_predictions[df_predictions['ExecutionDate'] == last_execution_date]
    #if not subset.empty:
    #    fig.add_trace(go.Scatter(
    #        x=subset['ForecastDate'], 
    #        y=subset['Prediction'], 
    #        mode='markers', 
    #        name=f"Forecast {last_execution_date.date()}",
    #        marker=dict(color='red', size=4)  # Larger red dots
    #    ))

    # Add the actual beta series as a bold black line
    fig.add_trace(go.Scatter(
        x=actual_series.dropna().index, 
        y=actual_series.dropna().values, 
        mode='lines', 
        #name="Actual Beta", 
        line=dict(color='black', width=2)  # Black line for actual beta
    ))

    # Update layout for better visualization
    fig.update_layout(
        title=f"{model}: Forecasted vs Actual {selected_target.capitalize()}",
        xaxis_title="Date",
        #yaxis_title=selected_target.capitalize(),
        #legend=dict(font=dict(size=10)),
        template="plotly_white",
        showlegend=False,
        height=700,
        width=1200
    )

    return fig



# Load RMSE data for Factors
@st.cache_data
def load_rmse_data(output_folder):
    filepath = os.path.join(output_folder, "factors_rmse_metrics.csv")
    #st.write(f"Loading RMSE data from: {filepath}")  # Debugging output
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            st.warning(f"RMSE file {filepath} is empty.")
        return df
    except Exception as e:
        st.error(f"Error loading RMSE file: {e}")
        return pd.DataFrame()

# Load forecast data for Factors
@st.cache_data
def load_forecast_data(base_path, selected_country):
    data = []
    country_folder = os.path.join(base_path, selected_country)
    #st.write(f"Loading forecast data from: {country_folder}")  # Debugging output
    for root, _, files in os.walk(country_folder):
        for file in files:
            if file.endswith('.csv') and 'forecasts' in file:
                filepath = os.path.join(root, file)
                #st.write(f"Processing file: {filepath}")  # Debugging output
                try:
                    target_variable, model = file.split('_forecasts_')
                    model = model.replace('.csv', '')
                    df = pd.read_csv(filepath)
                    if df.empty:
                        st.warning(f"File {file} is empty.")
                    df['TargetVariable'] = target_variable
                    df['Model'] = model
                    data.append(df)
                except Exception as e:
                    st.error(f"Error processing file {file}: {e}")
    if data:
        return pd.concat(data, ignore_index=True)
    else:
        st.warning("No forecast data found.")
        return pd.DataFrame()

# Load RMSE data for Yields
@st.cache_data
def load_yield_rmse_data(output_folder, selected_country):
    """
    Load RMSE data for yield models and observed yield metrics for the selected country.

    Args:
        output_folder (str): Base folder path for the selected country.
        selected_country (str): Selected country (e.g., "US" or "EA").

    Returns:
        pd.DataFrame: Combined DataFrame containing RMSE data for models and observed yields.
    """
    # Define the filenames and corresponding model names
    files = {
        "metrics_timeseries_AR_1.csv": "AR_1",
        "metrics_timeseries_AR_1_Output_Gap_Direct_Inflation_UCSV.csv": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "metrics_timeseries_Mixed_Model.csv": "Mixed_Model",
    }

    # Observed yield metrics file for the selected country
    observed_file = f"{selected_country}_observed_yields_metrics_AR_1.csv"

    data = []
    # Load standard model files
    for file, model_name in files.items():
        filepath = os.path.join(output_folder, file)
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                st.warning(f"File {file} is empty.")
            else:
                df["Model"] = model_name  # Add a column for the model name
                df = df[df['Metric'] == 'RMSE']
                df['Maturity (Years)'] = df['Maturity (Years)'].astype(float)
                df['Execution Date'] = pd.to_datetime(df['Execution Date'], errors="coerce")
                df = df.sort_values(by=["Execution Date"])
                data.append(df)
        except Exception as e:
            st.error(f"Error loading file {file}: {e}")

    # Load observed yield metrics file
    observed_filepath = os.path.join(output_folder, observed_file)
    try:
        df_observed = pd.read_csv(observed_filepath)
        if df_observed.empty:
            st.warning(f"Observed file {observed_file} is empty.")
        else:
            df_observed["Model"] = "Observed_Yields_AR_1"  # Fixed model name for observed yields
            df_observed['Maturity (Years)'] = df_observed['Maturity (Years)'].str.replace(" years", "").astype(float)
            df_observed['Execution Date'] = pd.to_datetime(df_observed['Execution Date'], errors="coerce")
            df_observed['Horizon (Years)'] = df_observed['Horizon (Years)'].astype(float)
            data.append(df_observed)
    except Exception as e:
        st.error(f"Error loading observed file {observed_file}: {e}")

    # Combine all data into a single DataFrame
    if data:
        return pd.concat(data, ignore_index=True)
    else:
        st.warning("No data available for the yield models.")
        return pd.DataFrame()
    
# Factors Page
# Factors Page
def factors_page():
    #st.title("Factors Dashboard")

    # Sidebar: Country selection
    st.sidebar.header("Country Selection")
    available_countries = ["US", "EA"]  # Update this list based on your folder structure
    selected_country = st.sidebar.selectbox("Select a Country", available_countries)

    # Load precomputed RMSE data
    output_folder = os.path.join(BASE_PATH, selected_country)
    rmse = load_rmse_data(output_folder)

    # Load raw forecast data for Actual vs. Projected Paths
    forecast_data = load_forecast_data(BASE_PATH, selected_country)

    # Tabs for "Model Comparison" and "Model Details"
    tab1, tab2 = st.tabs(["Model Comparison", "Model Details"])

    # Tab 1: Model Comparison
    with tab1:
        st.subheader("Model Comparison")

        # Dropdown for model selection (multi-select)
        all_models = rmse['Model'].unique()
        default_models = ['AR_1', 'AR_1_Output_Gap_Direct', 'AR_1_Output_Gap_Direct_Inflation_UCSV']
        selected_models = st.multiselect("Select Models for Comparison", all_models, default=default_models)

        # Target variables
        target_variables = rmse['TargetVariable'].unique()

        # Row 1: RMSE Across Horizons
        st.subheader("RMSE Across Horizons")
        cols = st.columns(3)
        for i, target_variable in enumerate(target_variables):
            with cols[i]:
                #st.write(f"**{target_variable.capitalize()}**")

                # Filter data based on selected models and target variable
                filtered_rmse = rmse[
                    (rmse['TargetVariable'] == target_variable) &
                    (rmse['Model'].isin(selected_models))
                ]

                # Calculate RMSE by Horizon
                rmse_by_horizon = filtered_rmse.groupby(['Horizon', 'Model'])['RMSE'].mean().reset_index()

                # Create a new figure using plotly.graph_objects
                fig_horizon = go.Figure()

                # Add traces for each model
                for i, model in enumerate(selected_models):
                    model_data = rmse_by_horizon[rmse_by_horizon['Model'] == model]
                    color = list(COLORS.values())[i % len(COLORS)]  # Cycle through colors if needed
                    fig_horizon.add_trace(go.Scatter(
                        x=model_data['Horizon'],
                        y=model_data['RMSE'],
                        mode='lines',
                        name=model,
                        line=dict(color=color)
                    ))

                # Update layout for better visualization
                fig_horizon.update_layout(
                    title=f"{target_variable.capitalize()}",
                    xaxis_title="Horizon",
                    #yaxis_title="RMSE",
                    xaxis=dict(showgrid=True),
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        yanchor="top"
                    ),
                    template="plotly_white"
                )
                st.plotly_chart(fig_horizon, use_container_width=True)

        # Row 2: RMSE Across Execution Dates
        st.subheader("RMSE Across Execution Dates")
        cols = st.columns(3)
        for i, target_variable in enumerate(target_variables):
            with cols[i]:
                #st.write(f"**{target_variable.capitalize()}**")

                # Filter data based on selected models and target variable
                filtered_rmse = rmse[
                    (rmse['TargetVariable'] == target_variable) &
                    (rmse['Model'].isin(selected_models))
                ]

                # Calculate RMSE by Execution Date
                rmse_by_execution = filtered_rmse.groupby(['ExecutionDate', 'Model'])['RMSE'].mean().reset_index()

                # Create a new figure using plotly.graph_objects
                fig_execution = go.Figure()

                # Add traces for each model
                for j, model in enumerate(selected_models):
                    model_data = rmse_by_execution[rmse_by_execution['Model'] == model]
                    color = list(COLORS.values())[j % len(COLORS)]
                    fig_execution.add_trace(go.Scatter(
                        x=model_data['ExecutionDate'],
                        y=model_data['RMSE'],
                        mode='lines',
                        name=model,
                        line=dict(color=color)
                    ))
                
                # Update layout for better visualization
                fig_execution.update_layout(
                    title=f"{target_variable.capitalize()}",
                    xaxis_title="Execution Date",
                    #yaxis_title="RMSE",
                    xaxis=dict(showgrid=True),
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        yanchor="top"
                    ),
                    template="plotly_white"
                )
                st.plotly_chart(fig_execution, use_container_width=True)

    # Tab 2: Model Details
    # Tab 2: Model Details
    with tab2:
        st.subheader("Model Details")

        col1, col2 = st.columns(2)
        with col1:
            # Dropdown for specific target variable selection
            target_variables = forecast_data['TargetVariable'].unique()
            selected_target = st.selectbox("Select a Target Variable", target_variables)

        with col2:
            # Dropdown for specific model selection
            models = forecast_data['Model'].unique()
            selected_model = st.selectbox("Select a Model", models)

        col1, col2 = st.columns(2)
        with col1:
            # Row 1: Actual vs. Projected Paths
            #st.subheader(f"Actual vs. Projected Paths for {selected_target} and {selected_model}")

            # Generate the plot
            fig = plot_forecasts_with_actuals_dots_and_line(
                df_predictions=forecast_data,
                model=selected_model,
                selected_target=selected_target
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Row 2: 3D RMSE Surface Plot
            #st.subheader(f"3D RMSE Surface Plot for {selected_target} and {selected_model}")
            selected_horizons = [6, 12, 24, 36, 48, 60]
            filtered_surface = rmse[
                (rmse['TargetVariable'] == selected_target) &
                (rmse['Model'] == selected_model) &
                (rmse['Horizon'].isin(selected_horizons))
            ]

            if not filtered_surface.empty:
                # Ensure ExecutionDate is in datetime format
                filtered_surface['ExecutionDate'] = pd.to_datetime(filtered_surface['ExecutionDate'], errors="coerce")

                # Sort by ExecutionDate and Horizon
                filtered_surface = filtered_surface.sort_values(by=["ExecutionDate", "Horizon"])

                # Pivot the data to create a grid for execution dates, horizons, and RMSE values
                surface_pivot = filtered_surface.pivot(index="ExecutionDate", columns="Horizon", values="RMSE")
                surface_pivot = surface_pivot.fillna(0)  # Handle missing values

                # Convert execution dates to numeric (ordinal format) for plotting
                execution_dates_ordinal = surface_pivot.index.map(pd.Timestamp.toordinal)

                # Create the grid for the 3D plot
                X, Y = np.meshgrid(execution_dates_ordinal, surface_pivot.columns)
                Z = surface_pivot.T.values

                # Convert execution dates back to human-readable strings for hover labels
                execution_dates_str = surface_pivot.index.strftime('%Y-%m-%d')

                # Create the 3D surface plot
                fig_surface = go.Figure(data=[go.Surface(
                    z=Z,
                    x=execution_dates_str,
                    y=surface_pivot.columns,
                    colorscale="Viridis",
                    contours=dict(  # Add gridlines
                        x=dict(show=True, color="white", width=2),
                        y=dict(show=True, color="white", width=2),
                        #z=dict(show=True, color="white", width=2),
                    ),
                    colorbar=dict(len=0.5, thickness=15),
                    #colorbar=dict(title="RMSE"),
                )])

                # Update layout for better visualization
                fig_surface.update_layout(
                    title=f"3D RMSE Surface Plot",
                    scene=dict(
                        xaxis=dict(title="Execution Date", tickangle=45),
                        yaxis_title="Horizon",
                        zaxis_title="RMSE",
                        aspectratio=dict(x=1, y=1, z=0.7)
                    ),
                    xaxis=dict(showgrid=True),
                    width=1000,
                    height=800,
                    margin=dict(l=0, r=0, b=0, t=30)
                )

                # Display the 3D surface plot in Streamlit
                st.plotly_chart(fig_surface, use_container_width=False)
            else:
                st.warning("Surface data not available for the selected target and model.")
# Yields Page
def yields_page():
    # Sidebar: Country selection
    st.sidebar.header("Country Selection")
    available_countries = ["US", "EA", "UK"]  # Dynamically update this list if needed
    selected_country = st.sidebar.selectbox("Select a Country", available_countries)

    # Load data for the selected country
    output_folder = os.path.join(BASE_PATH, selected_country)
    yield_rmse = load_yield_rmse_data(output_folder, selected_country)

    

    # Tabs for "Model Comparison" and "Model Details"
    tab1, tab2 = st.tabs(["Model Comparison", "Model Details"])

    # Tab 1: Model Comparison
    with tab1:
        st.subheader("Model Comparison")

        # Filter the data for RMSE only
        rmse_data = yield_rmse[yield_rmse["Metric"] == "RMSE"]

        # Allow the user to select models to compare
        all_models = rmse_data["Model"].unique()
        default_models = ["AR_1", "AR_1_Output_Gap_Direct_Inflation_UCSV", "Mixed_Model"]
        # Ensure default models are part of the available models
        valid_defaults = [model for model in default_models if model in all_models]
        selected_models = st.multiselect("Select Models to Compare", all_models, default=valid_defaults)

        # Maturities of interest
        maturities = [0.25, 2, 5, 10]  # 3 months, 2 years, 5 years, 10 years

        # Row 1: RMSE Across Horizons
        st.subheader("RMSE Across Horizons")
        cols_horizon = st.columns(4)  # Create 4 columns for the maturities
        for i, maturity in enumerate(maturities):
            with cols_horizon[i]:
                st.write(f"**Maturity: {maturity} Years**")

                # Filter data for the specific maturity and selected models
                filtered_data = rmse_data[
                    (rmse_data["Maturity (Years)"] == maturity) &
                    (rmse_data["Model"].isin(selected_models))
                ]

                # Separate observed data for overlay
                observed_data = rmse_data[
                    (rmse_data["Maturity (Years)"] == maturity) &
                    (rmse_data["Model"] == "Observed_Yields_AR_1")  # Filter observed yields
                ]

                # Calculate RMSE by Horizon for selected models
                rmse_by_horizon = filtered_data.groupby(["Horizon (Years)", "Model"])["Value"].mean().reset_index()
                rmse_by_horizon_obs = observed_data.groupby(["Horizon (Years)", "Model"])["Value"].mean().reset_index()

                # Load STR and LTR data (replace with your actual data loading logic)
                forecast_data = pd.read_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_horizon_all_countries_indicators.csv")  # Replace with the correct path
                str_data = forecast_data[
                    (forecast_data["country"] == selected_country) & (forecast_data["indicator"] == "STR") & (forecast_data["horizon"].isin([1,2,3,4,5]))
                ]
                ltr_data = forecast_data[
                    (forecast_data["country"] == selected_country) & (forecast_data["indicator"] == "LTR") & (forecast_data["horizon"].isin([1,2,3,4,5]))
                ]

                # Create a new figure using plotly.graph_objects
                fig_horizon = go.Figure()

                # Add traces for each model
                for model in selected_models:
                    model_data = rmse_by_horizon[rmse_by_horizon["Model"] == model]
                    fig_horizon.add_trace(go.Scatter(
                        x=model_data["Horizon (Years)"],
                        y=model_data["Value"],
                        mode="lines+markers",
                        name=model
                    ))

                # Add observed yield metrics as a separate trace
                #if not rmse_by_horizon_obs.empty:
                #    fig_horizon.add_trace(go.Scatter(
                #        x=rmse_by_horizon_obs["Horizon (Years)"],
                #        y=rmse_by_horizon_obs["Value"],
                #        mode="lines+markers",
                #        name="Observed_Yields_AR_1",
                #        line=dict(dash="dot", color="red")  # Use a dashed red line for observed metrics
                #    ))

                # Overlay STR results for 0.25 maturity
                if maturity == 0.25 and not str_data.empty:
                    fig_horizon.add_trace(go.Scatter(
                        x=str_data["horizon"],
                        y=str_data["RMSE_AR1"]/100,
                        mode="lines+markers",
                        name="STR AR(1) Forecast",
                        line=dict(color="brown", width=2)
                    ))
                    fig_horizon.add_trace(go.Scatter(
                        x=str_data["horizon"],
                        y=str_data["RMSE_consensus"]/100,
                        mode="lines+markers",
                        name="STR Consensus Forecast",
                        line=dict(color="orange", width=2)
                    ))

                # Overlay LTR results for 10 maturity
                if maturity == 10 and not ltr_data.empty:
                    fig_horizon.add_trace(go.Scatter(
                        x=ltr_data["horizon"],
                        y=ltr_data["RMSE_AR1"]/100,
                        mode="lines+markers",
                        name="LTR AR(1) Forecast",
                        line=dict(color="brown", width=2)
                    ))
                    fig_horizon.add_trace(go.Scatter(
                        x=ltr_data["horizon"],
                        y=ltr_data["RMSE_consensus"]/100,
                        mode="lines+markers",
                        name="LTR Consensus Forecast",
                        line=dict(color="orange", width=2)
                    ))

                # Update layout for better visualization
                fig_horizon.update_layout(
                    xaxis_title="Horizon (Years)",
                    yaxis_title="RMSE",
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    margin=dict(l=10, r=10, t=30, b=10),  # Adjust margins for compact layout
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        yanchor="top"
                    ),
                    template="plotly_white"
                )
                st.plotly_chart(fig_horizon, use_container_width=True)

        # Row 2: RMSE Across Execution Dates
        st.subheader("RMSE Across Execution Dates")
        cols_execution = st.columns(4)  # Create 4 columns for the maturities
        for i, maturity in enumerate(maturities):
            with cols_execution[i]:
                st.write(f"**Maturity: {maturity} Years**")

                # Filter data for the specific maturity and selected models
                filtered_data = rmse_data[
                    (rmse_data["Maturity (Years)"] == maturity) &
                    (rmse_data["Model"].isin(selected_models))
                ]

                # Separate observed data for overlay
                observed_data = rmse_data[
                    (rmse_data["Maturity (Years)"] == maturity) &
                    (rmse_data["Model"] == "Observed_Yields_AR_1")  # Filter observed yields
                ]

                # Load STR and LTR data (replace with your actual data loading logic)
                forecast_data = pd.read_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_exec_all_countries_indicators.csv")  # Replace with the correct path
                str_data = forecast_data[
                    (forecast_data["country"] == selected_country) & (forecast_data["indicator"] == "STR")
                ]
                ltr_data = forecast_data[
                    (forecast_data["country"] == selected_country) & (forecast_data["indicator"] == "LTR")
                ]

                # Calculate RMSE by Execution Date for selected models
                rmse_by_execution = filtered_data.groupby(["Execution Date", "Model"])["Value"].mean().reset_index()
                rmse_by_execution_obs = observed_data.groupby(["Execution Date", "Model"])["Value"].mean().reset_index()

                # Ensure Execution Date is in datetime format
                rmse_by_execution = rmse_by_execution.sort_values(by=["Execution Date"])
                rmse_by_execution_obs = rmse_by_execution_obs.sort_values(by=["Execution Date"])

                # Create a new figure using plotly.graph_objects
                fig_execution = go.Figure()

                # Add traces for each model
                for model in selected_models:
                    model_data = rmse_by_execution[rmse_by_execution["Model"] == model]
                    fig_execution.add_trace(go.Scatter(
                        x=model_data["Execution Date"],
                        y=model_data["Value"],
                        mode="lines",
                        name=model
                    ))

                # Add observed yield metrics as a separate trace
                #if not rmse_by_execution_obs.empty:
                #    fig_execution.add_trace(go.Scatter(
                #        x=rmse_by_execution_obs["Execution Date"],
                #        y=rmse_by_execution_obs["Value"],
                #        mode="lines",
                #        name="Observed_Yields_AR_1",
                #        line=dict(dash="dot", color="red")  # Use a dashed red line for observed metrics
                #    ))

                # Overlay STR results for 0.25 maturity
                if maturity == 0.25 and not str_data.empty:
                    fig_execution.add_trace(go.Scatter(
                        x=str_data["executionDate"],
                        y=str_data["RMSE_AR1"]/100,
                        mode="lines",
                        name="STR AR(1) Forecast",
                        line=dict(color="brown", width=2)
                    ))
                    fig_execution.add_trace(go.Scatter(
                        x=str_data["executionDate"],
                        y=str_data["RMSE_consensus"]/100,
                        mode="lines",
                        name="STR Consensus Forecast",
                        line=dict(color="orange", width=2)
                    ))

                # Overlay LTR results for 10 maturity
                if maturity == 10 and not ltr_data.empty:
                    fig_execution.add_trace(go.Scatter(
                        x=ltr_data["executionDate"],
                        y=ltr_data["RMSE_AR1"]/100,
                        mode="lines",
                        name="LTR AR(1) Forecast",
                        line=dict(color="brown", width=2)
                    ))
                    fig_execution.add_trace(go.Scatter(
                        x=ltr_data["executionDate"],
                        y=ltr_data["RMSE_consensus"]/100,
                        mode="lines",
                        name="LTR Consensus Forecast",
                        line=dict(color="orange", width=2)
                    ))

                # Update layout for better visualization
                fig_execution.update_layout(
                    xaxis_title="Execution Date",
                    yaxis_title="RMSE",
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    margin=dict(l=10, r=10, t=30, b=10),  # Adjust margins for compact layout
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        yanchor="top"
                    ),
                    template="plotly_white"
                )
                st.plotly_chart(fig_execution, use_container_width=True)
    # Tab 2: Model Details
    with tab2:
    # Row 2: 3D RMSE Surface Plot
        st.subheader(f"3D RMSE Surface Plot for Selected Model and Horizon")

        # Allow the user to select a single model for the 3D graph
        selected_model = st.selectbox("Select a Model for the 3D Surface Plot", all_models)

        # Allow the user to select a horizon
        #available_horizons = rmse_data["Horizon (Years)"].unique()
        available_horizons = [0.5, 1, 2, 3, 4, 5]
        selected_horizon = st.selectbox("Select a Horizon for the 3D Surface Plot", sorted(available_horizons))

        # Filter the data for the selected model and horizon
        filtered_surface = rmse_data[
            (rmse_data["Model"] == selected_model) &  # Single selected model
            (rmse_data["Horizon (Years)"] == selected_horizon) &  # Selected horizon
            (rmse_data["Metric"] == "RMSE")  # Filter for RMSE metric
        ]

        if not filtered_surface.empty:
            # Ensure Execution Date is in datetime format
            filtered_surface["Execution Date"] = pd.to_datetime(filtered_surface["Execution Date"], errors="coerce")

            # Check for duplicates and resolve them by aggregating (e.g., taking the mean)
            duplicates = filtered_surface.duplicated(subset=["Execution Date", "Maturity (Years)"], keep=False)
            if duplicates.any():
                st.warning(f"Duplicate entries found for model '{selected_model}' and horizon '{selected_horizon}'. Aggregating duplicate values by taking the mean.")
                filtered_surface = filtered_surface.groupby(["Execution Date", "Maturity (Years)"], as_index=False).mean()

            # Pivot the data to create a grid for execution dates, maturities, and RMSE values
            surface_pivot = filtered_surface.pivot(index="Execution Date", columns="Maturity (Years)", values="Value")

            # Handle missing values by filling NaN with 0
            surface_pivot = surface_pivot.fillna(0)

            # Create the grid for the 3D plot
            X, Y = np.meshgrid(surface_pivot.index.map(pd.Timestamp.toordinal), surface_pivot.columns)
            Z = surface_pivot.T.values  # Transpose the Z matrix to match the swapped axes

            # Convert execution dates back to human-readable strings for hover labels
            execution_dates_str = surface_pivot.index.strftime('%Y-%m-%d')

            # Create the 3D surface plot
            fig_surface = go.Figure(data=[go.Surface(
                z=Z,
                x=execution_dates_str,  # Use human-readable dates for the x-axis
                y=surface_pivot.columns,  # Maturities for the y-axis
                colorscale="Viridis",
                contours=dict(  # Add gridlines
                    x=dict(show=True, color="white", width=2),
                    y=dict(show=True, color="white", width=2),
                    #z=dict(show=True, color="white", width=2),
                ),
                colorbar=dict(len=0.5, thickness=15, title="RMSE")
            )])

            # Update layout for better visualization
            fig_surface.update_layout(
                #title=f"3D RMSE Surface Plot", # <br>(Model: {selected_model}, Horizon: {selected_horizon} Years)
                scene=dict(
                    xaxis_title="Execution Date",
                    yaxis_title="Maturity (Years)",
                    zaxis_title="RMSE",
                    xaxis=dict(tickangle=45),  # Rotate x-axis labels for better readability
                ),
                width=1000,  # Increase plot width
                height=800,  # Increase plot height
                margin=dict(l=0, r=0, b=0, t=50)  # Adjust margins
            )

            # Display the 3D surface plot
            st.plotly_chart(fig_surface, use_container_width=False)
        else:
            st.warning(f"No surface data available for model '{selected_model}' and horizon '{selected_horizon}'.")

import plotly.express as px

import plotly.graph_objects as go

# Plot RMSE Across Horizons
def plot_rmse_horizon(rmse_horizon, country, indicator):
    filtered_data = rmse_horizon[(rmse_horizon["country"] == country) & (rmse_horizon["indicator"] == indicator)]
    
    # Create a blank figure
    fig = go.Figure()
    
    # Add AR(1) RMSE trace
    fig.add_trace(
        go.Scatter(
            x=filtered_data["horizon"],
            y=filtered_data["RMSE_AR1"],
            mode="lines+markers",
            name="AR(1) RMSE"
        )
    )
    
    # Add Consensus RMSE trace
    fig.add_trace(
        go.Scatter(
            x=filtered_data["horizon"],
            y=filtered_data["RMSE_consensus"],
            mode="lines+markers",
            name="Consensus RMSE"
        )
    )
    
    # Update layout with gridlines and labels
    fig.update_layout(
        title="RMSE Across Horizons",
        xaxis=dict(
            title="Horizon",
            showgrid=True
        ),
        yaxis=dict(
            title="RMSE",
            showgrid=True
        ),
        template="plotly_white"
    )
    return fig

# Plot RMSE Across Execution Dates
def plot_rmse_exec(rmse_exec, country, indicator):
    filtered_data = rmse_exec[(rmse_exec["country"] == country) & (rmse_exec["indicator"] == indicator)]
    
    # Create a blank figure
    fig = go.Figure()
    
    # Add AR(1) RMSE trace
    fig.add_trace(
        go.Scatter(
            x=filtered_data["executionDate"],
            y=filtered_data["RMSE_AR1"],
            mode="lines",
            name="AR(1) RMSE"
        )
    )
    
    # Add Consensus RMSE trace
    fig.add_trace(
        go.Scatter(
            x=filtered_data["executionDate"],
            y=filtered_data["RMSE_consensus"],
            mode="lines",
            name="Consensus RMSE"
        )
    )
    
    # Update layout with gridlines and labels
    fig.update_layout(
        title="RMSE Across Execution Dates",
        xaxis=dict(
            title="Execution Date",
            showgrid=True
        ),
        yaxis=dict(
            title="RMSE",
            showgrid=True
        ),
        template="plotly_white"
    )
    return fig
# Plot RMSE by Horizon and Execution Date
def plot_rmse_horizon_exec(rmse_horizon_exec, country, indicator):
    filtered_data = rmse_horizon_exec[(rmse_horizon_exec["country"] == country) & (rmse_horizon_exec["indicator"] == indicator)]
    fig = px.scatter(
        filtered_data,
        x="executionDate",
        y="horizon",
        size="RMSE_AR1",
        color="RMSE_consensus",
        labels={"executionDate": "Execution Date", "horizon": "Horizon", "RMSE_consensus": "Consensus RMSE"},
        title="RMSE by Horizon and Execution Date",
        color_continuous_scale="Viridis"
    )
    return fig

# Plot Forecast vs. Realized Values
def plot_forecast_vs_realized(df_forecasts, country, indicator):
    """
    Plot forecasted paths as dots and realized values as a line for a specific country and indicator.

    Args:
        df_forecasts (pd.DataFrame): DataFrame containing forecasted paths with columns:
            - forecast_date
            - forecasted_month
            - monthly_forecast
            - actual
        country (str): The country code (e.g., 'US', 'EA').
        indicator (str): The indicator name (e.g., 'GDP', 'CPI').

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure object.
    """
    fig = go.Figure()

    # Ensure forecast_date and forecasted_month are datetime objects
    df_forecasts['forecast_date'] = pd.to_datetime(df_forecasts['executionDate'])
    df_forecasts['forecasted_month'] = pd.to_datetime(df_forecasts['forecastedDate'])

    # Sort the data for proper plotting
    df_forecasts = df_forecasts.sort_values(by=["forecast_date", "forecasted_month"])

    # Extract the realized series (aligned with forecasted_month)
    actual_series = df_forecasts.drop_duplicates(subset=['forecasted_month']).set_index('forecasted_month')['actual']

    # Loop through forecast dates and plot forecasts as dots
    unique_forecast_dates = df_forecasts['forecast_date'].unique()
    for i, forecast_date in enumerate(unique_forecast_dates):
        subset = df_forecasts[df_forecasts['forecast_date'] == forecast_date]

        if not subset.empty:
            # Plot forecasts as small gray dots
            fig.add_trace(go.Scatter(
                x=subset['forecasted_month'],
                y=subset['consensus_fcst'],  # Annualized forecast
                mode='lines+markers',  # Use markers only
                marker=dict(color='gray', size=2),  # Small gray dots
                showlegend=False,
                opacity=0.4
            ))

    # Add the realized series as a bold black line
    fig.add_trace(go.Scatter(
        x=actual_series.dropna().index,
        y=actual_series.dropna().values,
        mode='lines+markers',
        name="Realized",
        line=dict(color='black', width=2)  # Black line for realized values
    ))

    # Update layout for better visualization
    fig.update_layout(
        title=f"Forecasted vs Realized {indicator} for {country}",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        showlegend=False,
        height=700,
        width=1200
    )

    return fig

# Consensus Page
def consensus_page():
    st.title("Consensus Analysis")

    rmse_horizon = pd.read_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_horizon_all_countries_indicators.csv")
    rmse_exec = pd.read_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_exec_all_countries_indicators.csv")
    rmse_horizon_exec = pd.read_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\rmse_horizon_exec_all_countries_indicators.csv")
    forecast_data = pd.read_csv(r"\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\backtest_results_all_countries_indicators.csv")

    # Sidebar: Dropdowns for Country and Indicator
    st.sidebar.header("Filter Options")
    countries = forecast_data["country"].unique()
    indicators = forecast_data["indicator"].unique()
    selected_country = st.sidebar.selectbox("Select a Country", countries)
    selected_indicator = st.sidebar.selectbox("Select an Indicator", indicators)

    # Display selected country and indicator
    st.markdown(f"### Country: {selected_country} | Indicator: {selected_indicator}")

    # 2x2 Grid of Plots
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_rmse_horizon(rmse_horizon, selected_country, selected_indicator), use_container_width=True)
    with col2:
        st.plotly_chart(plot_rmse_exec(rmse_exec, selected_country, selected_indicator), use_container_width=True)

    #col3, col4 = st.columns(2)
    #with col3:
    #    st.plotly_chart(plot_rmse_horizon_exec(rmse_horizon_exec, selected_country, selected_indicator), use_container_width=True)
    #with col4:
    # Filter data for the selected country and indicator
    filtered_data = forecast_data[
        (forecast_data['country'] == selected_country) & 
        (forecast_data['indicator'] == selected_indicator)
    ]

    # Generate the plot
    fig = plot_forecast_vs_realized(filtered_data, selected_country, selected_indicator)

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    #st.plotly_chart(plot_forecast_vs_realized(forecast_data, selected_country, selected_indicator), use_container_width=True)


# Main App
def main():
    # Enable wide mode
    st.set_page_config(layout="wide", page_title="RMSE Analysis")

    # Sidebar: Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Factors", "Yields", "Consensus"])

    # Page Navigation
    if page == "Factors":
        factors_page()
    elif page == "Yields":
        yields_page()
    elif page == "Consensus":
        consensus_page()

if __name__ == "__main__":
    main()
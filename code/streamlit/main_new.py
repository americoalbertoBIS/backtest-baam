import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# Helper Functions
def load_data(folder_path, model, target_variable, file_name):
    file_path = os.path.join(folder_path, model, target_variable, file_name)
    if os.path.exists(file_path):
        if file_name.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_name.endswith('.parquet'):
            return pd.read_parquet(file_path)
    else:
        st.warning(f"File not found: {file_path}")
        return None

# App Configuration
st.set_page_config(layout="wide", page_title="Backtesting Results Dashboard")

# Sidebar: Country Selection
st.sidebar.header("Configuration")
countries = ["US", "EA", "UK"]
selected_country = st.sidebar.selectbox("Select Country", countries, index=0)

# Dynamically update the folder path based on the selected country
base_folder = r"K:\RMAS\Users\Alberto\backtest-baam\data"
folder_path = os.path.join(base_folder, selected_country, "factors")

# Get available models and target variables
models = [f.name for f in os.scandir(folder_path) if f.is_dir() and f.name != "archive"]  # Exclude "archive"
target_variables = ["beta1", "beta2", "beta3"]

# Date picker for user to select the starting date
start_date = st.sidebar.date_input("Select Start Date", value=pd.to_datetime("2000-01-01"))

# Main Layout
st.title("Backtesting Results Dashboard")

# Create a 3-column layout for beta1, beta2, beta3
cols = st.columns(3)
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
        
        # Loop through the two plot types (RMSE by Horizon and RMSE by Execution Date)
        for plot_type in ["RMSE by Horizon", "RMSE by Execution Date"]:
            st.subheader(plot_type)
            
            # Combine data for all selected models
            combined_data = []
            for model in selected_models:
                if plot_type == "RMSE by Horizon":
                    data = load_data(folder_path, model, target_variable, "outofsample_metrics_by_horizon.csv")
                    if data is not None:
                        data["Model"] = model
                        combined_data.append(data)
                elif plot_type == "RMSE by Execution Date":
                    data = load_data(folder_path, model, target_variable, "outofsample_metrics_by_execution_date.csv")
                    if data is not None:
                        data["Model"] = model
                        combined_data.append(data)
            
            # Create the plot if data is available
            if combined_data:
                combined_data = pd.concat(combined_data, ignore_index=True)
                
                if plot_type == "RMSE by Horizon":
                    fig = px.line(
                        combined_data,
                        x="Horizon",
                        y="rmse",
                        color="Model",
                        markers=True,
                        title=f"{plot_type} ({target_variable})",
                        labels={"rmse": "RMSE", "Horizon": "Horizon"},
                    )
                elif plot_type == "RMSE by Execution Date":
                    fig = px.line(
                        combined_data,
                        x="ExecutionDate",
                        y="rmse",
                        color="Model",
                        title=f"{plot_type} ({target_variable})",
                        labels={"rmse": "RMSE", "ExecutionDate": "Execution Date"},
                    )
                
                # Update layout to place legends at the bottom
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
                st.warning(f"No data available for {plot_type} ({target_variable})")
        
        # Row: Adjusted R-Squared Metric (placed at the bottom)
        st.subheader("Adjusted R-Squared Metric")
        combined_data = []
        for model in selected_models:
            # Load insample_metrics.csv for the selected model and beta
            data = load_data(folder_path, model, target_variable, "insample_metrics.csv")
            if data is not None:
                # Filter data for the adjusted_r_squared metric
                filtered_data = data[data['metric'] == 'adjusted_r_squared'][['execution_date', 'value']].copy()
                filtered_data['Model'] = model
                combined_data.append(filtered_data)
        
        # Create the plot if data is available
        if combined_data:
            combined_data = pd.concat(combined_data, ignore_index=True)
            fig = px.line(
                combined_data,
                x="execution_date",
                y="value",
                color="Model",
                title=f"Adjusted R-Squared ({target_variable})",
                labels={"execution_date": "Execution Date", "value": "Adjusted R-Squared"},
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
            st.warning(f"No data available for Adjusted R-Squared ({target_variable})")
        
        # Row: Average RMSE by Horizon (from outofsample_metrics_by_row.csv)
        st.subheader("Average RMSE by Horizon (From Selected Start Date)")
        combined_data = []
        for model in selected_models:
            data = load_data(folder_path, model, target_variable, "outofsample_metrics_by_row.csv")
            if data is not None:
                filtered_data = data[pd.to_datetime(data['ExecutionDate']) >= pd.to_datetime(start_date)]
                filtered_data = filtered_data.groupby("Horizon")["RMSE_Row"].mean().reset_index()
                filtered_data["Model"] = model
                combined_data.append(filtered_data)
        
        if combined_data:
            combined_data = pd.concat(combined_data, ignore_index=True)
            st.write(combined_data)
        
        # Row: RMSE by Horizon (from forecasts.csv)
        st.subheader("RMSE by Horizon (From Selected Start Date)")
        combined_data = []
        for model in selected_models:
            data = load_data(folder_path, model, target_variable, "forecasts.csv")
            if data is not None:
                filtered_data = data[pd.to_datetime(data['ExecutionDate']) >= pd.to_datetime(start_date)]
                filtered_data = filtered_data.dropna(subset=["Actual"])
                rmse_data = (
                    filtered_data
                    .groupby("Horizon")
                    .apply(lambda x: np.sqrt(((x["Prediction"] - x["Actual"]) ** 2).mean()))
                    .reset_index(name="RMSE")
                )
                rmse_data["Model"] = model
                combined_data.append(rmse_data)
        
        if combined_data:
            combined_data = pd.concat(combined_data, ignore_index=True)
            fig = px.line(
                combined_data,
                x="Horizon",
                y="RMSE",
                color="Model",
                title=f"RMSE by Horizon (From Forecasts) ({target_variable})",
                labels={"Horizon": "Horizon", "RMSE": "RMSE"},
            )
            st.plotly_chart(fig, use_container_width=True)
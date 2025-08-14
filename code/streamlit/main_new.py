import streamlit as st
import pandas as pd
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

# Folder Path
folder_path = r"L:\RMAS\Users\Alberto\backtest-baam\data\US\factors"

# Get available models and target variables
models = [f.name for f in os.scandir(folder_path) if f.is_dir() and f.name != "archive"]  # Exclude "archive"
target_variables = ["beta1", "beta2", "beta3"]

# Sidebar: Select Models
st.sidebar.header("Configuration")
selected_models = st.sidebar.multiselect("Select Models", models, default=models[:1])

# Main Layout
st.title("Backtesting Results Dashboard")

# Create a 2x3 grid using st.columns and st.container
for row_label, plot_type in enumerate(["RMSE by Horizon", "RMSE by Execution Date"]):
    with st.container():
        st.subheader(plot_type)
        
        # Create 3 columns for beta1, beta2, beta3
        cols = st.columns(3)
        for col, target_variable in zip(cols, target_variables):
            with col:
                st.write(f"Results for {target_variable}")
                
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
                            y=-0.2,  # Adjust legend position below the plot
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {plot_type} ({target_variable})")
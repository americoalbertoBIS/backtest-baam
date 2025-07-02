import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

metrics_df_raw = pd.read_csv(r"C:\git\backtest-baam\data\US\metrics_timeseries.csv")
metrics_df_old = metrics_df_raw.iloc[1066:,:].copy()

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_3d_rmse_interactive_1(data, horizon, output_html="rmse_plot_AR1.html"):
    """
    Create an interactive 3D plot for RMSE values across maturities and execution dates
    for a specific horizon using Plotly, and save it as an HTML file.

    Args:
        data (pd.DataFrame): Dataset containing RMSE values with columns:
            - "Execution Date"
            - "Maturity (Years)"
            - "Horizon (Years)"
            - "Metric"
            - "Value"
        horizon (float): The forecasting horizon (in years) to plot.
        output_html (str): The file name to save the interactive plot as an HTML file.
    """
    # Ensure Execution Date is in datetime format
    data["Execution Date"] = pd.to_datetime(data["Execution Date"], errors="coerce")

    # Filter the data for the selected horizon and RMSE metric
    rmse_data = data[(data["Horizon (Years)"] == horizon) & (data["Metric"] == "RMSE")]

    # Remove duplicate rows for the same Execution Date and Maturity (Years)
    #rmse_data = rmse_data.sort_values(by=["Execution Date", "Maturity (Years)", "Value"])  # Sort by Value if needed
    #rmse_data = rmse_data.drop_duplicates(subset=["Execution Date", "Maturity (Years)"], keep="first")
    #rmse_data["Maturity (Years)"] = rmse_data["Maturity (Years)"].astype(float)
    # Pivot the data to create a grid for execution dates, maturities, and RMSE values
    rmse_pivot = rmse_data.pivot(index="Execution Date", columns="Maturity (Years)", values="Value")

    # Handle missing values by filling NaN with 0
    #rmse_pivot = rmse_pivot.fillna(0)

    # Convert execution dates to numeric (ordinal format) for plotting
    execution_dates_ordinal = rmse_pivot.index.map(pd.Timestamp.toordinal)

    # Create the grid for the 3D plot
    X, Y = np.meshgrid(execution_dates_ordinal, rmse_pivot.columns)  # X: Execution Dates, Y: Maturities
    Z = rmse_pivot.T.values  # Transpose the Z matrix to match the swapped axes

    # Replace zeros in Z with NaN
    #Z[Z == 0] = np.nan

    # Convert execution dates back to human-readable strings for hover labels
    execution_dates_str = rmse_pivot.index.strftime('%Y-%m-%d')

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(
        z=Z,
        x=execution_dates_str,  # Use human-readable dates for the x-axis
        y=rmse_pivot.columns,  # Maturities for the y-axis
        colorscale="Viridis",
        colorbar=dict(title="RMSE"),
    )])

    # Update layout for better visualization
    fig.update_layout(
        title=f"Interactive 3D Plot of RMSE by Maturity and Execution Date<br>(Horizon: {horizon} Years)",
        scene=dict(
            xaxis_title="Execution Date",
            yaxis_title="Maturity (Years)",
            zaxis_title="RMSE",
            xaxis=dict(tickangle=45),  # Rotate x-axis labels
        ),
    )

    # Save the plot as an HTML file
    fig.write_html(output_html)
    print(f"Interactive 3D plot saved as {output_html}")

    # Show the plot
    fig.show()

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_3d_rmse_interactive_2(metrics_df):
    """
    Create an interactive 3D plot for RMSE values using Plotly, with gridlines.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing RMSE values with columns:
            - "Maturity (Years)"
            - "Horizon (Years)"
            - "Metric"
            - "Value"
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    # Filter the DataFrame for RMSE values
    rmse_df = metrics_df[metrics_df["Metric"] == "RMSE"]
    rmse_df['Value'] = pd.to_numeric(rmse_df['Value']) * 100  # Convert to percentage if needed

    # Extract unique maturities and horizons
    maturities = sorted(rmse_df["Maturity (Years)"].unique())
    horizons = sorted(rmse_df["Horizon (Years)"].unique())

    # Create a meshgrid for maturities and horizons
    X, Y = np.meshgrid(maturities, horizons)

    # Map RMSE values to the grid
    Z = np.zeros_like(X, dtype=float)
    for i, horizon in enumerate(horizons):
        for j, maturity in enumerate(maturities):
            # Get RMSE value for the current maturity and horizon
            value = rmse_df[
                (rmse_df["Maturity (Years)"] == maturity) &
                (rmse_df["Horizon (Years)"] == horizon)
            ]["Value"].values
            Z[i, j] = value[0] if len(value) > 0 else np.nan  # Handle missing values

    # Create the 3D surface plot using Plotly
    fig = go.Figure(data=[go.Surface(
        z=Z,
        x=maturities,  # Maturities on the x-axis
        y=horizons,  # Horizons on the y-axis
        colorscale="Viridis",  # Color scheme
        colorbar=dict(title="RMSE (%)"),  # Add color bar
        contours=dict(  # Add gridlines
            x=dict(show=True, color="white", width=2),
            y=dict(show=True, color="white", width=2),
            #z=dict(show=True, color="white", width=2),
        )
    )])

    # Update layout for better visualization
    fig.update_layout(
        title="Interactive 3D Plot of RMSE with Gridlines",
        scene=dict(
            xaxis_title="Maturity (Years)",
            yaxis_title="Horizon (Years)",
            zaxis_title="RMSE (%)",
            xaxis=dict(tickmode="linear"),  # Ensure ticks are linear
            yaxis=dict(tickmode="linear"),
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        margin=dict(l=0, r=0, b=0, t=50),  # Reduce margins
    )
    
    # Save the plot as an HTML file
    output_html = r"C:\rmse_3d_plot_AR1_maturity_horizons.html"
    fig.write_html(output_html)
    print(f"Interactive 3D plot saved as {output_html}")

    # Optionally show the plot
    # fig.show()
    
plot_3d_rmse_interactive_2(metrics_df)

# Example usage
# Assuming `metrics_df` is your dataset with RMSE values
horizon_to_plot = 2.0  # Example: 5-year horizon
plot_3d_rmse_interactive_1(metrics_df, horizon_to_plot, output_html=r"C:\rmse_3d_plot_AR1_horizon2.html")


import matplotlib.pyplot as plt
import pandas as pd

def plot_var_cvar_over_execution_dates(metrics_df, maturity, horizon):
    """
    Plot VaR, CVaR, Observed Returns, and Expected Returns for a single horizon and maturity across all execution dates.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing precomputed metrics with columns:
            - "Execution Date"
            - "Horizon (Years)"
            - "Maturity (Years)"
            - "Metric"
            - "Value"
        maturity (float): The maturity (in years) to plot (e.g., 1.0).
        horizon (float): The horizon (in years) to plot (e.g., 1.0).
    """
    # Filter the DataFrame for the given horizon, maturity, and metrics
    filtered_df = metrics_df[
        (metrics_df["Horizon (Years)"] == horizon) &
        (metrics_df["Maturity (Years)"] == maturity) &
        (metrics_df["Metric"].isin(["VaR", "CVaR", "Observed Annual Return", "Expected Returns"]))
    ]

    # Check for duplicates
    duplicate_rows = filtered_df[filtered_df.duplicated(subset=["Execution Date", "Metric"], keep=False)]
    if not duplicate_rows.empty:
        print("Duplicate rows detected:")
        print(duplicate_rows)

        # Aggregate duplicates (e.g., take the mean of duplicate values)
        filtered_df = filtered_df.groupby(["Execution Date", "Metric"], as_index=False)["Value"].mean()

    # Pivot the data to align metrics by execution date
    pivot_df = filtered_df.pivot(index="Execution Date", columns="Metric", values="Value")

    # Ensure all required metrics are present
    required_metrics = ["VaR", "CVaR", "Observed Annual Return", "Expected Returns"]
    for metric in required_metrics:
        if metric not in pivot_df.columns:
            print(f"Missing metric: {metric}")
            return

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pivot_df.index, pivot_df["Observed Annual Return"], label="Observed Returns", color="blue", alpha=0.7)
    plt.plot(pivot_df.index, pivot_df["Expected Returns"], label="Expected Returns", color="green", alpha=0.7)
    plt.plot(pivot_df.index, pivot_df["VaR"], label="VaR (Threshold)", color="red", linestyle="--")
    plt.plot(pivot_df.index, pivot_df["CVaR"], label="CVaR (Tail Average)", color="orange", linestyle=":")

    # Highlight breaches where Observed Returns < VaR
    breaches = pivot_df["Observed Annual Return"] < pivot_df["VaR"]
    plt.scatter(pivot_df.index[breaches], pivot_df["Observed Annual Return"][breaches], 
                color="black", label="VaR Breaches", zorder=5)

    # Add labels and legend
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.title(f"VaR, CVaR, Observed, and Expected Returns Across Execution Dates (Horizon={horizon}y, Maturity={maturity}y)", fontsize=14)
    plt.xlabel("Execution Date", fontsize=12)
    plt.ylabel("Returns", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.6, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
plot_var_cvar_over_execution_dates(metrics_df, maturity=1.0, horizon=1.0)


rmse_by_horizon = metrics_df.groupby("Horizon (Years)").apply(
    lambda x: pd.Series({
        "RMSE_AR1": calculate_rmse(x, "AR1_fcst", "actual"),
        "RMSE_consensus": calculate_rmse(x, "consensus_fcst", "actual"),
        "Volatility_AR1": x["AR1_fcst"].std(),
        "Volatility_consensus": x["consensus_fcst"].std()
    })
).reset_index()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

metrics_df_raw = pd.read_csv(r"C:\git\backtest-baam\data\US\metrics_timeseries.csv")
metrics_df = metrics_df_raw.iloc[1066:,:].copy()

def plot_returns_vs_var_cvar_by_execution_date_long(data, maturity, horizon):
    """
    Plot observed historical annual returns vs VaR, CVaR, and expected returns
    for a given maturity and forecasting horizon over execution dates.

    Args:
        data (pd.DataFrame): Long-format dataset containing metrics in the "Metric" column.
        maturity (float): The maturity (in years) to plot.
        horizon (float): The forecasting horizon (in years) to plot.
    """
    # Filter data for the selected maturity and horizon
    filtered_data = data[
        (data["Maturity (Years)"] == maturity) &
        (data["Horizon (Years)"] == horizon)
    ]

    # Debugging: Check filtered data
    if filtered_data.empty:
        print(f"No data available for Maturity: {maturity} Years, Horizon: {horizon} Years")
        return
    
    print("Filtered Data Columns:", filtered_data.columns)
    print(filtered_data.head())

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Plot observed historical returns
    sns.lineplot(
        data=filtered_data[filtered_data["Metric"] == "Observed Annual Return"],
        x="Execution Date",
        y="Value",
        label="Observed Returns",
        color="blue",
    )

    # Plot VaR
    sns.lineplot(
        data=filtered_data[filtered_data["Metric"] == "VaR"],
        x="Execution Date",
        y="Value",
        label="VaR (Threshold)",
        color="red",
        linestyle="--",
    )

    # Plot CVaR
    sns.lineplot(
        data=filtered_data[filtered_data["Metric"] == "CVaR"],
        x="Execution Date",
        y="Value",
        label="CVaR (Tail Risk)",
        color="orange",
        linestyle=":",
    )

    # Plot Expected Returns
    sns.lineplot(
        data=filtered_data[filtered_data["Metric"] == "Expected Returns"],
        x="Execution Date",
        y="Value",
        label="Expected Returns",
        color="green",
        linestyle="-.",
    )

    # Highlight breaches (Observed < VaR)
    observed_data = filtered_data[filtered_data["Metric"] == "Observed Annual Return"]
    var_data = filtered_data[filtered_data["Metric"] == "VaR"]
    breaches = observed_data.merge(var_data, on=["Execution Date"], suffixes=("_obs", "_var"))
    breaches = breaches[breaches["Value_obs"] < breaches["Value_var"]]

    plt.scatter(
        breaches["Execution Date"],
        breaches["Value_obs"],
        color="black",
        label="VaR Breaches",
        zorder=5,
    )

    # Add labels, title, and legend
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.title(f"Observed Returns vs VaR, CVaR, and Expected Returns\n"
              f"(Maturity: {maturity} Years, Horizon: {horizon} Years)", fontsize=14)
    plt.xlabel("Execution Date", fontsize=12)
    plt.ylabel("Returns", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.6, linestyle="--")
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming `metrics_df` is the long-format DataFrame
maturity_to_plot = 2.0  # Example: 1-year maturity
horizon_to_plot = 5.0   # Example: 1-year forecasting horizon
metrics_df["Value"] = pd.to_numeric(metrics_df["Value"], errors="coerce")
metrics_df["Execution Date"] = pd.to_datetime(metrics_df["Execution Date"])
plot_returns_vs_var_cvar_by_execution_date_long(metrics_df, maturity_to_plot, horizon_to_plot)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_rmse_by_execution_date(data, horizon):
    """
    Create a 3D plot for RMSE values across execution dates and maturities for a specific horizon.

    Args:
        data (pd.DataFrame): Dataset containing RMSE values with columns:
            - "Execution Date"
            - "Maturity (Years)"
            - "Horizon (Years)"
            - "Metric"
            - "Value"
        horizon (float): The forecasting horizon (in years) to plot.
    """
    # Filter the data for the selected horizon and RMSE metric
    rmse_data = data[(data["Horizon (Years)"] == horizon) & (data["Metric"] == "RMSE")]

    # Pivot the data to create a grid for execution dates, maturities, and RMSE values
    rmse_pivot = rmse_data.pivot(index="Maturity (Years)", columns="Execution Date", values="Value")

    # Convert execution dates to numeric (ordinal format)
    rmse_pivot.columns = rmse_pivot.columns.map(pd.Timestamp.toordinal)

    # Handle missing values by filling NaN with 0
    rmse_pivot = rmse_pivot.fillna(0)

    # Create the grid for the 3D plot
    X, Y = np.meshgrid(rmse_pivot.columns, rmse_pivot.index)  # X: Execution Dates, Y: Maturities
    Z = rmse_pivot.values  # Z: RMSE values

    # Debugging: Check the data
    print("X (Execution Dates):", X)
    print("Y (Maturities):", Y)
    print("Z (RMSE Values):", Z)
    print("Z Data Type:", Z.dtype)

    # Initialize the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8
    )

    # Format the x-axis as dates
    ax.set_xticks(X[0][::len(X[0]) // 10])  # Show fewer date ticks for readability
    ax.set_xticklabels([pd.Timestamp.fromordinal(int(date)).strftime('%Y-%m-%d') for date in X[0][::len(X[0]) // 10]], rotation=45)

    # Add labels and title
    ax.set_xlabel("Execution Date", fontsize=12)
    ax.set_ylabel("Maturity (Years)", fontsize=12)
    ax.set_zlabel("RMSE", fontsize=12)
    ax.set_title(f"3D Plot of RMSE by Execution Date and Maturity\n(Horizon: {horizon} Years)", fontsize=14)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="RMSE")

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming `metrics_df` is your dataset with RMSE values
horizon_to_plot = 5.0  # Example: 5-year horizon
plot_3d_rmse_by_execution_date(metrics_df, horizon_to_plot)

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_3d_rmse_interactive(data, horizon, output_html="rmse_plot.html"):
    """
    Create an interactive 3D plot for RMSE values across execution dates and maturities
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
    # Filter the data for the selected horizon and RMSE metric
    rmse_data = data[(data["Horizon (Years)"] == horizon) & (data["Metric"] == "RMSE")]

    # Pivot the data to create a grid for execution dates, maturities, and RMSE values
    rmse_pivot = rmse_data.pivot(index="Maturity (Years)", columns="Execution Date", values="Value")

    # Convert execution dates to numeric (ordinal format) for plotting
    rmse_pivot.columns = rmse_pivot.columns.map(pd.Timestamp.toordinal)

    # Handle missing values by filling NaN with 0
    rmse_pivot = rmse_pivot.fillna(0)

    # Create the grid for the 3D plot
    X, Y = np.meshgrid(rmse_pivot.columns, rmse_pivot.index)  # X: Execution Dates, Y: Maturities
    Z = rmse_pivot.values  # Z: RMSE values

    # Convert execution dates back to human-readable strings for hover labels
    execution_dates = [pd.Timestamp.fromordinal(int(date)).strftime('%Y-%m-%d') for date in rmse_pivot.columns]

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(
        z=Z,
        x=execution_dates,  # Use human-readable dates for the x-axis
        y=rmse_pivot.index,  # Maturities
        colorscale="Viridis",
        colorbar=dict(title="RMSE"),
    )])

    # Update layout for better visualization
    fig.update_layout(
        title=f"Interactive 3D Plot of RMSE by Execution Date and Maturity<br>(Horizon: {horizon} Years)",
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

# Example usage
# Assuming `metrics_df` is your dataset with RMSE values
horizon_to_plot = 5.0  # Example: 5-year horizon
plot_3d_rmse_interactive(metrics_df, horizon_to_plot, output_html=r"C:\rmse_3d_plot.html")

import seaborn as sns
import matplotlib.pyplot as plt

# Filter for Pass/Fail metrics
pass_fail_data = metrics_df[metrics_df["Metric"].isin([
    "Kupiec POF Test Pass", "Christoffersen Independence Test Pass"
])]

# Plot the pass/fail counts
sns.countplot(data=pass_fail_data, x="Metric", hue="Value", palette="Set2")
plt.title("Kupiec and Christoffersen Test Pass/Fail Counts")
plt.xlabel("Metric")
plt.ylabel("Count")
plt.legend(title="Pass (1) / Fail (0)")
plt.show()

# Filter for test statistics
test_stat_data = metrics_df[metrics_df["Metric"].isin([
    "Kupiec POF Test Statistic", "Christoffersen Independence Test Statistic"
])]

# Create a pivot table for heatmap
kupiec_stat_pivot = test_stat_data[test_stat_data["Metric"] == "Kupiec POF Test Statistic"].pivot(
    index="Maturity (Years)", columns="Horizon (Years)", values="Value"
)
christoffersen_stat_pivot = test_stat_data[test_stat_data["Metric"] == "Christoffersen Independence Test Statistic"].pivot(
    index="Maturity (Years)", columns="Horizon (Years)", values="Value"
)

# Plot heatmap for Kupiec POF Test Statistic
plt.figure(figsize=(12, 6))
sns.heatmap(kupiec_stat_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Test Statistic"})
plt.title("Kupiec POF Test Statistic Heatmap")
plt.xlabel("Horizon (Years)")
plt.ylabel("Maturity (Years)")
plt.show()

# Plot heatmap for Christoffersen Independence Test Statistic
plt.figure(figsize=(12, 6))
sns.heatmap(christoffersen_stat_pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Test Statistic"})
plt.title("Christoffersen Independence Test Statistic Heatmap")
plt.xlabel("Horizon (Years)")
plt.ylabel("Maturity (Years)")
plt.show()
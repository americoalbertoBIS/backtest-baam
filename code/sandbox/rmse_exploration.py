import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages for saving plots to PDF

def calculate_rmse(predictions, actuals):
    """
    Calculate Root Mean Squared Error (RMSE) between predictions and actual values.
    """
    return np.sqrt(((predictions - actuals) ** 2).mean())

def generate_rmse_comparison_with_bands(df):
    """
    Calculate RMSE by observation, execution date, and horizon, and include median and percentile bands.

    Args:
        df (pd.DataFrame): DataFrame containing 'ExecutionDate', 'Horizon', 'Prediction', and 'Actual' columns.

    Returns:
        dict: Dictionary containing RMSE values, median, and percentiles.
    """
    df = df.dropna(subset=["Prediction", "Actual"])
    df["ExecutionDate"] = pd.to_datetime(df["ExecutionDate"])

    # RMSE by Observation (individual errors)
    df["RMSE_by_observation"] = np.sqrt((df["Prediction"] - df["Actual"]) ** 2)

    # RMSE by Execution Date
    grouped_by_execution_date = df.groupby("ExecutionDate")["RMSE_by_observation"]
    rmse_by_execution_date = grouped_by_execution_date.mean()
    median_rmse_by_execution_date = grouped_by_execution_date.median()
    percentile_25_execution_date = grouped_by_execution_date.quantile(0.25)
    percentile_75_execution_date = grouped_by_execution_date.quantile(0.75)

    # Direct RMSE by Execution Date
    direct_rmse_by_execution_date = df.groupby("ExecutionDate").apply(
        lambda group: calculate_rmse(group["Prediction"], group["Actual"])
    )

    # RMSE by Horizon
    grouped_by_horizon = df.groupby("Horizon")["RMSE_by_observation"]
    rmse_by_horizon = grouped_by_horizon.mean()
    median_rmse_by_horizon = grouped_by_horizon.median()
    percentile_25_horizon = grouped_by_horizon.quantile(0.25)
    percentile_75_horizon = grouped_by_horizon.quantile(0.75)

    # Direct RMSE by Horizon
    direct_rmse_by_horizon = df.groupby("Horizon").apply(
        lambda group: calculate_rmse(group["Prediction"], group["Actual"])
    )

    return {
        "RMSE_by_execution_date": rmse_by_execution_date,
        "Median_RMSE_by_execution_date": median_rmse_by_execution_date,
        "Percentile_25_Execution_Date": percentile_25_execution_date,
        "Percentile_75_Execution_Date": percentile_75_execution_date,
        "Direct_RMSE_by_execution_date": direct_rmse_by_execution_date,
        "RMSE_by_horizon": rmse_by_horizon,
        "Median_RMSE_by_horizon": median_rmse_by_horizon,
        "Percentile_25_Horizon": percentile_25_horizon,
        "Percentile_75_Horizon": percentile_75_horizon,
        "Direct_RMSE_by_horizon": direct_rmse_by_horizon,
    }

def plot_rmse_comparison_with_bands(results, target_variable, model_name, pdf):
    """
    Plot RMSE comparison with median, percentile bands, and direct RMSE overlay for execution date and horizon.
    Save the plot to a PDF file.

    Args:
        results (dict): Dictionary containing RMSE values and bands.
        target_variable (str): The beta variable (e.g., beta1, beta2, beta3).
        model_name (str): The model name (e.g., AR_1, AR_1_GDP).
        pdf (PdfPages): PdfPages object to save plots to the PDF.
    """
    # Extract data for plotting
    rmse_by_execution_date = results["RMSE_by_execution_date"]
    median_rmse_by_execution_date = results["Median_RMSE_by_execution_date"]
    percentile_25_execution_date = results["Percentile_25_Execution_Date"]
    percentile_75_execution_date = results["Percentile_75_Execution_Date"]
    direct_rmse_by_execution_date = results["Direct_RMSE_by_execution_date"]

    rmse_by_horizon = results["RMSE_by_horizon"]
    median_rmse_by_horizon = results["Median_RMSE_by_horizon"]
    percentile_25_horizon = results["Percentile_25_Horizon"]
    percentile_75_horizon = results["Percentile_75_Horizon"]
    direct_rmse_by_horizon = results["Direct_RMSE_by_horizon"]

    # Define improved colors
    mean_color = "#00008B"  # Dark Blue
    median_color = "#4682B4"  # Light Blue
    direct_color = "#8B0000"  # Dark Red
    percentile_band_color = "#D3D3D3"  # Light Grey

    # Create a 2-row, 1-column plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

    # Top plot: RMSE by Execution Date with bands
    axes[0].plot(rmse_by_execution_date.index, rmse_by_execution_date.values, label="Mean RMSE", color=mean_color)
    axes[0].plot(direct_rmse_by_execution_date.index, direct_rmse_by_execution_date.values, label="Direct RMSE", color=direct_color)
    axes[0].plot(median_rmse_by_execution_date.index, median_rmse_by_execution_date.values, label="Median RMSE", color=median_color)
    axes[0].fill_between(
        rmse_by_execution_date.index,
        percentile_25_execution_date.values,
        percentile_75_execution_date.values,
        color=percentile_band_color,
        alpha=0.5,
        label="25th-75th Percentile",
    )
    axes[0].set_title(f"RMSE by Execution Date ({target_variable}, {model_name})")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].grid(True)

    # Bottom plot: RMSE by Horizon with bands
    axes[1].plot(rmse_by_horizon.index, rmse_by_horizon.values, label="Mean RMSE", color=mean_color)
    axes[1].plot(direct_rmse_by_horizon.index, direct_rmse_by_horizon.values, label="Direct RMSE", color=direct_color)
    axes[1].plot(median_rmse_by_horizon.index, median_rmse_by_horizon.values, label="Median RMSE", color=median_color)
    axes[1].fill_between(
        rmse_by_horizon.index,
        percentile_25_horizon.values,
        percentile_75_horizon.values,
        color=percentile_band_color,
        alpha=0.5,
        label="25th-75th Percentile",
    )
    axes[1].set_title(f"RMSE by Horizon ({target_variable}, {model_name})")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout and save the plot to the PDF
    plt.tight_layout()
    pdf.savefig(fig)  # Save the current figure to the PDF
    plt.close(fig)  # Close the figure to free memory

def process_files_and_save_to_pdf(data_dir, country):
    """
    Process all beta forecast files in the directory and save plots to a PDF by country.

    Args:
        data_dir (str): Directory containing the beta forecast files (e.g., 'data/US/factors/').
        country (str): Name of the country (e.g., 'US', 'EA').
    """
    forecast_files = glob.glob(os.path.join(data_dir, "beta*_forecasts_*.csv"))
    output_pdf = fr"C:\git\backtest-baam\graphs\RMSE_comparison_{country}.pdf"

    print(f"Found {len(forecast_files)} files to process for {country}...")

    with PdfPages(output_pdf) as pdf:
        for file_path in forecast_files:
            # Extract model name and target variable from the filename
            file_name = os.path.basename(file_path)
            target_variable = file_name.split("_")[0]
            model_name = file_name.replace(f"{target_variable}_forecasts_", "").replace(".csv", "")

            # Load the forecast file
            df = pd.read_csv(file_path)

            # Generate RMSE comparison results with bands
            results = generate_rmse_comparison_with_bands(df)

            # Plot RMSE comparison and save to PDF
            plot_rmse_comparison_with_bands(results, target_variable, model_name, pdf)

    print(f"Saved all plots to {output_pdf}")

# Example usage
for country in ['EA', 'US']:
    data_dir = rf"C:\git\backtest-baam\data\{country}\factors"  # Directory containing the beta forecast files
    process_files_and_save_to_pdf(data_dir, country)
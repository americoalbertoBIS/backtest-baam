# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:05:02 2025

@author: al005366
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:27:19 2025

@author: al005366
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from scipy.stats import linregress
import numpy as np
import ruptures as rpt  # For Bai-Perron test (multiple structural breaks)
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# Helper Functions
# ==========================================

def preprocess_data(file_path):
    """Load and preprocess the time series data."""
    merged_df = pd.read_csv(file_path)
    grouped_df = merged_df.groupby('ForecastDate')['Actual'].last().reset_index()
    grouped_df['ForecastDate'] = pd.to_datetime(grouped_df['ForecastDate'])
    grouped_df = grouped_df.sort_values(by='ForecastDate')
    grouped_df = grouped_df.dropna(subset=['Actual'])  # Remove missing values
    grouped_df = grouped_df[grouped_df['Actual'].apply(np.isfinite)]  # Remove infinite values
    return grouped_df


def plot_time_series(grouped_df, pdf, title="Original Time Series"):
    """Plot the time series with the mean."""
    plt.figure(figsize=(12, 6))
    plt.plot(grouped_df['ForecastDate'], grouped_df['Actual'], label='Time Series', color='black')
    plt.axhline(grouped_df['Actual'].mean(), color='red', linestyle='--', label='Mean')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Actual')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if pdf:
        pdf.savefig()
    plt.show()


def adf_test(series):
    """Perform the Augmented Dickey-Fuller (ADF) test."""
    result = adfuller(series)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Number of Lags Used": result[2],
        "Number of Observations Used": result[3],
        "Critical Values": result[4]
    }


def kpss_test(series):
    """Perform the KPSS test."""
    result = kpss(series, regression='c', nlags="auto")
    return {
        "KPSS Statistic": result[0],
        "p-value": result[1],
        "Number of Lags Used": result[2],
        "Critical Values": result[3]
    }


def zivot_andrews_test(series):
    """
    Perform the Zivot-Andrews test for a single structural break.
    Handles missing or infinite values by dropping them before running the test.
    """
    series_cleaned = series.dropna()
    series_cleaned = series_cleaned[np.isfinite(series_cleaned)]

    if len(series_cleaned) < 3:
        raise ValueError("Time series is too short after cleaning to perform the Zivot-Andrews test.")

    result = zivot_andrews(series_cleaned, trim=0.15)
    return {
        "Test Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[2],
        "Lag Length Used": result[3],
        "Break Point Index": result[4]
    }


def plot_zivot_andrews_breakpoint(grouped_df, za_results, pdf=None, title="Zivot-Andrews Breakpoint"):
    """Plot the Zivot-Andrews breakpoint on the time series."""
    break_point_index = za_results["Break Point Index"]
    break_point_date = grouped_df['ForecastDate'].iloc[break_point_index]

    plt.figure(figsize=(12, 6))
    plt.plot(grouped_df['ForecastDate'], grouped_df['Actual'], label='Time Series', color='black')
    plt.axvline(x=break_point_date, color='red', linestyle='--', label='Structural Break (Zivot-Andrews)')
    plt.text(break_point_date, max(grouped_df['Actual']),
             break_point_date.strftime('%Y-%m-%d'),
             color='red', fontsize=10, rotation=0, ha='right')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Actual')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save to PDF if provided
    if pdf:
        pdf.savefig()
    plt.show()


def bai_perron_test(series, model="rbf", penalty=20):
    """Perform the Bai-Perron test for multiple structural breaks."""
    breaks = rpt.Pelt(model=model).fit(series).predict(pen=penalty)
    return breaks


def plot_bai_perron_breakpoints(grouped_df, bp_breaks, pdf=None, title="Bai-Perron Breakpoints"):
    """Plot the Bai-Perron breakpoints on the time series."""
    time_series = grouped_df['Actual'].values
    break_dates = [grouped_df['ForecastDate'].iloc[idx] for idx in bp_breaks[:-1]]  # Exclude the last index
    segment_start_dates = [grouped_df['ForecastDate'].iloc[0]] + break_dates  # Include the start of the series
    segment_end_dates = break_dates + [grouped_df['ForecastDate'].iloc[-1]]  # Include the end of the series
    colors = ['lightblue', 'lightcoral']  # Alternating colors for segments

    plt.figure(figsize=(12, 6))
    plt.plot(grouped_df['ForecastDate'], time_series, label='Time Series', color='black')
    for i, (start, end) in enumerate(zip(segment_start_dates, segment_end_dates)):
        color = colors[i % 2]
        plt.axvspan(start, end, color=color, alpha=0.3)
    for idx, date in zip(bp_breaks[:-1], break_dates):
        plt.axvline(x=date, color='red', linestyle='--', alpha=0.7)
        plt.text(date, max(time_series), date.strftime('%Y-%m-%d'), color='black', fontsize=10, rotation=0)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Actual')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()

    # Save to PDF if provided
    if pdf:
        pdf.savefig()
    plt.show()


def difference_series(series):
    """Perform first-order differencing and ensure no NaN or inf values remain."""
    differenced_series = series.diff().dropna()
    return differenced_series[np.isfinite(differenced_series)]


def is_stationary(adf_results, kpss_results):
    """Check if the series is stationary based on ADF and KPSS tests."""
    adf_stationary = adf_results['p-value'] < 0.05
    kpss_stationary = kpss_results['p-value'] >= 0.05
    return adf_stationary, kpss_stationary


def generate_report(adf_results, kpss_results, za_results, bp_breaks, grouped_df, transformation="Original"):
    """Generate a textual report summarizing the results."""
    report = []

    # Transformation type
    report.append(f"### {transformation} Series Analysis\n")

    # Stationarity Analysis
    report.append("#### Stationarity Analysis\n")
    report.append("**ADF Test**:\n")
    report.append("- Null Hypothesis: The series has a unit root (non-stationary).\n")
    report.append("- Alternative Hypothesis: The series is stationary.\n")
    report.append(f"- ADF Test p-value: {adf_results['p-value']:.4f} (Stationary: {'Yes' if adf_results['p-value'] < 0.05 else 'No'})\n")

    report.append("**KPSS Test**:\n")
    report.append("- Null Hypothesis: The series is stationary.\n")
    report.append("- Alternative Hypothesis: The series is non-stationary.\n")
    report.append(f"- KPSS Test p-value: {kpss_results['p-value']:.4f} (Stationary: {'Yes' if kpss_results['p-value'] >= 0.05 else 'No'})\n")

    # Structural Breaks
    report.append("\n#### Structural Breaks Analysis\n")
    report.append(f"- Zivot-Andrews Breakpoint: {grouped_df['ForecastDate'].iloc[za_results['Break Point Index']]}\n")
    report.append("Bai-Perron Break Dates:\n")
    for idx in bp_breaks[:-1]:
        report.append(f"  - {grouped_df['ForecastDate'].iloc[idx]}\n")

    return "\n".join(report)

# ==========================================
# Main Execution
# ==========================================

# File path
#beta = 'beta3'
for country in ['US', 'EA', 'UK']:
    for beta in ['beta1', 'beta2', 'beta3']:
        file_path = rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\{country}\factors\AR_1\{beta}\forecasts.csv'
        pdf_output_path = rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\graphs\NSfactors_starionarity_{country}_{beta}.pdf'
        
        # Preprocess the data
        grouped_df = preprocess_data(file_path)
        
        # Create a PDF file to save the plots and report
        with PdfPages(pdf_output_path) as pdf:
            # Plot original time series
            plot_time_series(grouped_df, pdf=pdf, title="Original Time Series")
        
            # Perform structural break tests
            za_results = zivot_andrews_test(grouped_df['Actual'])
            bp_breaks = bai_perron_test(grouped_df['Actual'].values, model="rbf", penalty=20)
        
            # Plot Zivot-Andrews breakpoint
            plot_zivot_andrews_breakpoint(grouped_df, za_results, pdf=pdf)
        
            # Plot Bai-Perron breakpoints
            plot_bai_perron_breakpoints(grouped_df, bp_breaks, pdf=pdf)
        
            # Perform stationarity tests
            adf_results = adf_test(grouped_df['Actual'])
            kpss_results = kpss_test(grouped_df['Actual'])
        
            # Check stationarity and apply differencing if needed
            adf_stationary, kpss_stationary = is_stationary(adf_results, kpss_results)
            report = generate_report(adf_results, kpss_results, za_results, bp_breaks, grouped_df, transformation="Original")
        
            # Save the report for the original series
            plt.figure(figsize=(8.5, 11))
            plt.text(0.01, 0.99, report, fontsize=10, va='top', family='monospace')
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
            if not (adf_stationary and kpss_stationary):
                print("The series is non-stationary. Applying first-order differencing.")
        
                # Apply differencing and clean the series
                grouped_df['Actual'] = difference_series(grouped_df['Actual'])
                grouped_df = grouped_df.dropna()
                
                # Plot differenced series
                plot_time_series(grouped_df, pdf=pdf, title="Differenced Time Series")
        
                # Re-run stationarity tests
                adf_results = adf_test(grouped_df['Actual'])
                kpss_results = kpss_test(grouped_df['Actual'])
        
                # Re-run structural break tests
                za_results = zivot_andrews_test(grouped_df['Actual'])
                bp_breaks = bai_perron_test(grouped_df['Actual'].values, model="rbf", penalty=5)
        
                # Plot Zivot-Andrews breakpoint for differenced series
                plot_zivot_andrews_breakpoint(grouped_df, za_results, pdf=pdf, title="Zivot-Andrews Breakpoint (Differenced)")
        
                # Plot Bai-Perron breakpoints for differenced series
                plot_bai_perron_breakpoints(grouped_df, bp_breaks, pdf=pdf, title="Bai-Perron Breakpoints (Differenced)")
        
                # Generate and save the report for the differenced series
                report = generate_report(adf_results, kpss_results, za_results, bp_breaks, grouped_df, transformation="Differenced")
                plt.figure(figsize=(8.5, 11))
                plt.text(0.01, 0.99, report, fontsize=10, va='top', family='monospace')
                plt.axis('off')
                pdf.savefig()
                plt.close()
        
        print(f"Analysis complete. Results saved to {pdf_output_path}")
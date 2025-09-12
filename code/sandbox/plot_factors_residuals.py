# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 15:38:14 2025

@author: al005366
"""

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# Helper Functions
# ==========================================

def load_and_merge_data(beta1_path, beta2_path, beta3_path):
    """Load and merge the datasets for beta1, beta2, and beta3."""
    df_beta1 = pd.read_csv(beta1_path).rename(columns={'residual': 'residual_beta1'})
    df_beta2 = pd.read_csv(beta2_path).rename(columns={'residual': 'residual_beta2'})
    df_beta3 = pd.read_csv(beta3_path).rename(columns={'residual': 'residual_beta3'})
    
    # Merge datasets on execution_date and date
    merged_df = pd.merge(df_beta1, df_beta2, on=['execution_date', 'date'])
    merged_df = pd.merge(merged_df, df_beta3, on=['execution_date', 'date'])
    return merged_df


def filter_data_by_execution_date(merged_df, execution_date):
    """Filter the merged dataset for a specific execution date."""
    filtered_df = merged_df[merged_df['execution_date'] == execution_date]
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    return filtered_df


def plot_correlation_matrix(filtered_df, pdf, execution_date):
    """Plot and save the correlation matrix as a heatmap."""
    correlation_matrix = filtered_df[['residual_beta1', 'residual_beta2', 'residual_beta3']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Matrix for Execution Date = {execution_date}')
    pdf.savefig()
    plt.close()


def plot_pairplot(filtered_df, pdf, execution_date):
    """Plot and save the pair plot of residuals."""
    sns.pairplot(filtered_df[['residual_beta1', 'residual_beta2', 'residual_beta3']])
    plt.suptitle(f'Pair Plot of Residuals for Execution Date = {execution_date}', y=1.02)
    pdf.savefig()
    plt.close()


def plot_residuals_over_time(filtered_df, pdf, execution_date):
    """Plot and save line plots of residuals over time."""
    plt.figure(figsize=(12, 8))

    # Residual Beta1
    plt.subplot(3, 1, 1)
    plt.plot(filtered_df['date'], filtered_df['residual_beta1'], label='Residual Beta1 (SSR)', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel('Residual Beta1')
    plt.title(f'Residuals Over Time for Execution Date = {execution_date}')
    plt.grid()
    plt.legend()

    # Residual Beta2
    plt.subplot(3, 1, 2)
    plt.plot(filtered_df['date'], filtered_df['residual_beta2'], label='Residual Beta2 (Slope)', color='red')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel('Residual Beta2')
    plt.grid()
    plt.legend()

    # Residual Beta3
    plt.subplot(3, 1, 3)
    plt.plot(filtered_df['date'], filtered_df['residual_beta3'], label='Residual Beta3 (Curvature)', color='green')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Residual Beta3')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    pdf.savefig()
    plt.close()


def plot_histograms(filtered_df, pdf, execution_date):
    """Plot and save histograms of residuals."""
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df['residual_beta1'], kde=True, color='blue', label='Residual Beta1 (SSR)', bins=100)
    sns.histplot(filtered_df['residual_beta2'], kde=True, color='red', label='Residual Beta2 (Slope)', bins=100)
    sns.histplot(filtered_df['residual_beta3'], kde=True, color='green', label='Residual Beta3 (Curvature)', bins=100)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Residuals for Execution Date = {execution_date}')
    plt.legend()
    pdf.savefig()
    plt.close()


def plot_boxplot(filtered_df, pdf, execution_date):
    """Plot and save a box plot of residuals."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=filtered_df[['residual_beta1', 'residual_beta2', 'residual_beta3']])
    plt.xticks([0, 1, 2], ['Residual Beta1 (SSR)', 'Residual Beta2 (Slope)', 'Residual Beta3 (Curvature)'])
    plt.title(f'Box Plot of Residuals for Execution Date = {execution_date}')
    plt.ylabel('Residual Value')
    pdf.savefig()
    plt.close()


def plot_moving_correlations(filtered_df, pdf, window_size=60):
    """Plot and save moving correlations between residuals."""
    # Compute moving correlations
    filtered_df['corr_beta1_beta2'] = filtered_df['residual_beta1'].rolling(window=window_size).corr(filtered_df['residual_beta2'])
    filtered_df['corr_beta1_beta3'] = filtered_df['residual_beta1'].rolling(window=window_size).corr(filtered_df['residual_beta3'])
    filtered_df['corr_beta2_beta3'] = filtered_df['residual_beta2'].rolling(window=window_size).corr(filtered_df['residual_beta3'])

    # Create subplots
    plt.figure(figsize=(12, 12))

    # Subplot 1: Beta1 vs Beta2
    plt.subplot(3, 1, 1)
    plt.plot(filtered_df['date'], filtered_df['corr_beta1_beta2'], label='Corr(Beta1, Beta2)', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel('Correlation')
    plt.title(f'Moving Correlation: Beta1 vs Beta2 (Window Size = {window_size})')
    plt.legend()
    plt.grid()

    # Subplot 2: Beta1 vs Beta3
    plt.subplot(3, 1, 2)
    plt.plot(filtered_df['date'], filtered_df['corr_beta1_beta3'], label='Corr(Beta1, Beta3)', color='red')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel('Correlation')
    plt.title(f'Moving Correlation: Beta1 vs Beta3 (Window Size = {window_size})')
    plt.legend()
    plt.grid()

    # Subplot 3: Beta2 vs Beta3
    plt.subplot(3, 1, 3)
    plt.plot(filtered_df['date'], filtered_df['corr_beta2_beta3'], label='Corr(Beta2, Beta3)', color='green')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.title(f'Moving Correlation: Beta2 vs Beta3 (Window Size = {window_size})')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    pdf.savefig()
    plt.close()


def analyze_correlation(filtered_df, pdf, execution_date):
    """Analyze and save the correlation matrix and textual assertions."""
    correlation_matrix = filtered_df[['residual_beta1', 'residual_beta2', 'residual_beta3']].corr()

    # Plot the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Matrix for Execution Date = {execution_date}')
    pdf.savefig()
    plt.close()

    # Analyze correlation
    text = ["### Correlation Analysis\n"]
    text.append("Correlation Matrix:\n")
    text.append(correlation_matrix.to_string())
    text.append("\n\n")

    # Check for strong correlation
    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2:
                corr_value = correlation_matrix.loc[col1, col2]
                if abs(corr_value) > 0.7:
                    text.append(f"{col1} and {col2} are strongly correlated (correlation = {corr_value:.2f}).\n")
                else:
                    text.append(f"{col1} and {col2} are weakly correlated (correlation = {corr_value:.2f}).\n")

    # Save the textual analysis
    plt.figure(figsize=(8.5, 11))
    plt.text(0.01, 0.99, "\n".join(text), fontsize=10, va='top', family='monospace')
    plt.axis('off')
    pdf.savefig()
    plt.close()


def analyze_serial_correlation(filtered_df, pdf):
    """Analyze and save serial correlation results."""
    text = ["### Serial Correlation Analysis\n"]

    # Durbin-Watson Test
    text.append("#### Durbin-Watson Test Results:\n")
    for col in ['residual_beta1', 'residual_beta2', 'residual_beta3']:
        dw_stat = durbin_watson(filtered_df[col].dropna())
        text.append(f"{col}: Durbin-Watson Statistic = {dw_stat:.4f}")
        if dw_stat < 1.5:
            text.append(" (Evidence of positive autocorrelation).\n")
        elif dw_stat > 2.5:
            text.append(" (Evidence of negative autocorrelation).\n")
        else:
            text.append(" (No significant autocorrelation).\n")
    text.append("\n")

    # Ljung-Box Test
    text.append("#### Ljung-Box Test Results:\n")
    for col in ['residual_beta1', 'residual_beta2', 'residual_beta3']:
        lb_test = acorr_ljungbox(filtered_df[col].dropna(), lags=[10], return_df=True)
        p_value = lb_test['lb_pvalue'].iloc[0]
        text.append(f"{col}: Ljung-Box p-value = {p_value:.4f}")
        if p_value < 0.05:
            text.append(" (Reject null hypothesis: Residuals are autocorrelated).\n")
        else:
            text.append(" (Fail to reject null hypothesis: No significant autocorrelation).\n")
    text.append("\n")

    # Save the textual analysis
    plt.figure(figsize=(8.5, 11))
    plt.text(0.01, 0.99, "\n".join(text), fontsize=10, va='top', family='monospace')
    plt.axis('off')
    pdf.savefig()
    plt.close()


def plot_acf_plots(filtered_df, pdf):
    """Plot and save autocorrelation plots for residuals."""
    for col in ['residual_beta1', 'residual_beta2', 'residual_beta3']:
        plt.figure(figsize=(8, 4))
        plot_acf(filtered_df[col].dropna(), lags=30, title=f"ACF for {col}")
        pdf.savefig()
        plt.close()


# ==========================================
# Main Execution
# ==========================================

for country in ['US', 'EA', 'UK']:
    # File paths
    beta1_path = rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\{country}\factors\AR_1\beta1\residuals.csv'
    beta2_path = rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\{country}\factors\AR_1\beta2\residuals.csv'
    beta3_path = rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\{country}\factors\AR_1\beta3\residuals.csv'
    pdf_output_path = rf'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\graphs\residual_analysis_report_{country}.pdf'
    
    # Execution date
    execution_date = '2025-01-01'
    
    # Load and preprocess the data
    merged_df = load_and_merge_data(beta1_path, beta2_path, beta3_path)
    filtered_df = filter_data_by_execution_date(merged_df, execution_date)
    
    # Create a PDF to save the results
    with PdfPages(pdf_output_path) as pdf:
        # Correlation matrix and analysis
        analyze_correlation(filtered_df, pdf, execution_date)
    
        # Pair plot
        plot_pairplot(filtered_df, pdf, execution_date)
    
        # Residuals over time
        plot_residuals_over_time(filtered_df, pdf, execution_date)
    
        # Histograms of residuals
        plot_histograms(filtered_df, pdf, execution_date)
    
        # Box plot of residuals
        plot_boxplot(filtered_df, pdf, execution_date)
    
        # Moving correlations
        plot_moving_correlations(filtered_df, pdf)
    
        # Serial correlation analysis
        analyze_serial_correlation(filtered_df, pdf)
    
        # Autocorrelation plots
        plot_acf_plots(filtered_df, pdf)
    
    print(f"Analysis complete. Results saved to {pdf_output_path}")
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# 1) BASIC HELPERS
# ----------------------------

def _as_array(x):
    return np.asarray(x).reshape(-1)

def _safe_prob(p):
    eps = 1e-12
    return np.clip(p, eps, 1 - eps)

# ----------------------------
# 2) KUPIEC POF TEST
# ----------------------------

def kupiec_pof_test(realized, var_series, alpha):
    """
    Kupiec (POF) unconditional coverage test.
    H0: P(violation) = 1 - alpha
    Returns dict with LR statistic, p-value, exceptions, and hit rate.
    """
    r = _as_array(realized)
    v = _as_array(var_series)
    T = r.size
    I = (r < v).astype(int)
    T1 = I.sum()  # Number of exceptions
    T0 = T - T1  # Non-exceptions

    p_hat = _safe_prob(T1 / T)
    p0 = _safe_prob(1 - alpha)

    # Likelihood ratio
    lr = -2 * (
        (T0 * np.log(1 - p0) + T1 * np.log(p0)) -
        (T0 * np.log(1 - p_hat) + T1 * np.log(p_hat))
    )
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR": lr, "p_value": pval, "exceptions": T1, "hit_rate": T1 / T}

# ----------------------------
# 3) WORKFLOW FOR POF TEST
# ----------------------------

# Load the dataset
df = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual_metrics.csv')

# Ensure execution_date is a datetime object
df['execution_date'] = pd.to_datetime(df['execution_date'])

# Define parameters
maturities = df['maturity_years'].unique()
horizons = df['horizon_years'].unique()
alpha_levels = [0.95, 0.975, 0.99]

# Initialize results list
results = []

# Iterate over maturities and horizons
for maturity in maturities:
    for horizon in horizons:
        # Filter data for the current maturity and horizon
        df_subset = df[(df['maturity_years'] == maturity) & (df['horizon_years'] == horizon)]
        df_subset = df_subset.sort_values(by='execution_date')
        
        # Drop duplicates, keeping the first occurrence
        df_subset = df_subset.drop_duplicates(subset=['execution_date', 'metric'], keep='first')
        
        # Reattempt pivot
        pivoted_df = df_subset.pivot(index='execution_date', columns='metric', values='value')
                
        # Reattempt pivot
        pivoted_df = df_subset.pivot(index='execution_date', columns='metric', values='value')
        
        required_columns = ['Observed Annual Return', 'VaR 95', 'VaR 97', 'VaR 99']
        pivoted_df = pivoted_df[required_columns]
        
        # Drop rows where 'Observed Annual Return' is NaN
        pivoted_df = pivoted_df.dropna(subset=['Observed Annual Return'])
                
        # Prepare realized returns
        #obs_dates = pivoted_df['Observed Annual Return'].dropna()['execution_date']
        realized = pivoted_df['Observed Annual Return'].values

        # Skip if no realized returns are available
        if realized.size == 0:
            continue

        # Prepare VaR dictionary
        for alpha in alpha_levels:
            var_series = pivoted_df[f'VaR {int(alpha * 100)}'].values

            # Skip if sizes do not match
            if realized.size != var_series.size:
                print(f"Size mismatch for maturity {maturity}, horizon {horizon}, alpha {alpha}")
                continue

            # Run Kupiec POF test
            pof_result = kupiec_pof_test(realized, var_series, alpha)

            # Append results
            results.append({
                "Model": "Mixed_Model",
                "Maturity": maturity,
                "Horizon": horizon,
                "Alpha": alpha,
                "LR": pof_result["LR"],
                "p_value": pof_result["p_value"],
                "Exceptions": pof_result["exceptions"],
                "Hit Rate": pof_result["hit_rate"]
            })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# ----------------------------
# 4) VISUALIZATION
# ----------------------------

# Example: Heatmap for POF Test p-values by Maturity and Horizon
for alpha in alpha_levels:
    pof_results = results_df[results_df['Alpha'] == alpha]
    heatmap_data = pof_results.pivot(index='Maturity', columns='Horizon', values='p_value')

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1)
    plt.title(f"POF Test \( p \)-Values by Maturity and Horizon (Alpha = {alpha})")
    plt.xlabel("Horizon")
    plt.ylabel("Maturity")
    plt.show()
    
    heatmap_data = pof_results.pivot(index='Maturity', columns='Horizon', values='Hit Rate')

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1)
    plt.title(f"Hit Rate by Maturity and Horizon (Alpha = {alpha})")
    plt.xlabel("Horizon")
    plt.ylabel("Maturity")
    plt.show()
    
    heatmap_data = pof_results.pivot(index='Maturity', columns='Horizon', values='Exceptions')

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r")
    plt.title(f"Exceptions by Maturity and Horizon (Alpha = {alpha})")
    plt.xlabel("Horizon")
    plt.ylabel("Maturity")
    plt.show()
    
    

# Example: Summary of Results
print("Summary of Kupiec POF Test Results:")
print(results_df)

for maturity in maturities:
    # Filter data for the current maturity
    df_maturity = df[df['maturity_years'] == maturity].sort_values(by='execution_date')

    # Create a figure with subplots (1 row per horizon, 5 columns for VaR levels)
    horizons = list(df_maturity['horizon_years'].unique())
    n_horizons = len(horizons)
    horizons.sort()
    fig, axes = plt.subplots(n_horizons, 1, figsize=(15, 5 * n_horizons), sharex=False)

    if n_horizons == 1:  # If there's only one horizon, wrap axes in a list for consistency
        axes = [axes]

    for i, horizon in enumerate(horizons):
        # Filter data for the current horizon
        df_horizon = df_maturity[df_maturity['horizon_years'] == horizon]

        # Pivot the data for this horizon
        df_horizon = df_horizon.drop_duplicates(subset=['execution_date', 'metric'], keep='first')
        pivoted_df = df_horizon.pivot(index='execution_date', columns='metric', values='value')

        # Ensure required columns exist
        required_columns = ['Observed Annual Return', 'VaR 95', 'VaR 97', 'VaR 99']
        if not all(col in pivoted_df.columns for col in required_columns):
            print(f"Skipping horizon {horizon} for maturity {maturity}: Missing required columns.")
            continue

        # Drop rows with NaN in 'Observed Annual Return'
        pivoted_df = pivoted_df.dropna(subset=['Observed Annual Return'])

        # Plot Observed Annual Return and VaR levels
        ax = axes[i]
        ax.plot(pivoted_df.index, pivoted_df['Observed Annual Return'], label='Observed Annual Return', color='black', linewidth=1.5)
        ax.plot(pivoted_df.index, pivoted_df['VaR 95'], label='VaR 95', color='blue', linestyle='--')
        ax.plot(pivoted_df.index, pivoted_df['VaR 97'], label='VaR 97', color='orange', linestyle='--')
        ax.plot(pivoted_df.index, pivoted_df['VaR 99'], label='VaR 99', color='red', linestyle='--')
        ax.grid()
        # Add labels and legend
        ax.set_title(f"Maturity: {maturity}, Horizon: {horizon}")
        ax.set_ylabel("Returns / VaR")
        ax.legend(loc='upper left')

    # Add x-axis label to the last subplot
    axes[-1].set_xlabel("Execution Date")

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f"Observed Returns and VaR Levels for Maturity {maturity}", y=1.02, fontsize=16)
    plt.show()
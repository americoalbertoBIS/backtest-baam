# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:32:43 2025

@author: al005366
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# ----------------------------
# 1) BASIC HELPERS
# ----------------------------

def _as_array(x):
    return np.asarray(x).reshape(-1)

def _safe_prob(p):
    # avoid log(0) in likelihoods
    eps = 1e-12
    return np.clip(p, eps, 1 - eps)

def empirical_cdf_from_sims(sims_row, value):
    """
    Empirical CDF F_hat(value) from a vector of simulations.
    Uses plotting position (rank + 1)/(N + 1) to avoid 0/1.
    """
    sims = _as_array(sims_row)
    N = sims.size
    k = np.sum(sims <= value)
    return (k + 1.0) / (N + 1.0)

# ----------------------------
# 2) VAR BACKTESTS
# ----------------------------

def kupiec_pof_test(realized, var_series, alpha):
    """
    Kupiec (POF) unconditional coverage test.
    H0: P(violation) = 1 - alpha
    Returns dict with LR statistic, p-value, counts.
    """
    r = _as_array(realized)
    v = _as_array(var_series)
    assert r.size == v.size
    T = r.size
    I = (r < v).astype(int)
    T1 = I.sum()
    T0 = T - T1

    p_hat = _safe_prob(T1 / T)
    p0 = _safe_prob(1 - alpha)

    # Likelihood ratio
    lr = -2 * (
        (T0 * np.log(1 - p0) + T1 * np.log(p0))
        - (T0 * np.log(1 - p_hat) + T1 * np.log(p_hat))
    )
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR": lr, "p_value": pval, "alpha": alpha, "T": T, "exceptions": int(T1), "hit_rate": T1 / T}

def christoffersen_independence_test(realized, var_series):
    """
    Christoffersen (1998) independence test for violation clustering.
    First-order Markov test on the hit sequence.
    H0: Independence (no clustering).
    """
    r = _as_array(realized)
    v = _as_array(var_series)
    assert r.size == v.size
    I = (r < v).astype(int)

    # Transition counts n_ij for Markov chain of hits
    I_lag = I[:-1]
    I_now = I[1:]
    n00 = np.sum((I_lag == 0) & (I_now == 0))
    n01 = np.sum((I_lag == 0) & (I_now == 1))
    n10 = np.sum((I_lag == 1) & (I_now == 0))
    n11 = np.sum((I_lag == 1) & (I_now == 1))

    # MLEs
    pi01 = _safe_prob(n01 / max(n00 + n01, 1))
    pi11 = _safe_prob(n11 / max(n10 + n11, 1))
    pi1  = _safe_prob((n01 + n11) / max(n00 + n01 + n10 + n11, 1))

    # Likelihoods
    L_indep = ((1 - pi1) ** (n00 + n10)) * (pi1 ** (n01 + n11))
    L_markov = ((1 - pi01) ** n00) * (pi01 ** n01) * ((1 - pi11) ** n10) * (pi11 ** n11)

    lr = -2 * np.log(_safe_prob(L_indep) / _safe_prob(L_markov))
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR": lr, "p_value": pval, "counts": {"n00": int(n00), "n01": int(n01), "n10": int(n10), "n11": int(n11)}}

def christoffersen_conditional_coverage(realized, var_series, alpha):
    """
    Christoffersen joint test = POF (coverage) + independence.
    H0: correct unconditional coverage and independence.
    """
    pof = kupiec_pof_test(realized, var_series, alpha)
    indep = christoffersen_independence_test(realized, var_series)
    lr = pof["LR"] + indep["LR"]
    pval = 1 - stats.chi2.cdf(lr, df=2)
    return {"LR": lr, "p_value": pval, "components": {"POF": pof, "Independence": indep}}

# ----------------------------
# 3) ES (CVaR) BACKTEST (simple, transparent) # ----------------------------

def es_mean_exceedance_test(realized, var_series, es_series, alpha):
    """
    Intuition: below-VaR losses (tail losses) should average ES - VaR.
    Define TL_t = (r_t - VaR_t) given r_t < VaR_t.
    Test H0: mean(TL_t) == (ES_t - VaR_t) (one-sample t-test on differences).
    This is a practical, intuitive ES backtest (not the full AS14 scoring test).
    """
    r = _as_array(realized)
    v = _as_array(var_series)
    e = _as_array(es_series)
    assert r.size == v.size == e.size

    mask = r < v
    if mask.sum() < 5:
        return {"alpha": alpha, "n_tail": int(mask.sum()), "p_value": np.nan, "t_stat": np.nan,
                "message": "Too few tail observations for a meaningful ES test."}

    tl = r[mask] - v[mask]
    target = (e[mask] - v[mask])

    diff = tl - target
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(diff.size)
    t_stat = mean_diff / se if se > 0 else np.inf
    df = diff.size - 1
    # two-sided
    pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

    return {"alpha": alpha, "n_tail": int(mask.sum()), "t_stat": float(t_stat), "p_value": float(pval),
            "avg_actual_tail": float(tl.mean()), "avg_modeled_tail": float(target.mean())}

def es_exceedance_ratio_test(realized, var_series, es_series, alpha):
    """
    Alternative quick check: Z_t = (r_t - VaR_t) / (ES_t - VaR_t) on violations.
    Under a well-calibrated ES, E[ Z_t | r_t < VaR_t ] ≈ 1.
    Tests mean(Z) == 1 with a t-test on (Z - 1).
    """
    r = _as_array(realized)
    v = _as_array(var_series)
    e = _as_array(es_series)
    mask = r < v
    if mask.sum() < 5:
        return {"alpha": alpha, "n_tail": int(mask.sum()), "p_value": np.nan, "t_stat": np.nan,
                "message": "Too few tail observations for a meaningful ES test."}
    denom = e[mask] - v[mask]
    # avoid division by zero (should be negative number)
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
    Z = (r[mask] - v[mask]) / denom
    Y = Z - 1.0
    meanY = Y.mean()
    se = Y.std(ddof=1) / np.sqrt(Y.size)
    t_stat = meanY / se if se > 0 else np.inf
    df = Y.size - 1
    pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    return {"alpha": alpha, "n_tail": int(mask.sum()), "t_stat": float(t_stat), "p_value": float(pval),
            "mean_Z": float(Z.mean())}

# ----------------------------
# 4) PIT-BASED DISTRIBUTION TESTS
# ----------------------------

def compute_pit_from_simulations(realized, simulations):
    """
    realized: (T,)
    simulations: (T, N) array where simulations[t, :] are the draws for date t.
    Returns u in (0,1), the PITs per date.
    """
    r = _as_array(realized)
    sims = np.asarray(simulations)
    assert sims.ndim == 2 and sims.shape[0] == r.size
    T = r.size
    u = np.empty(T, dtype=float)
    for t in range(T):
        u[t] = empirical_cdf_from_sims(sims[t, :], r[t])
    return u

def pit_uniformity_tests(pit_u, lags_for_lb=(1, 5, 10)):
    """
    Tests whether PITs are ~ U(0,1) and independent.
    Approach:
    - Transform z = Φ^{-1}(u) and test normality (KS, Anderson-Darling, Jarque-Bera)
    - Ljung-Box on z for serial independence
    """
    u = _as_array(pit_u)
    # Guard against 0/1
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    z = stats.norm.ppf(u)

    # Normality tests
    ks_stat, ks_p = stats.kstest(z, 'norm')
    ad = stats.anderson(z, dist='norm')  # returns statistic and critical values
    jb_stat, jb_p = stats.jarque_bera(z)

    # Serial independence (Ljung-Box) on z
    lb = {}
    for L in lags_for_lb:
        lb_stat, lb_p = acorr_ljungbox(z, lags=[L], return_df=False)
        lb[L] = {"LB_stat": float(lb_stat[0]), "p_value": float(lb_p[0])}

    return {
        "KS_norm": {"stat": float(ks_stat), "p_value": float(ks_p)},
        "AD_norm": {"stat": float(ad.statistic), "critical_values": ad.critical_values.tolist(), "significance": ad.significance_level.tolist()},
        "JB_norm": {"stat": float(jb_stat), "p_value": float(jb_p)},
        "LjungBox_on_z": lb
    }

# ----------------------------
# 5) WRAPPERS TO RUN EVERYTHING OVER MULTIPLE ALPHAS # ----------------------------

def run_var_es_backtests(realized, var_dict, es_dict=None, alphas=(0.95, 0.975, 0.99)):
    """
    realized: (T,)
    var_dict: {alpha: array_like of shape (T,)} for VaR at each alpha
    es_dict:  {alpha: array_like of shape (T,)} for ES at each alpha (optional for ES tests)
    Returns a structured dict of results.
    """
    results = {"VaR": {}, "ES": {}}
    for a in alphas:
        var_series = var_dict[a]
        # VaR tests
        pof = kupiec_pof_test(realized, var_series, a)
        indep = christoffersen_independence_test(realized, var_series)
        cc = christoffersen_conditional_coverage(realized, var_series, a)
        results["VaR"][a] = {"POF": pof, "Independence": indep, "ConditionalCoverage": cc}

        # ES tests if provided
        if es_dict is not None and a in es_dict:
            es_series = es_dict[a]
            es_mean = es_mean_exceedance_test(realized, var_series, es_series, a)
            es_ratio = es_exceedance_ratio_test(realized, var_series, es_series, a)
            results["ES"][a] = {"MeanTail_vs_ES": es_mean, "Zratio_test": es_ratio}
    return results

# Load the dataset
df = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual_metrics.csv')

# Ensure execution_date is a datetime object
df['execution_date'] = pd.to_datetime(df['execution_date'])

# Example: Prepare data for one maturity and multiple horizons
# Define maturities and horizons
maturities = df['maturity_years'].unique()
horizons = df['horizon_years'].unique()
alpha_levels = [0.95, 0.975, 0.99]

# Create an empty list to store results
results = []

# Iterate over maturities and horizons
for maturity in maturities:
    for horizon in horizons:
        # Filter data for the current maturity and horizon
        df_subset = df[(df['maturity_years'] == maturity) & (df['horizon_years'] == horizon)]

        # Prepare realized returns
        realized = df_subset[df_subset['metric'] == 'Observed Annual Return']['value'].values

        # Skip if no realized returns are available
        if realized.size == 0:
            continue

        # Prepare VaR and ES dictionaries
        var_dict = {}
        es_dict = {}
        for alpha in alpha_levels:
            var_dict[alpha] = df_subset[df_subset['metric'] == f'VaR {int(alpha * 100)}']['value'].values
            es_dict[alpha] = df_subset[df_subset['metric'] == f'CVaR {int(alpha * 100)}']['value'].values

        # Run backtests
        backtest_results = run_var_es_backtests(realized, var_dict, es_dict, alphas=alpha_levels)

        # Store results in a structured format
        for alpha in alpha_levels:
            # VaR results
            pof = backtest_results["VaR"][alpha]["POF"]
            indep = backtest_results["VaR"][alpha]["Independence"]
            cc = backtest_results["VaR"][alpha]["ConditionalCoverage"]

            results.append({
                "Maturity": maturity,
                "Horizon": horizon,
                "Alpha": alpha,
                "Test": "POF",
                "LR": pof["LR"],
                "p_value": pof["p_value"],
                "Exceptions": pof["exceptions"],
                "Hit Rate": pof["hit_rate"]
            })
            results.append({
                "Maturity": maturity,
                "Horizon": horizon,
                "Alpha": alpha,
                "Test": "Independence",
                "LR": indep["LR"],
                "p_value": indep["p_value"],
                "n00": indep["counts"]["n00"],
                "n01": indep["counts"]["n01"],
                "n10": indep["counts"]["n10"],
                "n11": indep["counts"]["n11"]
            })
            results.append({
                "Maturity": maturity,
                "Horizon": horizon,
                "Alpha": alpha,
                "Test": "ConditionalCoverage",
                "LR": cc["LR"],
                "p_value": cc["p_value"],
                "POF_LR": cc["components"]["POF"]["LR"],
                "Independence_LR": cc["components"]["Independence"]["LR"]
            })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Summarize key metrics
summary = (
    results_df.groupby(["Maturity", "Horizon", "Alpha", "Test"])
    .agg({
        "p_value": "mean",
        "Hit Rate": "mean",
        "Exceptions": "sum"
    })
    .reset_index()
)

# Pivot the table for better readability
summary_pivot = summary.pivot(
    index=["Maturity", "Horizon", "Alpha"],
    columns="Test",
    values=["p_value", "Hit Rate", "Exceptions"]
)

# Display the summarized DataFrame
print(summary_pivot)

# Filter for poor performance (e.g., p_value < 0.05)
poor_performance = summary[
    (summary["p_value"] > 0.05) & (summary["Test"] == "POF")
]

print("Poorly Performing Areas (POF Test):")
print(poor_performance)

poor_independence = summary[
    (summary["p_value"] > 0.05) & (summary["Test"] == "Independence")
]
print("Poorly Performing Areas (Independence Test):")
print(poor_independence)

import seaborn as sns
import matplotlib.pyplot as plt

# Filter for POF test results
pof_results = results_df[results_df["Test"] == "POF"]

# Pivot the data for heatmap plotting
heatmap_data = pof_results.pivot_table(
    index="Maturity", columns="Horizon", values="p_value", aggfunc="mean"
)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1)
plt.title("POF Test \( p \)-Values by Maturity and Horizon")
plt.xlabel("Horizon")
plt.ylabel("Maturity")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Filter for POF test results
pof_results = results_df[results_df['Test'] == 'POF']

# Pivot the data for heatmap plotting (e.g., maturity vs horizon for alpha = 0.95)
alpha = 0.99
heatmap_data = pof_results[pof_results['Alpha'] == alpha].pivot(
    index='Maturity', columns='Horizon', values='p_value'
)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1)
plt.title(f"POF Test \( p \)-Values (Alpha = {alpha})")
plt.xlabel("Horizon")
plt.ylabel("Maturity")
plt.show()

# Aggregate exceptions by maturity and horizon
exceptions_data = (
    results_df[results_df["Test"] == "POF"]
    .groupby(["Maturity", "Horizon"])["Exceptions"]
    .sum()
    .reset_index()
)

# Pivot for plotting
exceptions_pivot = exceptions_data.pivot(index="Maturity", columns="Horizon", values="Exceptions")

# Plot bar chart
exceptions_pivot.plot(kind="bar", figsize=(12, 8), stacked=True)
plt.title("Total Exceptions by Maturity and Horizon")
plt.xlabel("Maturity")
plt.ylabel("Number of Exceptions")
plt.legend(title="Horizon")
plt.show()


# Filter for POF test results
exceptions_data = pof_results.pivot(
    index='Maturity', columns='Horizon', values='Exceptions'
)

# Define the alpha levels
alpha_levels = [0.95, 0.975, 0.99]

# Prepare an empty list to store results
results_by_date = []

# Group data by execution date
execution_dates = df['execution_date'].unique()

# Iterate over execution dates
for date in execution_dates:
    # Filter data for the current execution date
    df_date = df[df['execution_date'] == date]

    # Prepare realized returns
    realized = df_date[df_date['metric'] == 'Observed Annual Return']['value'].values

    # Skip if no realized returns are available
    if realized.size == 0:
        continue

    # Prepare VaR and ES dictionaries
    var_dict = {}
    es_dict = {}
    for alpha in alpha_levels:
        var_dict[alpha] = df_date[df_date['metric'] == f'VaR {int(alpha * 100)}']['value'].values
        es_dict[alpha] = df_date[df_date['metric'] == f'CVaR {int(alpha * 100)}']['value'].values

    # Run backtests
    backtest_results = run_var_es_backtests(realized, var_dict, es_dict, alphas=alpha_levels)

    # Store results in a structured format
    for alpha in alpha_levels:
        # VaR results
        pof = backtest_results["VaR"][alpha]["POF"]
        indep = backtest_results["VaR"][alpha]["Independence"]
        cc = backtest_results["VaR"][alpha]["ConditionalCoverage"]

        results_by_date.append({
            "Execution Date": date,
            "Alpha": alpha,
            "Test": "POF",
            "LR": pof["LR"],
            "p_value": pof["p_value"],
            "Exceptions": pof["exceptions"],
            "Hit Rate": pof["hit_rate"]
        })
        results_by_date.append({
            "Execution Date": date,
            "Alpha": alpha,
            "Test": "Independence",
            "LR": indep["LR"],
            "p_value": indep["p_value"],
            "n00": indep["counts"]["n00"],
            "n01": indep["counts"]["n01"],
            "n10": indep["counts"]["n10"],
            "n11": indep["counts"]["n11"]
        })
        results_by_date.append({
            "Execution Date": date,
            "Alpha": alpha,
            "Test": "ConditionalCoverage",
            "LR": cc["LR"],
            "p_value": cc["p_value"],
            "POF_LR": cc["components"]["POF"]["LR"],
            "Independence_LR": cc["components"]["Independence"]["LR"]
        })

# Convert results to a DataFrame
results_by_date_df = pd.DataFrame(results_by_date)

# Display the DataFrame
print(results_by_date_df)

# Example: Rolling empirical coverage for alpha = 0.95
alpha = 0.95

# Filter data for the selected alpha level
pof_results = results_by_date_df[(results_by_date_df['Test'] == 'POF') & (results_by_date_df['Alpha'] == alpha)]

# Plot rolling empirical coverage
plt.figure(figsize=(12, 6))
plt.plot(pof_results['Execution Date'], pof_results['Hit Rate'], label=f"Empirical Coverage (Alpha = {alpha})")
plt.axhline(1 - alpha, color='red', linestyle='--', label="Target Coverage")
plt.title(f"Empirical Coverage Over Time (Alpha = {alpha})")
plt.xlabel("Execution Date")
plt.ylabel("Coverage")
plt.legend()
plt.show()

# Pivot the data for heatmap plotting
heatmap_data = results_by_date_df[results_by_date_df['Test'] == 'POF'].pivot(
    index='Execution Date', columns='Alpha', values='p_value'
)
heatmap_data.index = heatmap_data.index.astype(str)
# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data.T, annot=False, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1)
plt.title("POF Test \( p \)-Values Over Time")
plt.xlabel("Alpha")
plt.ylabel("Execution Date")
plt.show()

# Plot exceptions over time
plt.figure(figsize=(12, 6))
plt.plot(pof_results['Execution Date'], pof_results['Exceptions'], label=f"Exceptions (Alpha = {alpha})")
plt.title(f"Exceptions Over Time (Alpha = {alpha})")
plt.xlabel("Execution Date")
plt.ylabel("Number of Exceptions")
plt.legend()
plt.show()


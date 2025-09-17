# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:24:25 2025

@author: al005366
"""

import pandas as pd
from scipy.stats import chi2, weibull_min
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\monthly_metrics.csv')

# Ensure execution_date is a datetime object
df['execution_date'] = pd.to_datetime(df['execution_date'])

# Pivot the data for easier analysis
df_wide = df.pivot_table(
    index=['execution_date', 'horizon_years'],
    columns='metric',
    values='value'
).reset_index()

# Rename columns to be Python-friendly
df_wide.columns = [c.strip().lower().replace(' ', '_').replace('%', 'pct') for c in df_wide.columns]

obs_returns_col = 'observed_annual_return'
exp_returns_col = 'expected_annual_returns'
# Drop rows with missing observed_annual_return or zero expected_annual_returns
df_wide = df_wide.dropna(subset=[obs_returns_col])
df_wide = df_wide[df_wide[exp_returns_col] != 0]

# Define helper functions for tests
def to_viol_array(viol):
    """Coerce list/ndarray/Series/DataFrame of 0/1 or bools to 1-d np.int8 array."""
    arr = np.asarray(viol).flatten()
    if not np.all(np.isin(arr, [0, 1, True, False])):
        raise ValueError("All values must be 0/1 or boolean.")
    return arr.astype(np.int8)

def kupiec_test(violations, var_conf_level=0.99):
    """Unconditional coverage LR test."""
    viol = to_viol_array(violations)
    n = len(viol)
    nex = int(viol.sum())  # Number of breaches
    p0 = 1 - var_conf_level
    ph = nex / n if n > 0 else 0  # Observed breach rate

    if ph in (0, 1) or n == 0:
        return {"LR_uc": np.inf, "p_uc": 0.0}  # Invalid test due to zero breaches

    # Likelihood ratio calculation
    LR = -2 * (
        nex * np.log(p0) + (n - nex) * np.log(1 - p0)
        - nex * np.log(ph) - (n - nex) * np.log(1 - ph)
    )
    pvalue = 1 - chi2.cdf(LR, df=1)
    return {"LR_uc": LR, "p_uc": pvalue}

def christoffersen_test(violations):
    """Independence (no-clustering) LR test."""
    viol = to_viol_array(violations)
    if len(viol) < 2 or viol.sum() == 0:
        return {"LR_ind": np.nan, "p_ind": np.nan}  # Invalid test due to zero breaches

    v0, v1 = viol[:-1], viol[1:]
    n00 = ((v0 == 0) & (v1 == 0)).sum()
    n01 = ((v0 == 0) & (v1 == 1)).sum()
    n10 = ((v0 == 1) & (v1 == 0)).sum()
    n11 = ((v0 == 1) & (v1 == 1)).sum()

    # Transition probabilities
    Tm1 = len(viol) - 1
    pi = (n01 + n11) / Tm1 if Tm1 > 0 else 0.0
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    # Handle zero probabilities
    if pi in (0, 1) or pi0 in (0, 1) or pi1 in (0, 1):
        return {"LR_ind": np.nan, "p_ind": np.nan}

    # Likelihood ratio calculation
    ll_ind = n00 * np.log(1 - pi) + n01 * np.log(pi) + n10 * np.log(1 - pi) + n11 * np.log(pi)
    ll_dep = n00 * np.log(1 - pi0) + n01 * np.log(pi0) + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    LR_ind = -2 * (ll_ind - ll_dep)
    p_ind = 1 - chi2.cdf(LR_ind, df=1)
    return {"LR_ind": LR_ind, "p_ind": p_ind}

# Christoffersen‐Pelletier duration test: Weibull vs Exponential
def duration_test(violations, conf_level=0.95):
    """Christoffersen‐Pelletier duration test: Weibull vs Exponential."""
    viol = to_viol_array(violations)
    hits = np.nonzero(viol)[0] + 1  # Get indices of breaches (1-based indexing)
    if len(hits) < 2:
        return {"LR_dur": np.nan, "p_dur": np.nan, "decision": "too few hits"}  # Not enough breaches

    
    D_head = hits[0]
    D_mid = np.diff(hits)
    D_tail = len(viol) + 1 - hits[-1]
    D = np.r_[D_head, D_mid, D_tail]
    C = np.array([1] + [0] * len(D_mid) + [1])
    N = len(D)

    def nll(b):
        a = ((D**b).sum() / (N - C.sum())) ** (-1.0 / b)
        ll = 0.0
        for Di, Ci in zip(D, C):
            if Ci:
                ll += weibull_min.logsf(Di, b, scale=1 / a)
            else:
                ll += weibull_min.logpdf(Di, b, scale=1 / a)
        return -ll

    res = optimize.minimize_scalar(nll, bounds=(0.1, 10), method='bounded')
    b_hat = res.x
    ll_u = -nll(b_hat)
    ll_r = -nll(1.0)
    LR = 2 * (ll_u - ll_r)
    pval = 1 - chi2.cdf(LR, df=1)
    decision = "reject" if pval < (1 - conf_level) else "fail to reject"
    return {"b_hat": b_hat, "LR_dur": LR, "p_dur": pval, "decision": decision}

# Define parameters
alphas = [0.95, 0.975, 0.99]
horizons = df_wide['horizon_years'].unique()

# Group data by maturity and run backtesting for each group
maturities = df['maturity_years'].unique()

results = []
for maturity in maturities:
    print(f"Maturity: {maturity}")
    # Filter data for the current maturity
    df_maturity = df[df['maturity_years'] == maturity]
    
    # Pivot the data for easier analysis
    df_wide = df_maturity.pivot_table(
        index=['execution_date', 'horizon_years'],
        columns='metric',
        values='value'
    ).reset_index()
    
    # Rename columns to be Python-friendly
    df_wide.columns = [
        c.strip().lower().replace(' ', '_').replace('%', 'pct') for c in df_wide.columns
    ]
    
    # Drop rows with missing observed_annual_return or zero expected_annual_returns
    df_wide = df_wide.dropna(subset=[obs_returns_col])
    df_wide = df_wide[df_wide[exp_returns_col] != 0]
    
    # Run backtesting for each horizon and alpha
    rows = []
    for h in sorted(df_wide['horizon_years'].unique()):
        dfh = df_wide[df_wide.horizon_years == h]
        obs = dfh[obs_returns_col].values
        
        for α in alphas:
            varcol = f'var_{int(α*100)}'
            if varcol not in dfh.columns or f'cvar_{int(α*100)}' not in dfh.columns:
                continue  # Skip if required columns are missing
            
            # Calculate violations
            viol = (obs < dfh[varcol]).astype(int)
            
            # Run Kupiec test
            kupiec_result = kupiec_test(viol, 1 - α)
            LR_uc, p_uc = kupiec_result["LR_uc"], kupiec_result["p_uc"]
            
            # Run Christoffersen test
            christoffersen_result = christoffersen_test(viol)
            LR_ind, p_ind = christoffersen_result["LR_ind"], christoffersen_result["p_ind"]
            
            # Combined conditional coverage test
            LR_cc = LR_uc + LR_ind
            p_cc = 1 - chi2.cdf(LR_cc, df=2)
    
            # Run duration test
            duration_result = duration_test(viol)
            LR_dur = duration_result["LR_dur"]
            p_dur = duration_result["p_dur"]
            decision = duration_result["decision"]
            
            # Calculate average width
            raw_widths = np.abs(dfh[varcol] - dfh[f'cvar_{int(α*100)}'])
            # Print diagnostics
            print(f"Alpha: {α}, Horizon: {h}")
            print(f"Observed Breaches: {viol.sum()}, Expected Breaches: {len(viol) * (1 - α)}")
            print(f"LR_uc: {LR_uc}, p_uc: {p_uc}")
            print(f"LR_ind: {LR_ind}, p_ind: {p_ind}")
            print(f"LR_dur: {LR_dur}, p_dur: {p_dur}")
            
            # Append results
            rows.append({
                'maturity': maturity,
                'horizon': h,
                'alpha': α,
                'empirical_cov': 1 - viol.mean(),
                'viol_num': len(viol[viol!=0]),
                'obs_num': len(obs),
                'Kupiec_p': p_uc,
                'Christoffersen_p': p_cc,
                'Duration_p': p_dur,
                'Duration_decision': decision,
                'avg_width': raw_widths.mean()
            })
    
    # Append results for this maturity
    results.extend(rows)

# Create a DataFrame with all results
summary = pd.DataFrame(results)

# Analyze results
print(summary)

# Filter and print results
sig = 0.05
mask = (summary['Kupiec_p'] > sig) & (summary['Christoffersen_p'] > sig) & (summary['Duration_p'] > sig)
passers = summary[mask].copy()
print(passers[['horizon', 'alpha', 'empirical_cov', 'Kupiec_p', 'Christoffersen_p', 'Duration_p', 'avg_width']])

# Plot well-calibrated combos ranked by sharpness
labels = passers.apply(lambda r: f"{int(r.horizon)}y @ {int(r.alpha*100)}%", axis=1)
plt.figure(figsize=(8, 6))
plt.barh(labels, passers['avg_width'], color='C2')
plt.xlabel("Average Width")
plt.title("Well‐calibrated combos (p>0.05) ranked by sharpness")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Heatmaps for Kupiec, Christoffersen, and Duration test p-values
for col, title in [
    ('Kupiec_p', 'Kupiec p-values'),
    ('Christoffersen_p', 'Christoffersen p-values'),
    ('Duration_p', 'Duration p-values'),
]:
    pt = summary.pivot_table(values=col, index='horizon', columns='alpha')
    plt.figure(figsize=(6, 4))
    sns.heatmap(pt, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1)
    plt.title(title)
    plt.ylabel("Horizon (years)")
    plt.xlabel("VaR level α")
    plt.show()

# Empirical coverage heatmap
pt = summary.pivot(index='horizon', columns='alpha', values='empirical_cov')
plt.figure(figsize=(6, 4))
sns.heatmap(
    pt,
    annot=True,
    fmt=".3f",
    cmap="RdYlBu_r",
    center=1 - 0.95,   # center on target coverage (e.g. 0.05 distance from 1)
    vmin=pt.min().min(),
    vmax=pt.max().max()
)
plt.title("Empirical Coverage Heatmap")
plt.ylabel("Horizon (years)")
plt.xlabel("VaR level α")
plt.tight_layout()
plt.show()
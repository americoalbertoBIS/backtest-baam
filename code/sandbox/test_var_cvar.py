# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:24:25 2025

@author: al005366
"""

import pandas as pd

df = pd.read_csv(r'L:\RMAS\Users\Alberto\backtest-baam\data_joint\US\returns\estimated_returns\Mixed_Model\annual_metrics.csv')


import pandas as pd
import numpy as np

# assume df has columns: maturity_years, execution_date, horizon_years, metric, value
# and that the "Observed Annual Return" rows have been filled in
df['execution_date'] = pd.to_datetime(df['execution_date'])

# pivot so each row is (execution_date, horizon_years), cols are metrics
df_wide = df.pivot_table(
    index=['execution_date','horizon_years'],
    columns='metric',
    values='value'
).reset_index()


# rename columns to legal Python names
df_wide.columns = [c.strip().lower().replace(' ','_').replace('%','pct') for c in df_wide.columns]
# e.g. df_wide now has columns:
# 'execution_date','horizon_years','expected_annual_returns','volatility',
# 'observed_annual_return','var_95','cvar_95','var_97_5','cvar_97_5','var_99','cvar_99'

df_wide = df_wide.dropna(subset=['observed_annual_return'])
df_wide = df_wide[df_wide['expected_annual_returns']!=0]

from scipy.stats import chi2

from scipy.stats import chi2
import numpy as np

def kupiec_test(violations, alpha):
    T = len(violations)
    N = violations.sum()
    p_hat = N / T
    # log‐likelihood under H0: P(failure)=alpha
    ll0 = (T-N)*np.log(1-alpha) + N*np.log(alpha)
    # log‐likelihood under H1: P(failure)=p_hat
    # guard against p_hat=0 or 1
    if p_hat in (0,1):
        ll1 = -np.inf
    else:
        ll1 = (T-N)*np.log(1-p_hat) + N*np.log(p_hat)
    LR_uc = -2*(ll0 - ll1)
    p_uc = 1 - chi2.cdf(LR_uc, df=1)
    return LR_uc, p_uc

import numpy as np
from scipy.stats import chi2

def christoffersen_test(violations):
    """
    violations: 1d array of 0/1 exception indicators
    returns: LR_ind, p_ind  (test of independence of exceptions)
    """
    v0 = violations[:-1]
    v1 = violations[1:]
    n00 = int(np.sum((v0==0)&(v1==0)))
    n01 = int(np.sum((v0==0)&(v1==1)))
    n10 = int(np.sum((v0==1)&(v1==0)))
    n11 = int(np.sum((v0==1)&(v1==1)))
    Tm1 = len(violations) - 1

    # unconditional exception rate
    pi = (n01 + n11) / Tm1

    # state‐dependent rates
    pi0 = n01 / (n00 + n01) if (n00 + n01)>0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11)>0 else 0.0

    # helper to do count*log(prob) safely
    def safe_ll(count, prob):
        if count == 0:
            return 0.0
        else:
            # we know prob is in (0,1), so log(prob) is finite
            return count * np.log(prob)

    # log‐likelihood under independence (all transitions use pi)
    ll_ind = (
        safe_ll(n00, 1 - pi) +
        safe_ll(n01, pi) +
        safe_ll(n10, 1 - pi) +
        safe_ll(n11, pi)
    )

    # log‐likelihood under dependence (use pi0 and pi1)
    ll_dep = (
        safe_ll(n00, 1 - pi0) +
        safe_ll(n01, pi0) +
        safe_ll(n10, 1 - pi1) +
        safe_ll(n11, pi1)
    )

    LR_ind = -2.0 * (ll_ind - ll_dep)
    p_ind = 1.0 - chi2.cdf(LR_ind, df=1)

    return LR_ind, p_ind

alphas = [0.95, 0.975, 0.99]
horizons = df_wide['horizon_years'].unique()

rows = []
for h in sorted(horizons):
    dfh = df_wide[df_wide.horizon_years==h].dropna(subset=['observed_annual_return'])
    obs = dfh['observed_annual_return'].values

    for α in alphas:
        varcol = f'var_{int(α*100)}'
        viol = (obs < dfh[varcol]).astype(int)

        # Kupiec
        LR_uc, p_uc = kupiec_test(viol, 1-α)

        # Christoffersen (independence)
        LR_ind, p_ind = christoffersen_test(viol)

        # Combined conditional‐coverage
        LR_cc = LR_uc + LR_ind
        p_cc = 1 - chi2.cdf(LR_cc, df=2)

        # average interval‐width if you have both tails, else skip or approximate
        raw_widths = np.abs(dfh[f'var_{int(α*100)}'] - dfh[f'cvar_{int(α*100)}'])

        rows.append({
            'horizon': h,
            'alpha': α,
            'empirical_cov': 1 - viol.mean(),
            'Kupiec_p': p_uc,
            'Christoffersen_p': p_cc,
            'avg_width': raw_widths.mean()
        })

summary = pd.DataFrame(rows)

sig = 0.05
mask = (summary['Kupiec_p']        > sig) & \
       (summary['Christoffersen_p'] > sig)
passers = summary[mask].copy()
print(passers[['horizon','alpha','empirical_cov','Kupiec_p','Christoffersen_p','avg_width']])

passers = passers.sort_values('avg_width')

import matplotlib.pyplot as plt

labels = passers.apply(lambda r: f"{int(r.horizon)}y @ {int(r.alpha*100)}%",axis=1)
plt.barh(labels, passers['avg_width'], color='C2')
plt.xlabel("Average Width")
plt.title("Well‐calibrated combos (p>0.05) ranked by sharpness")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

for col, title in [('Kupiec_p','Kupiec p-values'),
                   ('Christoffersen_p','Christoffersen p-values')]:
    pt = summary.pivot_table('{}'.format(col),
                             index='horizon',
                             columns='alpha')
    plt.figure(figsize=(6,4))
    sns.heatmap(pt,
                annot=True,
                fmt=".2f",
                cmap="RdYlBu_r",
                vmin=0,
                vmax=1)
    plt.title(title)
    plt.ylabel("Horizon (years)")
    plt.xlabel("VaR level α")
    plt.show()
    
import matplotlib.pyplot as plt
W = 100  # roll‐window size in months
fig, ax = plt.subplots(1,1,figsize=(10,4))

for α in alphas:
    varcol = f'var_{int(α*100)}'
    dfw = df_wide.sort_values('execution_date')
    viol = (dfw['observed_annual_return'] < dfw[varcol]).astype(int)
    roll_cov = 1 - viol.rolling(W).mean()
    ax.plot(dfw['execution_date'], roll_cov, label=f'α={α:.3f}')

ax.axhline(1-alphas[0], color='black', linestyle='--', label='target coverage')
ax.legend()
ax.set_title(f'Rolling {W}-month empirical coverage (horizon={h}y)')
ax.set_ylabel('Coverage')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Option A: using pivot with keywords
pt = summary.pivot(
    index='horizon',
    columns='alpha',
    values='empirical_cov'
)

# Option B: or equivalently via pivot_table
# pt = summary.pivot_table(
#     index='horizon',
#     columns='alpha',
#     values='empirical_cov'
# )

plt.figure(figsize=(6,4))
sns.heatmap(
    pt,
    annot=True,
    fmt=".3f",
    cmap="RdYlBu_r",
    center=1-0.95,   # center on target coverage (e.g. 0.05 distance from 1)
    vmin=pt.min().min(),
    vmax=pt.max().max()
)
plt.title("Empirical Coverage Heatmap")
plt.ylabel("Horizon (years)")
plt.xlabel("VaR level α")
plt.tight_layout()
plt.show()
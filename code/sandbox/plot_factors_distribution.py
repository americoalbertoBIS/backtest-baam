# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:56:36 2025

@author: al005366
"""

import pandas as pd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data_sim_beta1 = pd.read_parquet(r'L:\RMAS\Users\Alberto\data\simulations_beta1.parquet')
data_sim_beta2 = pd.read_parquet(r'L:\RMAS\Users\Alberto\data\simulations_beta2.parquet')
data_sim_beta3 = pd.read_parquet(r'L:\RMAS\Users\Alberto\data\simulations_beta3.parquet')

data_sim_beta1_old = pd.read_parquet(r'L:\RMAS\Users\Alberto\backtest-baam\data\US\factors\AR_1\beta1\simulations.parquet')
data_sim_beta2_old = pd.read_parquet(r'L:\RMAS\Users\Alberto\backtest-baam\data\US\factors\AR_1\beta2\simulations.parquet')
data_sim_beta3_old = pd.read_parquet(r'L:\RMAS\Users\Alberto\backtest-baam\data\US\factors\AR_1\beta3\simulations.parquet')


# Assume your dataframes are already standardized and filtered as in previous steps

betas = [
    ('Beta1', data_sim_beta1, data_sim_beta1_old),
    ('Beta2', data_sim_beta2, data_sim_beta2_old),
    ('Beta3', data_sim_beta3, data_sim_beta3_old)
]

# Get 100 random common dates
dates1 = set(data_sim_beta1['ExecutionDate']).intersection(set(data_sim_beta1_old['ExecutionDate']))
dates2 = set(data_sim_beta2['ExecutionDate']).intersection(set(data_sim_beta2_old['ExecutionDate']))
dates3 = set(data_sim_beta3['ExecutionDate']).intersection(set(data_sim_beta3_old['ExecutionDate']))
common_dates = sorted(list(dates1 & dates2 & dates3))
np.random.seed(42)
sample_dates = np.random.choice(common_dates, size=100, replace=False)
sample_dates.sort()
# Color map
colors_new = ['#1f77b4', '#2ca02c', '#9467bd']
colors_old = ['#ff7f0e', '#d62728', '#8c564b']

# Create subplots: 3 rows, 1 column
fig = make_subplots(rows=3, cols=1, subplot_titles=[b[0] for b in betas], shared_xaxes=False, shared_yaxes=False)

# Initial traces for the first date
for i, (beta_name, beta_new, beta_old) in enumerate(betas):
    vals_new = beta_new[beta_new['ExecutionDate'] == sample_dates[0]]['SimulatedValue']
    vals_old = beta_old[beta_old['ExecutionDate'] == sample_dates[0]]['SimulatedValue']
    fig.add_trace(go.Histogram(x=vals_new, nbinsx=50, name=f"{beta_name}_new", marker_color=colors_new[i], opacity=0.6, histnorm='probability density', showlegend=True), row=i+1, col=1)
    fig.add_trace(go.Histogram(x=vals_old, nbinsx=50, name=f"{beta_name}_old", marker_color=colors_old[i], opacity=0.6, histnorm='probability density', showlegend=True), row=i+1, col=1)

# Create animation frames with dynamic axis ranges
frames = []
for date in sample_dates:
    frame_data = []
    x_ranges = []
    y_ranges = []
    for i, (beta_name, beta_new, beta_old) in enumerate(betas):
        vals_new = beta_new[beta_new['ExecutionDate'] == date]['SimulatedValue'].dropna()
        vals_old = beta_old[beta_old['ExecutionDate'] == date]['SimulatedValue'].dropna()
        frame_data.append(go.Histogram(x=vals_new, nbinsx=50, marker_color=colors_new[i], opacity=0.6, histnorm='probability density', name=f"{beta_name}_new", showlegend=False))
        frame_data.append(go.Histogram(x=vals_old, nbinsx=50, marker_color=colors_old[i], opacity=0.6, histnorm='probability density', name=f"{beta_name}_old", showlegend=False))
        
        # Handle empty arrays
        if len(vals_new) == 0 and len(vals_old) == 0:
            x_ranges.append((0, 1))
            y_ranges.append((0, 1))
        else:
            all_vals = np.concatenate([vals_new.values, vals_old.values]) if len(vals_new) > 0 and len(vals_old) > 0 else \
                       vals_new.values if len(vals_new) > 0 else vals_old.values
            x_min, x_max = np.nanmin(all_vals), np.nanmax(all_vals)
            # If still not finite, set a default range
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                x_min, x_max = 0, 1
            x_ranges.append((x_min, x_max))
            # Estimate y max for density
            counts_new = np.histogram(vals_new, bins=50, density=True)[0] if len(vals_new) > 0 else np.array([0])
            counts_old = np.histogram(vals_old, bins=50, density=True)[0] if len(vals_old) > 0 else np.array([0])
            y_max = max(np.nanmax(counts_new), np.nanmax(counts_old))
            if not np.isfinite(y_max):
                y_max = 1
            y_ranges.append((0, y_max * 1.05))
    
    # Layout updates for axes
    layout_updates = {}
    for i in range(3):
        layout_updates[f'xaxis{i+1}'] = dict(range=x_ranges[i], autorange=True)
        layout_updates[f'yaxis{i+1}'] = dict(range=y_ranges[i], autorange=True)
    frames.append(go.Frame(data=frame_data, name=str(date), layout=layout_updates))

# Update layout for animation
fig.update_layout(
    title="Simulated Value Distributions: Beta1, Beta2, Beta3 (Animated by Execution Date)",
    barmode='overlay',
    sliders=[dict(
        steps=[dict(method="animate",
                    args=[[str(date)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    label=str(date)) for date in sample_dates],
        active=0,
        transition={"duration": 0},
        x=0.1,
        y=0,
        len=0.9
    )]
)

fig.frames = frames

fig.update_xaxes(title_text="Simulated Value")
fig.update_yaxes(title_text="Density")

# Save as HTML
fig.write_html(r"C:\\animated_beta_distributions.html")
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:22:04 2025

@author: al005366
"""

import pandas as pd
base_path = r'L:\RMAS\Users\Alberto\backtest-baam\data\US\returns\estimated_returns'
data_1  = pd.read_parquet(fr'{base_path}\AR_1\annual\2.0_years\simulations_01122024.parquet')
data_2  = pd.read_parquet(fr'{base_path}\Mixed_Model\annual\2.0_years\simulations_01122024.parquet')

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure
plt.figure(figsize=(12, 8))

# Get unique horizons
horizons = data_1['Horizon (Years)'].unique()
horizons = [5]
# Loop through each horizon and plot KDE for both dataframes
for horizon in horizons:
    # Filter data for the current horizon
    data_1_filtered = data_1[data_1['Horizon (Years)'] == horizon]
    data_2_filtered = data_2[data_2['Horizon (Years)'] == horizon]
    
    # Plot KDE for data_1
    sns.kdeplot(data_1_filtered['AnnualReturn'], label=f'Data 1 - Horizon {horizon} Years', bw_adjust=1)
    
    # Plot KDE for data_2
    sns.kdeplot(data_2_filtered['AnnualReturn'], label=f'Data 2 - Horizon {horizon} Years', bw_adjust=1)

# Add title and labels
plt.title('KDE of Annual Returns by Horizon (Years)', fontsize=16)
plt.xlabel('Annual Return', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

plt.figure(figsize=(12, 8))

data_1_filtered = data_1#[data_1['Horizon (Years)'] == horizon]
data_2_filtered = data_2#[data_2['Horizon (Years)'] == horizon]

# Plot KDE for data_1
sns.kdeplot(data_1_filtered['AnnualReturn'], label=f'Data 1 - Horizon {horizon} Years', bw_adjust=1)

# Plot KDE for data_2
sns.kdeplot(data_2_filtered['AnnualReturn'], label=f'Data 2 - Horizon {horizon} Years', bw_adjust=1)

# Add title and labels
plt.title('KDE of Annual Returns by Horizon (Years)', fontsize=16)
plt.xlabel('Annual Return', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import math

# Get unique horizons
horizons = sorted(data_1['Horizon (Years)'].unique())

# Define the grid layout (2 rows x 3 columns)
n_rows = 2
n_cols = 3

# Create subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 10), sharey=True)

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Loop through each horizon and plot KDE for both dataframes
for i, horizon in enumerate(horizons):
    # Filter data for the current horizon
    data_1_filtered = data_1[data_1['Horizon (Years)'] == horizon]
    data_2_filtered = data_2[data_2['Horizon (Years)'] == horizon]
    
    # Plot KDE for data_1 on the current axis
    sns.kdeplot(data_1_filtered['AnnualReturn'], label=f'Data 1', bw_adjust=1, ax=axes[i], color='blue')
    
    # Plot KDE for data_2 on the current axis
    sns.kdeplot(data_2_filtered['AnnualReturn'], label=f'Data 2', bw_adjust=1, ax=axes[i], color='orange')
    
    # Add title and legend for the current subplot
    axes[i].set_title(f'Horizon {horizon} Years', fontsize=14)
    axes[i].legend()
    axes[i].grid(True)

# Hide any unused subplots if horizons < n_rows * n_cols
for j in range(len(horizons), len(axes)):
    axes[j].axis('off')

# Add global x and y labels
fig.text(0.5, 0.04, 'Annual Return', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)

# Adjust layout to fit titles and labels
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# Show the plot
plt.show()
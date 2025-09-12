import pandas as pd
import matplotlib.pyplot as plt
import glob

# Define base path
base_path = r'\\msfsshared\bnkg\RMAS\Users\Alberto\backtest-baam\data\US\factors'

# Get all subfolders
folders = glob.glob(base_path + "/*/")
models = [folder.rstrip("\\").split("\\")[-1] for folder in folders if 'archive' not in folder]

# Define betas
betas = ['beta1', 'beta2', 'beta3']

# Loop over each model to create a grid of plots
for model in models:
    # Initialize a list to store data for all betas
    all_beta_data = []

    # Load and merge data for all betas
    for beta in betas:
        try:
            # Load the data for the current beta
            data = pd.read_csv(rf'{base_path}\{model}\{beta}\insample_metrics.csv')
            data['execution_date'] = pd.to_datetime(data['execution_date'])
            data['beta'] = beta  # Add a column to distinguish betas
            all_beta_data.append(data)
        except FileNotFoundError:
            print(f"File not found for model: {model}, beta: {beta}")
            continue

    # If no data is available for any beta, skip this model
    if not all_beta_data:
        print(f"No data available for model: {model}")
        continue

    # Concatenate all beta data into a single DataFrame
    merged_data = pd.concat(all_beta_data)

    # Filter data to include only rows from 1965 onward
    merged_data = merged_data[merged_data['execution_date'] >= '1965-01-01']

    # Rename indicators `beta1`, `beta2`, `beta3` to `lagged_beta`
    merged_data['indicator'] = merged_data['indicator'].replace(['beta1', 'beta2', 'beta3'], 'lagged_beta')

    # Drop the "model" indicator if it exists
    merged_data = merged_data[merged_data['indicator'] != 'model']
    merged_data = merged_data[merged_data['indicator'] != 'const']

    # Dynamically determine which indicators are present in the merged data
    available_indicators = sorted(merged_data['indicator'].unique().tolist())

    # Dynamically calculate the grid size: rows = indicators, columns = betas
    num_rows = len(available_indicators)
    num_cols = len(betas)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)

    # Loop through each indicator (row)
    for row_idx, indicator in enumerate(available_indicators):
        # Filter data for the current indicator
        coefficient_data = merged_data[
            (merged_data['indicator'] == indicator) & 
            (merged_data['metric'] == 'coefficient')
        ]
        p_value_data = merged_data[
            (merged_data['indicator'] == indicator) & 
            (merged_data['metric'] == 'p_value') & 
            (merged_data['value'] < 0.1)  # Only p_values < 0.1
        ]

        # Loop through each beta (column)
        for col_idx, beta in enumerate(betas):
            ax1 = axes[row_idx, col_idx]

            # Filter data for the current beta
            beta_coefficient_data = coefficient_data[coefficient_data['beta'] == beta]
            beta_p_value_data = p_value_data[p_value_data['beta'] == beta]

            # Add a thin solid line at y=0 if coefficient values take negative values
            if not beta_coefficient_data.empty and beta_coefficient_data['value'].min() < 0:
                ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

            # Plot coefficient values on the primary y-axis
            if not beta_coefficient_data.empty:
                ax1.plot(
                    beta_coefficient_data['execution_date'],
                    beta_coefficient_data['value'],
                    label='Coefficient',
                    color='k',  # Red color for coefficients
                    linewidth=2
                )

            # Create a secondary y-axis for p-values
            ax2 = ax1.twinx()
            if not beta_p_value_data.empty:
                ax2.plot(
                    beta_p_value_data['execution_date'],
                    beta_p_value_data['value'],
                    label='P-Value (< 0.1)',
                    linestyle='-',
                    color='#aa322f',  # Red color for coefficients
                    alpha=0.7
                )

            # Customize the subplot
            ax1.set_title(f"{indicator.capitalize()} ({beta})", fontsize=10)
            ax1.set_xlabel("Execution Date")
            ax1.set_ylabel("Coefficient Value", color='k')
            ax2.set_ylabel("P-Value", color='#aa322f')
            #ax1.tick_params(axis='y', labelcolor='#aa322f')
            ax2.tick_params(axis='y', labelcolor='#aa322f')
            ax1.grid(True)

            # Add legends
            ax1.legend(loc='upper left', fontsize=8)
            ax2.legend(loc='upper right', fontsize=8)

    # Turn off any unused axes (e.g., if there are fewer indicators or betas than grid cells)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if row_idx >= len(available_indicators) or col_idx >= len(betas):
                fig.delaxes(axes[row_idx, col_idx])

    # Adjust layout and show the plot
    fig.suptitle(f"Model: {model} - Coefficients and P-Values", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
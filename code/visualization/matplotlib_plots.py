import matplotlib.pyplot as plt

def plot_forecasts_with_actuals(df_predictions, realized_beta, model='AR(1)', save_path=None):
    plt.figure(figsize=(10, 6))
    unique_execution_dates = df_predictions['ExecutionDate'].unique()

    for i, execution_date in enumerate(unique_execution_dates):
        subset = df_predictions[df_predictions['ExecutionDate'] == execution_date]
        if i % 60 == 0:
            plt.plot(subset['ForecastDate'], subset['Prediction'], linewidth=1.5, label=f"Forecast {execution_date.date()}")
        else:
            plt.plot(subset['ForecastDate'], subset['Prediction'], color='gray', alpha=0.2)

    plt.plot(realized_beta[(realized_beta.index > '1990') & (realized_beta.index < '2025')], color='black', linewidth=2, label="Actual Beta 1")
    plt.title(f"{model}: Forecasted vs Actual Beta 1")
    plt.xlabel("Date")
    plt.ylabel("Beta 1")
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
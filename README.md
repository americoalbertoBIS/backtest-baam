
# backtest-baam

## Overview

**backtest-baam** is a modular Python framework for backtesting, forecasting, and evaluating financial models, with a focus on yield curves, macroeconomic factors, and returns. It supports robust out-of-sample and in-sample analysis, risk metrics, and interactive visualization. The codebase is designed for research and production workflows in financial modeling, especially for government bond markets and macroeconomic forecasting.

---

## Folder Structure

```
code/
│
├── backtesting_observed_returns.py
├── backtesting_observed_yields.py
├── config_paths.py
├── main_factors_backtesting.py
├── main_factors_processing.py
├── main_returns_crps.py
├── progress.log
├── requirements.txt
│
├── backtesting/
├── data_preparation/
├── MathWorks/
├── mlruns/
├── modeling/
├── sandbox/
├── streamlit/
├── visualization/
└── __pycache__/
```

---

## Main Scripts

- **backtesting_observed_returns.py / backtesting_observed_yields.py**Run backtests on observed returns and yields.

  - Load data, fit models (e.g., AR(1)), compute metrics (RMSE, R²), and save results.
  - Organize outputs by country, frequency, and model.
- **main_factors_backtesting.py**Orchestrates backtesting for factor models, including logging and MLflow experiment tracking.
- **main_factors_processing.py**Processes and transforms factor data for modeling and analysis.
- **main_returns_crps.py**Computes CRPS (Continuous Ranked Probability Score) for probabilistic model evaluation.
- **config_paths.py**
  Centralizes configuration for data and output paths.

---

## Subfolders

- **backtesting/**Core logic for backtesting workflows:

  - `backtesting_logging.py`: Logging utilities.
  - `backtesting_pipeline.py`: Orchestrates backtesting steps.
  - `config_models.py`: Model configurations (e.g., AR(1), ARX).
  - `factors_processing.py`: Factor transformation and yield prediction.
- **data_preparation/**Data loading, cleaning, and consensus forecast generation:

  - `conensus_forecast.py`: Consensus forecast class.
  - `data_loader.py`: Load macro and yield curve data.
- **modeling/**Model definitions and routines:

  - `yield_curve_modeling.py`: Nelson-Siegel yield curve models.
  - `macro_modeling.py`: Macroeconomic variable modeling.
  - `time_series_modeling.py`: AR(1) and ARX models.
- **mlruns/**Stores experiment tracking data (e.g., MLflow artifacts, metrics, forecasts).
- **sandbox/**Prototyping and exploratory scripts for metrics, risk analysis, bootstrapping, and statistical tests:

  - `test_var_cvar.py`, `test_output_gap.py`, `generate_factors_metrics.py`, etc.
- **streamlit/**Interactive dashboards for visualizing model results, metrics, and simulations:

  - `main_new.py`: Main Streamlit app for exploring factors, yields, returns, and consensus forecasts.
- **visualization/**Scripts for generating publication-ready plots and charts:

  - `plots_presentation.py`, `plots_davos_2025.py`: Plot RMSE, CRPS, and other metrics.
- **MathWorks/**MATLAB-related scripts or interoperability files.
- **__pycache__/**
  Python bytecode cache for faster execution.

---

## Requirements

All dependencies are listed in [`requirements.txt`](requirements.txt).
Install them with:

```sh
pip install -r requirements.txt
```

---

## Usage

1. **Data Preparation**

   - Place macroeconomic and yield curve data in the expected directories.
   - Use scripts in `data_preparation/` to load and preprocess data.
2. **Modeling & Backtesting**

   - Run `backtesting_observed_returns.py` or `backtesting_observed_yields.py` to perform backtests.
   - Use `main_factors_backtesting.py` for factor model backtesting.
3. **Metrics & Analysis**

   - Use `main_returns_crps.py` for CRPS analysis.
   - Use scripts in `sandbox/` and `visualization/` for RMSE, VaR, CVaR, and other metrics.
4. **Interactive Dashboards**

   - Launch dashboards from the `streamlit/` folder:
     ```sh
     streamlit run streamlit/main_new.py
     ```
   - Explore tabs for factors, yields, returns, simulations, and model comparisons.

---

## Example Workflow

```sh
# Install dependencies
pip install -r requirements.txt

# Run a backtest on observed returns
python backtesting_observed_returns.py

# Launch the Streamlit dashboard
streamlit run streamlit/main_new.py
```

---

## Contributing

- Fork the repo and submit pull requests for improvements.
- Use the `sandbox/` folder for prototyping new metrics or analysis methods.

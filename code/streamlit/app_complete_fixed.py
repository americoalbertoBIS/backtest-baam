# app_complete_fixed.py
# Complete Streamlit app with original tabs restored + fixes:
# - Robust IO and column normalization
# - Cached folder reads
# - Portable base path & country selection
# - Factors overview (beta1, beta2, beta3)
# - Out-of-sample & in-sample metrics
# - Simulations (fan chart + PIT heatmap)
# - Yields overview & backtesting
# - Yields simulations
# - Returns & forward-looking distributions (with per-model KDE)
#
# Run: streamlit run app_complete_fixed.py

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional: multi-group KDE
try:
    import plotly.figure_factory as ff
    _HAS_FF = True
except Exception:
    _HAS_FF = False

import streamlit as st


# --------------------------- Page config ---------------------------
st.set_page_config(page_title="Backtesting & Simulations — Complete", layout="wide")


# --------------------------- Helpers: standardization & IO ---------------------------
def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and types so downstream code is consistent."""
    rename_mapping = {
        # Dates
        "execution_date": "ExecutionDate",
        "forecast_date": "ForecastDate",
        "forecasted_date": "ForecastDate",
        # Values
        "prediction": "Prediction",
        "pred": "Prediction",
        "actual": "Actual",
        # Metadata
        "horizon": "Horizon",
        "maturity": "maturity",
        # Metrics
        "rmse": "rmse",
        "RMSE": "rmse",
        # Factors (if present, keep names as-is but ensure consistent case)
        "Beta1": "beta1", "Beta2": "beta2", "Beta3": "beta3",
    }
    df = df.rename(columns=rename_mapping)

    # Dates
    for c in ("ExecutionDate", "ForecastDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Ints
    if "Horizon" in df.columns:
        df["Horizon"] = pd.to_numeric(df["Horizon"], errors="coerce").astype("Int64")

    # Numerics
    for c in ("Prediction", "Actual", "rmse", "beta1", "beta2", "beta3", "SimulatedValue"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "maturity" in df.columns and df["maturity"].dtype == object:
        df["maturity"] = df["maturity"].astype(str).strip()

    return df


def _resolve_file(base_path: str, subfolder: str, model: str, file_name: str) -> Path:
    """Support both absolute and relative subfolders."""
    base = Path(base_path)
    sub = Path(subfolder)
    if sub.is_absolute():
        return sub / model / file_name
    return base / subfolder / model / file_name


def load_data(base_path: str, subfolder: str, model: str, file_name: str) -> Optional[pd.DataFrame]:
    """Robust single-file load with normalization."""
    file_path = _resolve_file(base_path, subfolder, model, file_name)
    if not file_path.exists():
        st.warning(f"File not found: {file_path}")
        return None
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        st.warning(f"Unsupported file format: {file_path.name}")
        return None
    return _standardize_df(df)


@st.cache_data(show_spinner=False)
def load_and_cache_models(
    models: List[str],
    base_path: str,
    subfolder: str,
    file_name: str,
    tag: str,
) -> Dict[str, pd.DataFrame]:
    """Load per-model data from a given subfolder, tagging each with Model + tag."""
    out: Dict[str, pd.DataFrame] = {}
    for m in models:
        df = load_data(base_path, subfolder, m, file_name)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["Model"] = m if not tag else f"{m}{tag}"
        out[df["Model"].iloc[0]] = df
    return out


@st.cache_data(show_spinner=False)
def read_simulation_folder(folder: str) -> Optional[pd.DataFrame]:
    """Concatenate all parquet/csv files from a simulations folder."""
    p = Path(folder)
    files = list(p.glob("*.parquet")) + list(p.glob("*.csv"))
    if not files:
        return None
    frames = []
    for f in files:
        if f.suffix.lower() == ".parquet":
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        frames.append(_standardize_df(df))
    return pd.concat(frames, ignore_index=True)


# --------------------------- Plot helpers ---------------------------
def apply_layout(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        title=title or None,
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=60, r=20, b=40, l=20),
        hovermode="x unified",
    )
    return fig


def kde_by_model(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """One KDE per model using figure_factory if available; fallback to hist-lines."""
    series_list: List[np.ndarray] = []
    labels: List[str] = []
    for model, g in df.groupby("Model"):
        arr = pd.to_numeric(g[value_col], errors="coerce").dropna().values
        if arr.size >= 10:
            series_list.append(arr)
            labels.append(model)

    if not series_list:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for KDE", showarrow=False, y=0.5, x=0.5)
        return apply_layout(fig, title)

    if _HAS_FF:
        fig = ff.create_distplot(series_list, group_labels=labels, show_hist=False, show_rug=False)
        fig.update_traces(mode="lines")
        return apply_layout(fig, title)

    # Fallback
    fig = go.Figure()
    for arr, lab in zip(series_list, labels):
        hist, bins = np.histogram(arr, bins="fd", density=True)
        centers = 0.5 * (bins[1:] + bins[:-1])
        fig.add_trace(go.Scatter(x=centers, y=hist, mode="lines", name=lab))
    return apply_layout(fig, title)


# --------------------------- Metrics ---------------------------
def compute_rmse_by(
    df: pd.DataFrame,
    value_col_pred: str = "Prediction",
    value_col_act: str = "Actual",
    group_cols: Tuple[str, ...] = ("Model", "Horizon"),
) -> pd.DataFrame:
    x = df.dropna(subset=[value_col_pred, value_col_act]).copy()
    if x.empty:
        return pd.DataFrame(columns=list(group_cols) + ["RMSE"])
    x["sqerr"] = (x[value_col_pred] - x[value_col_act]) ** 2
    out = (
        x.groupby(list(group_cols))["sqerr"]
        .mean()
        .reset_index()
        .rename(columns={"sqerr": "RMSE"})
    )
    out["RMSE"] = np.sqrt(out["RMSE"])
    return out


def compute_dynamic_rmse(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Dynamic RMSE within a date window (ExecutionDate)."""
    x = df.copy()
    if "ExecutionDate" not in x.columns:
        return pd.DataFrame(columns=["Model", "Horizon", "RMSE"])
    if start_date is not None:
        x = x[x["ExecutionDate"] >= start_date]
    if end_date is not None:
        x = x[x["ExecutionDate"] <= end_date]
    return compute_rmse_by(x, group_cols=("Model", "Horizon"))


# --------------------------- Sim plots ---------------------------
def build_percentiles(df: pd.DataFrame, value_col: str = "SimulatedValue") -> pd.DataFrame:
    q = (
        df.groupby("ExecutionDate")[value_col]
        .quantile([0.05, 0.5, 0.95])
        .unstack()
        .rename(columns={0.05: "p5", 0.5: "p50", 0.95: "p95"})
    )
    return q


def plot_fan_chart(percentiles: pd.DataFrame, actuals_ts: pd.Series, title: str) -> go.Figure:
    combo = percentiles.join(actuals_ts.rename("Actual"), how="inner")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combo.index, y=combo["p95"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip", name="p95"))
    fig.add_trace(go.Scatter(x=combo.index, y=combo["p5"], mode="lines", fill="tonexty",
                             fillcolor="rgba(0, 100, 200, 0.20)", line=dict(width=0),
                             name="5–95% band"))
    fig.add_trace(go.Scatter(x=combo.index, y=combo["p50"], mode="lines", name="Median Simulation"))
    fig.add_trace(go.Scatter(x=combo.index, y=combo["Actual"], mode="lines", name="Actual"))
    return apply_layout(fig, title)


def plot_pit_heatmap(sim_df: pd.DataFrame, forecasts_df: pd.DataFrame, title: str) -> go.Figure:
    actuals = (
        forecasts_df[["ForecastDate", "Actual"]]
        .dropna()
        .drop_duplicates(subset=["ForecastDate"])
        .set_index("ForecastDate")["Actual"]
    )
    tmp = sim_df.merge(actuals.rename("Realized"), left_on="ForecastDate", right_index=True, how="left")
    pit = (
        tmp.groupby(["ExecutionDate", "Horizon"])
           .apply(lambda g: (g["SimulatedValue"] <= g["Realized"]).mean())
           .unstack()
    )
    fig = px.imshow(
        pit.T, labels={"x": "Execution Date", "y": "Horizon", "color": "PIT"},
        title=title, origin="lower", color_continuous_scale="RdBu", aspect="auto"
    )
    return apply_layout(fig)


# --------------------------- Sidebar ---------------------------
st.sidebar.header("Configuration")

default_base = r"\\msfsshared\BNKG\\RMAS\Users\Alberto\backtest-baam\data"
base_root = st.sidebar.text_input("Base data folder", value=default_base)

# Try to list countries (subfolders) if base exists
countries = []
try:
    countries = [p.name for p in Path(base_root).iterdir() if p.is_dir()]
except Exception:
    pass
selected_country = st.sidebar.selectbox("Country folder", countries) if countries else st.sidebar.text_input("Country folder", value="")

# File names (override if needed)
factors_file = st.sidebar.text_input("Factors file name (per model)", value="factors.parquet")
yields_file = st.sidebar.text_input("Yields file name (per model)", value="yields.parquet")

# Models: try to infer from factors folder; allow manual override
factors_folder = str(Path(base_root) / selected_country / "factors") if selected_country else str(Path(base_root) / "factors")
try:
    inferred_models = [p.name for p in Path(factors_folder).iterdir() if p.is_dir() and p.name != "archive"]
except Exception:
    inferred_models = []
models_str = st.sidebar.text_input("Models (comma-separated)", value=", ".join(inferred_models[:5]) if inferred_models else "AR1, RW, SimpleMean")
models = [m.strip() for m in models_str.split(",") if m.strip()]

benchmark_model = st.sidebar.text_input("Benchmark model name", value="Observed")
include_benchmark = st.sidebar.checkbox("Include benchmark (observed_yields)", value=True)

# Simulations folder
sim_folder_default = str(Path(base_root) / selected_country / "simulations") if selected_country else ""
sim_folder = st.sidebar.text_input("Simulations folder (parquet/csv files)", value=sim_folder_default)

# Filters
maturity_filter = st.sidebar.text_input("Filter: maturity (exact label, optional)", value="")
horizon_filter = st.sidebar.text_input("Filter: horizon (integer, optional)", value="")
date_min = st.sidebar.date_input("Start date (ExecutionDate)", value=None)
date_max = st.sidebar.date_input("End date (ExecutionDate)", value=None)


# --------------------------- Load: factors & yields ---------------------------
# Factors per model (no benchmark)
factors_map = load_and_cache_models(
    models=models,
    base_path=str(Path(base_root) / selected_country),
    subfolder="factors",
    file_name=factors_file,
    tag="",
)

# Yields per model + optional benchmark
yields_map_est = load_and_cache_models(
    models=models,
    base_path=str(Path(base_root) / selected_country),
    subfolder="estimated_yields",
    file_name=yields_file,
    tag="",
)
yields_map_obs = load_and_cache_models(
    models=[benchmark_model] if include_benchmark else [],
    base_path=str(Path(base_root) / selected_country),
    subfolder="observed_yields",
    file_name=yields_file,
    tag=" (Benchmark)",
)
yields_map = {**yields_map_est, **yields_map_obs}

# Combined frames
factors_df = pd.concat(factors_map.values(), ignore_index=True) if factors_map else pd.DataFrame()
yields_df = pd.concat(yields_map.values(), ignore_index=True) if yields_map else pd.DataFrame()

# Apply optional filters
def _apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if not x.empty and maturity_filter and "maturity" in x.columns:
        x = x[x["maturity"] == maturity_filter]
    if not x.empty and horizon_filter and "Horizon" in x.columns:
        try:
            h_val = int(horizon_filter)
            x = x[x["Horizon"] == h_val]
        except Exception:
            st.warning("Horizon filter must be an integer.")
    if not x.empty and "ExecutionDate" in x.columns:
        if date_min:
            x = x[x["ExecutionDate"] >= pd.to_datetime(date_min)]
        if date_max:
            x = x[x["ExecutionDate"] <= pd.to_datetime(date_max)]
    return x

factors_df_f = _apply_common_filters(factors_df)
yields_df_f = _apply_common_filters(yields_df)


# --------------------------- Tabs (full set restored) ---------------------------
(tab_factors,
 tab_out_of_sample,
 tab_in_sample,
 tab_simulations,
 tab_yields_overview,
 tab_yields_backtesting,
 tab_yields_simulations,
 tab_returns,
 tab_returns_fwd) = st.tabs([
    "Factors overview",
    "Out-of-Sample Model Comparison",
    "In-Sample Model Specifics",
    "Simulations",
    "Yields overview",
    "Yields Backtesting",
    "Yields Simulations",
    "Returns",
    "Returns forward looking distributions",
])


# ========================= Tab: Factors overview =========================
with tab_factors:
    st.subheader("Actuals vs Predictions for Factors (beta1, beta2, beta3)")

    if factors_df_f.empty:
        st.info("No factors loaded. Check base path, country, models, and file names.")
    else:
        # Layout: three columns for the three factors
        cols = st.columns(3)
        target_variables = ["beta1", "beta2", "beta3"]

        for col, tv in zip(cols, target_variables):
            with col:
                st.markdown(f"**{tv}: Actuals vs Predictions**")
                # Select models for this factor (limit to 3 to keep plot readable)
                available_models = sorted(factors_df_f["Model"].dropna().unique().tolist())
                sel = st.multiselect(f"Models for {tv} (max 3)", available_models, default=available_models[:1], key=f"sel_{tv}")
                if len(sel) > 3:
                    st.warning("Please select up to 3 models. Using the first 3.")
                    sel = sel[:3]

                fig = go.Figure()
                for m in sel:
                    d = factors_df_f[(factors_df_f["Model"] == m) & (factors_df_f[tv].notna())]
                    # Plot Prediction if available
                    if "Prediction" in d.columns and d["Prediction"].notna().any():
                        fig.add_trace(go.Scatter(x=d["ForecastDate"], y=d["Prediction"], mode="lines", name=f"{m} — Pred"))
                    # Plot Actual if available (once)
                # Global actuals by ForecastDate (dedup)
                if tv in factors_df_f.columns:
                    act = (factors_df_f.dropna(subset=["ForecastDate", tv])
                                      .sort_values("ForecastDate")
                                      .drop_duplicates(subset=["ForecastDate"]))
                    fig.add_trace(go.Scatter(x=act["ForecastDate"], y=act[tv], mode="lines", name=f"{tv} — Actual", line=dict(color="black", dash="dot")))

                apply_layout(fig, f"{tv}: Actual vs Pred")
                st.plotly_chart(fig, use_container_width=True)


# ========================= Tab: Out-of-Sample Model Comparison =========================
with tab_out_of_sample:
    st.subheader("Out-of-Sample Metrics (from Yields)")

    if yields_df_f.empty:
        st.info("No yields loaded. Check 'estimated_yields' and 'observed_yields' paths.")
    else:
        rmse_df = compute_rmse_by(yields_df_f, group_cols=("Model", "Horizon"))
        if not rmse_df.empty:
            fig1 = px.line(rmse_df.sort_values(["Model", "Horizon"]), x="Horizon", y="RMSE", color="Model", markers=True,
                           title="RMSE by Horizon")
            apply_layout(fig1)
            st.plotly_chart(fig1, use_container_width=True)
            st.download_button("Download RMSE by Horizon (CSV)", rmse_df.to_csv(index=False).encode(), "rmse_by_horizon.csv", "text/csv")
        else:
            st.info("Need Prediction & Actual to compute RMSE.")

        if "ExecutionDate" in yields_df_f.columns:
            c1, c2 = st.columns(2)
            mind = pd.to_datetime(yields_df_f["ExecutionDate"].min())
            maxd = pd.to_datetime(yields_df_f["ExecutionDate"].max())
            with c1:
                dstart = st.date_input("Dynamic window start", value=mind.date() if pd.notna(mind) else None, key="dyn_start_oos")
            with c2:
                dend = st.date_input("Dynamic window end", value=maxd.date() if pd.notna(maxd) else None, key="dyn_end_oos")
            dyn = compute_dynamic_rmse(yields_df_f, pd.to_datetime(dstart) if dstart else None, pd.to_datetime(dend) if dend else None)
            if not dyn.empty:
                fig2 = px.line(dyn.sort_values(["Model", "Horizon"]), x="Horizon", y="RMSE", color="Model", markers=True,
                               title="Dynamic RMSE by Horizon (ExecutionDate window)")
                apply_layout(fig2)
                st.plotly_chart(fig2, use_container_width=True)
                st.download_button("Download Dynamic RMSE (CSV)", dyn.to_csv(index=False).encode(), "dynamic_rmse.csv", "text/csv")
            else:
                st.info("Dynamic RMSE unavailable for the selected window.")


# ========================= Tab: In-Sample Model Specifics =========================
with tab_in_sample:
    st.subheader("In-Sample Model Specifics (Yields)")

    if yields_df_f.empty:
        st.info("No yields loaded.")
    else:
        models_av = sorted(yields_df_f["Model"].dropna().unique().tolist())
        msel = st.selectbox("Model", models_av, key="ins_model")
        maturities = sorted(yields_df_f["maturity"].dropna().astype(str).unique().tolist()) if "maturity" in yields_df_f.columns else []
        mty = st.selectbox("Maturity", maturities, key="ins_mat") if maturities else None

        sub = yields_df_f[yields_df_f["Model"] == msel].copy()
        if mty:
            sub = sub[sub["maturity"] == mty]

        if sub.empty or "ForecastDate" not in sub.columns:
            st.info("Missing ForecastDate or no data after filters.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig_ts = go.Figure()
                if "Prediction" in sub.columns and sub["Prediction"].notna().any():
                    fig_ts.add_trace(go.Scatter(x=sub["ForecastDate"], y=sub["Prediction"], mode="lines", name="Prediction"))
                if "Actual" in sub.columns and sub["Actual"].notna().any():
                    act = sub.dropna(subset=["ForecastDate", "Actual"]).drop_duplicates("ForecastDate")
                    fig_ts.add_trace(go.Scatter(x=act["ForecastDate"], y=act["Actual"], mode="lines", name="Actual", line=dict(color="black", dash="dot")))
                apply_layout(fig_ts, "Time Series")
                st.plotly_chart(fig_ts, use_container_width=True)

            with c2:
                if {"Prediction", "Actual"}.issubset(sub.columns):
                    xy = sub.dropna(subset=["Prediction", "Actual"])
                    fig_sc = px.scatter(xy, x="Prediction", y="Actual", trendline="ols", title="Prediction vs Actual")
                    apply_layout(fig_sc)
                    st.plotly_chart(fig_sc, use_container_width=True)
                else:
                    st.info("Need Prediction & Actual for scatter.")


# ========================= Tab: Simulations =========================
with tab_simulations:
    st.subheader("Simulations: Fan Chart & PIT Heatmap")

    if not sim_folder:
        st.info("Set a simulations folder in the sidebar (parquet/csv files).")
    else:
        sims = read_simulation_folder(sim_folder)
        if sims is None or sims.empty:
            st.info("No simulations found in the specified folder.")
        else:
            sims_f = _apply_common_filters(sims)

            # Execution month select
            if "ExecutionDate" in sims_f.columns:
                months = sorted(sims_f["ExecutionDate"].dropna().dt.to_period("M").unique())
                if months:
                    sel_month = st.selectbox("Execution month", options=months, format_func=lambda p: p.strftime("%Y-%m"), key="sim_month")
                    sims_month = sims_f[sims_f["ExecutionDate"].dt.to_period("M") == sel_month]
                else:
                    sims_month = sims_f
            else:
                sims_month = sims_f

            # Fan chart needs percentiles + actuals from yields
            if "SimulatedValue" in sims_f.columns and not yields_df.empty and "ForecastDate" in yields_df.columns:
                perc = build_percentiles(sims_f)
                act_ts = (yields_df.dropna(subset=["ForecastDate", "Actual"])
                                    .sort_values("ForecastDate")
                                    .drop_duplicates(subset=["ForecastDate"])
                                    .set_index("ForecastDate")["Actual"])
                if not perc.empty and not act_ts.empty:
                    fig_fan = plot_fan_chart(perc, act_ts, "Simulation Fan Chart vs Actual")
                    st.plotly_chart(fig_fan, use_container_width=True)
                else:
                    st.info("Insufficient data for fan chart.")
            else:
                st.info("Missing SimulatedValue or actuals to overlay.")

            # PIT heatmap
            if "SimulatedValue" in sims_f.columns and not yields_df.empty:
                try:
                    fig_hm = plot_pit_heatmap(sims_f, yields_df, "PIT Heatmap (Horizon × ExecutionDate)")
                    st.plotly_chart(fig_hm, use_container_width=True)
                except Exception as e:
                    st.warning(f"PIT heatmap skipped: {e}")

            st.download_button("Download simulations (filtered) CSV",
                               sims_month.to_csv(index=False).encode(),
                               "simulations_filtered.csv", "text/csv")


# ========================= Tab: Yields overview =========================
with tab_yields_overview:
    st.subheader("Yields overview")

    if yields_df_f.empty:
        st.info("No yields loaded.")
    else:
        maturities = sorted(yields_df_f["maturity"].dropna().astype(str).unique().tolist()) if "maturity" in yields_df_f.columns else []
        sel_mat = st.selectbox("Select maturity", maturities, key="yo_mat") if maturities else None

        plot_df = yields_df_f.copy()
        if sel_mat:
            plot_df = plot_df[plot_df["maturity"] == sel_mat]

        if plot_df.empty or "ForecastDate" not in plot_df.columns:
            st.info("Nothing to plot (check ForecastDate/maturity).")
        else:
            fig = go.Figure()
            for m in sorted(plot_df["Model"].dropna().unique()):
                d = plot_df[plot_df["Model"] == m]
                if "Prediction" in d.columns and d["Prediction"].notna().any():
                    fig.add_trace(go.Scatter(x=d["ForecastDate"], y=d["Prediction"], mode="lines", name=f"{m} — Prediction"))
            if "Actual" in plot_df.columns and plot_df["Actual"].notna().any():
                act = plot_df.dropna(subset=["ForecastDate", "Actual"]).drop_duplicates("ForecastDate").sort_values("ForecastDate")
                fig.add_trace(go.Scatter(x=act["ForecastDate"], y=act["Actual"], mode="lines", name="Actual", line=dict(color="black", dash="dot")))
            apply_layout(fig, f"Yields at {sel_mat}" if sel_mat else "Yields (All)")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button("Download plotted data (CSV)", plot_df.to_csv(index=False).encode(), "yields_overview.csv", "text/csv")


# ========================= Tab: Yields Backtesting =========================
with tab_yields_backtesting:
    st.subheader("Yields Backtesting — Out-of-Sample")

    if yields_df_f.empty:
        st.info("No yields loaded.")
    else:
        rmse_df = compute_rmse_by(yields_df_f, group_cols=("Model", "Horizon"))
        if not rmse_df.empty:
            fig1 = px.line(rmse_df.sort_values(["Model", "Horizon"]), x="Horizon", y="RMSE", color="Model", markers=True,
                           title="RMSE by Horizon")
            apply_layout(fig1)
            st.plotly_chart(fig1, use_container_width=True)

        if "ExecutionDate" in yields_df_f.columns:
            c1, c2 = st.columns(2)
            mind = pd.to_datetime(yields_df_f["ExecutionDate"].min())
            maxd = pd.to_datetime(yields_df_f["ExecutionDate"].max())
            with c1:
                dstart = st.date_input("Dynamic RMSE start", value=mind.date() if pd.notna(mind) else None, key="dyn_start_yb")
            with c2:
                dend = st.date_input("Dynamic RMSE end", value=maxd.date() if pd.notna(maxd) else None, key="dyn_end_yb")
            dyn = compute_dynamic_rmse(yields_df_f, pd.to_datetime(dstart) if dstart else None, pd.to_datetime(dend) if dend else None)
            if not dyn.empty:
                fig2 = px.line(dyn.sort_values(["Model", "Horizon"]), x="Horizon", y="RMSE", color="Model", markers=True,
                               title="Dynamic RMSE by Horizon")
                apply_layout(fig2)
                st.plotly_chart(fig2, use_container_width=True)


# ========================= Tab: Yields Simulations =========================
with tab_yields_simulations:
    st.subheader("Yields Simulations — Fan Charts by Maturity")

    if not sim_folder:
        st.info("Set a simulations folder in the sidebar.")
    else:
        sims = read_simulation_folder(sim_folder)
        if sims is None or sims.empty:
            st.info("No simulations found.")
        else:
            if "maturity" in sims.columns:
                maturities = sorted(sims["maturity"].dropna().astype(str).unique().tolist())
            else:
                maturities = []
            sel_mats = st.multiselect("Select maturities (max 3)", maturities, default=maturities[:1])
            if len(sel_mats) > 3:
                st.warning("Please select up to 3 maturities; showing first 3.")
                sel_mats = sel_mats[:3]

            # Actuals from yields
            act_ts = (yields_df.dropna(subset=["ForecastDate", "Actual"])
                                .sort_values("ForecastDate")
                                .drop_duplicates(subset=["ForecastDate"])
                                .set_index("ForecastDate")["Actual"]) if not yields_df.empty else pd.Series(dtype=float)

            cols = st.columns(max(1, len(sel_mats)))
            for col, mty in zip(cols, sel_mats):
                with col:
                    sub = sims[sims.get("maturity", "").astype(str) == str(mty)].copy()
                    if sub.empty:
                        st.info(f"No simulations for maturity {mty}.")
                        continue
                    perc = build_percentiles(sub)
                    if not perc.empty and not act_ts.empty:
                        fig = plot_fan_chart(perc, act_ts, f"Maturity {mty}: 5–95% band vs Actual")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Percentiles or actuals missing for maturity {mty}.")


# ========================= Tab: Returns =========================
with tab_returns:
    st.subheader("Returns (derived from yields)")

    if yields_df_f.empty or "ForecastDate" not in yields_df_f.columns:
        st.info("Need yields with ForecastDate to compute returns.")
    else:
        # Choose model & maturity
        models_av = sorted(yields_df_f["Model"].dropna().unique().tolist())
        msel = st.selectbox("Model", models_av, key="ret_model")
        maturities = sorted(yields_df_f["maturity"].dropna().astype(str).unique().tolist()) if "maturity" in yields_df_f.columns else []
        mty = st.selectbox("Maturity", maturities, key="ret_mat") if maturities else None

        sub = yields_df_f[yields_df_f["Model"] == msel].copy()
        if mty:
            sub = sub[sub["maturity"] == mty]

        if sub.empty:
            st.info("No data after filters.")
        else:
            # Compute simple yield changes as proxy 'returns' (Δy)
            sub = sub.sort_values("ForecastDate")
            sub["Return"] = sub["Prediction"].diff() * -1  # negative change in yields ~ price return proxy
            # Show TS
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=sub["ForecastDate"], y=sub["Return"], mode="lines", name="Return (proxy)"))
            apply_layout(fig_ts, f"Return series — {msel} @ {mty}")
            st.plotly_chart(fig_ts, use_container_width=True)

            # KDE per model (single model here, but allow compare by letting user pick multiple)
            compare = st.multiselect("Compare models (optional)", [m for m in models_av if m != msel], default=[])
            ret_df = sub[["ForecastDate", "Model", "Return"]].dropna().copy()
            for m in compare:
                sub_m = yields_df_f[yields_df_f["Model"] == m].copy()
                if mty:
                    sub_m = sub_m[sub_m["maturity"] == mty]
                sub_m = sub_m.sort_values("ForecastDate")
                sub_m["Return"] = sub_m["Prediction"].diff() * -1
                ret_df = pd.concat([ret_df, sub_m[["ForecastDate", "Model", "Return"]]], ignore_index=True)

            kde_fig = kde_by_model(ret_df.dropna(subset=["Return"]), "Return", f"KDE of returns @ {mty}")
            st.plotly_chart(kde_fig, use_container_width=True)

            st.download_button("Download returns (CSV)", ret_df.to_csv(index=False).encode(), "returns.csv", "text/csv")


# ========================= Tab: Returns forward looking distributions =========================
with tab_returns_fwd:
    st.subheader("Forward-looking distributions from simulations")

    if not sim_folder:
        st.info("Set a simulations folder in the sidebar.")
    else:
        sims = read_simulation_folder(sim_folder)
        if sims is None or sims.empty:
            st.info("No simulations found.")
        else:
            # If simulations include 'Model' we can group; otherwise treat as one group
            if "Model" not in sims.columns and not yields_df.empty and "Model" in yields_df.columns:
                # Try to merge model via nearest ExecutionDate/ForecastDate if present (best-effort)
                # Otherwise, leave as single group.
                pass

            # Build "forward changes" per model: Δ SimulatedValue across ExecutionDate
            s = sims.copy()
            if "SimulatedValue" in s.columns and "ExecutionDate" in s.columns:
                s = s.sort_values(["Model", "ExecutionDate"]) if "Model" in s.columns else s.sort_values("ExecutionDate")
                s["ForwardChange"] = s.groupby(s["Model"] if "Model" in s.columns else 0)["SimulatedValue"].diff()
                # KDE per model
                df_kde = s.dropna(subset=["ForwardChange"]).copy()
                title = "KDE of forward changes (simulated) — one curve per model" if "Model" in df_kde.columns else "KDE of forward changes (simulated)"
                fig = kde_by_model(df_kde.rename(columns={"ForwardChange": "VAL"}), "VAL", title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Simulations need SimulatedValue and ExecutionDate to compute forward changes.")

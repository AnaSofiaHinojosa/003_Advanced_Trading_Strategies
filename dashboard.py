# dashboard.py
# Streamlit Data Drift Dashboard
# Runs backtests on TEST and VAL, then computes all drift analytics in the dashboard

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import ks_2samp

# --- Imports from your repo ---
from utils import get_data, split_data, get_target, load_model
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data
from backtest import backtest


# =============================================================================
# Helper functions for dashboard plots
# =============================================================================

def ks_pairwise(reference_df, other_df, alpha=0.05):
    """KS test for each feature."""
    common_cols = [c for c in reference_df.columns if c in other_df.columns]
    results = []
    for col in common_cols:
        ref, oth = reference_df[col].dropna(), other_df[col].dropna()
        if len(ref) < 2 or len(oth) < 2 or ref.nunique() < 2 or oth.nunique() < 2:
            p_val, drift = np.nan, False
        else:
            _, p_val = ks_2samp(ref, oth)
            drift = p_val < alpha
        results.append({"feature": col, "p_value": p_val, "drift_detected": drift})
    return pd.DataFrame(results).sort_values("p_value", na_position="last")


def make_overlay_histograms(train, test, val, feature_name):
    """Overlay histograms (Plotly) for a single feature."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train, name="Train", histnorm="probability density", opacity=0.6))
    fig.add_trace(go.Histogram(x=test,  name="Test",  histnorm="probability density", opacity=0.6))
    fig.add_trace(go.Histogram(x=val,   name="Val",   histnorm="probability density", opacity=0.6))
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution by period • {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def melt_drift_windows(drift_df, split_name):
    """Convert drift_df from backtest into long form for heatmap."""
    if drift_df is None or drift_df.empty:
        return pd.DataFrame(columns=["feature", "window_mid", "drift", "split"])
    value_cols = [c for c in drift_df.columns if c not in ("start_idx", "end_idx", "Close")]
    df_long = drift_df.melt(
        id_vars=["start_idx", "end_idx"],
        value_vars=value_cols,
        var_name="feature",
        value_name="drift"
    )
    df_long["window_mid"] = (df_long["start_idx"] + df_long["end_idx"]) // 2
    df_long["split"] = split_name
    return df_long


def explain_top_features(long_df, top_n=5):
    """Summarize top drifted features."""
    if long_df.empty:
        return pd.DataFrame(columns=["feature", "drift_windows", "total_windows", "pct_windows_drift"])
    grp = long_df.groupby("feature")["drift"].agg(["sum", "count"]).reset_index()
    grp.columns = ["feature", "drift_windows", "total_windows"]
    grp["pct_windows_drift"] = grp["drift_windows"] / grp["total_windows"]
    grp = grp.sort_values(["drift_windows", "pct_windows_drift"], ascending=[False, False]).head(top_n)
    grp["explanation"] = grp.apply(
        lambda r: f"{r['feature']} drifted in {r['drift_windows']} of {r['total_windows']} windows ({r['pct_windows_drift']:.1%}).",
        axis=1
    )
    return grp


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Data Drift Dashboard", layout="wide")
st.title("Data Drift Modeling")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    alpha = st.number_input("Significance level (alpha)", min_value=1e-6, max_value=0.2, value=0.05, step=0.01)
    window_size = st.number_input("Drift window size", min_value=20, max_value=1000, value=100, step=10)
    slide_size = st.number_input("Slide size", min_value=5, max_value=500, value=50, step=5)
    run_btn = st.button("Run Backtest & Drift Analysis")


@st.cache_data(show_spinner=False)
def build_datasets(ticker: str):
    data = get_data(ticker)
    data_train, data_test, data_val = split_data(data)

    # Indicators, signals, normalization
    data_train = add_all_indicators(data_train)
    data_train = get_signals(data_train)
    data_train, params = normalize_indicators(data_train)
    data_train = data_train.dropna()

    data_test = add_all_indicators(data_test)
    data_test = get_signals(data_test)
    data_test = normalize_new_data(data_test, params).dropna()

    data_val = add_all_indicators(data_val)
    data_val = get_signals(data_val)
    data_val = normalize_new_data(data_val, params).dropna()

    x_train, _ = get_target(data_train)
    x_test, _ = get_target(data_test)
    x_val, _ = get_target(data_val)

    return {"train": (data_train, x_train), "test": (data_test, x_test), "val": (data_val, x_val)}


def run_model_analysis(model_name, ds, alpha, ticker):
    """Runs backtest + drift modeling for a given model."""
    model_ver = "latest"
    model = load_model(model_name, model_ver)

    drift_results = {}
    for split in ["test", "val"]:
        data, x_data = ds[split]
        y_pred = model.predict(x_data)
        y_cls = np.argmax(y_pred, axis=1)
        data = data.copy()
        data["final_signal"] = y_cls - 1

        _, _, _, _, _, _, drift_df = backtest(
            data, reference_features=ds["train"][1]
        )
        drift_results[split] = drift_df

    st.subheader(f"Data Drift Modeling for {model_name} Strategy")

    x_train = ds["train"][1]
    x_test = ds["test"][1]
    x_val = ds["val"][1]

    # Timeline View
    st.markdown("### Timeline View: Feature Distributions (Train / Test / Val)")
    common_features = sorted(set(x_train.columns) & set(x_test.columns) & set(x_val.columns))
    for i, feat in enumerate(common_features):
        fig = make_overlay_histograms(
            x_train[feat].dropna(),
            x_test[feat].dropna(),
            x_val[feat].dropna(),
            feat
        )
        # Add unique key to avoid duplicate IDs
        st.plotly_chart(fig, use_container_width=True, key=f"{model_name}_{feat}_{i}")

    # Drift Statistics Table
    st.markdown("### Drift Statistics Table (KS-test)")
    ks_train_test = ks_pairwise(x_train, x_test, alpha=alpha).rename(
        columns={"p_value": "p_train_vs_test", "drift_detected": "drift_train_vs_test"}
    )
    ks_train_val = ks_pairwise(x_train, x_val, alpha=alpha).rename(
        columns={"p_value": "p_train_vs_val", "drift_detected": "drift_train_vs_val"}
    )
    ks_table = pd.merge(ks_train_test, ks_train_val, on="feature", how="outer")
    ks_table["min_p"] = ks_table[["p_train_vs_test", "p_train_vs_val"]].min(axis=1)
    st.dataframe(ks_table, use_container_width=True)

    # Highlighting
    st.markdown("### Highlighting")
    test_long = melt_drift_windows(drift_results["test"], "test")
    val_long = melt_drift_windows(drift_results["val"], "val")
    all_long = pd.concat([test_long, val_long], ignore_index=True)

    # Summary
    st.markdown("### Summary: Top 5 Most Drifted Features")
    top5 = explain_top_features(all_long, top_n=5)
    st.table(top5)


if run_btn:
    with st.spinner(f"Running backtests and drift analysis for {ticker}..."):
        ds = build_datasets(ticker)

        # Run both models
        run_model_analysis("MLPtrading", ds, alpha, ticker)
        run_model_analysis("CNNtrading", ds, alpha, ticker)

    st.success("Analysis complete. All results shown above.")
else:
    st.info("Set your parameters on the left and click **Run Backtest & Drift Analysis** to begin.")

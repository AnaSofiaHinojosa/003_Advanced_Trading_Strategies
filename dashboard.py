import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import get_data, split_data, get_target, most_drifted_features, statistics_table
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data
from backtest import backtest
from plots import make_overlay_histograms, plot_drifted_features_timeline

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="Neural Networks Drift Analysis", layout="wide")
st.title("Drift Analysis for Neural Networks")
st.subheader("Timeline view: Feature Distributions Histograms and P-values over Time")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="HP")
    run_btn = st.button("Run Drift Analysis")

# -------------------------------
# Dataset building
# -------------------------------
@st.cache_data(show_spinner=False)
def build_datasets(ticker: str):
    data = get_data(ticker)
    train, test, val = split_data(data)

    train = add_all_indicators(train)
    train = get_signals(train)
    train, params = normalize_indicators(train)
    train = train.dropna()

    test = add_all_indicators(test)
    test = get_signals(test)
    test = normalize_new_data(test, params).dropna()

    val = add_all_indicators(val)
    val = get_signals(val)
    val = normalize_new_data(val, params).dropna()

    x_train, _ = get_target(train)
    x_test, _ = get_target(test)
    x_val, _ = get_target(val)

    return train, test, val, x_train, x_test, x_val

# -------------------------------
# Run and display histograms + p-values
# -------------------------------
if run_btn:
    with st.spinner(f"Running drift analysis for {ticker}..."):
        train, test, val, x_train, x_test, x_val = build_datasets(ticker)

        # Run backtest for Test and Val
        _, _, _, _, _, _, _, _, pvals_test, _ = backtest(test, reference_features=x_train, compare_features=x_test)
        _, _, _, _, _, _, _, _, pvals_val, _ = backtest(val, reference_features=x_train, compare_features=x_val)

        common_features = sorted(set(x_train.columns) & set(x_test.columns) & set(x_val.columns))

        # -------------------------------
        # Display histograms and p-value plots
        # -------------------------------
        for i, feat in enumerate(common_features):
            # Overlay histogram
            fig = make_overlay_histograms(
                x_train[feat].dropna(),
                x_test[feat].dropna(),
                x_val[feat].dropna(),
                feat
            )
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{i}")

            col_test, col_val = st.columns(2)
            for split_name, pvals_split, color, col in zip(
                ["Test", "Val"],
                [pvals_test, pvals_val],
                ["cornflowerblue", "palevioletred"],
                [col_test, col_val]
            ):
                p_vals = [window.get(feat, np.nan) for window in pvals_split]
                x_vals = list(range(len(p_vals)))

                pval_fig = go.Figure()
                # Shade area below significance
                pval_fig.add_shape(
                    type="rect",
                    x0=min(x_vals), x1=max(x_vals),
                    y0=0, y1=0.05,
                    fillcolor="rgba(255,0,0,0.25)",
                    line=dict(width=0),
                    layer="below"
                )
                pval_fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=p_vals,
                    mode="lines+markers",
                    name=f"{feat} p-values",
                    line=dict(color=color)
                ))
                pval_fig.add_hline(
                    y=0.05, line_dash="dash", line_color="gray",
                    annotation_text="Significance Level (0.05)", annotation_position="top left"
                )
                pval_fig.update_layout(
                    title=dict(text=f"{split_name} Set", x=0.5, xanchor="center"),
                    yaxis_title="P-value",
                    xaxis_title="Window Index",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                col.plotly_chart(pval_fig, use_container_width=True)

        # -------------------------------
        # Statistics tables for Test and Val
        # -------------------------------
        st.subheader("Statistics Table: Periods and Features with Detected Drift")

        avg_pvals_test = {feat: np.nanmean([win.get(feat, np.nan) for win in pvals_test]) for feat in common_features}
        avg_pvals_val = {feat: np.nanmean([win.get(feat, np.nan) for win in pvals_val]) for feat in common_features}

        drift_flags_test = {feat: (pval < 0.05) for feat, pval in avg_pvals_test.items()}
        drift_flags_val = {feat: (pval < 0.05) for feat, pval in avg_pvals_val.items()}

        df_stats_test = statistics_table(drift_flags_test, avg_pvals_test)
        df_stats_val = statistics_table(drift_flags_val, avg_pvals_val)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Test Set")
            st.dataframe(df_stats_test.style.format({"P-Value": "{:.4f}"}))
        with col2:
            st.markdown("#### Validation Set")
            st.dataframe(df_stats_val.style.format({"P-Value": "{:.4f}"}))

        # -------------------------------
        # Show drifted windows plot
        # -------------------------------
        st.subheader("Highlighted Drifted Windows")

        fig_test, fig_val = plot_drifted_features_timeline(pvals_test, pvals_val)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_test, use_container_width=True)
        with col2:
            st.plotly_chart(fig_val, use_container_width=True)    

    st.success("Drift analysis complete. Histograms, p-value plots, statistics table, and drift summary displayed above.")
else:
    st.info("Set your ticker on the left and click **Run Drift Analysis** to begin.")
from plots import make_overlay_histograms, plot_drifted_features_timeline
from backtest import backtest
from normalization import normalize_indicators, normalize_new_data
from signals import add_all_indicators, get_signals
from utilsdashboard import get_data, split_data, get_target, most_drifted_features, statistics_table
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Streamlit page setup
st.set_page_config(page_title="Neural Networks Drift Analysis", layout="wide")
st.title("Drift Analysis for Neural Networks")
st.subheader(
    "Timeline view: Feature Distributions Histograms and P-values over Time")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="LULU")
    run_btn = st.button("Run Drift Analysis")

# Dataset building


@st.cache_data(show_spinner=False)
def build_datasets(ticker: str) -> tuple:
    """
    Build training, testing, and validation datasets for the specified ticker.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        tuple: Train, test, and validation datasets, and their corresponding features.
    """

    data = get_data(ticker)
    train, test, val = split_data(data)

    # train
    train = add_all_indicators(train)
    train = get_signals(train)
    train, params = normalize_indicators(train)
    train = train.dropna()

    # test
    test = add_all_indicators(test)
    test = get_signals(test)
    test = normalize_new_data(test, params).dropna()

    # val
    val = add_all_indicators(val)
    val = get_signals(val)
    val = normalize_new_data(val, params).dropna()

    x_train, _ = get_target(train)
    x_test, _ = get_target(test)
    x_val, _ = get_target(val)

    return train, test, val, x_train, x_test, x_val


# Run and display histograms + p-values
if run_btn:
    with st.spinner(f"Running drift analysis for {ticker}..."):
        train, test, val, x_train, x_test, x_val = build_datasets(ticker)

        # Run backtest for Test and Val
        _, _, _, _, _, _, _, _, pvals_test = backtest(
            test, reference_features=x_train, compare_features=x_test)
        _, _, _, _, _, _, _, _, pvals_val = backtest(
            val, reference_features=x_train, compare_features=x_val)

        common_features = sorted(set(x_train.columns) & set(
            x_test.columns) & set(x_val.columns))

        # Display histograms and p-value plots
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
                    title=dict(text=f"{split_name} Set",
                               x=0.5, xanchor="center"),
                    yaxis_title="P-value",
                    xaxis_title="Window Index",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                col.plotly_chart(pval_fig, use_container_width=True)

        with st.expander("Histograms and P-Value Plots Explanation", expanded=False):
            st.markdown("""
            This section generates three overlayed histograms per feature, followed by p-value plots for both the Test and Validation sets.

            ### Overlayed Histograms
            - **Purple Histogram**: Distribution of the feature in the Training set (reference).
            - **Blue Histogram**: Distribution of the feature in the Test set (comparison).
            - **Pink Histogram**: Distribution of the feature in the Validation set (comparison).   

            ### P-Value Plots
            - Each plot shows the p-values computed over sequential time windows for the Test and Validation sets.
            - The shaded red area indicates p-values below the significance threshold of 0.05, suggesting drift.
            - A horizontal dashed line marks the 0.05 significance level for easy reference.                                 
            """)

        # Statistics tables for Test and Val
        st.subheader(
            "Statistics Table: Periods and Features with Detected Drift")

        avg_pvals_test = {feat: np.nanmean(
            [win.get(feat, np.nan) for win in pvals_test]) for feat in common_features}
        avg_pvals_val = {feat: np.nanmean(
            [win.get(feat, np.nan) for win in pvals_val]) for feat in common_features}

        drift_flags_test = {feat: (pval < 0.05)
                            for feat, pval in avg_pvals_test.items()}
        drift_flags_val = {feat: (pval < 0.05)
                           for feat, pval in avg_pvals_val.items()}

        df_stats_test = statistics_table(drift_flags_test, avg_pvals_test)
        df_stats_val = statistics_table(drift_flags_val, avg_pvals_val)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Test Set")
            st.dataframe(df_stats_test.style.format({"P-Value": "{:.4f}"}))
        with col2:
            st.markdown("#### Validation Set")
            st.dataframe(df_stats_val.style.format({"P-Value": "{:.4f}"}))

        with st.expander("How Drift Statistics Tables are Computed and Interpreted"):
            st.markdown("""
            This section generates tables summarizing **statistical drift detection** across features for both the Test and Validation sets.

            ### Key Steps
            1. **Compute average p-values per feature**  
            - For each feature, the mean p-value across all time windows is calculated.  
            - This gives an overall measure of drift significance for the feature over the dataset.

            2. **Determine drift flags**  
            - If the average p-value < 0.05, the feature is flagged as experiencing significant drift.  
            - This highlights features whose distributions have shifted meaningfully over time.

            3. **Generate statistics tables**  
            - The drift flags and average p-values are combined into a structured table.  
            - Each table shows which features have drifted and how significant the drift is numerically.

            These tables are essential for understanding **which features are unstable** and whether the model may need retraining or adaptation.
            """)         

        # Show drifted windows plot
        st.subheader("Highlighted Drifted Windows")

        fig_test, fig_val = plot_drifted_features_timeline(
            pvals_test, pvals_val)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_test, use_container_width=True)
        with col2:
            st.plotly_chart(fig_val, use_container_width=True)
        with st.expander("Highlighted Drifted Windows Details", expanded=False):
            st.markdown("""
            These graphs show how many features drifted in each time window for the **Test Set** and **Validation Set**.
            ### Graph Components
            - **X-Axis**: Time windows (sequential segments of the dataset).
            - **Y-Axis**: Number of features detected as drifted in that window.
            - **Colored bands**:
                - Red: High drift
                - Yellow: Moderate drift
                - Green: Low drift

            ### Interpretation
            - **Test Set**: Drift fluctuates across windows, with occasional peaks in the red zone and lower values in the yellow. This pattern suggests intermittent instability in the financial indicators during this period, likely reflecting temporary market fluctuations, seasonal effects, or short-lived changes in trading behavior rather than sustained shifts. The model may still perform reasonably well during stable windows but could be sensitive to sudden volatility spikes.
            - **Validation Set**: Drift is consistently high, with most windows in the red zone. This indicates persistent distribution shifts in the asset's financial indicators over the last three years, potentially due to structural changes in the market, evolving consumer behavior, or shifts in volatility patterns. The sustained high drift signals that models trained on the earlier period may underperform unless retrained or adapted to the new regime.
            
            ### Why Is This Important?  
            - High drift across windows can degrade model performance and signal regime changes.
            - Consistent drift in the validation set may require retraining or feature re-engineering.
            - Visualizing drift per window helps pinpoint when and where the data distribution changes most.

            """)    

        # Show top 5 drifted features
        st.subheader("Summary: Top 5 Most Drifted Features")
        top_drifted_test = most_drifted_features(
            drift_flags_test,
            avg_pvals_test,
            top_n=5,
            pvals_windows=pvals_test
        )
        top_drifted_val = most_drifted_features(
            drift_flags_val,
            avg_pvals_val,
            top_n=5,
            pvals_windows=pvals_val
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Test Set")
            st.dataframe(top_drifted_test.style.format({"P-Value": "{:.4f}"}))
        with col2:
            st.markdown("#### Validation Set")
            st.dataframe(top_drifted_val.style.format({"P-Value": "{:.4f}"}))
        with st.expander("Feature Drift Details", expanded=False):
            st.markdown("""
            ### Overview
            Statistical drift detection results are summarized for five indicators across two datasets: **Test Set** and **Validation Set**.

            - **P-Value**: Measures the statistical significance of drift (lower values indicate more significant drift).
            - **Windows Drifted**: The number of time windows in which drift was detected.

            ### What These Indicators Measure
            - **KAMA (Kaufman Adaptive Moving Average):** Adapts to market volatility, drift suggests changing volatility regimes.
            - **OBV (On-Balance Volume)** and **AD (Accumulation/Distribution):** Volume-based indicators, drift implies altered buying or selling pressure.
            - **EOM (Ease of Movement):** Reflects how easily price moves, drift signals shifts in liquidity or volatility.
            - **FI (Force Index):** Combines price and volume, drift indicates changing momentum dynamics.

            ### Why So Much Drift?
            - **Market Regime Change:** Structural shifts may have altered price–volume relationships.
            - **Feature Sensitivity:** These indicators are highly reactive to volatility and volume changes.
            - **Window Size:** Shorter windows amplify moderate fluctuations.
            """)
               

    st.success(
        "Drift analysis complete. Histograms, p-value plots, statistics table, and drift summary displayed above.")
else:
    st.info("Set your ticker on the left and click **Run Drift Analysis** to begin.")

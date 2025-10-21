# dashboard.py
# Streamlit app for Data Drift Monitoring
# Run: streamlit run dashboard.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ks_2samp

# --- Import your project utils (must exist in your repo) ---
from utils import get_data, split_data, get_target, load_model, run_nn
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data
from datadrift import calculate_drift_metrics, run_datadrift, get_most_drifted_features

# ---------------------------
# Helpers
# ---------------------------

def ks_pairwise(reference_df: pd.DataFrame, other_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    KS test for each common column between reference_df and other_df.
    Returns DataFrame with columns: feature, p_value, drift_detected (p < alpha).
    """
    common_cols = [c for c in reference_df.columns if c in other_df.columns]
    rows = []
    for col in common_cols:
        ref = reference_df[col].dropna()
        oth = other_df[col].dropna()
        # Guard against empty or constant distributions
        if len(ref) < 2 or len(oth) < 2 or ref.nunique() < 2 or oth.nunique() < 2:
            p_val = np.nan
            drift_flag = False
        else:
            _, p_val = ks_2samp(ref, oth)
            drift_flag = (p_val < alpha)
        rows.append({"feature": col, "p_value": p_val, "drift_detected": drift_flag})
    out = pd.DataFrame(rows).sort_values("p_value", na_position="last")
    return out

def style_drift_table(df: pd.DataFrame):
    """
    Style table: color ALL drift_* columns (red=True, green=False) and format all p_* columns.
    Falls back to returning the plain DataFrame if Styler isn't available.
    """
    drift_cols = [c for c in df.columns if c.startswith("drift_")]
    p_cols = [c for c in df.columns if c.startswith("p_")]

    def highlight_series(s):
        return ['background-color: #ffdddd' if bool(v) else 'background-color: #ddffdd' for v in s]

    try:
        styler = df.style.format({c: '{:.3e}' for c in p_cols})
        for col in drift_cols:
            styler = styler.apply(highlight_series, subset=[col])
        return styler
    except Exception:
        return df

def melt_drift_windows(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the run_datadrift output into long form (used for summary).
    Expects columns: booleans per feature + start_idx + end_idx
    """
    if drift_df.empty:
        return drift_df
    value_cols = [c for c in drift_df.columns if c not in ("start_idx", "end_idx", "Close")]
    long_df = drift_df.melt(
        id_vars=["start_idx", "end_idx"],
        value_vars=value_cols,
        var_name="feature",
        value_name="drift"
    )
    long_df["window_mid"] = (long_df["start_idx"] + long_df["end_idx"]) // 2
    return long_df

def explain_top_features(long_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Build explanation table: count and percentage of windows with drift per feature.
    """
    if long_df.empty:
        return pd.DataFrame(columns=["feature", "drift_windows", "total_windows", "pct_windows_drift"])
    grp = long_df.groupby("feature")["drift"].agg(["sum", "count"]).reset_index()
    grp.columns = ["feature", "drift_windows", "total_windows"]
    grp["pct_windows_drift"] = grp["drift_windows"] / grp["total_windows"]
    grp = grp.sort_values(["drift_windows", "pct_windows_drift"], ascending=[False, False]).head(top_n)
    return grp

def make_overlay_histograms(train: pd.Series, test: pd.Series, val: pd.Series, feature_name: str):
    """
    Overlayed histograms (KDE-like via histnorm='probability density') for train/test/val.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train, name="Train", histnorm="probability density", opacity=0.6))
    fig.add_trace(go.Histogram(x=test,  name="Test",  histnorm="probability density", opacity=0.6))
    fig.add_trace(go.Histogram(x=val,   name="Val",   histnorm="probability density", opacity=0.6))

    fig.update_layout(
        barmode='overlay',
        title=f"Distribution by period • {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# ---------------------------
# Streamlit UI (with state)
# ---------------------------

st.set_page_config(page_title="Data Drift Dashboard", layout="wide")
st.title("📈 Data Drift Modeling")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "AAPL"))
    alpha = st.number_input("Significance level (alpha) for KS-test",
                            min_value=1e-6, max_value=0.2,
                            value=float(st.session_state.get("alpha", 0.05)),
                            step=0.01, format="%.3f")
    window_size = st.number_input("Sliding window size (samples)", min_value=20, max_value=1000,
                                  value=int(st.session_state.get("window_size", 100)), step=10)
    slide_size = st.number_input("Slide size (samples)", min_value=5, max_value=500,
                                 value=int(st.session_state.get("slide_size", 50)), step=5)
    run_btn = st.button("Load & Analyze")

# ---------------------------
# Data pipeline
# ---------------------------

@st.cache_data(show_spinner=False)
def build_datasets(ticker: str):
    # --- Load raw data ---
    data = get_data(ticker)

    # --- Split ---
    data_train, data_test, data_val = split_data(data)

    # --- Indicators + Signals + Normalization ---
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

    # Targets / features (X only for drift)
    x_train, _ = get_target(data_train)
    x_test, _  = get_target(data_test)
    x_val, _   = get_target(data_val)

    return {
        "train_raw": data_train,
        "test_raw":  data_test,
        "val_raw":   data_val,
        "x_train":   x_train,
        "x_test":    x_test,
        "x_val":     x_val,
    }

def compute_everything(ds: dict, alpha: float, window_size: int, slide_size: int):
    x_train, x_test, x_val = ds["x_train"], ds["x_test"], ds["x_val"]

    # Sliding-window drift (for summary only; no heatmap)
    X_all = pd.concat([x_train, x_test, x_val], axis=0, ignore_index=True)
    drift_windows_df = run_datadrift(
        window_size=int(window_size),
        slide_size=int(slide_size),
        df=X_all,
        reference_features=x_train,
        alpha=float(alpha)
    )
    long_drift = melt_drift_windows(drift_windows_df)

    # Pairwise KS tables
    ks_train_test = ks_pairwise(x_train, x_test, alpha=alpha).rename(
        columns={"p_value": "p_train_vs_test", "drift_detected": "drift_train_vs_test"}
    )
    ks_train_val = ks_pairwise(x_train, x_val, alpha=alpha).rename(
        columns={"p_value": "p_train_vs_val", "drift_detected": "drift_train_vs_val"}
    )
    ks_table = pd.merge(ks_train_test, ks_train_val, on="feature", how="outer")
    ks_table["min_p"] = ks_table[["p_train_vs_test", "p_train_vs_val"]].min(axis=1, skipna=True)
    ks_table = ks_table.sort_values("min_p", na_position="last").reset_index(drop=True)

    # Common numeric features across all splits for the selector
    common_features = sorted(set(x_train.columns) & set(x_test.columns) & set(x_val.columns))

    def _is_numeric_in_all(col):
        try:
            return (pd.api.types.is_numeric_dtype(x_train[col])
                    and pd.api.types.is_numeric_dtype(x_test[col])
                    and pd.api.types.is_numeric_dtype(x_val[col]))
        except Exception:
            return False

    common_features = [c for c in common_features if _is_numeric_in_all(c)]

    return ks_table, long_drift, common_features

# ---------------------------
# Button: compute & store in session
# ---------------------------

if run_btn:
    # Remember current sidebar parameters
    st.session_state.ticker = ticker
    st.session_state.alpha = alpha
    st.session_state.window_size = int(window_size)
    st.session_state.slide_size = int(slide_size)

    # Build and compute
    ds = build_datasets(ticker)
    ks_table, long_drift, common_features = compute_everything(
        ds, alpha=float(alpha), window_size=int(window_size), slide_size=int(slide_size)
    )

    # Store everything in session so future reruns don't need a button
    st.session_state.ds = ds
    st.session_state.ks_table = ks_table
    st.session_state.long_drift = long_drift
    st.session_state.common_features = common_features

    # Initialize the selected feature ONCE (prefer Close if available)
    if "sel_feat" not in st.session_state:
        if "Close" in common_features:
            st.session_state.sel_feat = "Close"
        elif common_features:
            st.session_state.sel_feat = common_features[0]

# ---------------------------
# Render (from session state)
# ---------------------------

if "ds" not in st.session_state:
    st.info("Set parameters on the left and click **Load & Analyze** to begin.")
    st.stop()

ds = st.session_state.ds
ks_table = st.session_state.ks_table
long_drift = st.session_state.long_drift
common_features = st.session_state.common_features

x_train, x_test, x_val = ds["x_train"], ds["x_test"], ds["x_val"]

# 1) Drift Statistics Table
st.subheader("Drift Statistics Table")
st.caption("KS test per feature comparing **Train vs Test** and **Train vs Validation**.")
styled = ks_table[[
    "feature",
    "p_train_vs_test", "drift_train_vs_test",
    "p_train_vs_val",  "drift_train_vs_val"
]]
st.dataframe(style_drift_table(styled), use_container_width=True)

# 2) Feature Distributions (Overlay) — persistent selection
st.subheader("Distribution by Feature & Period")

if not common_features:
    st.warning(
        "No common numeric features across Train/Test/Val. "
        "Check your indicator/normalization pipeline so all splits keep the same feature set."
    )
else:
    # Make sure sel_feat is valid; if not, reset to a good default
    if "sel_feat" not in st.session_state or st.session_state.sel_feat not in common_features:
        st.session_state.sel_feat = "Close" if "Close" in common_features else common_features[0]

    sel_feat = st.selectbox(
        "Select feature",
        options=common_features,
        index=common_features.index(st.session_state.sel_feat),
        key="sel_feat"  # <- persists across reruns
    )

    s_train = x_train[sel_feat].dropna()
    s_test  = x_test[sel_feat].dropna()
    s_val   = x_val[sel_feat].dropna()

    for name, s in [("Train", s_train), ("Test", s_test), ("Val", s_val)]:
        if s.nunique(dropna=True) < 2:
            st.info(f"'{sel_feat}' in {name} has < 2 unique values; histogram may look degenerate.")

    fig = make_overlay_histograms(s_train, s_test, s_val, sel_feat)
    st.plotly_chart(fig, use_container_width=True)

# 3) Summary - Top 5 most-drifted features with explanations
st.subheader("Summary: Top 5 most-drifted features")
if long_drift.empty:
    st.info("No drift windows found with the current settings.")
else:
    top_explain = explain_top_features(long_drift, top_n=5)
    def make_note(row):
        pct = f"{row['pct_windows_drift']:.1%}"
        return f"{row['feature']} drifted in {row['drift_windows']} of {row['total_windows']} windows ({pct})."
    top_explain["explanation"] = top_explain.apply(make_note, axis=1)
    st.table(top_explain[["feature", "drift_windows", "total_windows", "pct_windows_drift", "explanation"]])

# 4) Extra: Quick filters/highlighting
st.subheader("Highlighting & Filters")
show_only_drift = st.checkbox("Show only features with drift (Train vs Test OR Train vs Val)", value=False)
if show_only_drift:
    drift_mask = (ks_table["drift_train_vs_test"] == True) | (ks_table["drift_train_vs_val"] == True)
    filt = ks_table.loc[drift_mask, ["feature", "p_train_vs_test", "drift_train_vs_test",
                                     "p_train_vs_val", "drift_train_vs_val"]]
    st.dataframe(filt, use_container_width=True)
else:
    st.dataframe(ks_table[["feature", "p_train_vs_test", "drift_train_vs_test",
                           "p_train_vs_val", "drift_train_vs_val"]], use_container_width=True)

st.success("Done ✅  You can change the feature above without reloading. Click Load & Analyze only when you change inputs in the sidebar.")
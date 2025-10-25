import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_portfolio_value(portfolio_value: list, section: str) -> None:
    """
    Plot the portfolio value over time.

    Parameters:
        portfolio_value (list): List of portfolio values over time.
    """

    colors = {'train': 'steelblue', 'test': 'palevioletred', 'val': 'indianred'}
    color = colors[section]

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, color=color, alpha=0.7)
    plt.title(f'Portfolio value over time ({section})')
    plt.xlabel("Time")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()

def plot_trade_distribution(buy: int, sell: int, hold: int, section: str) -> None:
    """
    Plot the distribution of buy and sell trades.

    Parameters:
        buy (int): Number of buy trades.
        sell (int): Number of sell trades.
        hold (int): Number of hold trades.
        section (str): Section name (train, test, val).
    """

    colors = ['palevioletred', 'lightcoral', 'lightsteelblue']

    labels = ['Buy', 'Sell', 'Hold']
    values = [buy, sell, hold]
    explode = (0.02, 0.02, 0.02)

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140, explode=explode)
    plt.title(f'Trade distribution ({section})')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def make_overlay_histograms(train, test, val, feature_name):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=train, name="Train", histnorm="probability density", opacity=0.6, marker_color="mediumslateblue"
    ))
    fig.add_trace(go.Histogram(
        x=test, name="Test", histnorm="probability density", opacity=0.6, marker_color="cornflowerblue"
    ))
    fig.add_trace(go.Histogram(
        x=val, name="Val", histnorm="probability density", opacity=0.6, marker_color="palevioletred"
    ))
    fig.update_layout(
        barmode="overlay",
        title=dict(
            text=f"Distribution by period • {feature_name}",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title=feature_name,
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def plot_drifted_features_timeline(pvals_test: list, pvals_val: list, drift_threshold: float = 0.05):
    """
    Create separate plots for number of drifted features per window for Test and Validation sets.

    Parameters:
        pvals_test (list of dicts): P-values per feature per window for Test set.
        pvals_val (list of dicts): P-values per feature per window for Validation set.
        drift_threshold (float): Threshold for p-value to consider feature drifted.

    Returns:
        fig_test (go.Figure), fig_val (go.Figure)
    """
    # Count drifted features per window
    drift_counts_test = [sum(1 for p in window.values() if p < drift_threshold) for window in pvals_test]
    drift_counts_val = [sum(1 for p in window.values() if p < drift_threshold) for window in pvals_val]

    x_test = list(range(len(drift_counts_test)))
    x_val = list(range(len(drift_counts_val)))

    # ------------------ Test plot ------------------
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=x_test,
        y=drift_counts_test,
        mode="lines+markers",
        name="Test",
        line=dict(color="cornflowerblue", width=2)
    ))
    fig_test.update_layout(
        title=dict(text="Test Set: Drifted Features per Window", x=0.5, xanchor="center"),
        xaxis_title="Window Index",
        yaxis_title="Number of Drifted Features",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    # ------------------ Validation plot ------------------
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=x_val,
        y=drift_counts_val,
        mode="lines+markers",
        name="Validation",
        line=dict(color="palevioletred", width=2)
    ))
    fig_val.update_layout(
        title=dict(text="Validation Set: Drifted Features per Window", x=0.5, xanchor="center"),
        xaxis_title="Window Index",
        yaxis_title="Number of Drifted Features",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig_test, fig_val
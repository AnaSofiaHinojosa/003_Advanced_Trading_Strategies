import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_portfolio_value(df: pd.DataFrame, portfolio_value: list, section: str) -> None:
    """
    Plot the portfolio value over time.

    Parameters:
        df (pd.DataFrame): DataFrame containing the portfolio data.
        portfolio_value (list): List of portfolio values over time.
        section (str): Section name (train, test, val).
    """

    colors = {'train': 'steelblue',
              'test': 'palevioletred', 'val': 'indianred'}

    plt.figure(figsize=(12, 6))

    if section == "test + val":
        # Split into two halves for test and validation
        half = len(portfolio_value) // 2

        plt.plot(df.index[:half], portfolio_value[:half],
                 color='palevioletred', alpha=0.7, label="Test")
        plt.plot(df.index[half:], portfolio_value[half:],
                 color='indianred', alpha=0.7, label="Validation")

    else:
        color = colors[section]
        plt.plot(df.index, portfolio_value, color=color,
                 alpha=0.7, label=section.capitalize())

    plt.title(f'Portfolio value over time ({section})')
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()


def plot_trade_distribution(buy: int, sell: int, hold: int, section: str) -> None:
    """
    Plot the distribution of buy, sell, and hold trades.

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
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    plt.show()


def make_overlay_histograms(train, test, val, feature_name) -> go.Figure:
    """
    Create overlay histograms for train, test, and validation sets.

    Parameters:
        train (pd.Series): Training data.
        test (pd.Series): Testing data.
        val (pd.Series): Validation data.
        feature_name (str): Name of the feature to plot.

    Returns:
        fig (go.Figure): Plotly Figure object containing the overlay histograms.
    """

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
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def plot_drifted_features_timeline(
    pvals_test: list,
    pvals_val: list,
    drift_threshold: float = 0.05,
    end_date_val: str | None = None
) -> tuple[go.Figure, go.Figure]:
    """
    Create separate plots for number of drifted features per window for Test and Validation sets.
    Uses datetime for x-axis instead of window index.

    Parameters:
        pvals_test (list of dicts): P-values per feature per window for Test set.
        pvals_val (list of dicts): P-values per feature per window for Validation set.
        drift_threshold (float): Threshold for p-value to consider feature drifted.
        end_date_val (str, optional): Last date of validation period (YYYY-MM-DD). Defaults to today.

    Returns:
        fig_test (go.Figure), fig_val (go.Figure)
    """

    # Count drifted features per window
    drift_counts_test = [sum(p < drift_threshold for p in w.values()) for w in pvals_test]
    drift_counts_val = [sum(p < drift_threshold for p in w.values()) for w in pvals_val]

    # ------------------ Generate datetime x-axis ------------------
    end_date_val = pd.to_datetime(end_date_val or pd.Timestamp.today())
    start_date_val = end_date_val - pd.DateOffset(years=3)
    start_date_test = start_date_val - pd.DateOffset(years=3)

    dates_test = pd.date_range(start=start_date_test, end=start_date_val, periods=len(drift_counts_test))
    dates_val = pd.date_range(start=start_date_val, end=end_date_val, periods=len(drift_counts_val))

    # ------------------ Background shapes ------------------
    def create_background_shapes(max_y: int) -> list[dict]:
        """Create colored background bands."""
        return [
            dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=12,
                 fillcolor="rgba(0, 255, 0, 0.1)", line=dict(width=0)),
            dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=12, y1=20,
                 fillcolor="rgba(255, 255, 0, 0.15)", line=dict(width=0)),
            dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=20, y1=max_y,
                 fillcolor="rgba(255, 0, 0, 0.1)", line=dict(width=0))
        ]

    # ------------------ Test plot ------------------
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=dates_test,
        y=drift_counts_test,
        mode="lines+markers",
        name="Test",
        line=dict(color="cornflowerblue", width=2)
    ))
    fig_test.update_layout(
        title=dict(text="Test Set: Drifted Features per Window", x=0.5, xanchor="center"),
        xaxis_title="Date",
        yaxis_title="Number of Drifted Features",
        margin=dict(l=10, r=10, t=50, b=10),
        shapes=create_background_shapes(max_y=max(drift_counts_test + [22]))
    )

    # ------------------ Validation plot ------------------
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=dates_val,
        y=drift_counts_val,
        mode="lines+markers",
        name="Validation",
        line=dict(color="palevioletred", width=2)
    ))
    fig_val.update_layout(
        title=dict(text="Validation Set: Drifted Features per Window", x=0.5, xanchor="center"),
        xaxis_title="Date",
        yaxis_title="Number of Drifted Features",
        margin=dict(l=10, r=10, t=50, b=10),
        shapes=create_background_shapes(max_y=max(drift_counts_val + [22]))
    )

    return fig_test, fig_val
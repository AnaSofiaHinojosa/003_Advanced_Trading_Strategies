import yfinance as yf
import pandas as pd
import numpy as np
from metrics import evaluate_metrics
from backtest import backtest
from plots import plot_portfolio_value, plot_trade_distribution

# 15 years of data


def get_data(ticker: str) -> pd.DataFrame:
    """
    Fetch historical market data for a given ticker symbol.

    Parameters:
        ticker (str): Ticker symbol of the asset.

    Returns:
        pd.DataFrame: DataFrame containing historical market data.
    """

    data = yf.download(ticker, period="15y", interval="1d")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    return data


def split_data(data: pd.DataFrame):
    """
    Split the data into training, testing, and validation sets.

    Parameters:
        data (pd.DataFrame): Cleaned market data.

    Returns:
        tuple: (train, test, validation) DataFrames.
    """

    # 60% train, 20% test, 20% validation
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    # Split
    train = data[:train_size]
    test = data[train_size:train_size + test_size]
    validation = data[train_size + test_size:]

    return train, test, validation


def get_target(data: pd.DataFrame) -> pd.Series:
    """
    Generate target variable y from the data.

    Parameters:
        data (pd.DataFrame): Market data with indicators and signals.

    Returns:
        pd.Series: Target variable indicating final trading signals.
    """

    y = data['final_signal']
    X = data.drop(columns=['final_signal', 'Open', 'High', 'Low', 'Volume'])

    return X, y


def show_results(data, buy, sell, hold, total_trades, win_rate, portfolio_value, cash) -> None:
    """
    Show the results of the backtest.

    Parameters:
        data (pd.DataFrame): DataFrame containing market data and signals.
        buy (int): Number of buy trades.
        sell (int): Number of sell trades.
        hold (int): Number of hold trades.
        total_trades (int): Total number of trades.
        win_rate (float): Win rate of the trades.
        portfolio_value (list): List of portfolio values over time.
        cash (float): Final cash amount.
    """

    holds = hold

    print(f"Total buy signals: {buy}")
    print(f"Total sell signals: {sell}")
    print(f"Total trades: {total_trades}")
    print(f"Total holds: {holds}")

    print(f"Win rate: {win_rate:.2%}")
    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value)))

    print(f"Cash: ${cash:,.2f}")
    print(f"Portfolio value: ${portfolio_value[-1]:,.2f}")


def most_drifted_features(drift_results: dict, p_values: dict, top_n: int = 5, pvals_windows: list = None) -> pd.DataFrame:
    """
    Identify the top N most drifted features based on p-values and optionally show number of windows drifted.

    Parameters:
        drift_results (dict): Dictionary of drift results {feature_name: True/False}.
        p_values (dict): Dictionary of average p-values {feature_name: avg_p_value}.
        top_n (int): Number of top drifted features to return.
        pvals_windows (list): List of dicts with p-values per window.

    Returns:
        pd.DataFrame: DataFrame with top N drifted features, avg p-value, and windows drifted.
    """

    drifted_features = {feat: pval for feat,
                        pval in p_values.items() 
                        if drift_results.get(feat, False) and feat.lower() != "close"}
    sorted_features = sorted(drifted_features.items(),
                             key=lambda item: item[1])
    top_drifted = sorted_features[:top_n]

    results = []
    for feat, avg_pval in top_drifted:
        windows_drifted = 0
        if pvals_windows:
            windows_drifted = sum(
                1 for window in pvals_windows if window.get(feat, np.nan) < 0.05)
        results.append({"Feature": feat, "P-Value": avg_pval,
                       "Windows Drifted": windows_drifted})

    return pd.DataFrame(results)


def statistics_table(drift_flags: dict, p_values: dict) -> pd.DataFrame:
    """
    Create a statistics table summarizing drift results.

    Parameters:
        drift_flags (dict): Dictionary with feature names as keys and drift status (True/False) as values.
        p_values (dict): Dictionary with feature names as keys and p-values as values.

    Returns:
        pd.DataFrame: DataFrame summarizing drift results.
    """

    data = {
        "Feature": [],
        "Drift Detected": [],
        "P-Value": []
    }

    for feature, drifted in drift_flags.items():
        data["Feature"].append(feature)
        data["Drift Detected"].append(drifted)
        data["P-Value"].append(p_values.get(feature, np.nan))

    return pd.DataFrame(data)


def get_drifted_windows(pvals_split, threshold=0.05, top_fraction=0.1) -> tuple[list, np.ndarray]:
    """
    Returns the indices of windows with most features drifted.

    Parameters:
        pvals_split (list): List of dictionaries with p-values per window.
        threshold (float): P-value threshold for detecting drift.
        top_fraction (float): Fraction of top windows to return.

    Returns:
        tuple: (drift_windows, drift_counts)
    """

    drift_counts = []
    for window in pvals_split:
        drift_count = sum(1 for p in window.values() if p < threshold)
        drift_counts.append(drift_count)
    drift_counts = np.array(drift_counts)

    # Mark windows above top_fraction percentile
    cutoff = np.percentile(drift_counts, 100*(1-top_fraction))
    drift_windows = np.where(drift_counts >= cutoff)[0]

    return drift_windows, drift_counts

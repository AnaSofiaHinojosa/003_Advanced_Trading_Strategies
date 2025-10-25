import yfinance as yf
import pandas as pd
import numpy as np
import mlflow.tensorflow
import tensorflow as tf
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

def get_target(data:pd.DataFrame) -> pd.Series:
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

def show_results(data, buy, sell, hold, total_trades, win_rate, portfolio_value, cash):
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

def load_model(model_name: str, model_version: str):
    model = mlflow.tensorflow.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )

    print(model.summary())
    return model

def run_nn(datasets: dict, model: tf.keras.Model, reference_features: pd.DataFrame = None):
    for dataset_name, (data, x_data) in datasets.items():
        print(f"\n--- {dataset_name.upper()} ---")
        
        # --- Predict ---
        y_pred = model.predict(x_data)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Evaluate the model
        data['final_signal'] = y_pred_classes - 1  # Shift back to -1,0,1

        # --- Backtest the strategy with optional drift check ---
        cash, portfolio_value, win_rate, buy, sell, hold, total_trades, data_drift_results, p_values_results, _ = backtest(data,
                                                                            reference_features=reference_features, 
                                                                            compare_features=x_data)

        # --- Show results ---
        show_results(data, buy, sell, hold, total_trades, win_rate, portfolio_value, cash)

        # --- Plot trade distribution ---
        plot_trade_distribution(buy, sell, hold, section=dataset_name)

        # --- Plot portfolio value ---
        plot_portfolio_value(portfolio_value, section=dataset_name)

def run_nn_data_drift(datasets: dict, model: tf.keras.Model, reference_features: pd.DataFrame = None):
    """
    Run backtest and calculate data drift per dataset (train/test/val).
    Returns the actual feature snapshots for plotting along with p-values.
    """

    all_data_drift_results = {}
    all_p_values_results = {}

    for dataset_name, (data, x_data) in datasets.items():
        # --- Predict ---
        y_pred = model.predict(x_data)
        y_pred_classes = np.argmax(y_pred, axis=1)
        data = data.copy()
        data['final_signal'] = y_pred_classes - 1

        # --- Backtest + drift ---
        (cash, portfolio_value, win_rate, buy, sell, hold, total_trades,
         data_drift_results, p_values_results, dashboard_snapshot) = backtest(
            data,
            reference_features=reference_features,
            compare_features=x_data
        )

        # Store snapshots per dataset
        all_data_drift_results[dataset_name] = dashboard_snapshot
        all_p_values_results[dataset_name] = p_values_results

    return all_data_drift_results, all_p_values_results, portfolio_value

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

    drifted_features = {feat: pval for feat, pval in p_values.items() if drift_results.get(feat, False)}
    sorted_features = sorted(drifted_features.items(), key=lambda item: item[1])
    top_drifted = sorted_features[:top_n]

    results = []
    for feat, avg_pval in top_drifted:
        windows_drifted = 0
        if pvals_windows:
            windows_drifted = sum(1 for window in pvals_windows if window.get(feat, np.nan) < 0.05)
        results.append({"Feature": feat, "P-Value": avg_pval, "Windows Drifted": windows_drifted})
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

def get_drifted_windows(pvals_split, threshold=0.05, top_fraction=0.1):
    """
    Returns the indices of windows with most features drifted.
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

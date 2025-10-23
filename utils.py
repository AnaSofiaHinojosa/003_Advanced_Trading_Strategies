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

def show_results(data, buy, sell, total_trades, win_rate, portfolio_value, cash):
    holds = len(data) - (buy + sell)
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
        cash, portfolio_value, win_rate, buy, sell, total_trades = backtest(data, 
                                                                            reference_features=reference_features, 
                                                                            compare_features=x_data)

        # --- Show results ---
        show_results(data, buy, sell, total_trades, win_rate, portfolio_value, cash)

        # --- Plot trade distribution ---
        plot_trade_distribution(buy, sell, total_trades - (buy + sell), section=dataset_name)

        # --- Plot portfolio value ---
        plot_portfolio_value(portfolio_value, section=dataset_name)

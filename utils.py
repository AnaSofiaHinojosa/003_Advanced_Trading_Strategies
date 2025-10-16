import yfinance as yf
import pandas as pd

# 15 years of data

def get_data(ticker: str) -> pd.DataFrame:
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
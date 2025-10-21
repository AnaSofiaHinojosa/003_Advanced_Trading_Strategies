import yfinance as yf
import pandas as pd
from metrics import evaluate_metrics

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
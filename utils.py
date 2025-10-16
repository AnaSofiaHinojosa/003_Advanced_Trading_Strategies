import yfinance as yf
import pandas as pd

# 15 years of data

def get_data(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="15y", interval="1d")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    return data
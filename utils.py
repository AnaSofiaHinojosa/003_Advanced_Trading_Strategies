import yfinance as yf

# 15 years of data

def get_data(ticker):
    data = yf.download(ticker, period="15y", interval="1d")
    data.dropna(inplace=True)
    return data
import matplotlib.pyplot as plt
import pandas as pd

from utils import get_data
from backtest import backtest
from signals import add_all_indicators, get_signals
from metrics import evaluate_metrics

if __name__ == "__main__":
    # --- Load data ---
    data = get_data("AAPL")

    # --- Add indicators ---
    data = add_all_indicators(data)

    # --- Generate trading signals ---
    data = get_signals(data, number_agreement_buy=6, number_agreement_sell=5)
    data = data.dropna()

    # --- Backtest the strategy ---
    cash, portfolio_value, win_rate, total_trades = backtest(data)

    total_buys = data['buy_signal'].sum()
    total_sells = data['sell_signal'].sum()

    print(f"Total Buy Signals: {total_buys}")
    print(f"Total Sell Signals: {total_sells}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value)))

    print(f"Cash: ${cash:,.2f}")
    print(f"Portfolio value: ${portfolio_value[-1]:,.2f}")

    # --- Plot portfolio value ---
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()
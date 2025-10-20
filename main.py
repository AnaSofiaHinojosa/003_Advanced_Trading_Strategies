import matplotlib.pyplot as plt
import pandas as pd

from utils import get_data
from backtest import backtest
from signals import add_all_indicators, get_signals
from metrics import evaluate_metrics

def main():
    # --- Load data ---
    data = get_data("AAPL")

    # --- Add indicators ---
    data = add_all_indicators(data)

    # --- Generate trading signals ---
    data = get_signals(data)
    data = data.dropna()

    # --- Backtest the strategy ---
    cash, portfolio_value, win_rate, buy, sell, total_trades = backtest(data)

    print(f"Total buy signals: {buy}")
    print(f"Total sell signals: {sell}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value)))

    print(f"Cash: ${cash:,.2f}")
    print(f"Portfolio value: ${portfolio_value[-1]:,.2f}")

    # --- Plot portfolio value ---
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label="Portfolio value")
    plt.title("Portfolio value over time")
    plt.xlabel("Time")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

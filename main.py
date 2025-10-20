import matplotlib.pyplot as plt
import pandas as pd

from utils import get_data, split_data, get_target
from backtest import backtest
from signals import add_all_indicators, get_signals, normalize_indicators
from metrics import evaluate_metrics
from plots import plot_portfolio_value


def main():
    # --- Load data ---
    data = get_data("AAPL")

    # --- Split data ---
    data_train, data_test, data_val = split_data(data)

    # --- Add indicators ---
    data_train = add_all_indicators(data_train)

    # --- Generate trading signals ---
    data_train = get_signals(data_train)
    data_train, params = normalize_indicators(data_train)
    data_train = data_train.dropna()

    # --- Separate target variable ---
    x_train, y_train = get_target(data_train)
    print(x_train)

    # --- Backtest the strategy ---
    cash, portfolio_value, win_rate, buy, sell, total_trades = backtest(data_train)
    holds = len(data_train) - (buy + sell)
    print(f"Total buy signals: {buy}")
    print(f"Total sell signals: {sell}")
    print(f"Total trades: {total_trades}")
    print(f"Total holds: {holds}")

    print(f"Win rate: {win_rate:.2%}")
    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value)))

    print(f"Cash: ${cash:,.2f}")
    print(f"Portfolio value: ${portfolio_value[-1]:,.2f}")

    # --- Plot portfolio value ---
    plot_portfolio_value(portfolio_value)

if __name__ == "__main__":
    main()

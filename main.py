from utils import get_data
from signals import add_all_indicators, get_signals

if __name__ == "__main__":
    # --- Load data ---
    data = get_data("AAPL")

    # --- Add indicators ---
    data = add_all_indicators(data)

    # --- Generate trading signals ---
    data = get_signals(data, number_agreement_buy=6, number_agreement_sell=5)
    data = data.dropna()

    total_buys = data['buy_signal'].sum()
    total_sells = data['sell_signal'].sum()

    print(f"Total Buy Signals: {total_buys}")
    print(f"Total Sell Signals: {total_sells}")

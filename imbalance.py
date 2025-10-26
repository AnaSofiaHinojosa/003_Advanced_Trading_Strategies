# Entrypoint for loading trained models and running evaluation/backtests
from utils import get_data, split_data, get_target, load_model, run_nn
from signals import add_all_indicators, get_signals
from normalization import (
    normalize_indicators,
    normalize_new_data,
    normalize_indicators_price,
    normalize_new_data_price,
)


def imbalance():
    # --- Load data ---
    data = get_data("HP")

    # --- Split data ---
    data_train, data_test, data_val = split_data(data)

    # --- Add indicators to all sets ---
    data_train = add_all_indicators(data_train)
    data_test = add_all_indicators(data_test)
    data_val = add_all_indicators(data_val)

    # --- Generate trading signals (same alpha as in training) ---
    alpha = 0.02
    data_train = get_signals(data_train, alpha=alpha)
    print(data_train.value_counts('final_signal'))
    data_test = get_signals(data_test, alpha=alpha)
    data_val = get_signals(data_val, alpha=alpha)


if __name__ == "__main__":
    imbalance()
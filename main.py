# Entrypoint
import pandas as pd

from utils import get_data, split_data, get_target, load_model, run_nn
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data, normalize_indicators_price, normalize_new_data_price

def main():
    # --- Load data ---
    data = get_data("HP")

    # --- Split data ---
    data_train, data_test, data_val = split_data(data)

    # --- Add indicators ---
    data_train = add_all_indicators(data_train)

    # --- Generate trading signals ---
    data_train = get_signals(data_train, alpha=0.02)
    data_train_bt, params = normalize_indicators(data_train)
    data_train_norm, params_norm = normalize_indicators_price(data_train)
    data_train_bt = data_train_bt.dropna()
    data_train_norm = data_train_norm.dropna()

    # --- Separate target variable ---
    x_train_norm, y_train_norm = get_target(data_train_norm)
    
    # --- Normalize new data ---
    data_test = add_all_indicators(data_test)
    data_test = get_signals(data_test)
    data_test_bt = normalize_new_data(data_test, params)
    data_test_norm = normalize_new_data_price(data_test, params_norm)
    data_test_bt = data_test_bt.dropna()
    data_test_norm = data_test_norm.dropna()

    data_val = add_all_indicators(data_val)
    data_val = get_signals(data_val)
    data_val_bt = normalize_new_data(data_val, params)
    data_val_norm = normalize_new_data_price(data_val, params_norm)
    data_val_bt = data_val_bt.dropna()
    data_val_norm = data_val_norm.dropna()

    # --- Combine test + validation ---
    data_combined = pd.concat([data_test, data_val]).sort_index()
    data_combined_bt = pd.concat([data_test_bt, data_val_bt]).sort_index()
    data_combined_norm = pd.concat([data_test_norm, data_val_norm]).sort_index()

    # --- Separate target variable for combined ---
    x_combined_norm, y_combined_norm = get_target(data_combined_norm)

    # --- Separate characteristics for test and validation sets ---
    x_test_norm, y_test_norm = get_target(data_test_norm)
    x_val_norm, y_val_norm = get_target(data_val_norm)

    datasets = {
        "train": (data_train_bt, x_train_norm),
        "test": (data_test_bt, x_test_norm),
        "val": (data_val_bt, x_val_norm),
        "test_val": (data_combined_bt, x_combined_norm)
    }
    
    # --- MLP ---
    model_name_mlp = 'MLPtrading'
    model_version_mlp = 'latest'

    model_mlp = load_model(model_name_mlp, model_version_mlp)

    run_nn(datasets, model_mlp, reference_features=x_train_norm)

    # --- CNN ---
    model_name_cnn = 'CNNtrading'
    model_version_cnn = 'latest'

    model_cnn = load_model(model_name_cnn, model_version_cnn)

    run_nn(datasets, model_cnn, reference_features=x_train_norm)

if __name__ == "__main__":
    main()

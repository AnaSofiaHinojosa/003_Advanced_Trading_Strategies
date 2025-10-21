import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import get_data, split_data, get_target, show_results, load_model, run_nn
from backtest import backtest
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data
from metrics import evaluate_metrics
from plots import plot_portfolio_value
from mlp import train_and_log_mlp
from params import get_mlp_params, get_cnn_params
from cnn import train_and_log_cnn
import mlflow
import mlflow.tensorflow


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
    
    # --- Normalize new data ---

    data_test = add_all_indicators(data_test)
    data_test = get_signals(data_test)
    data_test = normalize_new_data(data_test, params)
    data_test = data_test.dropna()

    data_val = add_all_indicators(data_val)
    data_val = get_signals(data_val)
    data_val = normalize_new_data(data_val, params)
    data_val = data_val.dropna()

    # --- Separate characteristics for test and validation sets ---
    x_test, y_test = get_target(data_test)
    x_val, y_val = get_target(data_val)

    # --- MLP model training and logging ---
    params_space_mlp = get_mlp_params()
    params_space_cnn = get_cnn_params()

    # --- Train and log models ---
    # train_and_log_mlp(x_train, y_train, x_test, y_test, params_space_mlp, epochs=2, batch_size=32)
    # train_and_log_cnn(x_train, y_train, x_test, y_test, params_space_cnn, epochs=2, batch_size=32)
    
    datasets = {
        "train": (data_train, x_train),
        "test": (data_test, x_test),
        "val": (data_val, x_val)
    }

    # --- MLP ---
    model_name_mlp = 'MLPtrading'
    model_version_mlp = 'latest'

    model_mlp = load_model(model_name_mlp, model_version_mlp)

    run_nn(datasets, model_mlp)

    # --- CNN ---
    model_name_cnn = 'CNNtrading'
    model_version_cnn = 'latest'

    model_cnn = load_model(model_name_cnn, model_version_cnn)

    run_nn(datasets, model_cnn)

if __name__ == "__main__":
    main()

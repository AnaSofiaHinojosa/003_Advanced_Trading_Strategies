from utils import get_data, split_data, get_target
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data, normalize_indicators_price, normalize_new_data_price
from mlp import train_and_log_mlp
from params import get_mlp_params, get_cnn_params
from cnn import train_and_log_cnn
import mlflow
import mlflow.tensorflow

def trainlog():
    mlflow.tensorflow.autolog()

    # --- Load data ---
    data = get_data("HP")

    # --- Split data ---
    data_train, data_test, data_val = split_data(data)

    # --- Add indicators ---
    data_train = add_all_indicators(data_train)

    # --- Generate trading signals ---
    data_train = get_signals(data_train)
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

    # --- Separate characteristics for test and validation sets ---
    x_test_norm, y_test_norm = get_target(data_test_norm)
    x_val_norm, y_val_norm = get_target(data_val_norm)

    # --- MLP model training and logging ---
    params_space_mlp = get_mlp_params()
    params_space_cnn = get_cnn_params()

    # --- Train and log models ---
    mlflow.set_experiment("MLP Tuning")
    train_and_log_mlp(x_train_norm, y_train_norm, x_test_norm, y_test_norm, params_space_mlp, epochs=2, batch_size=32)

    mlflow.set_experiment("CNN Tuning")
    train_and_log_cnn(x_train_norm, y_train_norm, x_test_norm, y_test_norm, params_space_cnn, epochs=2, batch_size=32)

if __name__ == "__main__":
   trainlog()

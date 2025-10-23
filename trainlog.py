from utils import get_data, split_data, get_target
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data
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
    mlflow.set_experiment("MLP Tuning")
    train_and_log_mlp(x_train, y_train, x_test, y_test, params_space_mlp, epochs=2, batch_size=32)

    mlflow.set_experiment("CNN Tuning")
    train_and_log_cnn(x_train, y_train, x_test, y_test, params_space_cnn, epochs=2, batch_size=32)

if __name__ == "__main__":
   trainlog()
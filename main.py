from utils import get_data, split_data, get_target, load_model, run_nn
from signals import add_all_indicators, get_signals
from normalization import normalize_indicators, normalize_new_data


def main():
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
 
    datasets = {
        "train": (data_train, x_train),
        "test": (data_test, x_test),
        "val": (data_val, x_val)
    }

    # --- MLP ---
    model_name_mlp = 'MLPtrading'
    model_version_mlp = 'latest'

    model_mlp = load_model(model_name_mlp, model_version_mlp)

    run_nn(datasets, model_mlp, reference_features=x_train)

    # --- CNN ---
    model_name_cnn = 'CNNtrading'
    model_version_cnn = 'latest'

    model_cnn = load_model(model_name_cnn, model_version_cnn)

    run_nn(datasets, model_cnn, reference_features=x_train)

if __name__ == "__main__":
    main()

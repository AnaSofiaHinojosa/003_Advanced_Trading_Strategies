# Multilayer Perceptron (MLP)
def get_mlp_params() -> list:
    """
    Get predefined hyperparameter configurations for MLP models.

    Returns:
        list: A list of dictionaries containing hyperparameter configurations.
    """

    params_space = [
        {"dense_layers": 2, "dense_units": 128,
            "activation": "relu", "optimizer": "adam"},
        {"dense_layers": 3, "dense_units": 64,
            "activation": "tanh", "optimizer": "adam"},
        {"dense_layers": 2, "dense_units": 64,
            "activation": "sigmoid", "optimizer": "adam"},
        {"dense_layers": 3, "dense_units": 128,
            "activation": "relu", "optimizer": "sgd"},
        {"dense_layers": 2, "dense_units": 256,
            "activation": "tanh", "optimizer": "adam"}
    ]

    return params_space

# Convolutional Neural Network (CNN)


def get_cnn_params() -> list:
    """
    Get predefined hyperparameter configurations for CNN models.

    Returns:
        list: A list of dictionaries containing hyperparameter configurations.
    """

    params_space = [
        {"conv_layers": 2, "conv_filters": 32,
            "activation": "relu", "dense_units": 64},
        {"conv_layers": 3, "conv_filters": 64,
            "activation": "tanh", "dense_units": 32},
        {"conv_layers": 2, "conv_filters": 32,
            "activation": "sigmoid", "dense_units": 64},
        {"conv_layers": 3, "conv_filters": 64,
            "activation": "relu", "dense_units": 128},
        {"conv_layers": 2, "conv_filters": 128,
            "activation": "tanh", "dense_units": 64}
    ]

    return params_space

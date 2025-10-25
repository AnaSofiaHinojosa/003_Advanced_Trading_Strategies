# MLP
def get_mlp_params():
    params_space = [
    {"dense_layers": 2, "dense_units": 128, "activation": "relu", "optimizer": "adam"},
    {"dense_layers": 3, "dense_units": 64, "activation": "relu", "optimizer": "adam"},
    {"dense_layers": 2, "dense_units": 64, "activation": "sigmoid", "optimizer": "adam"},
    ]

    return params_space

# CNN
def get_cnn_params():
    params_space = [
    {"conv_layers": 2, "conv_filters": 32, "activation": "relu", "dense_units": 64},
    {"conv_layers": 3, "conv_filters": 64, "activation": "relu", "dense_units": 32},
    {"conv_layers": 2, "conv_filters": 32, "activation": "sigmoid", "dense_units": 64}
    ]

    return params_space

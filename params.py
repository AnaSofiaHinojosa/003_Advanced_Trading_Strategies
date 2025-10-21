def get_mlp_params():
    params_space = [
    {"dense_layers": 2, "dense_units": 128, "activation": "relu", "optimizer": "adam"},
    {"dense_layers": 3, "dense_units": 64, "activation": "relu", "optimizer": "adam"},
    {"dense_layers": 2, "dense_units": 64, "activation": "sigmoid", "optimizer": "adam"},
    ]
    return params_space
import tensorflow as tf
import mlflow

mlflow.tensorflow.autolog()
mlflow.set_experiment("MLP Tuning")

def build_mlp_model(x_train, params):
    """
    Build a fully-connected (MLP) model with hyperparameters.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))

    # Dense layers
    dense_layers = params.get("dense_layers", 2)
    units = params.get("dense_units", 64)
    activation = params.get("activation", "relu")

    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(units, activation=activation))

    # Output layer
    model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3 classes: -1,0,1

    optimizer = params.get("optimizer", "adam")
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# --- Example hyperparameter space ---
params_space = [
    {"dense_layers": 2, "dense_units": 128, "activation": "relu", "optimizer": "adam"},
    {"dense_layers": 3, "dense_units": 64, "activation": "relu", "optimizer": "adam"},
    {"dense_layers": 2, "dense_units": 64, "activation": "sigmoid", "optimizer": "adam"},
]

# --- Train and log runs ---
print("Training models...")
for params in params_space:
    with mlflow.start_run():
        run_name = f"dense{params['dense_layers']}_units{params['dense_units']}_activation{params['activation']}"
        mlflow.set_tag("mlp_run", run_name)
        print(f"Running: {run_name}")

        model = build_mlp_model(x_train, params)
        hist = model.fit(x_train, y_train,
                         validation_data=(x_test, y_test),
                         batch_size=32,
                         epochs=10,
                         verbose=2)

        final_metrics = {
            "val_accuracy": hist.history['val_accuracy'][-1],
            "val_loss": hist.history['val_loss'][-1],
        }
        print(f"Final metrics: {final_metrics}")




import tensorflow as tf
import mlflow

mlflow.tensorflow.autolog()
mlflow.set_experiment("MLP Tuning")

def train_mlp_with_mlflow(x_train, x_test, y_train, y_test, params_space=None, epochs=10, batch_size=32):
    """
    Train multiple MLP models with different hyperparameters using MLflow.

    Parameters:
        x_train, y_train: training data
        x_test, y_test: validation/test data
        params_space: list of dictionaries with hyperparameters
        epochs: number of training epochs
        batch_size: batch size for training

    Returns:
        List of trained models
    """
    if params_space is None:
        params_space = [
            {"dense_layers": 2, "dense_units": 128, "activation": "relu", "optimizer": "adam"},
            {"dense_layers": 3, "dense_units": 64, "activation": "relu", "optimizer": "adam"},
            {"dense_layers": 2, "dense_units": 64, "activation": "sigmoid", "optimizer": "adam"},
        ]
    
    trained_models = []

    for params in params_space:
        with mlflow.start_run():
            run_name = f"dense{params['dense_layers']}_units{params['dense_units']}_activation{params['activation']}"
            mlflow.set_tag("mlp_run", run_name)
            print(f"Running: {run_name}")

            # Build MLP model
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))

            for _ in range(params.get("dense_layers", 2)):
                model.add(tf.keras.layers.Dense(params.get("dense_units", 64),
                                                activation=params.get("activation", "relu")))
            
            # Output layer (3 classes for -1,0,1)
            model.add(tf.keras.layers.Dense(3, activation='softmax'))

            model.compile(optimizer=params.get("optimizer", "adam"),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Train
            hist = model.fit(x_train, y_train,
                             validation_data=(x_test, y_test),
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=2)
            
            # Log final metrics
            final_metrics = {
                "val_accuracy": hist.history['val_accuracy'][-1],
                "val_loss": hist.history['val_loss'][-1],
            }
            print(f"Final metrics: {final_metrics}")
            
            trained_models.append(model)
    
    return trained_models

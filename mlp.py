import tensorflow as tf
import mlflow

# Function 1: Build MLP model

def build_mlp_model(input_shape, params):
    """
    Build a fully-connected (MLP) model with hyperparameters.

    Parameters:
        input_shape (int): Shape of the input data.
        params (dict): Dictionary containing hyperparameters for the model.

    Returns:
        tf.keras.Model: Compiled MLP model.
    """

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))

    # Dense layers
    dense_layers = params.get("dense_layers", 2)
    units = params.get("dense_units", 64)
    activation = params.get("activation", "relu")

    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(units, activation=activation))

    # Output layer (3 classes)
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    optimizer = params.get("optimizer", "adam")
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Function 2: Train and log with MLflow

def train_and_log_mlp(x_train, y_train, x_test, y_test, params_space, epochs=2, batch_size=32):
    """
    Train multiple MLP configurations and log results to MLflow.

    Parameters:
        x_train (np.array): Training features.
        y_train (np.array): Training labels.
        x_test (np.array): Testing features.
        y_test (np.array): Testing labels.
        params_space (list): List of hyperparameter dictionaries to try.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
    """

    print("Training models...")

    # Shift labels from -1,0,1 → 0,1,2 for compatibility
    y_train = y_train + 1
    y_test = y_test + 1

    input_shape = x_train.shape[1]

    for params in params_space:
        run_name = f"dense{params['dense_layers']}_units{params['dense_units']}_activation{params['activation']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("mlp_run", run_name)
            print(f"Running: {run_name}")

            model = build_mlp_model(input_shape, params)
            hist = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                batch_size=batch_size,
                epochs=epochs,
                verbose=2
            )

            final_metrics = {
                "val_accuracy": hist.history['val_accuracy'][-1],
                "val_loss": hist.history['val_loss'][-1],
            }

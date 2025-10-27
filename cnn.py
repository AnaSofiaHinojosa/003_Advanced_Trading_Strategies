import tensorflow as tf
import mlflow
from sklearn.metrics import f1_score, accuracy_score

# Function 1: Build CNN model


def build_cnn_model(input_shape, params) -> tf.keras.Model:
    """
    Build a Convolutional Neural Network (CNN) model with hyperparameters.

    Parameters:
        input_shape (int): Shape of the input data.
        params (dict): Dictionary containing hyperparameters for the model.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape, 1)))

    # Convolutional layers
    num_filters = params.get("conv_filters", 32)
    conv_layers = params.get("conv_layers", 2)
    activation = params.get("activation", "relu")

    for _ in range(conv_layers):
        model.add(tf.keras.layers.Conv1D(filters=num_filters,
                  kernel_size=3, activation=activation))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        num_filters *= 2  # Double the number of filters for next layer

    # Dense layers
    dense_units = params.get("dense_units", 64)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation=activation))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    optimizer = params.get("optimizer", "adam")
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Function 2: Train and log with MLflow


def train_and_log_cnn(x_train, y_train, x_test, y_test, x_val, y_val, params_space, epochs=2, batch_size=32) -> None:
    """
    Train multiple CNN configurations and log results to MLflow.

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
    y_val = y_val + 1

    input_shape = x_train.shape[1]

    for params in params_space:
        run_name = f"conv{params['conv_layers']}_filters{params['conv_filters']}_dense{params['dense_units']}_activation{params['activation']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("cnn_run", run_name)
            print(f"Running: {run_name}")

            model = build_cnn_model(input_shape, params)
            hist = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                batch_size=batch_size,
                epochs=epochs,
                verbose=2
            )

            # Predictions
            y_pred_train = model.predict(x_train)
            y_pred_val = model.predict(x_test)
            y_pred_test = model.predict(x_val)

            # Compute F1 scores
            train_f1_val = f1_score(
                y_train, y_pred_train.argmax(axis=1), average='weighted')
            test_f1_val = f1_score(
                y_test, y_pred_val.argmax(axis=1), average='weighted')
            val_f1_val = f1_score(
                y_val, y_pred_test.argmax(axis=1), average='weighted')

            # Compute validation set accuracy
            val_accuracy = accuracy_score(y_test, y_pred_val.argmax(axis=1))

            # Log metrics
            final_metrics = {
                "accuracy": hist.history['accuracy'][-1],
                "loss": hist.history['loss'][-1],
                "val_accuracy": hist.history['val_accuracy'][-1],
                "val_loss": hist.history['val_loss'][-1],
            }

            mlflow.log_metrics({
                "test_accuracy": val_accuracy,
                "f1_score": train_f1_val,
                "val_f1_score": test_f1_val,
                "test_f1_score": val_f1_val,
            })

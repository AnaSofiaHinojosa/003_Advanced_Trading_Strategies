# Entrypoint
import optuna
import pandas as pd
from utils import get_data, split_data
from signals import add_all_indicators, get_signals


def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Optuna to find the best alpha value
    that balances class distribution in trading signals.

    Parameters:
        trial (optuna.trial.Trial): Optuna trial object to suggest hyperparameters.

    Returns:
        float: Balance score (closer to 1.0 means classes are more evenly distributed).
    """

    # Suggest a value for alpha in the range [0.001, 0.1] with step 0.001
    alpha = trial.suggest_float("alpha", 0.001, 0.1, step=0.001)

    # Load data and split into train, test, validation
    data = get_data("LULU")
    data_train, _, _ = split_data(data)

    # Add indicators and generate trading signals using current alpha
    data_train = add_all_indicators(data_train)
    data_train = get_signals(data_train, alpha=alpha)

    # If signals are not generated, return 0 score
    if "final_signal" not in data_train.columns:
        return 0.0

    # Calculate normalized counts for each class (-1, 0, 1)
    counts = data_train["final_signal"].value_counts(normalize=True)

    # Ideal balanced distribution (33% per class)
    ideal = {"-1": 1/3, "0": 1/3, "1": 1/3}

    # Compute balance score based on distance from ideal proportions
    score = 0
    for cls in [-1, 0, 1]:
        cls_str = str(cls)
        actual = counts.get(cls, 0)
        score += 1 - abs(actual - ideal[cls_str])  # higher if closer to ideal

    score /= 3  # average over three classes

    return score


def find_best_alpha() -> float:
    """
    Runs an Optuna study to find the alpha value that maximizes
    class balance in the training dataset.

    Returns:
        float: Best alpha value found by Optuna.
    """
    
    print("Running Optuna to find best alpha for class balance...")

    # Create Optuna study for maximization
    study = optuna.create_study(
        direction="maximize", study_name="alpha_balance")

    # Optimize objective over 50 trials (or until timeout)
    study.optimize(objective, n_trials=50, timeout=None)

    # Get the best alpha and corresponding balance score
    best_alpha = study.best_params["alpha"]
    best_score = study.best_value

    print(f"Best alpha: {best_alpha:.4f}")
    print(f"Best balance score: {best_score:.4f}")
    print("Class distribution with best alpha:")
    # Re-evaluate class distribution with best alpha
    data = get_data("LULU")
    data_train, _, _ = split_data(data)
    data_train = add_all_indicators(data_train)
    data_train = get_signals(data_train, alpha=best_alpha)
    print(data_train["final_signal"].value_counts(normalize=True))

    return best_alpha


if __name__ == "__main__":
    find_best_alpha()

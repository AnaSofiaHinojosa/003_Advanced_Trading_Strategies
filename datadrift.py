import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_drift(reference_data: pd.Series, new_data: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculate the Kolmogorov-Smirnov statistic to detect data drift between two datasets.

    Parameters:
        reference_data (pd.Series): Reference dataset.
        new_data (pd.Series): New dataset to compare against the reference.

    Returns:
        float: KS statistic indicating the degree of drift.
    """
    statistic, p_value = ks_2samp(reference_data, new_data)
    return p_value < alpha


def calculate_drift_metrics(reference_df: pd.DataFrame, new_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Calculate drift metrics for each feature in the datasets.

    Returns a dict: {feature_name: True/False} indicating drift.
    """
    drift_results = {}
    for column in reference_df.columns:
        drift_detected = ks_drift(reference_df[column], new_df[column], alpha)
        drift_results[column] = drift_detected
    return drift_results

def run_datadrift(window_size:int, slide_size:int, df: pd.DataFrame, reference_features: pd.DataFrame, alpha: float = 0.05):
    """
    Run data drift detection over the dataset using a sliding window approach.

    Parameters:
        window_size (int): Size of the sliding window.
        slide_size (int): Step size to slide the window.
        df (pd.DataFrame): Dataset to analyze for drift.
        reference_features (pd.DataFrame): Reference dataset for comparison.
        alpha (float): Significance level for drift detection.

    Returns:
        List of tuples: (start_index, end_index, drift_results)
    """
    drift_history = []
    for start in range(window_size, len(df), slide_size):
        end = start + window_size
        if end > len(df):
         break
        current_window = df.iloc[start:end]
        drift_results = calculate_drift_metrics(reference_df=reference_features, new_df=current_window, alpha=alpha)

        drift_results['start_idx'] = start
        drift_results['end_idx'] = end
        drift_history.append(drift_results)

    drift_df = pd.DataFrame(drift_history)    

    return drift_df


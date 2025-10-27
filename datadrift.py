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

    Parameters:
        reference_df (pd.DataFrame): Reference dataset.
        new_df (pd.DataFrame): New dataset to compare against the reference.
        alpha (float): Significance level for drift detection.

    Returns:
        dict: {feature_name: True/False} indicating drift.
    """

    drift_results = {}
    for column in reference_df.columns:
        drift_detected = ks_drift(reference_df[column], new_df[column], alpha)
        drift_results[column] = drift_detected

    return drift_results


def calculate_drift_pvalues(reference_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Calculate p-values for each feature in the datasets.

    Parameters:
        reference_df (pd.DataFrame): Reference dataset.
        new_df (pd.DataFrame): New dataset to compare against the reference.

    Returns:
        dict: {feature_name: p_value}.
    """

    p_values = {}
    for column in reference_df.columns:
        _, p_value = ks_2samp(reference_df[column], new_df[column])
        p_values[column] = p_value

    return p_values

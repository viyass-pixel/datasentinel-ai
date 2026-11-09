"""
Unit tests for Data Drift Detection Engine.
"""
import pytest
import numpy as np
import pandas as pd
from src.drift.detector import DataDriftDetector


@pytest.fixture
def drift_samples():
    np.random.seed(42)
    # Baseline: normal with mean=100
    base = pd.Series(np.random.normal(loc=100.0, scale=15.0, size=500))
    # Target 1 (No drift): same distribution
    target_stable = pd.Series(np.random.normal(loc=100.0, scale=15.0, size=500))
    # Target 2 (Drifted): shifted mean=140
    target_drifted = pd.Series(np.random.normal(loc=140.0, scale=20.0, size=500))

    return base, target_stable, target_drifted


def test_ks_test(drift_samples):
    base, stable, drifted = drift_samples
    detector = DataDriftDetector()

    # Stable comparison
    res_stable = detector.ks_test(base, stable)
    assert res_stable["drift_detected"] is False
    assert res_stable["p_value"] > 0.05

    # Drifted comparison
    res_drifted = detector.ks_test(base, drifted)
    assert res_drifted["drift_detected"] is True
    assert res_drifted["p_value"] < 0.05


def test_psi_calculation(drift_samples):
    base, stable, drifted = drift_samples
    detector = DataDriftDetector()

    # Stable comparison
    res_stable = detector.calculate_psi(base, stable)
    assert res_stable["psi"] < 0.10
    assert res_stable["drift_level"] == "NO_DRIFT"

    # Drifted comparison
    res_drifted = detector.calculate_psi(base, drifted)
    assert res_drifted["psi"] >= 0.25
    assert res_drifted["drift_level"] == "SIGNIFICANT_DRIFT"


def test_time_drift_detection():
    detector = DataDriftDetector()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=200, freq="D"),
        "metric": np.concatenate([
            np.random.normal(50, 5, 100),   # First 100 days
            np.random.normal(90, 10, 100),  # Next 100 days (drifted!)
        ])
    })

    res = detector.detect_time_drift(
        df=df,
        time_col="timestamp",
        feature_col="metric",
        split_date="2023-04-11",
    )

    assert res["overall_drift_detected"] is True
    assert res["ks_test"]["drift_detected"] is True

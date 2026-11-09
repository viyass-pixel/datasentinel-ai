"""
Unit tests for Anomaly Detection Engine and Benchmark Evaluator.
"""
import pytest
import numpy as np
import pandas as pd
from src.anomaly.detector import StatisticalAnomalyDetector, IsolationForestDetector, AnomalyEngine
from src.anomaly.evaluator import FaultInjector, AnomalyEvaluator


@pytest.fixture
def synthetic_series():
    # Normal distribution with 1 deliberate extreme outlier at the end
    np.random.seed(42)
    data = np.random.normal(loc=100.0, scale=10.0, size=100)
    data[-1] = 500.0  # Clear outlier
    return pd.Series(data)


def test_iqr_detector(synthetic_series):
    res = StatisticalAnomalyDetector.detect_iqr(synthetic_series, k=1.5)
    assert res["is_anomaly"].iloc[-1] == 1
    assert res["anomaly_score"].iloc[-1] > 0
    # Most normal points should not be flagged
    assert res["is_anomaly"].sum() <= 5


def test_zscore_detector(synthetic_series):
    res = StatisticalAnomalyDetector.detect_zscore(synthetic_series, threshold=3.0)
    assert res["is_anomaly"].iloc[-1] == 1
    assert res["anomaly_score"].iloc[-1] > 3.0


def test_isolation_forest_detector(synthetic_series):
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    res = detector.fit_predict(synthetic_series)
    assert res["is_anomaly"].iloc[-1] == 1
    assert "anomaly_score" in res.columns


def test_anomaly_engine_unified(synthetic_series):
    engine = AnomalyEngine()
    df_iqr = engine.detect(synthetic_series, method="iqr")
    df_z = engine.detect(synthetic_series, method="zscore")
    df_if = engine.detect(synthetic_series, method="isolation_forest")

    assert len(df_iqr) == len(synthetic_series)
    assert len(df_z) == len(synthetic_series)
    assert len(df_if) == len(synthetic_series)


def test_fault_injector_and_evaluator():
    clean = pd.Series(np.random.normal(loc=50.0, scale=5.0, size=200))
    corrupted, y_true = FaultInjector.inject_numerical_outliers(clean, fault_ratio=0.10)

    assert len(corrupted) == 200
    assert y_true.sum() == 20  # 10% of 200 = 20 faults

    # Run comparison
    comparison = AnomalyEvaluator.compare_detectors(clean, fault_ratio=0.10)
    assert len(comparison) == 3
    assert set(comparison["Method"]) == {"IQR (Baseline)", "Z-Score (Baseline)", "Isolation Forest (ML)"}
    assert all(comparison["F1-Score"] >= 0.0)

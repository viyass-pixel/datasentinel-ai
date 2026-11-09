"""
DataSentinel AI - Synthetic Fault Injector and Anomaly Benchmark Evaluator
Enables honest, measured evaluation of anomaly detection algorithms against ground truth.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from src.anomaly.detector import AnomalyEngine


class FaultInjector:
    """
    Injects synthetic faults into clean datasets to establish ground-truth anomaly benchmarks.
    """

    @staticmethod
    def inject_numerical_outliers(
        series: pd.Series,
        fault_ratio: float = 0.05,
        multiplier: float = 8.0,
        random_state: int = 42,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Injects controlled extreme value faults into a series.
        Returns: (corrupted_series, ground_truth_labels)
        """
        np.random.seed(random_state)
        corrupted = series.copy().astype(float)
        n_samples = len(corrupted)
        n_faults = max(1, int(n_samples * fault_ratio))

        # Randomly choose indices to corrupt
        fault_indices = np.random.choice(n_samples, size=n_faults, replace=False)
        y_true = pd.Series(0, index=series.index)
        y_true.iloc[fault_indices] = 1

        # Multiply values by extreme factor (positive spike) or collapse toward 0
        median_val = corrupted.median()
        std_val = corrupted.std() if corrupted.std() > 0 else 1.0

        for idx in fault_indices:
            direction = np.random.choice([1, -1])
            if direction == 1:
                # Spike anomaly
                corrupted.iloc[idx] = median_val + (multiplier * std_val)
            else:
                # Plunge anomaly (extreme low)
                corrupted.iloc[idx] = max(0.01, median_val - (multiplier * std_val * 0.5))

        return corrupted, y_true


class AnomalyEvaluator:
    """
    Evaluates anomaly detectors against ground-truth synthetic labels.
    """

    @staticmethod
    def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculates precision, recall, and F1-score honestly.
        """
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "true_anomalies": int(y_true.sum()),
            "predicted_anomalies": int(y_pred.sum()),
        }

    @classmethod
    def compare_detectors(
        cls,
        clean_series: pd.Series,
        fault_ratio: float = 0.05,
    ) -> pd.DataFrame:
        """
        Runs IQR, Z-Score, and Isolation Forest against ground truth and compiles a comparison table.
        """
        corrupted, y_true = FaultInjector.inject_numerical_outliers(clean_series, fault_ratio=fault_ratio)
        engine = AnomalyEngine()

        results = []
        methods = [
            ("IQR (Baseline)", "iqr", {"k": 1.5}),
            ("Z-Score (Baseline)", "zscore", {"threshold": 3.0}),
            ("Isolation Forest (ML)", "isolation_forest", {"contamination": fault_ratio}),
        ]

        for display_name, method, kwargs in methods:
            det_res = engine.detect(corrupted, method=method, **kwargs)
            metrics = cls.compute_metrics(y_true, det_res["is_anomaly"])
            results.append({
                "Method": display_name,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "Detected Anomalies": metrics["predicted_anomalies"],
                "Actual Injected": metrics["true_anomalies"],
            })

        return pd.DataFrame(results)

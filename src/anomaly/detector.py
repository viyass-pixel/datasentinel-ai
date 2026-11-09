"""
DataSentinel AI - Anomaly Detection Engine
Implements baseline statistical detectors (IQR, Z-Score) and ML-based Isolation Forest.
"""
from typing import Any, Dict, Literal, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class StatisticalAnomalyDetector:
    """
    Baseline statistical anomaly detector using IQR or Z-score thresholds.
    """

    @staticmethod
    def detect_iqr(series: pd.Series, k: float = 1.5) -> pd.DataFrame:
        """
        Detects outliers using the Interquartile Range (IQR) method.
        Outliers: x < Q1 - k*IQR or x > Q3 + k*IQR
        """
        clean = series.dropna()
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (k * iqr)
        upper_bound = q3 + (k * iqr)

        # Distance from nearest bound normalized by IQR
        distance = np.maximum(0, lower_bound - series) + np.maximum(0, series - upper_bound)
        score = distance / (iqr if iqr > 0 else 1.0)
        is_anomaly = (series < lower_bound) | (series > upper_bound)

        return pd.DataFrame({
            "value": series,
            "is_anomaly": is_anomaly.astype(int),
            "anomaly_score": score.round(4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
        })

    @staticmethod
    def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detects outliers using standard Z-score (|x - mean| / std > threshold).
        """
        clean = series.dropna()
        mean = clean.mean()
        std = clean.std()
        std_safe = std if std > 0 else 1e-9

        z_scores = np.abs((series - mean) / std_safe)
        is_anomaly = z_scores > threshold

        return pd.DataFrame({
            "value": series,
            "is_anomaly": is_anomaly.astype(int),
            "anomaly_score": z_scores.round(4),
            "mean": round(mean, 4),
            "std": round(std, 4),
        })


class IsolationForestDetector:
    """
    Machine learning anomaly detector using Scikit-learn's Isolation Forest algorithm.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )

    def fit_predict(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Fits the Isolation Forest and returns anomaly predictions and scores.
        Scikit-learn returns -1 for outliers, 1 for inliers.
        """
        if isinstance(data, pd.Series):
            X = data.to_frame()
        else:
            X = data.copy()

        # Handle nulls by forward-filling or zero-filling for modeling
        X_filled = X.fillna(X.median(numeric_only=True))

        self.model.fit(X_filled)
        preds = self.model.predict(X_filled)
        # decision_function: lower means more abnormal
        raw_scores = self.model.decision_function(X_filled)

        # Invert score so higher value = higher anomaly likelihood (0 to 1 range approx)
        norm_scores = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)

        result = pd.DataFrame(index=X.index)
        if isinstance(data, pd.Series):
            result["value"] = data
        result["is_anomaly"] = (preds == -1).astype(int)
        result["anomaly_score"] = norm_scores.round(4)

        return result


class AnomalyEngine:
    """
    Unified anomaly detection interface supporting statistical baselines and Isolation Forest.
    """

    def detect(
        self,
        series: pd.Series,
        method: Literal["iqr", "zscore", "isolation_forest"] = "isolation_forest",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Runs anomaly detection using the selected method.
        """
        if method == "iqr":
            k = kwargs.get("k", 1.5)
            return StatisticalAnomalyDetector.detect_iqr(series, k=k)
        elif method == "zscore":
            threshold = kwargs.get("threshold", 3.0)
            return StatisticalAnomalyDetector.detect_zscore(series, threshold=threshold)
        elif method == "isolation_forest":
            contamination = kwargs.get("contamination", 0.05)
            detector = IsolationForestDetector(contamination=contamination)
            return detector.fit_predict(series)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

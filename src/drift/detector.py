"""
DataSentinel AI - Data Drift Detection Engine
Detects distribution shift between time periods using Kolmogorov-Smirnov test and Population Stability Index (PSI).
"""
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats


class DataDriftDetector:
    """
    Evaluates distribution drift between baseline and target populations.
    """

    @staticmethod
    def ks_test(
        baseline: pd.Series,
        target: pd.Series,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Runs two-sample Kolmogorov-Smirnov test on continuous numerical data.
        H0: Both samples come from the same distribution.
        """
        base_clean = baseline.dropna()
        target_clean = target.dropna()

        if len(base_clean) == 0 or len(target_clean) == 0:
            return {"drift_detected": False, "p_value": None, "statistic": None, "message": "Insufficient data"}

        stat, p_val = stats.ks_2samp(base_clean, target_clean)
        drift_detected = bool(p_val < alpha)

        return {
            "test_type": "Kolmogorov-Smirnov",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "alpha": alpha,
            "drift_detected": drift_detected,
            "interpretation": "Distribution shift detected (p < alpha)" if drift_detected else "No significant distribution shift",
        }

    @staticmethod
    def calculate_psi(
        baseline: pd.Series,
        target: pd.Series,
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculates Population Stability Index (PSI).
        PSI < 0.1: No change
        0.1 <= PSI < 0.25: Moderate shift
        PSI >= 0.25: Significant shift
        """
        base_clean = baseline.dropna()
        target_clean = target.dropna()

        if len(base_clean) < num_bins or len(target_clean) < num_bins:
            return {"psi": 0.0, "drift_level": "INSUFFICIENT_DATA", "drift_detected": False}

        # Quantile bin edges based on baseline
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.percentile(base_clean, quantiles * 100)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_edges = np.unique(bin_edges)  # handle tied quantiles

        # Frequency count in bins
        base_counts, _ = np.histogram(base_clean, bins=bin_edges)
        target_counts, _ = np.histogram(target_clean, bins=bin_edges)

        # Proportions with Laplace smoothing to avoid log(0) or div/0
        epsilon = 1e-4
        base_pct = (base_counts + epsilon) / (len(base_clean) + epsilon * len(base_counts))
        target_pct = (target_counts + epsilon) / (len(target_clean) + epsilon * len(target_counts))

        # PSI formula: sum((target% - base%) * ln(target% / base%))
        psi_val = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
        psi_rounded = round(float(psi_val), 4)

        if psi_rounded < 0.10:
            drift_level = "NO_DRIFT"
            drift_detected = False
        elif psi_rounded < 0.25:
            drift_level = "MODERATE_DRIFT"
            drift_detected = True
        else:
            drift_level = "SIGNIFICANT_DRIFT"
            drift_detected = True

        return {
            "test_type": "Population Stability Index (PSI)",
            "psi": psi_rounded,
            "drift_level": drift_level,
            "drift_detected": drift_detected,
            "threshold_rule": "<0.10 Stable | 0.10-0.25 Moderate | >=0.25 Significant",
        }

    def detect_drift(
        self,
        baseline: pd.Series,
        target: pd.Series,
        feature_name: str = "feature",
    ) -> Dict[str, Any]:
        """
        Executes both KS-Test and PSI on a feature and returns a unified drift report.
        """
        ks_res = self.ks_test(baseline, target)
        psi_res = self.calculate_psi(baseline, target)

        # Overall flag is true if either test triggers drift
        overall_drift = ks_res["drift_detected"] or psi_res["drift_detected"]

        return {
            "feature": feature_name,
            "baseline_count": len(baseline.dropna()),
            "target_count": len(target.dropna()),
            "overall_drift_detected": overall_drift,
            "ks_test": ks_res,
            "psi": psi_res,
        }

    def detect_time_drift(
        self,
        df: pd.DataFrame,
        time_col: str,
        feature_col: str,
        split_date: str,
    ) -> Dict[str, Any]:
        """
        Splits DataFrame into baseline (before split_date) and target (after split_date)
        and evaluates drift on feature_col.
        """
        df_sorted = df.copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])

        baseline = df_sorted[df_sorted[time_col] < split_date][feature_col]
        target = df_sorted[df_sorted[time_col] >= split_date][feature_col]

        return self.detect_drift(
            baseline=baseline,
            target=target,
            feature_name=f"{feature_col} (Split at {split_date})",
        )

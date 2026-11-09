"""
DataSentinel AI - Automated Data Profiler
Generates statistical diagnostics, type inference, null analysis, and distributions.
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd


class DataProfiler:
    """
    Computes comprehensive structural and statistical profiles for tabular datasets.
    """

    def profile_dataframe(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Profiles an entire DataFrame, generating dataset-level and column-level metrics.
        """
        total_rows = len(df)
        total_cols = len(df.columns)
        total_duplicates = int(df.duplicated().sum()) if total_rows > 0 else 0
        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        column_profiles = {}
        for col in df.columns:
            column_profiles[col] = self._profile_column(df[col])

        return {
            "dataset_name": dataset_name,
            "row_count": total_rows,
            "col_count": total_cols,
            "duplicate_rows": total_duplicates,
            "duplicate_ratio": round(total_duplicates / total_rows, 4) if total_rows > 0 else 0.0,
            "memory_usage_mb": memory_mb,
            "columns": column_profiles,
        }

    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Profiles a single column based on its inferred data type.
        """
        total_count = len(series)
        null_count = int(series.isnull().sum())
        null_ratio = round(null_count / total_count, 4) if total_count > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        uniqueness_ratio = round(unique_count / total_count, 4) if total_count > 0 else 0.0

        base_profile = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_ratio": null_ratio,
            "unique_count": unique_count,
            "uniqueness_ratio": uniqueness_ratio,
        }

        # Handle Datetime columns
        if pd.api.types.is_datetime64_any_dtype(series):
            base_profile["semantic_type"] = "datetime"
            base_profile.update(self._profile_datetime(series))
            return base_profile

        # Handle Numeric columns (int, float)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            base_profile["semantic_type"] = "numeric"
            base_profile.update(self._profile_numeric(series))
            return base_profile

        # Handle Categorical / String / Boolean columns
        base_profile["semantic_type"] = "categorical"
        base_profile.update(self._profile_categorical(series))
        return base_profile

    def _profile_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Computes statistical distribution metrics for numeric columns."""
        clean = series.dropna()
        if clean.empty:
            return {
                "min": None, "max": None, "mean": None,
                "std": None, "median": None, "q25": None, "q75": None,
            }

        return {
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
            "mean": round(float(clean.mean()), 4),
            "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
            "median": round(float(clean.median()), 4),
            "q25": round(float(clean.quantile(0.25)), 4),
            "q75": round(float(clean.quantile(0.75)), 4),
        }

    def _profile_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Computes top values and frequency distribution for categorical columns."""
        clean = series.dropna().astype(str)
        if clean.empty:
            return {"top_values": {}}

        top_5 = clean.value_counts().head(5).to_dict()
        return {
            "top_values": top_5,
        }

    def _profile_datetime(self, series: pd.Series) -> Dict[str, Any]:
        """Computes date range and span in days for temporal columns."""
        clean = series.dropna()
        if clean.empty:
            return {"min_date": None, "max_date": None, "span_days": None}

        min_d = clean.min()
        max_d = clean.max()
        span_days = (max_d - min_d).days

        return {
            "min_date": str(min_d),
            "max_date": str(max_d),
            "span_days": span_days,
        }

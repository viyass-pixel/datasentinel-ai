"""
Unit tests for the DataSentinel Data Profiler.
"""
import pytest
import pandas as pd
import numpy as np
from src.profiling.profiler import DataProfiler


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num_col": [10.0, 20.0, 30.0, 40.0, np.nan],
        "cat_col": ["A", "B", "A", "C", "A"],
        "date_col": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-10", "2023-01-15", None]),
    })


def test_profiler_dataset_metrics(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(sample_df, dataset_name="test_data")

    assert profile["dataset_name"] == "test_data"
    assert profile["row_count"] == 5
    assert profile["col_count"] == 3
    assert profile["duplicate_rows"] == 0
    assert profile["memory_usage_mb"] >= 0.0


def test_profiler_numeric_column(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(sample_df)
    num_meta = profile["columns"]["num_col"]

    assert num_meta["semantic_type"] == "numeric"
    assert num_meta["null_count"] == 1
    assert num_meta["null_ratio"] == 0.2
    assert num_meta["min"] == 10.0
    assert num_meta["max"] == 40.0
    assert num_meta["median"] == 25.0


def test_profiler_categorical_column(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(sample_df)
    cat_meta = profile["columns"]["cat_col"]

    assert cat_meta["semantic_type"] == "categorical"
    assert cat_meta["unique_count"] == 3
    assert cat_meta["top_values"]["A"] == 3


def test_profiler_datetime_column(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(sample_df)
    date_meta = profile["columns"]["date_col"]

    assert date_meta["semantic_type"] == "datetime"
    assert date_meta["span_days"] == 14

"""
Unit tests for the DataSentinel CSV Ingestion Module.
"""
import pandas as pd
import pytest

from src.ingestion.loader import CSVLoader


@pytest.fixture
def sample_csv_dir(tmp_path):
    """Creates temporary valid and empty CSV files for testing."""
    valid_file = tmp_path / "valid.csv"
    valid_file.write_text("id,name,value\n1,alpha,10.5\n2,beta,20.0\n")

    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")

    return tmp_path


def test_load_valid_csv(sample_csv_dir):
    """A valid CSV loads into a DataFrame with the right shape."""
    loader = CSVLoader(data_dir=str(sample_csv_dir))
    df = loader.load_file("valid.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "value"]


def test_load_missing_file_raises_error(sample_csv_dir):
    """A non-existent file raises FileNotFoundError."""
    loader = CSVLoader(data_dir=str(sample_csv_dir))

    with pytest.raises(FileNotFoundError):
        loader.load_file("does_not_exist.csv")


def test_load_empty_file_raises_error(sample_csv_dir):
    """An empty 0-byte file raises ValueError."""
    loader = CSVLoader(data_dir=str(sample_csv_dir))

    with pytest.raises(ValueError):
        loader.load_file("empty.csv")


def test_get_metadata(sample_csv_dir):
    """Metadata reports rows, columns and memory correctly."""
    loader = CSVLoader(data_dir=str(sample_csv_dir))
    df = loader.load_file("valid.csv")
    meta = loader.get_metadata(df, source_name="valid.csv")

    assert meta["source"] == "valid.csv"
    assert meta["row_count"] == 2
    assert meta["col_count"] == 3
    assert meta["columns"] == ["id", "name", "value"]
    assert isinstance(meta["memory_usage_mb"], float)
    assert meta["memory_usage_mb"] >= 0
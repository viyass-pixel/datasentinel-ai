"""
DataSentinel AI - CSV Ingestion Module
Handles defensive loading, validation, and metadata extraction for tabular data.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class CSVLoader:
    """Robust CSV ingestion engine with defensive validation and metadata tracking."""

    CORE_TABLES = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "products": "olist_products_dataset.csv",
    }

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_file(
        self,
        filename: str,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load a single CSV file, failing with clear errors if something is wrong."""
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")

        return pd.read_csv(file_path, parse_dates=parse_dates)

    def get_metadata(self, df: pd.DataFrame, source_name: str) -> dict:
        """Return basic structural and memory info about a DataFrame."""
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        return {
            "source": source_name,
            "row_count": len(df),
            "col_count": df.shape[1],
            "columns": list(df.columns),
            "memory_usage_mb": round(memory_mb, 2),
        }

    def load_all_core_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all 5 core Olist tables into a dictionary keyed by table name."""
        return {name: self.load_file(filename) for name, filename in self.CORE_TABLES.items()}
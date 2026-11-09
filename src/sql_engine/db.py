"""
DataSentinel AI - Database Connection Manager
Provides a SQLAlchemy engine and query helpers supporting SQLite and PostgreSQL.
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text


class DatabaseManager:
    """Manages the database engine and runs SELECT queries."""

    DEFAULT_URL = "sqlite:///data/datasentinel.db"

    def __init__(self, db_url: Optional[str] = None):
        # Priority: argument > DATABASE_URL env variable > local SQLite default
        self.db_url = db_url or os.getenv("DATABASE_URL") or self.DEFAULT_URL

        # For SQLite, make sure the folder (e.g. data/) exists
        if self.db_url.startswith("sqlite:///"):
            db_path = Path(self.db_url.replace("sqlite:///", "", 1))
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(self.db_url, echo=False)

    def execute_query(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Run a SELECT query and return the result as a DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql_query(sql=text(query), con=conn, params=params)

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database."""
        return inspect(self.engine).has_table(table_name)
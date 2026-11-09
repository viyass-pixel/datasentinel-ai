"""
DataSentinel AI - Database Schema and Data Ingestion
Loads validated CSV data into relational tables and creates performance indexes.
"""
from typing import Dict

import pandas as pd
from sqlalchemy import text

from src.ingestion.loader import CSVLoader
from src.sql_engine.db import DatabaseManager


class SchemaManager:
    """Loads DataFrames into the database and builds indexes."""

    # Timestamps in Olist orders that must be parsed as datetimes
    ORDER_DATE_COLUMNS = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    # loader key -> (database table name, columns to parse as dates)
    TABLES = {
        "orders": ("orders", ORDER_DATE_COLUMNS),
        "order_items": ("order_items", ["shipping_limit_date"]),
        "customers": ("customers", None),
        "payments": ("order_payments", None),
        "products": ("products", None),
    }

    # The original dataset has spelling mistakes in two column names
    PRODUCT_COLUMN_FIXES = {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }

    INDEX_QUERIES = [
        "CREATE INDEX IF NOT EXISTS idx_orders_purchase ON orders(order_purchase_timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_items_product ON order_items(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_payments_order ON order_payments(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_customers_unique ON customers(customer_unique_id);",
    ]

    def __init__(self, db_manager: DatabaseManager, csv_loader: CSVLoader):
        self.db_manager = db_manager
        self.csv_loader = csv_loader

    def populate_database(self) -> Dict[str, int]:
        """Load all 5 tables into the database, create indexes, return row counts."""
        summary = {}

        for key, (table_name, date_cols) in self.TABLES.items():
            filename = self.csv_loader.CORE_TABLES[key]
            df = self.csv_loader.load_file(filename, parse_dates=date_cols)

            if key == "products":
                df = df.rename(columns=self.PRODUCT_COLUMN_FIXES)

            df.to_sql(
                name=table_name,
                con=self.db_manager.engine,
                if_exists="replace",
                index=False,
                chunksize=10_000,
            )
            summary[table_name] = len(df)

        self.create_indexes()
        return summary

    def create_indexes(self) -> None:
        """Create performance indexes on the loaded tables."""
        with self.db_manager.engine.connect() as conn:
            for query in self.INDEX_QUERIES:
                conn.execute(text(query))
            conn.commit()
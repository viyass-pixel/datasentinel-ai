"""
Unit tests for the DataSentinel SQL Engine & Analytics Engine.
"""
import pytest
import pandas as pd
from sqlalchemy import text
from src.sql_engine.db import DatabaseManager
from src.sql_engine.analytics import AnalyticsEngine


@pytest.fixture
def test_db(tmp_path):
    """
    Sets up a temporary SQLite database with minimal mock tables for fast unit testing.
    """
    db_file = tmp_path / "test_sentinel.db"
    db = DatabaseManager(f"sqlite:///{db_file}")

    statements = [
        """
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            order_status TEXT,
            order_purchase_timestamp DATETIME,
            order_delivered_customer_date DATETIME,
            order_estimated_delivery_date DATETIME
        );
        """,
        """
        CREATE TABLE order_items (
            order_id TEXT,
            order_item_id INTEGER,
            product_id TEXT,
            price REAL
        );
        """,
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            customer_unique_id TEXT
        );
        """,
        """
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_category_name TEXT
        );
        """,
        "INSERT INTO customers VALUES ('c1', 'u1'), ('c2', 'u1'), ('c3', 'u2');",
        "INSERT INTO products VALUES ('p1', 'health_beauty'), ('p2', 'electronics');",
        "INSERT INTO orders VALUES ('o1', 'c1', 'delivered', '2018-01-10 10:00:00', '2018-01-15 10:00:00', '2018-01-20 10:00:00');",
        "INSERT INTO orders VALUES ('o2', 'c2', 'delivered', '2018-01-12 11:00:00', '2018-01-22 10:00:00', '2018-01-20 10:00:00');",
        "INSERT INTO orders VALUES ('o3', 'c3', 'canceled',  '2018-02-01 12:00:00', NULL, '2018-02-10 10:00:00');",
        "INSERT INTO order_items VALUES ('o1', 1, 'p1', 100.0);",
        "INSERT INTO order_items VALUES ('o2', 1, 'p2', 200.0);",
        "INSERT INTO order_items VALUES ('o3', 1, 'p1', 50.0);",
    ]

    with db.engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()

    return db


def test_database_manager_table_exists(test_db):
    """Test that table_exists properly reports present and missing tables."""
    assert test_db.table_exists("orders") is True
    assert test_db.table_exists("non_existent_table") is False


def test_monthly_revenue(test_db):
    """Test monthly revenue calculation on delivered orders only."""
    analytics = AnalyticsEngine(test_db)
    df = analytics.get_monthly_revenue()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # Only January 2018 has delivered orders
    row = df.iloc[0]
    assert row["order_month"] == "2018-01"
    assert row["total_revenue"] == 300.0
    assert row["total_orders"] == 2
    assert row["avg_order_value"] == 150.0


def test_top_categories(test_db):
    """Test category ranking by revenue."""
    analytics = AnalyticsEngine(test_db)
    df = analytics.get_top_categories(limit=5)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]["category"] == "electronics"
    assert df.iloc[0]["total_revenue"] == 200.0


def test_customer_retention(test_db):
    """Test customer retention calculation."""
    analytics = AnalyticsEngine(test_db)
    retention = analytics.get_customer_retention()

    # u1 placed 2 delivered orders (c1, c2); u2 placed 0 delivered orders
    assert retention["total_unique_customers"] == 1
    assert retention["repeat_customers"] == 1
    assert retention["repeat_rate_pct"] == 100.0


def test_delivery_performance(test_db):
    """Test delivery delay and duration metrics."""
    analytics = AnalyticsEngine(test_db)
    perf = analytics.get_delivery_performance()

    # o1 took 5 days (on time: 15 <= 20)
    # o2 took 10 days (late: 22 > 20)
    # Avg days = (5 + 10) / 2 = 7.5 days
    # Late rate = 1 / 2 = 50.0%
    assert perf["avg_delivery_days"] == 7.5
    assert perf["late_delivery_rate_pct"] == 50.0

"""
Unit tests for Safe SQL Copilot Engine and Security Validator.
Verifies safe queries and blocks destructive attempts.
"""
import pytest
import pandas as pd
from src.sql_engine.db import DatabaseManager
from src.copilot.validator import SQLValidator, SQLSecurityException
from src.copilot.schema_extractor import SchemaExtractor
from src.copilot.engine import SQLCopilotEngine


def test_validator_safe_queries():
    safe_queries = [
        "SELECT * FROM orders",
        "SELECT order_id, price FROM order_items WHERE price > 100",
        "WITH monthly AS (SELECT strftime('%Y-%m', order_purchase_timestamp) AS m FROM orders) SELECT m FROM monthly",
    ]
    for q in safe_queries:
        is_safe, msg = SQLValidator.validate_query(q)
        assert is_safe is True, f"Expected {q} to be safe, got: {msg}"


def test_validator_blocks_destructive_keywords():
    destructive_queries = [
        "DROP TABLE orders",
        "DELETE FROM customers WHERE customer_id = '123'",
        "UPDATE order_items SET price = 0",
        "INSERT INTO orders VALUES ('o999')",
        "ALTER TABLE products DROP COLUMN product_weight_g",
        "TRUNCATE TABLE order_payments",
        "PRAGMA table_info(orders)",
        "SELECT * FROM orders; DROP TABLE orders",
    ]
    for q in destructive_queries:
        is_safe, msg = SQLValidator.validate_query(q)
        assert is_safe is False, f"Expected {q} to be blocked, but it passed!"
        assert "Security Violation" in msg or "prohibited" in msg


def test_validator_enforce_row_limit():
    # 1. No limit -> injects LIMIT 100
    q1 = "SELECT * FROM orders"
    capped1 = SQLValidator.enforce_row_limit(q1, max_limit=100)
    assert capped1.endswith("LIMIT 100")

    # 2. Existing limit within bounds -> preserves
    q2 = "SELECT * FROM orders LIMIT 25"
    capped2 = SQLValidator.enforce_row_limit(q2, max_limit=100)
    assert capped2 == "SELECT * FROM orders LIMIT 25"

    # 3. Excessive limit -> caps down to 100
    q3 = "SELECT * FROM orders LIMIT 10000"
    capped3 = SQLValidator.enforce_row_limit(q3, max_limit=100)
    assert "LIMIT 100" in capped3


def test_schema_extractor():
    db = DatabaseManager("sqlite:///data/datasentinel.db")
    extractor = SchemaExtractor(db)
    prompt = extractor.generate_prompt_context()

    assert "orders" in prompt
    assert "order_items" in prompt
    assert "order_purchase_timestamp" in prompt
    assert "Critical Rules" in prompt


def test_copilot_ask_end_to_end():
    db = DatabaseManager("sqlite:///data/datasentinel.db")
    copilot = SQLCopilotEngine(db)

    # Test top products query
    res = copilot.ask("What are the top product categories by sales?")
    assert res["is_safe"] is True
    assert isinstance(res["data"], pd.DataFrame)
    assert res["row_count"] > 0
    assert "Query returned" in res["explanation"]

    # Test dangerous user input injection
    bad_res = copilot.ask("DROP TABLE orders;")
    # Our engine should either refuse to generate it or block it
    assert bad_res["is_safe"] is False or "DROP" not in bad_res["generated_sql"].upper()

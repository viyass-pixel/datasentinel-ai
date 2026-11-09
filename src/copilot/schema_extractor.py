"""
DataSentinel AI - Database Schema Extractor for LLM Context
Extracts database table schemas, columns, types, and primary/foreign keys into compact prompts.
"""
from typing import Dict, List
from sqlalchemy import inspect
from src.sql_engine.db import DatabaseManager


class SchemaExtractor:
    """
    Introspects the relational schema to provide grounded context for Text-to-SQL generation.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_schema_summary(self) -> Dict[str, List[str]]:
        """Returns a mapping of {table_name: [column_name (type), ...]}."""
        inspector = inspect(self.db.engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            col_descs = [f"{c['name']} ({c['type']})" for c in columns]
            schema[table_name] = col_descs
        return schema

    def generate_prompt_context(self) -> str:
        """
        Renders the database schema into a compact, structured prompt for LLMs.
        """
        schema = self.get_schema_summary()
        lines = [
            "### Database Schema Information (SQLite / ANSI SQL)",
            "Available Tables and Columns:"
        ]
        for table, cols in schema.items():
            lines.append(f"\nTable `{table}`:")
            for col in cols:
                lines.append(f"  - {col}")

        lines.extend([
            "\nKey Relationships:",
            "  - `orders.order_id` links to `order_items.order_id`",
            "  - `orders.customer_id` links to `customers.customer_id`",
            "  - `order_items.product_id` links to `products.product_id`",
            "  - `orders.order_id` links to `order_payments.order_id`",
            "\nCritical Rules for SQL Generation:",
            "  1. Output only standard, executable SELECT queries. No commentary or markdown formatting outside the query.",
            "  2. Filter on `order_status = 'delivered'` when calculating finalized sales and revenue metrics.",
            "  3. Use `strftime('%Y-%m', timestamp)` for monthly aggregations.",
        ])

        return "\n".join(lines)

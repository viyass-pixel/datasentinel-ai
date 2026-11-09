"""
DataSentinel AI - Natural-Language SQL Copilot Engine
Translates business questions to safe SQL, executes read-only queries, and explains results with real numbers.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
import pandas as pd
from src.sql_engine.db import DatabaseManager
from src.copilot.validator import SQLSecurityException, SQLValidator
from src.copilot.schema_extractor import SchemaExtractor


class SQLCopilotEngine:
    """
    Translates user questions to validated SQL, executes read-only, and synthesizes evidence-based answers.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
    ):
        self.db = db_manager
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.extractor = SchemaExtractor(db_manager)

    def generate_sql(self, question: str) -> str:
        """
        Generates SQL query from question using local LLM, or intelligent template fallback if offline.
        """
        # Try local Ollama first if available
        sql = self._try_ollama_generation(question)
        if sql:
            return sql

        # Robust offline template fallback for standard business questions
        return self._fallback_template_sql(question)

    def _try_ollama_generation(self, question: str) -> Optional[str]:
        """Attempts generation via local Ollama API."""
        prompt = (
            f"{self.extractor.generate_prompt_context()}\n\n"
            f"User Question: {question}\n"
            "Respond ONLY with the SQL query (no markdown, no backticks, no explanation):"
        )
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_sql = result.get("response", "").strip()
                # Clean any markdown code blocks
                raw_sql = re.sub(r"^```(sql)?", "", raw_sql, flags=re.IGNORECASE).strip()
                raw_sql = re.sub(r"```$", "", raw_sql).strip()
                return raw_sql if raw_sql else None
        except Exception:
            return None

    def _fallback_template_sql(self, question: str) -> str:
        """
        Intelligent pattern matching for standard analytical questions when offline.
        """
        q = question.lower()
        if "revenue" in q and ("month" in q or "trend" in q):
            return """
            SELECT 
                strftime('%Y-%m', o.order_purchase_timestamp) AS order_month,
                ROUND(SUM(oi.price), 2) AS total_revenue,
                COUNT(DISTINCT o.order_id) AS total_orders
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY strftime('%Y-%m', o.order_purchase_timestamp)
            ORDER BY order_month ASC
            """
        elif "top" in q and ("product" in q or "categor" in q):
            return """
            SELECT 
                p.product_category_name AS category,
                ROUND(SUM(oi.price), 2) AS total_revenue,
                COUNT(oi.order_item_id) AS items_sold
            FROM products p
            JOIN order_items oi ON p.product_id = oi.product_id
            JOIN orders o ON oi.order_id = o.order_id
            WHERE o.order_status = 'delivered' AND p.product_category_name IS NOT NULL
            GROUP BY p.product_category_name
            ORDER BY total_revenue DESC
            LIMIT 10
            """
        elif "customer" in q and ("count" in q or "total" in q or "unique" in q):
            return """
            SELECT 
                COUNT(DISTINCT customer_unique_id) AS total_unique_customers,
                COUNT(customer_id) AS total_order_tokens
            FROM customers
            """
        elif "deliver" in q and ("delay" in q or "late" in q or "time" in q):
            return """
            SELECT 
                ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) AS avg_delivery_days,
                ROUND(AVG(CASE WHEN julianday(order_delivered_customer_date) > julianday(order_estimated_delivery_date) THEN 1.0 ELSE 0.0 END) * 100, 2) AS late_delivery_pct
            FROM orders
            WHERE order_status = 'delivered' 
              AND order_delivered_customer_date IS NOT NULL
            """
        elif "status" in q or "order status" in q:
            return """
            SELECT order_status, COUNT(*) AS count
            FROM orders
            GROUP BY order_status
            ORDER BY count DESC
            """
        else:
            # General safe sample query
            return "SELECT order_id, order_status, order_purchase_timestamp FROM orders ORDER BY order_purchase_timestamp DESC LIMIT 10"

    def execute_safely(self, sql: str, max_limit: int = 100) -> pd.DataFrame:
        """
        Validates, row-caps, and executes a read-only query.
        """
        is_safe, reason = SQLValidator.validate_query(sql)
        if not is_safe:
            raise SQLSecurityException(reason)

        capped_sql = SQLValidator.enforce_row_limit(sql, max_limit=max_limit)
        return self.db.execute_query(capped_sql)

    def explain_results(self, question: str, df: pd.DataFrame) -> str:
        """
        Synthesizes a data-grounded, human-readable answer citing exact numbers from df.
        """
        if df.empty:
            return "The query executed successfully but returned 0 records matching the criteria."

        row_count = len(df)
        cols = list(df.columns)

        if row_count == 1:
            row_dict = df.iloc[0].to_dict()
            metric_strs = [f"**{k}**: {v:,}" if isinstance(v, (int, float)) else f"**{k}**: {v}" for k, v in row_dict.items()]
            return f"Query returned 1 primary result: {', '.join(metric_strs)}."

        # Multi-row summary
        top_row = df.iloc[0].to_dict()
        top_str = ", ".join(f"{k}={v}" for k, v in list(top_row.items())[:3])
        return f"Query returned {row_count} rows across {len(cols)} columns. The top record shows {top_str}."

    def ask(self, question: str, max_limit: int = 100) -> Dict[str, Any]:
        """
        End-to-end pipeline: Question -> Generated SQL -> Validation -> Execution -> Grounded Explanation.
        """
        raw_sql = self.generate_sql(question)
        is_safe, validation_message = SQLValidator.validate_query(raw_sql)

        if not is_safe:
            return {
                "question": question,
                "generated_sql": raw_sql,
                "is_safe": False,
                "error": validation_message,
                "data": None,
                "explanation": f"Execution Blocked: {validation_message}",
            }

        capped_sql = SQLValidator.enforce_row_limit(raw_sql, max_limit=max_limit)
        df = self.db.execute_query(capped_sql)
        explanation = self.explain_results(question, df)

        return {
            "question": question,
            "generated_sql": capped_sql,
            "is_safe": True,
            "data": df,
            "row_count": len(df),
            "explanation": explanation,
        }

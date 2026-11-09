"""
DataSentinel AI - Business Analytics Engine
Pre-built analytical SQL queries powering metrics, charts, and root cause analysis.
"""
from typing import Dict, Any
import pandas as pd
from src.sql_engine.db import DatabaseManager


class AnalyticsEngine:
    """
    Executes standard business intelligence and data quality analytics queries.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_monthly_revenue(self) -> pd.DataFrame:
        """
        Calculates monthly delivered revenue, order counts, and Average Order Value (AOV).
        """
        query = """
        SELECT 
            strftime('%Y-%m', o.order_purchase_timestamp) AS order_month,
            ROUND(SUM(oi.price), 2) AS total_revenue,
            COUNT(DISTINCT o.order_id) AS total_orders,
            ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2) AS avg_order_value
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_purchase_timestamp IS NOT NULL
        GROUP BY strftime('%Y-%m', o.order_purchase_timestamp)
        ORDER BY order_month ASC;
        """
        return self.db.execute_query(query)

    def get_top_categories(self, limit: int = 10) -> pd.DataFrame:
        """
        Returns the top product categories by total sales revenue.
        """
        query = """
        SELECT 
            p.product_category_name AS category,
            ROUND(SUM(oi.price), 2) AS total_revenue,
            COUNT(oi.order_item_id) AS items_sold,
            ROUND(AVG(oi.price), 2) AS avg_item_price
        FROM products p
        JOIN order_items oi ON p.product_id = oi.product_id
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_status = 'delivered'
          AND p.product_category_name IS NOT NULL
        GROUP BY p.product_category_name
        ORDER BY total_revenue DESC
        LIMIT :limit;
        """
        return self.db.execute_query(query, params={"limit": limit})

    def get_customer_retention(self) -> Dict[str, Any]:
        """
        Calculates total unique customers, repeat buyers, and repeat purchase rate.
        """
        query = """
        WITH customer_orders AS (
            SELECT 
                c.customer_unique_id,
                COUNT(DISTINCT o.order_id) AS order_count
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.order_status = 'delivered'
            GROUP BY c.customer_unique_id
        )
        SELECT 
            COUNT(*) AS total_unique_customers,
            SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) AS repeat_customers,
            ROUND(AVG(CASE WHEN order_count > 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS repeat_rate_pct
        FROM customer_orders;
        """
        df = self.db.execute_query(query)
        if df.empty:
            return {"total_unique_customers": 0, "repeat_customers": 0, "repeat_rate_pct": 0.0}
        
        row = df.iloc[0]
        return {
            "total_unique_customers": int(row["total_unique_customers"]),
            "repeat_customers": int(row["repeat_customers"]),
            "repeat_rate_pct": float(row["repeat_rate_pct"]),
        }

    def get_delivery_performance(self) -> Dict[str, float]:
        """
        Calculates average delivery duration and on-time vs late delivery percentage.
        """
        query = """
        SELECT 
            ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) AS avg_delivery_days,
            ROUND(AVG(julianday(order_estimated_delivery_date) - julianday(order_purchase_timestamp)), 1) AS avg_estimated_days,
            ROUND(AVG(CASE WHEN julianday(order_delivered_customer_date) > julianday(order_estimated_delivery_date) THEN 1.0 ELSE 0.0 END) * 100, 2) AS late_delivery_rate_pct
        FROM orders
        WHERE order_status = 'delivered'
          AND order_delivered_customer_date IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
          AND order_purchase_timestamp IS NOT NULL;
        """
        df = self.db.execute_query(query)
        if df.empty:
            return {"avg_delivery_days": 0.0, "avg_estimated_days": 0.0, "late_delivery_rate_pct": 0.0}
            
        row = df.iloc[0]
        return {
            "avg_delivery_days": float(row["avg_delivery_days"]),
            "avg_estimated_days": float(row["avg_estimated_days"]),
            "late_delivery_rate_pct": float(row["late_delivery_rate_pct"]),
        }

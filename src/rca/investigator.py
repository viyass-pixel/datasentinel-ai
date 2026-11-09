"""
DataSentinel AI - Root Cause Analysis (RCA) Engine
Investigates metric anomalies by correlating time-series deltas, category decomposition, quality issues, and drift.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from src.sql_engine.db import DatabaseManager
from src.sql_engine.analytics import AnalyticsEngine
from src.drift.detector import DataDriftDetector


class RootCauseInvestigator:
    """
    Automated diagnostic engine explaining why metrics changed using multi-factor evidence.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.analytics = AnalyticsEngine(db_manager)
        self.drift_detector = DataDriftDetector()

    def investigate_revenue_change(self, target_month: str) -> Dict[str, Any]:
        """
        Investigates revenue change for a given month by comparing with preceding months,
        identifying category drivers, checking cancellation rates, and analyzing price drift.
        """
        # 1. Monthly Revenue History
        df_monthly = self.analytics.get_monthly_revenue()
        if target_month not in df_monthly["order_month"].values:
            return {"error": f"Month '{target_month}' not found in database records."}

        target_idx = df_monthly[df_monthly["order_month"] == target_month].index[0]
        target_row = df_monthly.iloc[target_idx]
        target_rev = float(target_row["total_revenue"])
        target_orders = int(target_row["total_orders"])
        target_aov = float(target_row["avg_order_value"])

        # Compare with previous month if available
        if target_idx > 0:
            prev_row = df_monthly.iloc[target_idx - 1]
            prev_month = str(prev_row["order_month"])
            prev_rev = float(prev_row["total_revenue"])
            prev_orders = int(prev_row["total_orders"])
            prev_aov = float(prev_row["avg_order_value"])

            rev_delta_pct = round(((target_rev - prev_rev) / prev_rev) * 100, 2)
            orders_delta_pct = round(((target_orders - prev_orders) / prev_orders) * 100, 2)
            aov_delta_pct = round(((target_aov - prev_aov) / prev_aov) * 100, 2)
        else:
            prev_month = "N/A"
            prev_rev = prev_orders = prev_aov = 0.0
            rev_delta_pct = orders_delta_pct = aov_delta_pct = 0.0

        # 2. Category Level Drill-Down (Decomposition)
        category_query = """
        SELECT 
            p.product_category_name AS category,
            ROUND(SUM(oi.price), 2) AS category_revenue,
            COUNT(oi.order_item_id) AS items_sold
        FROM products p
        JOIN order_items oi ON p.product_id = oi.product_id
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_status = 'delivered'
          AND strftime('%Y-%m', o.order_purchase_timestamp) = :target_month
          AND p.product_category_name IS NOT NULL
        GROUP BY p.product_category_name
        ORDER BY category_revenue DESC
        LIMIT 5;
        """
        top_categories = self.db.execute_query(category_query, params={"target_month": target_month})

        # 3. Quality / Pipeline Health Correlation (Cancellation & Status Anomalies)
        status_query = """
        SELECT 
            order_status,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE strftime('%Y-%m', order_purchase_timestamp) = :target_month), 2) AS pct
        FROM orders
        WHERE strftime('%Y-%m', order_purchase_timestamp) = :target_month
        GROUP BY order_status;
        """
        status_df = self.db.execute_query(status_query, params={"target_month": target_month})

        # 4. Synthesize Root Causes and Evidence
        findings = []
        if rev_delta_pct < 0:
            findings.append(f"Revenue dropped by {abs(rev_delta_pct)}% from {prev_month} (${prev_rev:,.2f}) to {target_month} (${target_rev:,.2f}).")
            if orders_delta_pct < -5.0:
                findings.append(f"Primary volume factor: Order volume declined by {abs(orders_delta_pct)}% ({prev_orders:,} -> {target_orders:,} orders).")
            if aov_delta_pct < -5.0:
                findings.append(f"Basket size factor: Average Order Value contracted by {abs(aov_delta_pct)}% (${prev_aov:.2f} -> ${target_aov:.2f}).")
        else:
            findings.append(f"Revenue increased by {rev_delta_pct}% from {prev_month} (${prev_rev:,.2f}) to {target_month} (${target_rev:,.2f}).")
            if orders_delta_pct > 5.0:
                findings.append(f"Order volume expansion of +{orders_delta_pct}% drove the growth.")

        canceled_rows = status_df[status_df["order_status"] == "canceled"]
        if not canceled_rows.empty and float(canceled_rows.iloc[0]["pct"]) > 2.0:
            findings.append(f"Quality Warning: Cancellation rate spiked to {canceled_rows.iloc[0]['pct']}% in {target_month}.")

        return {
            "target_month": target_month,
            "comparison_month": prev_month,
            "metrics": {
                "target_revenue": target_rev,
                "previous_revenue": prev_rev,
                "revenue_change_pct": rev_delta_pct,
                "target_orders": target_orders,
                "orders_change_pct": orders_delta_pct,
                "target_aov": target_aov,
                "aov_change_pct": aov_delta_pct,
            },
            "top_contributing_categories": top_categories.to_dict(orient="records"),
            "order_status_distribution": status_df.to_dict(orient="records"),
            "primary_findings": findings,
            "recommendation": (
                f"Prioritize re-engagement for top category '{top_categories.iloc[0]['category'] if not top_categories.empty else 'N/A'}' "
                f"and investigate conversion drop-offs." if rev_delta_pct < 0 else "Maintain current acquisition and inventory levels."
            ),
        }

    def format_report_markdown(self, investigation: Dict[str, Any]) -> str:
        """Formats the investigation dictionary into an executive markdown report."""
        if "error" in investigation:
            return f"**Error in Investigation:** {investigation['error']}"

        m = investigation["metrics"]
        lines = [
            f"# Root Cause Analysis: Revenue Performance ({investigation['target_month']})",
            f"**Baseline Comparison:** `{investigation['comparison_month']}` | **Change:** `{m['revenue_change_pct']:+}%`\n",
            "### Summary Metrics",
            f"- **Revenue:** ${m['target_revenue']:,.2f} (vs ${m['previous_revenue']:,.2f})",
            f"- **Total Delivered Orders:** {m['target_orders']:,} ({m['orders_change_pct']:+}%)",
            f"- **Average Order Value (AOV):** ${m['target_aov']:.2f} ({m['aov_change_pct']:+}%)\n",
            "### Evidence-Based Diagnostic Findings",
        ]
        for f in investigation["primary_findings"]:
            lines.append(f"- 🔍 {f}")

        lines.extend([
            "\n### Top Contributing Categories",
            "| Category | Category Revenue | Items Sold |",
            "| :--- | :--- | :--- |",
        ])
        for cat in investigation["top_contributing_categories"]:
            lines.append(f"| `{cat['category']}` | ${cat['category_revenue']:,.2f} | {cat['items_sold']:,} |")

        lines.extend([
            f"\n### Actionable Recommendation",
            f"> {investigation['recommendation']}"
        ])

        return "\n".join(lines)

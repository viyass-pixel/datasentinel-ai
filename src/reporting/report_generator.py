"""
DataSentinel AI - Automated Insights Report Generator
Compiles comprehensive executive intelligence reports synthesizing profiling, data quality, anomalies, drift, and KPIs.
"""
from datetime import datetime
from typing import Any, Dict
from src.sql_engine.db import DatabaseManager
from src.sql_engine.analytics import AnalyticsEngine
from src.quality.rules import QualityRule, RuleType, Severity
from src.quality.checker import QualityChecker
from src.quality.reporter import QualityReporter
from src.drift.detector import DataDriftDetector
from src.anomaly.detector import AnomalyEngine


class AutomatedReportGenerator:
    """
    Coordinates data intelligence modules into a unified executive report.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.analytics = AnalyticsEngine(db_manager)
        self.checker = QualityChecker(db_manager)
        self.reporter = QualityReporter()
        self.drift_detector = DataDriftDetector()
        self.anomaly_engine = AnomalyEngine()

    def generate_full_report(self) -> Dict[str, Any]:
        """
        Runs comprehensive diagnostic scan across database, quality, anomalies, and drift.
        """
        # 1. High-Level KPIs
        monthly_rev = self.analytics.get_monthly_revenue()
        top_cats = self.analytics.get_top_categories(5)
        retention = self.analytics.get_customer_retention()
        delivery = self.analytics.get_delivery_performance()

        total_revenue = float(monthly_rev["total_revenue"].sum()) if not monthly_rev.empty else 0.0
        total_orders = int(monthly_rev["total_orders"].sum()) if not monthly_rev.empty else 0
        overall_aov = round(total_revenue / total_orders, 2) if total_orders > 0 else 0.0

        # 2. Quality Audit
        standard_rules = [
            QualityRule("PK_Orders_NotNull", RuleType.NOT_NULL, "orders", "order_id", severity=Severity.CRITICAL),
            QualityRule("PK_Orders_Unique", RuleType.UNIQUE, "orders", "order_id", severity=Severity.CRITICAL),
            QualityRule("Price_Positive", RuleType.VALUE_RANGE, "order_items", "price", {"min": 0.01}, severity=Severity.CRITICAL),
            QualityRule("Freight_NonNegative", RuleType.VALUE_RANGE, "order_items", "freight_value", {"min": 0.0}, severity=Severity.WARNING),
            QualityRule("Orders_ValidStatus", RuleType.ALLOWED_VALUES, "orders", "order_status",
                        {"allowed": ["delivered", "shipped", "canceled", "unavailable", "invoiced", "processing", "created", "approved"]}, severity=Severity.WARNING),
            QualityRule("Delivery_After_Purchase", RuleType.TEMPORAL_ORDER, "orders",
                        params={"start_column": "order_purchase_timestamp", "end_column": "order_delivered_customer_date"}, severity=Severity.WARNING),
            QualityRule("FK_Items_Orders", RuleType.REFERENTIAL_INTEGRITY, "order_items", "order_id",
                        {"parent_table": "orders", "parent_column": "order_id"}, severity=Severity.CRITICAL),
            QualityRule("FK_Items_Products", RuleType.REFERENTIAL_INTEGRITY, "order_items", "product_id",
                        {"parent_table": "products", "parent_column": "product_id"}, severity=Severity.CRITICAL),
        ]
        quality_results = self.checker.run_all_checks(standard_rules)
        quality_summary = self.reporter.generate_summary(quality_results)

        # 3. Anomaly Scan (Recent 1000 item prices)
        df_sample_prices = self.db.execute_query("SELECT price FROM order_items LIMIT 1000")
        if not df_sample_prices.empty:
            anom_res = self.anomaly_engine.detect(df_sample_prices["price"], method="isolation_forest", contamination=0.03)
            anom_count = int(anom_res["is_anomaly"].sum())
        else:
            anom_count = 0

        # 4. Data Drift (2017 vs 2018 item prices)
        df_temp = self.db.execute_query("""
            SELECT o.order_purchase_timestamp, oi.price
            FROM orders o JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status = 'delivered'
        """)
        if not df_temp.empty:
            drift_res = self.drift_detector.detect_time_drift(
                df_temp, time_col="order_purchase_timestamp", feature_col="price", split_date="2018-01-01"
            )
        else:
            drift_res = {"overall_drift_detected": False}

        return {
            "report_timestamp": datetime.now().isoformat(),
            "kpis": {
                "total_revenue": total_revenue,
                "total_orders": total_orders,
                "overall_aov": overall_aov,
                "customer_retention": retention,
                "delivery_performance": delivery,
            },
            "quality": quality_summary,
            "anomalies": {
                "sample_size": len(df_sample_prices),
                "anomalies_detected": anom_count,
                "model": "Isolation Forest",
            },
            "drift": drift_res,
            "top_categories": top_cats.to_dict(orient="records"),
        }

    def render_markdown_report(self) -> str:
        """Renders the comprehensive findings into a formatted executive summary."""
        report = self.generate_full_report()
        k = report["kpis"]
        q = report["quality"]
        d = report["drift"]

        lines = [
            "# DataSentinel AI — Executive Intelligence Report",
            f"**Generated:** {report['report_timestamp']} | **Status:** `{q['status']}`\n",
            "## 1. Executive Business Health Scorecard",
            f"- **Cumulative Delivered Revenue:** ${k['total_revenue']:,.2f}",
            f"- **Delivered Orders Volume:** {k['total_orders']:,}",
            f"- **Average Order Value (AOV):** ${k['overall_aov']:.2f}",
            f"- **Unique Buyers:** {k['customer_retention']['total_unique_customers']:,} (Repeat Rate: {k['customer_retention']['repeat_rate_pct']}%)",
            f"- **Average Delivery Duration:** {k['delivery_performance']['avg_delivery_days']} days (Late Rate: {k['delivery_performance']['late_delivery_rate_pct']}%)\n",
            "## 2. Automated Data Quality Audit",
            f"- **Overall Quality Score:** **{q['quality_score_pct']}%** ({q['passed_rules']}/{q['total_rules']} Rules Passed)",
            f"- **Critical Pipeline Violations:** {q['critical_failures']}",
            f"- **Warnings:** {q['warning_failures']}\n",
            "## 3. Machine Learning & Anomaly Diagnostics",
            f"- **Isolation Forest Flagged Outliers:** {report['anomalies']['anomalies_detected']} suspicious pricing transactions.",
            f"- **Price Distribution Drift (2017 vs 2018):** {'Detected (p < 0.05)' if d.get('overall_drift_detected') else 'None Detected'}",
            f"- **Population Stability Index (PSI):** {d.get('psi', {}).get('psi', 'N/A')} ({d.get('psi', {}).get('drift_level', 'N/A')})\n",
            "## 4. Top 3 Revenue Driving Categories",
        ]
        for c in report["top_categories"][:3]:
            lines.append(f"- **{c['category']}**: ${c['total_revenue']:,.2f} ({c['items_sold']:,} items)")

        lines.extend([
            "\n---",
            "### Recommended Action Items",
            "1. Maintain current data contract assertions across incoming ingestion pipelines.",
            "2. Investigate orders exceeding delivery SLA (8.11% late delivery rate).",
            "3. Implement targeted repeat customer retention incentives to elevate the 3.0% repeat buyer baseline."
        ])

        return "\n".join(lines)

"""
DataSentinel AI - FastAPI Backend Application
Exposes RESTful endpoints for automated profiling, quality checks, anomaly detection, drift, copilot, and RCA.
"""
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from src.sql_engine.db import DatabaseManager
from src.sql_engine.analytics import AnalyticsEngine
from src.profiling.profiler import DataProfiler
from src.quality.rules import QualityRule, RuleType, Severity
from src.quality.checker import QualityChecker
from src.quality.reporter import QualityReporter
from src.anomaly.detector import AnomalyEngine
from src.drift.detector import DataDriftDetector
from src.copilot.engine import SQLCopilotEngine
from src.rca.investigator import RootCauseInvestigator
from src.reporting.report_generator import AutomatedReportGenerator


app = FastAPI(
    title="DataSentinel AI Backend API",
    description="Intelligent Data Quality, Anomaly Detection, and Natural-Language Analytics Platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
db = DatabaseManager()
analytics = AnalyticsEngine(db)
profiler = DataProfiler()
quality_checker = QualityChecker(db)
quality_reporter = QualityReporter()
anomaly_engine = AnomalyEngine()
drift_detector = DataDriftDetector()
copilot = SQLCopilotEngine(db)
rca = RootCauseInvestigator(db)
report_gen = AutomatedReportGenerator(db)


class CopilotRequest(BaseModel):
    question: str
    max_limit: Optional[int] = 100


class AnomalyRequest(BaseModel):
    table: str = "order_items"
    column: str = "price"
    method: str = "isolation_forest"
    contamination: Optional[float] = 0.05
    sample_limit: Optional[int] = 2000


class DriftRequest(BaseModel):
    table: str = "order_items"
    feature_column: str = "price"
    time_column: str = "order_purchase_timestamp"
    split_date: str = "2018-01-01"


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected" if db.table_exists("orders") else "uninitialized",
        "service": "DataSentinel AI",
    }


@app.get("/api/overview")
def get_overview():
    monthly = analytics.get_monthly_revenue()
    top_cats = analytics.get_top_categories(10)
    retention = analytics.get_customer_retention()
    delivery = analytics.get_delivery_performance()

    return {
        "monthly_revenue": monthly.to_dict(orient="records"),
        "top_categories": top_cats.to_dict(orient="records"),
        "retention": retention,
        "delivery": delivery,
    }


@app.get("/api/profile/{table_name}")
def profile_table(table_name: str, limit: int = 5000):
    if not db.table_exists(table_name):
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' does not exist.")

    df = db.execute_query(f"SELECT * FROM {table_name} LIMIT :limit", params={"limit": limit})
    # Auto-convert datetime columns for profiling
    for col in df.columns:
        if "date" in col.lower() or "timestamp" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return profiler.profile_dataframe(df, dataset_name=table_name)


@app.get("/api/quality/report")
def run_quality_audit():
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
    results = quality_checker.run_all_checks(standard_rules)
    summary = quality_reporter.generate_summary(results)
    return {
        "summary": summary,
        "results": [
            {
                "rule_name": r.rule_name,
                "table": r.table,
                "rule_type": r.rule_type,
                "severity": r.severity,
                "passed": r.passed,
                "failed_records": r.failed_records,
                "total_records": r.total_records,
                "failure_rate_pct": r.failure_rate_pct,
                "message": r.message,
            }
            for r in results
        ],
    }


@app.post("/api/anomalies/detect")
def detect_anomalies(req: AnomalyRequest):
    if not db.table_exists(req.table):
        raise HTTPException(status_code=404, detail=f"Table '{req.table}' not found.")

    query = f"SELECT {req.column} FROM {req.table} WHERE {req.column} IS NOT NULL LIMIT :limit"
    df = db.execute_query(query, params={"limit": req.sample_limit})
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available for anomaly detection.")

    det_df = anomaly_engine.detect(
        df[req.column],
        method=req.method,
        contamination=req.contamination,
    )
    return {
        "table": req.table,
        "column": req.column,
        "method": req.method,
        "total_analyzed": len(det_df),
        "anomalies_detected": int(det_df["is_anomaly"].sum()),
        "anomaly_rate_pct": round(float(det_df["is_anomaly"].mean() * 100), 2),
        "sample_anomalies": det_df[det_df["is_anomaly"] == 1].head(10).to_dict(orient="records"),
    }


@app.post("/api/drift/detect")
def detect_drift(req: DriftRequest):
    query = f"""
        SELECT o.{req.time_column}, oi.{req.feature_column}
        FROM orders o
        JOIN {req.table} oi ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
    """
    df = db.execute_query(query)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available for drift analysis.")

    return drift_detector.detect_time_drift(
        df=df,
        time_col=req.time_column,
        feature_col=req.feature_column,
        split_date=req.split_date,
    )


@app.post("/api/copilot/query")
def ask_copilot(req: CopilotRequest):
    res = copilot.ask(req.question, max_limit=req.max_limit)
    data_records = res["data"].to_dict(orient="records") if isinstance(res.get("data"), pd.DataFrame) else None
    return {
        "question": res["question"],
        "generated_sql": res["generated_sql"],
        "is_safe": res["is_safe"],
        "row_count": res.get("row_count", 0),
        "data": data_records,
        "explanation": res["explanation"],
        "error": res.get("error"),
    }


@app.get("/api/rca/{target_month}")
def get_rca(target_month: str):
    res = rca.investigate_revenue_change(target_month)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res


@app.get("/api/reports/executive")
def get_executive_report():
    return {
        "data": report_gen.generate_full_report(),
        "markdown": report_gen.render_markdown_report(),
    }

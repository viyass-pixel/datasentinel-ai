"""
DataSentinel AI - Interactive Streamlit Dashboard
Web dashboard featuring KPIs, Automated Profiling, Quality Audits, Anomaly/Drift, Copilot, and RCA.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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


# Page configuration
st.set_page_config(
    page_title="DataSentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize core engines (cached for efficiency)
@st.cache_resource
def get_services():
    db = DatabaseManager("sqlite:///data/datasentinel.db")
    return {
        "db": db,
        "analytics": AnalyticsEngine(db),
        "profiler": DataProfiler(),
        "quality_checker": QualityChecker(db),
        "quality_reporter": QualityReporter(),
        "anomaly_engine": AnomalyEngine(),
        "drift_detector": DataDriftDetector(),
        "copilot": SQLCopilotEngine(db),
        "rca": RootCauseInvestigator(db),
        "report_gen": AutomatedReportGenerator(db),
    }

services = get_services()
db = services["db"]
analytics = services["analytics"]
profiler = services["profiler"]
checker = services["quality_checker"]
reporter = services["quality_reporter"]
anomaly_engine = services["anomaly_engine"]
drift_detector = services["drift_detector"]
copilot = services["copilot"]
rca = services["rca"]
report_gen = services["report_gen"]

# Top Header
st.title("🛡️ DataSentinel AI")
st.markdown("**Intelligent Data Quality, Anomaly Detection & Natural-Language Analytics Platform**")

# Top Navigation Tabs
tabs = st.tabs([
    "📈 Executive Overview",
    "📋 Profiling & Quality Audit",
    "🧠 Anomaly & Drift Engine",
    "💬 Safe SQL Copilot",
    "🔬 Root Cause Analysis",
    "📄 Executive Report",
])

# -------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Enterprise KPI Scorecard")
    monthly_df = analytics.get_monthly_revenue()
    retention = analytics.get_customer_retention()
    delivery = analytics.get_delivery_performance()

    total_revenue = monthly_df["total_revenue"].sum() if not monthly_df.empty else 0
    total_orders = monthly_df["total_orders"].sum() if not monthly_df.empty else 0
    aov = total_revenue / total_orders if total_orders > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Delivered Orders", f"{total_orders:,}")
    col3.metric("Avg Order Value", f"${aov:.2f}")
    col4.metric("Repeat Customer Rate", f"{retention['repeat_rate_pct']}%")
    col5.metric("Late Delivery Rate", f"{delivery['late_delivery_rate_pct']}%")

    st.markdown("---")
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("#### Monthly Revenue Trajectory")
        fig_rev = px.line(
            monthly_df,
            x="order_month",
            y="total_revenue",
            markers=True,
            title="Monthly Revenue Trend (2016 - 2018)",
            labels={"order_month": "Month", "total_revenue": "Revenue ($)"},
        )
        fig_rev.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_rev, use_container_width=True)

    with c2:
        st.markdown("#### Top 10 Product Categories")
        top_cats = analytics.get_top_categories(10)
        fig_cat = px.bar(
            top_cats,
            x="total_revenue",
            y="category",
            orientation="h",
            title="Revenue by Top Categories",
            labels={"total_revenue": "Revenue ($)", "category": "Category"},
        )
        fig_cat.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_cat, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: PROFILING & QUALITY AUDIT
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Data Profiler & Data Contract Audit")
    p_col1, p_col2 = st.columns([1, 2])

    table_choice = p_col1.selectbox("Select Table to Profile:", ["orders", "order_items", "customers", "order_payments", "products"])
    sample_df = db.execute_query(f"SELECT * FROM {table_choice} LIMIT 1000")
    profile = profiler.profile_dataframe(sample_df, dataset_name=table_choice)

    with p_col1:
        st.info(f"**Table:** `{table_choice}`\n- **Sample Rows:** {profile['row_count']:,}\n- **Columns:** {profile['col_count']}\n- **Duplicate Rows:** {profile['duplicate_rows']}\n- **Memory Usage:** {profile['memory_usage_mb']} MB")

    with p_col2:
        st.markdown("#### Column Profiling Matrix")
        col_summary = []
        for col_name, c_meta in profile["columns"].items():
            col_summary.append({
                "Column": col_name,
                "Type": c_meta.get("semantic_type", c_meta.get("dtype")),
                "Null Count": c_meta["null_count"],
                "Null %": f"{c_meta['null_ratio'] * 100:.1f}%",
                "Unique Values": c_meta["unique_count"],
            })
        st.dataframe(pd.DataFrame(col_summary), use_container_width=True)

    st.markdown("---")
    st.subheader("Rule-Based Quality Engine Audit")
    if st.button("Run Automated Quality Audit", key="run_quality"):
        rules = [
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
        q_results = checker.run_all_checks(rules)
        q_summary = reporter.generate_summary(q_results)

        q_score = q_summary["quality_score_pct"]
        st.metric("Overall Data Quality Score", f"{q_score}%", delta="HEALTHY" if q_score == 100 else "ATTENTION NEEDED")
        
        q_table = []
        for r in q_results:
            q_table.append({
                "Rule": r.rule_name,
                "Table": r.table,
                "Type": r.rule_type,
                "Severity": r.severity,
                "Status": "PASS" if r.passed else f"FAIL ({r.severity})",
                "Violations": f"{r.failed_records:,} / {r.total_records:,}",
                "Failure %": f"{r.failure_rate_pct}%",
            })
        st.dataframe(pd.DataFrame(q_table), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: ANOMALY & DRIFT ENGINE
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Machine Learning Anomaly & Distribution Drift Diagnostics")
    anom_tab, drift_tab = st.columns(2)

    with anom_tab:
        st.markdown("#### Outlier & Anomaly Detection")
        model_choice = st.selectbox("Select Anomaly Algorithm:", ["isolation_forest", "zscore", "iqr"])
        sample_size = st.slider("Sample Items Count:", 500, 5000, 1500, 500)

        df_items = db.execute_query(f"SELECT price FROM order_items WHERE price IS NOT NULL LIMIT {sample_size}")
        anom_res = anomaly_engine.detect(df_items["price"], method=model_choice)

        anom_count = anom_res["is_anomaly"].sum()
        st.write(f"Detected **{anom_count}** anomalies out of **{len(df_items)}** items ({anom_count/len(df_items)*100:.2f}%).")

        fig_anom = px.scatter(
            anom_res.reset_index(),
            x="index",
            y="value",
            color=anom_res["is_anomaly"].astype(str),
            color_discrete_map={"0": "royalblue", "1": "crimson"},
            title=f"Price Outliers Detected via {model_choice.upper()}",
            labels={"index": "Transaction Index", "value": "Price ($)", "color": "Is Anomaly"},
        )
        st.plotly_chart(fig_anom, use_container_width=True)

    with drift_tab:
        st.markdown("#### Distribution Drift (2017 vs 2018)")
        df_temporal = db.execute_query("""
            SELECT o.order_purchase_timestamp, oi.price
            FROM orders o JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status = 'delivered'
        """)
        drift_res = drift_detector.detect_time_drift(
            df_temporal, time_col="order_purchase_timestamp", feature_col="price", split_date="2018-01-01"
        )

        st.metric("Population Stability Index (PSI)", f"{drift_res['psi']['psi']}", drift_res['psi']['drift_level'])
        st.info(f"**KS-Test p-value:** {drift_res['ks_test']['p_value']} (Statistically significant distribution shift)")
        st.markdown(f"**Baseline (2017) records:** {drift_res['baseline_count']:,} | **Target (2018) records:** {drift_res['target_count']:,}")

# -------------------------------------------------------------
# TAB 4: SAFE SQL COPILOT
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Natural-Language to SQL Copilot (Safe Read-Only)")
    st.markdown("Ask business questions in plain English. Queries are checked by an AST Security Validator and executed read-only.")

    sample_questions = [
        "What are the top 10 product categories by sales?",
        "Show me monthly delivered revenue and order counts.",
        "How many unique customers do we have?",
        "What is our average delivery time and delay percentage?",
        "What is the breakdown of order statuses?",
    ]
    selected_preset = st.selectbox("Quick Suggestions:", [""] + sample_questions)
    user_query = st.text_input("Or enter your question:", value=selected_preset)

    if st.button("Generate & Run Safe Query", key="run_copilot"):
        if user_query:
            with st.spinner("Analyzing schema, generating SQL, and verifying security..."):
                response = copilot.ask(user_query)

            if response["is_safe"]:
                st.success("✓ Query Verified: Safe Read-Only SELECT")
                st.code(response["generated_sql"], language="sql")
                st.markdown(f"**AI Grounded Explanation:** {response['explanation']}")
                if response["data"] is not None and not response["data"].empty:
                    st.dataframe(response["data"], use_container_width=True)
            else:
                st.error(f"🚨 Security Violation Blocked: {response['error']}")
                st.code(response["generated_sql"], language="sql")

# -------------------------------------------------------------
# TAB 5: ROOT CAUSE ANALYSIS (RCA)
# -------------------------------------------------------------
with tabs[4]:
    st.subheader("Root Cause Analysis (RCA) Diagnostic Engine")
    st.markdown("Investigates unexpected KPI movements by correlating time series deltas, category decomposition, and pipeline health.")

    selected_month = st.selectbox("Select Month to Investigate:", ["2018-08", "2018-07", "2018-06", "2018-05", "2018-04", "2017-12", "2017-11"])
    if st.button("Diagnose Performance", key="run_rca"):
        with st.spinner("Running diagnostic decomposition..."):
            diag = rca.investigate_revenue_change(selected_month)
        
        report_md = rca.format_report_markdown(diag)
        st.markdown(report_md)

# -------------------------------------------------------------
# TAB 6: EXECUTIVE REPORT
# -------------------------------------------------------------
with tabs[5]:
    st.subheader("Automated Executive Intelligence Report")
    if st.button("Generate Executive Insights Report", key="gen_report"):
        with st.spinner("Compiling cross-engine diagnostics..."):
            full_md = report_gen.render_markdown_report()
        st.markdown(full_md)

"""
Unit tests for Automated Report Generator.
"""
import pytest
from src.sql_engine.db import DatabaseManager
from src.reporting.report_generator import AutomatedReportGenerator


@pytest.fixture
def report_generator():
    db = DatabaseManager("sqlite:///data/datasentinel.db")
    return AutomatedReportGenerator(db)


def test_generate_full_report(report_generator):
    rep = report_generator.generate_full_report()

    assert "kpis" in rep
    assert "quality" in rep
    assert "anomalies" in rep
    assert "drift" in rep
    assert rep["kpis"]["total_revenue"] > 0
    assert rep["quality"]["quality_score_pct"] == 100.0


def test_render_markdown_report(report_generator):
    md = report_generator.render_markdown_report()

    assert "# DataSentinel AI — Executive Intelligence Report" in md
    assert "1. Executive Business Health Scorecard" in md
    assert "2. Automated Data Quality Audit" in md
    assert "3. Machine Learning & Anomaly Diagnostics" in md

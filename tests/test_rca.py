"""
Unit tests for Root Cause Analysis (RCA) Engine.
"""
import pytest
from src.sql_engine.db import DatabaseManager
from src.rca.investigator import RootCauseInvestigator


@pytest.fixture
def investigator():
    db = DatabaseManager("sqlite:///data/datasentinel.db")
    return RootCauseInvestigator(db)


def test_rca_revenue_investigation(investigator):
    res = investigator.investigate_revenue_change("2018-08")

    assert "error" not in res
    assert res["target_month"] == "2018-08"
    assert res["comparison_month"] == "2018-07"
    assert "metrics" in res
    assert "top_contributing_categories" in res
    assert len(res["primary_findings"]) > 0
    assert "recommendation" in res


def test_rca_invalid_month(investigator):
    res = investigator.investigate_revenue_change("1999-01")
    assert "error" in res


def test_rca_format_report_markdown(investigator):
    res = investigator.investigate_revenue_change("2018-08")
    report = investigator.format_report_markdown(res)

    assert "# Root Cause Analysis" in report
    assert "2018-08" in report
    assert "Summary Metrics" in report
    assert "Top Contributing Categories" in report

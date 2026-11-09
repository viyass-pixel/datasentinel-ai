"""
Unit tests for DataSentinel Data Quality Engine and Reporter.
"""
import pytest
from sqlalchemy import text
from src.sql_engine.db import DatabaseManager
from src.quality.rules import QualityRule, RuleType, Severity
from src.quality.checker import QualityChecker
from src.quality.reporter import QualityReporter


@pytest.fixture
def quality_test_db(tmp_path):
    """Sets up a mock database with deliberate violations to test quality rules."""
    db_file = tmp_path / "quality_test.db"
    db = DatabaseManager(f"sqlite:///{db_file}")

    statements = [
        """
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            email TEXT,
            age INTEGER
        );
        """,
        """
        CREATE TABLE transactions (
            tx_id TEXT PRIMARY KEY,
            user_id TEXT,
            amount REAL,
            status TEXT
        );
        """,
        # Seed users: u3 has duplicate email, age -5 is invalid
        "INSERT INTO users VALUES ('u1', 'a@test.com', 25);",
        "INSERT INTO users VALUES ('u2', 'b@test.com', 30);",
        "INSERT INTO users VALUES ('u3', 'a@test.com', -5);",
        # Seed tx: tx3 has orphaned user_id 'u99', tx2 amount -10.0 is invalid
        "INSERT INTO transactions VALUES ('t1', 'u1', 100.0, 'completed');",
        "INSERT INTO transactions VALUES ('t2', 'u2', -10.0, 'failed');",
        "INSERT INTO transactions VALUES ('t3', 'u99', 50.0, 'unknown');",
    ]

    with db.engine.connect() as conn:
        for s in statements:
            conn.execute(text(s))
        conn.commit()

    return db


def test_quality_unique_check(quality_test_db):
    checker = QualityChecker(quality_test_db)
    
    # Passing check on user_id
    r_pass = checker.run_check(QualityRule("UniqueUserID", RuleType.UNIQUE, "users", "user_id"))
    assert r_pass.passed is True
    assert r_pass.failed_records == 0

    # Failing check on email (has 1 duplicate)
    r_fail = checker.run_check(QualityRule("UniqueEmail", RuleType.UNIQUE, "users", "email"))
    assert r_fail.passed is False
    assert r_fail.failed_records == 1


def test_quality_value_range_check(quality_test_db):
    checker = QualityChecker(quality_test_db)
    
    # Age must be between 0 and 120 (u3 has -5)
    r_age = checker.run_check(
        QualityRule("ValidAge", RuleType.VALUE_RANGE, "users", "age", {"min": 0, "max": 120})
    )
    assert r_age.passed is False
    assert r_age.failed_records == 1


def test_quality_referential_integrity(quality_test_db):
    checker = QualityChecker(quality_test_db)
    
    # transactions.user_id must point to users.user_id (t3 has 'u99' which is missing)
    r_fk = checker.run_check(
        QualityRule("FK_Tx_Users", RuleType.REFERENTIAL_INTEGRITY, "transactions", "user_id",
                    {"parent_table": "users", "parent_column": "user_id"})
    )
    assert r_fk.passed is False
    assert r_fk.failed_records == 1


def test_quality_reporter(quality_test_db):
    checker = QualityChecker(quality_test_db)
    reporter = QualityReporter()

    rules = [
        QualityRule("UniqueUserID", RuleType.UNIQUE, "users", "user_id", severity=Severity.CRITICAL),
        QualityRule("UniqueEmail", RuleType.UNIQUE, "users", "email", severity=Severity.WARNING),
    ]

    results = checker.run_all_checks(rules)
    summary = reporter.generate_summary(results)

    assert summary["total_rules"] == 2
    assert summary["passed_rules"] == 1
    assert summary["failed_rules"] == 1
    assert summary["quality_score_pct"] == 50.0
    assert summary["status"] == "DEGRADED"

    json_str = reporter.to_json(results)
    assert "quality_score_pct" in json_str

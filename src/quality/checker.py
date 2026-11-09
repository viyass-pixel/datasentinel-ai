"""
DataSentinel AI - Quality Checker Engine
Executes data contract rules against SQL tables and returns detailed violation results.
"""
from typing import List
from sqlalchemy import text
from src.sql_engine.db import DatabaseManager
from src.quality.rules import QualityRule, QualityCheckResult, RuleType


class QualityChecker:
    """
    Executes rule-based data contract assertions on the database.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def run_check(self, rule: QualityRule) -> QualityCheckResult:
        """Executes a single quality rule."""
        if rule.rule_type == RuleType.NOT_NULL:
            return self._check_not_null(rule)
        elif rule.rule_type == RuleType.UNIQUE:
            return self._check_unique(rule)
        elif rule.rule_type == RuleType.VALUE_RANGE:
            return self._check_value_range(rule)
        elif rule.rule_type == RuleType.ALLOWED_VALUES:
            return self._check_allowed_values(rule)
        elif rule.rule_type == RuleType.TEMPORAL_ORDER:
            return self._check_temporal_order(rule)
        elif rule.rule_type == RuleType.REFERENTIAL_INTEGRITY:
            return self._check_referential_integrity(rule)
        else:
            raise ValueError(f"Unsupported rule type: {rule.rule_type}")

    def run_all_checks(self, rules: List[QualityRule]) -> List[QualityCheckResult]:
        """Executes a list of quality rules and returns all results."""
        return [self.run_check(r) for r in rules]

    def _check_not_null(self, rule: QualityRule) -> QualityCheckResult:
        query = f"""
        SELECT 
            COUNT(*) AS total_count,
            SUM(CASE WHEN {rule.column} IS NULL THEN 1 ELSE 0 END) AS null_count
        FROM {rule.table};
        """
        df = self.db.execute_query(query)
        total = int(df.iloc[0]["total_count"])
        nulls = int(df.iloc[0]["null_count"] or 0)
        rate = round((nulls / total) * 100, 2) if total > 0 else 0.0

        return QualityCheckResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            table=rule.table,
            column=rule.column,
            severity=rule.severity.value,
            passed=(nulls == 0),
            total_records=total,
            failed_records=nulls,
            failure_rate_pct=rate,
            message=f"{nulls} null records found in {rule.table}.{rule.column} ({rate}%)."
        )

    def _check_unique(self, rule: QualityRule) -> QualityCheckResult:
        query = f"""
        SELECT 
            COUNT(*) AS total_count,
            (COUNT(*) - COUNT(DISTINCT {rule.column})) AS duplicate_count
        FROM {rule.table}
        WHERE {rule.column} IS NOT NULL;
        """
        df = self.db.execute_query(query)
        total = int(df.iloc[0]["total_count"])
        dups = int(df.iloc[0]["duplicate_count"] or 0)
        rate = round((dups / total) * 100, 2) if total > 0 else 0.0

        return QualityCheckResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            table=rule.table,
            column=rule.column,
            severity=rule.severity.value,
            passed=(dups == 0),
            total_records=total,
            failed_records=dups,
            failure_rate_pct=rate,
            message=f"{dups} duplicates found in {rule.table}.{rule.column} ({rate}%)."
        )

    def _check_value_range(self, rule: QualityRule) -> QualityCheckResult:
        min_val = rule.params.get("min")
        max_val = rule.params.get("max")
        conditions = []
        if min_val is not None:
            conditions.append(f"{rule.column} < {min_val}")
        if max_val is not None:
            conditions.append(f"{rule.column} > {max_val}")
        
        where_clause = " OR ".join(conditions) if conditions else "1=0"
        query = f"""
        SELECT 
            COUNT(*) AS total_count,
            SUM(CASE WHEN ({where_clause}) THEN 1 ELSE 0 END) AS out_of_range_count
        FROM {rule.table}
        WHERE {rule.column} IS NOT NULL;
        """
        df = self.db.execute_query(query)
        total = int(df.iloc[0]["total_count"])
        failed = int(df.iloc[0]["out_of_range_count"] or 0)
        rate = round((failed / total) * 100, 2) if total > 0 else 0.0

        return QualityCheckResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            table=rule.table,
            column=rule.column,
            severity=rule.severity.value,
            passed=(failed == 0),
            total_records=total,
            failed_records=failed,
            failure_rate_pct=rate,
            message=f"{failed} out-of-range values in {rule.table}.{rule.column} [{min_val}, {max_val}] ({rate}%)."
        )

    def _check_allowed_values(self, rule: QualityRule) -> QualityCheckResult:
        allowed = rule.params.get("allowed", [])
        allowed_str = ", ".join(f"'{v}'" for v in allowed)
        query = f"""
        SELECT 
            COUNT(*) AS total_count,
            SUM(CASE WHEN {rule.column} NOT IN ({allowed_str}) THEN 1 ELSE 0 END) AS invalid_count
        FROM {rule.table}
        WHERE {rule.column} IS NOT NULL;
        """
        df = self.db.execute_query(query)
        total = int(df.iloc[0]["total_count"])
        failed = int(df.iloc[0]["invalid_count"] or 0)
        rate = round((failed / total) * 100, 2) if total > 0 else 0.0

        return QualityCheckResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            table=rule.table,
            column=rule.column,
            severity=rule.severity.value,
            passed=(failed == 0),
            total_records=total,
            failed_records=failed,
            failure_rate_pct=rate,
            message=f"{failed} records with unexpected values in {rule.table}.{rule.column} ({rate}%)."
        )

    def _check_temporal_order(self, rule: QualityRule) -> QualityCheckResult:
        col_start = rule.params.get("start_column")
        col_end = rule.params.get("end_column")
        query = f"""
        SELECT 
            COUNT(*) AS total_count,
            SUM(CASE WHEN julianday({col_end}) < julianday({col_start}) THEN 1 ELSE 0 END) AS order_violations
        FROM {rule.table}
        WHERE {col_start} IS NOT NULL AND {col_end} IS NOT NULL;
        """
        df = self.db.execute_query(query)
        total = int(df.iloc[0]["total_count"])
        failed = int(df.iloc[0]["order_violations"] or 0)
        rate = round((failed / total) * 100, 2) if total > 0 else 0.0

        return QualityCheckResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            table=rule.table,
            column=f"{col_start} -> {col_end}",
            severity=rule.severity.value,
            passed=(failed == 0),
            total_records=total,
            failed_records=failed,
            failure_rate_pct=rate,
            message=f"{failed} temporal violations where {col_end} occurred before {col_start} ({rate}%)."
        )

    def _check_referential_integrity(self, rule: QualityRule) -> QualityCheckResult:
        fk_col = rule.column
        parent_table = rule.params.get("parent_table")
        parent_col = rule.params.get("parent_column")

        query = f"""
        SELECT 
            COUNT(*) AS total_count,
            SUM(CASE WHEN p.{parent_col} IS NULL THEN 1 ELSE 0 END) AS orphaned_count
        FROM {rule.table} c
        LEFT JOIN {parent_table} p ON c.{fk_col} = p.{parent_col}
        WHERE c.{fk_col} IS NOT NULL;
        """
        df = self.db.execute_query(query)
        total = int(df.iloc[0]["total_count"])
        orphaned = int(df.iloc[0]["orphaned_count"] or 0)
        rate = round((orphaned / total) * 100, 2) if total > 0 else 0.0

        return QualityCheckResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            table=rule.table,
            column=fk_col,
            severity=rule.severity.value,
            passed=(orphaned == 0),
            total_records=total,
            failed_records=orphaned,
            failure_rate_pct=rate,
            message=f"{orphaned} orphaned records in {rule.table}.{fk_col} missing from {parent_table}.{parent_col} ({rate}%)."
        )

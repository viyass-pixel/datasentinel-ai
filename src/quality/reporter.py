"""
DataSentinel AI - Quality Report Generator
Generates structured JSON and readable Markdown summary reports with a Data Quality Score.
"""
import json
from datetime import datetime
from typing import Any, Dict, List
from src.quality.rules import QualityCheckResult


class QualityReporter:
    """
    Computes overall data quality scores and formats reports.
    """

    def generate_summary(self, results: List[QualityCheckResult]) -> Dict[str, Any]:
        """Calculates aggregate score and failure breakdown."""
        total_rules = len(results)
        passed_rules = sum(1 for r in results if r.passed)
        failed_rules = total_rules - passed_rules

        score_pct = round((passed_rules / total_rules) * 100, 1) if total_rules > 0 else 100.0

        critical_failures = sum(1 for r in results if not r.passed and r.severity == "CRITICAL")
        warning_failures = sum(1 for r in results if not r.passed and r.severity == "WARNING")

        return {
            "timestamp": datetime.now().isoformat(),
            "quality_score_pct": score_pct,
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "critical_failures": critical_failures,
            "warning_failures": warning_failures,
            "status": "HEALTHY" if failed_rules == 0 else ("CRITICAL" if critical_failures > 0 else "DEGRADED"),
        }

    def to_markdown(self, results: List[QualityCheckResult]) -> str:
        """Renders an executive summary report in Markdown table format."""
        summary = self.generate_summary(results)

        lines = [
            f"# Data Quality Report [{summary['status']}]",
            f"**Generated:** {summary['timestamp']} | **Status:** `{summary['status']}`\n",
            f"### **Overall Data Quality Score: {summary['quality_score_pct']}%**",
            f"- **Rules Evaluated:** {summary['total_rules']} | **Passed:** {summary['passed_rules']} | **Failed:** {summary['failed_rules']}",
            f"- **Critical Violations:** {summary['critical_failures']} | **Warnings:** {summary['warning_failures']}\n",
            "---",
            "### Rule Evaluation Details\n",
            "| Rule Name | Table | Type | Severity | Status | Violations | Failure Rate | Message |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in results:
            status_text = "PASS" if r.passed else f"FAIL ({r.severity})"
            lines.append(
                f"| `{r.rule_name}` | `{r.table}` | `{r.rule_type}` | `{r.severity}` | **{status_text}** | {r.failed_records:,} / {r.total_records:,} | {r.failure_rate_pct}% | {r.message} |"
            )

        return "\n".join(lines)

    def to_json(self, results: List[QualityCheckResult]) -> str:
        """Serializes the report to structured JSON."""
        summary = self.generate_summary(results)
        report = {
            "summary": summary,
            "results": [
                {
                    "rule_name": r.rule_name,
                    "rule_type": r.rule_type,
                    "table": r.table,
                    "column": r.column,
                    "severity": r.severity,
                    "passed": r.passed,
                    "total_records": r.total_records,
                    "failed_records": r.failed_records,
                    "failure_rate_pct": r.failure_rate_pct,
                    "message": r.message,
                }
                for r in results
            ],
        }
        return json.dumps(report, indent=2)

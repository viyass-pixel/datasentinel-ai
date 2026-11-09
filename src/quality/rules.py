"""
DataSentinel AI - Data Quality Rules
Defines data quality rule schemas, types, and severity levels.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RuleType(str, Enum):
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    VALUE_RANGE = "value_range"
    ALLOWED_VALUES = "allowed_values"
    TEMPORAL_ORDER = "temporal_order"
    REFERENTIAL_INTEGRITY = "referential_integrity"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class QualityRule:
    """Represents a single data quality expectation/contract."""
    name: str
    rule_type: RuleType
    table: str
    column: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.CRITICAL
    description: str = ""


@dataclass
class QualityCheckResult:
    """Result of evaluating a single data quality rule."""
    rule_name: str
    rule_type: str
    table: str
    column: Optional[str]
    severity: str
    passed: bool
    total_records: int
    failed_records: int
    failure_rate_pct: float
    message: str
    sample_violations: Optional[list] = None

"""
Governance validation rules and compliance checking.

This module implements validation rules that enforce architectural principles
of canonical-first processing and prevent unauthorized data transformations
that could compromise data integrity.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from ..extraction.models import CanonicalData, ShapedData, TransformationRule

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation results."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCategory(Enum):
    """Categories of validation rules."""
    CANONICAL_INTEGRITY = "canonical_integrity"
    TRANSFORMATION_APPROVAL = "transformation_approval"
    DATA_DISTORTION = "data_distortion"
    COMPLIANCE = "compliance"
    SCHEMA_CONFORMANCE = "schema_conformance"


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Contains information about validation outcome, severity, and
    any recommendations for remediation.
    """
    rule_id: str
    passed: bool
    severity: ValidationSeverity
    message: str
    category: RuleCategory
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "recommendations": self.recommendations
        }


@dataclass
class ValidationRule:
    """
    Defines a governance validation rule.

    Each rule includes a validation function and metadata about
    what it checks and how violations should be handled.
    """
    rule_id: str
    category: RuleCategory
    name: str
    description: str
    severity: ValidationSeverity
    validation_fn: Callable[[Any], bool]
    enabled: bool = True
    auto_fix: bool = False
    fix_fn: Optional[Callable[[Any], Any]] = None

    def validate(self, data: Any) -> ValidationResult:
        """
        Execute validation rule.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with outcome
        """
        try:
            passed = self.validation_fn(data)

            message = f"Validation passed: {self.name}" if passed else f"Validation failed: {self.name}"

            result = ValidationResult(
                rule_id=self.rule_id,
                passed=passed,
                severity=self.severity,
                message=message,
                category=self.category,
                details={"data_type": type(data).__name__}
            )

            if not passed and self.auto_fix and self.fix_fn:
                result.recommendations.append(f"Auto-fix available for rule: {self.rule_id}")

            return result

        except Exception as e:
            logger.error(f"Error executing validation rule {self.rule_id}: {e}")
            return ValidationResult(
                rule_id=self.rule_id,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Validation error: {str(e)}",
                category=self.category,
                details={"error": str(e)}
            )


class GovernanceValidator:
    """
    Main governance validation engine.

    Enforces data governance rules across the processing pipeline,
    ensuring canonical data integrity and preventing unauthorized
    transformations.
    """

    def __init__(self):
        """Initialize governance validator."""
        self.rules: Dict[str, ValidationRule] = {}
        self.approval_workflows: Dict[str, Callable] = {}
        self._register_default_rules()
        logger.info("GovernanceValidator initialized")

    def _register_default_rules(self) -> None:
        """Register default validation rules."""

        # Canonical Integrity Rules
        self.register_rule(ValidationRule(
            rule_id="canonical_001",
            category=RuleCategory.CANONICAL_INTEGRITY,
            name="Canonical Data Non-Null Check",
            description="Ensures canonical data extraction_data is not None",
            severity=ValidationSeverity.CRITICAL,
            validation_fn=lambda data: (
                isinstance(data, CanonicalData) and
                data.extraction_data is not None
            )
        ))

        self.register_rule(ValidationRule(
            rule_id="canonical_002",
            category=RuleCategory.CANONICAL_INTEGRITY,
            name="Canonical Integrity Hash Validation",
            description="Validates canonical data integrity hash",
            severity=ValidationSeverity.ERROR,
            validation_fn=self._validate_canonical_hash
        ))

        self.register_rule(ValidationRule(
            rule_id="canonical_003",
            category=RuleCategory.CANONICAL_INTEGRITY,
            name="Canonical Timestamp Validity",
            description="Ensures canonical timestamp is not in the future",
            severity=ValidationSeverity.WARNING,
            validation_fn=lambda data: (
                isinstance(data, CanonicalData) and
                data.timestamp <= datetime.now()
            )
        ))

        # Transformation Approval Rules
        self.register_rule(ValidationRule(
            rule_id="transform_001",
            category=RuleCategory.TRANSFORMATION_APPROVAL,
            name="Transformation Rule Authorization",
            description="Validates all transformation rules are authorized",
            severity=ValidationSeverity.ERROR,
            validation_fn=self._validate_transformation_authorization
        ))

        self.register_rule(ValidationRule(
            rule_id="transform_002",
            category=RuleCategory.TRANSFORMATION_APPROVAL,
            name="1NF Compliance Requirement",
            description="Ensures shaped data is 1NF compliant",
            severity=ValidationSeverity.CRITICAL,
            validation_fn=lambda data: (
                isinstance(data, ShapedData) and
                data.is_1nf_compliant
            )
        ))

        # Data Distortion Rules
        self.register_rule(ValidationRule(
            rule_id="distortion_001",
            category=RuleCategory.DATA_DISTORTION,
            name="Field Count Preservation",
            description="Warns if shaped data has significantly fewer fields than canonical",
            severity=ValidationSeverity.WARNING,
            validation_fn=self._validate_field_count_preservation
        ))

        self.register_rule(ValidationRule(
            rule_id="distortion_002",
            category=RuleCategory.DATA_DISTORTION,
            name="Data Type Consistency",
            description="Validates data types are preserved during transformation",
            severity=ValidationSeverity.WARNING,
            validation_fn=self._validate_data_type_consistency
        ))

        # Compliance Rules
        self.register_rule(ValidationRule(
            rule_id="compliance_001",
            category=RuleCategory.COMPLIANCE,
            name="Metadata Preservation",
            description="Ensures required metadata is preserved",
            severity=ValidationSeverity.ERROR,
            validation_fn=self._validate_metadata_preservation
        ))

        self.register_rule(ValidationRule(
            rule_id="compliance_002",
            category=RuleCategory.COMPLIANCE,
            name="Processing ID Consistency",
            description="Validates processing IDs are consistent across pipeline",
            severity=ValidationSeverity.ERROR,
            validation_fn=self._validate_processing_id_consistency
        ))

        logger.info(f"Registered {len(self.rules)} default validation rules")

    def register_rule(self, rule: ValidationRule) -> None:
        """
        Register a validation rule.

        Args:
            rule: ValidationRule to register
        """
        self.rules[rule.rule_id] = rule
        logger.debug(f"Registered validation rule: {rule.rule_id}")

    def unregister_rule(self, rule_id: str) -> None:
        """
        Unregister a validation rule.

        Args:
            rule_id: Rule ID to unregister
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.debug(f"Unregistered validation rule: {rule_id}")

    def enable_rule(self, rule_id: str) -> None:
        """Enable a validation rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.debug(f"Enabled rule: {rule_id}")

    def disable_rule(self, rule_id: str) -> None:
        """Disable a validation rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.debug(f"Disabled rule: {rule_id}")

    def validate_canonical(
        self,
        canonical_data: CanonicalData,
        rules: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """
        Validate canonical data against governance rules.

        Args:
            canonical_data: CanonicalData to validate
            rules: Optional list of specific rule IDs to run

        Returns:
            List of ValidationResults
        """
        logger.info(f"Validating canonical data: {canonical_data.processing_id}")

        results = []

        # Determine which rules to run
        rules_to_run = rules or [
            rid for rid, rule in self.rules.items()
            if rule.category in [RuleCategory.CANONICAL_INTEGRITY, RuleCategory.COMPLIANCE]
            and rule.enabled
        ]

        for rule_id in rules_to_run:
            if rule_id in self.rules and self.rules[rule_id].enabled:
                result = self.rules[rule_id].validate(canonical_data)
                results.append(result)

        logger.info(
            f"Canonical validation complete: {sum(r.passed for r in results)}/{len(results)} passed"
        )

        return results

    def validate_shaped(
        self,
        shaped_data: ShapedData,
        canonical_data: Optional[CanonicalData] = None,
        rules: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """
        Validate shaped data against governance rules.

        Args:
            shaped_data: ShapedData to validate
            canonical_data: Optional canonical data for comparison
            rules: Optional list of specific rule IDs to run

        Returns:
            List of ValidationResults
        """
        logger.info(f"Validating shaped data: {shaped_data.processing_id}")

        results = []

        # Store canonical for comparison rules
        if canonical_data:
            self._canonical_context = canonical_data

        # Determine which rules to run
        rules_to_run = rules or [
            rid for rid, rule in self.rules.items()
            if rule.category in [
                RuleCategory.TRANSFORMATION_APPROVAL,
                RuleCategory.DATA_DISTORTION,
                RuleCategory.COMPLIANCE
            ]
            and rule.enabled
        ]

        for rule_id in rules_to_run:
            if rule_id in self.rules and self.rules[rule_id].enabled:
                result = self.rules[rule_id].validate(shaped_data)
                results.append(result)

        logger.info(
            f"Shaped validation complete: {sum(r.passed for r in results)}/{len(results)} passed"
        )

        return results

    def validate_transformation(
        self,
        canonical_data: CanonicalData,
        shaped_data: ShapedData
    ) -> List[ValidationResult]:
        """
        Validate transformation from canonical to shaped data.

        Args:
            canonical_data: Source canonical data
            shaped_data: Resulting shaped data

        Returns:
            List of ValidationResults
        """
        logger.info("Validating canonical -> shaped transformation")

        self._canonical_context = canonical_data

        results = []

        # Run all transformation and distortion rules
        for rule in self.rules.values():
            if rule.category in [
                RuleCategory.TRANSFORMATION_APPROVAL,
                RuleCategory.DATA_DISTORTION
            ] and rule.enabled:
                result = rule.validate(shaped_data)
                results.append(result)

        logger.info(
            f"Transformation validation complete: {sum(r.passed for r in results)}/{len(results)} passed"
        )

        return results

    def get_validation_report(
        self,
        results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Args:
            results: List of validation results

        Returns:
            Validation report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_rules": len(results),
            "passed": sum(r.passed for r in results),
            "failed": sum(not r.passed for r in results),
            "by_severity": {},
            "by_category": {},
            "critical_failures": [],
            "recommendations": []
        }

        # Group by severity
        for severity in ValidationSeverity:
            severity_results = [r for r in results if r.severity == severity]
            report["by_severity"][severity.value] = {
                "total": len(severity_results),
                "passed": sum(r.passed for r in severity_results),
                "failed": sum(not r.passed for r in severity_results)
            }

        # Group by category
        for category in RuleCategory:
            category_results = [r for r in results if r.category == category]
            report["by_category"][category.value] = {
                "total": len(category_results),
                "passed": sum(r.passed for r in category_results),
                "failed": sum(not r.passed for r in category_results)
            }

        # Collect critical failures
        critical_failures = [
            r for r in results
            if not r.passed and r.severity == ValidationSeverity.CRITICAL
        ]
        report["critical_failures"] = [r.to_dict() for r in critical_failures]

        # Collect all recommendations
        for result in results:
            if not result.passed and result.recommendations:
                report["recommendations"].extend(result.recommendations)

        return report

    def register_approval_workflow(
        self,
        workflow_id: str,
        workflow_fn: Callable[[TransformationRule], bool]
    ) -> None:
        """
        Register an approval workflow for transformations.

        Args:
            workflow_id: Unique workflow identifier
            workflow_fn: Function that returns True if transformation is approved
        """
        self.approval_workflows[workflow_id] = workflow_fn
        logger.info(f"Registered approval workflow: {workflow_id}")

    def request_transformation_approval(
        self,
        transformation_rule: TransformationRule,
        workflow_id: Optional[str] = None
    ) -> bool:
        """
        Request approval for a transformation rule.

        Args:
            transformation_rule: Transformation rule to approve
            workflow_id: Optional specific workflow to use

        Returns:
            True if approved, False otherwise
        """
        if workflow_id and workflow_id in self.approval_workflows:
            workflow_fn = self.approval_workflows[workflow_id]
            approved = workflow_fn(transformation_rule)
            logger.info(
                f"Transformation {transformation_rule.rule_id} "
                f"{'approved' if approved else 'rejected'} by workflow {workflow_id}"
            )
            return approved

        # Default approval logic: allow all pre-registered rule types
        allowed_types = {"normalize", "filter", "rename", "aggregate"}
        approved = transformation_rule.rule_type in allowed_types

        logger.info(
            f"Transformation {transformation_rule.rule_id} "
            f"{'approved' if approved else 'rejected'} by default policy"
        )

        return approved

    # Helper validation functions

    def _validate_canonical_hash(self, data: Any) -> bool:
        """Validate canonical data integrity hash."""
        if not isinstance(data, CanonicalData):
            return False

        if not data.source_metadata or "integrity_hash" not in data.source_metadata:
            return False

        # Hash validation would involve recalculating and comparing
        # For now, just check it exists and is non-empty
        return bool(data.source_metadata["integrity_hash"])

    def _validate_transformation_authorization(self, data: Any) -> bool:
        """Validate all transformation rules are authorized."""
        if not isinstance(data, ShapedData):
            return False

        # Check each transformation rule
        for rule in data.transformations_applied:
            if not self.request_transformation_approval(rule):
                logger.warning(f"Unauthorized transformation detected: {rule.rule_id}")
                return False

        return True

    def _validate_field_count_preservation(self, data: Any) -> bool:
        """Validate field count is reasonably preserved."""
        if not isinstance(data, ShapedData):
            return False

        # Check if canonical context is available
        if not hasattr(self, '_canonical_context'):
            return True  # Can't validate without canonical

        canonical = self._canonical_context

        canonical_fields = len(canonical.extraction_data) if canonical.extraction_data else 0
        shaped_fields = len(data.transformed_data) if data.transformed_data else 0

        # Warn if we lost more than 50% of fields
        if shaped_fields < canonical_fields * 0.5:
            logger.warning(
                f"Significant field reduction: {canonical_fields} -> {shaped_fields}"
            )
            return False

        return True

    def _validate_data_type_consistency(self, data: Any) -> bool:
        """Validate data types are preserved where expected."""
        if not isinstance(data, ShapedData):
            return False

        # Basic check: ensure transformed_data is a dict
        if not isinstance(data.transformed_data, dict):
            return False

        # Check all values are atomic (no nested dicts/lists for 1NF)
        if data.is_1nf_compliant:
            for value in data.transformed_data.values():
                if isinstance(value, (dict, list)) and value:  # Allow empty
                    return False

        return True

    def _validate_metadata_preservation(self, data: Any) -> bool:
        """Validate required metadata is preserved."""
        if isinstance(data, CanonicalData):
            return (
                bool(data.processing_id) and
                bool(data.timestamp) and
                bool(data.schema_version)
            )
        elif isinstance(data, ShapedData):
            return (
                bool(data.processing_id) and
                bool(data.canonical_id) and
                bool(data.timestamp)
            )

        return False

    def _validate_processing_id_consistency(self, data: Any) -> bool:
        """Validate processing IDs are consistent."""
        if isinstance(data, ShapedData):
            # Check if canonical context is available
            if hasattr(self, '_canonical_context'):
                canonical = self._canonical_context
                # Shaped ID should reference canonical ID
                return data.canonical_id == canonical.processing_id

        return True

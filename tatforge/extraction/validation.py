"""
Extraction Result Validation System - COLPALI-602.

This module provides comprehensive validation for extraction results, ensuring
outputs conform to declared schemas and meet quality standards before downstream
processing. Includes schema conformance, data quality checks, missing field
detection, type validation, and detailed validation reporting.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

try:
    from pydantic import BaseModel, ValidationError, validator, create_model
    from pydantic.fields import FieldInfo
except ImportError:
    # Fallback if Pydantic not available
    BaseModel = None
    ValidationError = None

from ..core.schema_manager import BAMLDefinition, BAMLClass, BAMLFunction

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks."""
    SCHEMA_CONFORMANCE = "schema_conformance"
    TYPE_VALIDATION = "type_validation"
    REQUIRED_FIELDS = "required_fields"
    DATA_QUALITY = "data_quality"
    BUSINESS_RULES = "business_rules"
    FORMAT_VALIDATION = "format_validation"


@dataclass
class ValidationIssue:
    """Individual validation issue with details."""
    validation_type: ValidationType
    severity: ValidationSeverity
    field_path: str
    message: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    is_valid: bool
    validation_timestamp: datetime
    total_issues: int
    issues_by_severity: Dict[ValidationSeverity, int]
    issues: List[ValidationIssue]
    quality_score: float  # 0-1 score based on validation results
    schema_compliance_score: float
    data_completeness_score: float
    type_accuracy_score: float
    validation_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical validation issues."""
        return self.issues_by_severity.get(ValidationSeverity.CRITICAL, 0) > 0

    @property
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return self.issues_by_severity.get(ValidationSeverity.ERROR, 0) > 0

    def get_issues_by_type(self, validation_type: ValidationType) -> List[ValidationIssue]:
        """Get all issues of a specific validation type."""
        return [issue for issue in self.issues if issue.validation_type == validation_type]

    def get_issues_by_field(self, field_path: str) -> List[ValidationIssue]:
        """Get all issues for a specific field."""
        return [issue for issue in self.issues if issue.field_path == field_path]


class BaseValidator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate data and return list of issues.

        Args:
            data: Data to validate
            context: Validation context with schema and metadata

        Returns:
            List of validation issues
        """
        pass


class SchemaConformanceValidator(BaseValidator):
    """Validator for schema conformance using BAML definitions."""

    def __init__(self, baml_definition: Optional[BAMLDefinition] = None):
        """Initialize with optional BAML definition."""
        self.baml_definition = baml_definition

    async def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data against BAML schema definition."""
        issues = []

        # Get expected schema from context or stored definition
        expected_schema = context.get('expected_schema', {})
        baml_function = context.get('baml_function')

        if not expected_schema and not baml_function:
            issues.append(ValidationIssue(
                validation_type=ValidationType.SCHEMA_CONFORMANCE,
                severity=ValidationSeverity.WARNING,
                field_path="root",
                message="No schema definition available for validation"
            ))
            return issues

        # Validate required fields
        if baml_function and hasattr(baml_function, 'return_type'):
            issues.extend(await self._validate_return_type(data, baml_function.return_type))

        # Validate against expected schema structure
        if expected_schema:
            issues.extend(await self._validate_schema_structure(data, expected_schema, ""))

        return issues

    async def _validate_return_type(self, data: Dict[str, Any], return_type: str) -> List[ValidationIssue]:
        """Validate data against expected return type."""
        issues = []

        # Basic return type validation
        if return_type.lower() in ['object', 'dict'] and not isinstance(data, dict):
            issues.append(ValidationIssue(
                validation_type=ValidationType.SCHEMA_CONFORMANCE,
                severity=ValidationSeverity.ERROR,
                field_path="root",
                message=f"Expected object/dict, got {type(data).__name__}",
                expected_value="object",
                actual_value=type(data).__name__
            ))

        return issues

    async def _validate_schema_structure(
        self,
        data: Dict[str, Any],
        expected_schema: Dict[str, Any],
        path_prefix: str
    ) -> List[ValidationIssue]:
        """Recursively validate data structure against schema."""
        issues = []

        # Check for required fields
        required_fields = expected_schema.get('required', [])
        for field in required_fields:
            field_path = f"{path_prefix}.{field}" if path_prefix else field
            if field not in data:
                issues.append(ValidationIssue(
                    validation_type=ValidationType.REQUIRED_FIELDS,
                    severity=ValidationSeverity.ERROR,
                    field_path=field_path,
                    message=f"Required field '{field}' is missing",
                    suggestion=f"Add field '{field}' to the extracted data"
                ))

        # Check field types in properties
        properties = expected_schema.get('properties', {})
        for field_name, field_schema in properties.items():
            field_path = f"{path_prefix}.{field_name}" if path_prefix else field_name

            if field_name in data:
                field_value = data[field_name]
                expected_type = field_schema.get('type', 'string')

                # Type validation
                type_issues = await self._validate_field_type(
                    field_value, expected_type, field_path
                )
                issues.extend(type_issues)

                # Recursive validation for nested objects
                if expected_type == 'object' and isinstance(field_value, dict):
                    nested_issues = await self._validate_schema_structure(
                        field_value, field_schema, field_path
                    )
                    issues.extend(nested_issues)

                # Array validation
                elif expected_type == 'array' and isinstance(field_value, list):
                    array_issues = await self._validate_array_items(
                        field_value, field_schema.get('items', {}), field_path
                    )
                    issues.extend(array_issues)

        return issues

    async def _validate_field_type(
        self,
        value: Any,
        expected_type: str,
        field_path: str
    ) -> List[ValidationIssue]:
        """Validate individual field type."""
        issues = []

        type_mapping = {
            'string': str,
            'int': int,
            'integer': int,
            'float': float,
            'number': (int, float),
            'bool': bool,
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_python_type = type_mapping.get(expected_type.lower())
        if expected_python_type and not isinstance(value, expected_python_type):
            issues.append(ValidationIssue(
                validation_type=ValidationType.TYPE_VALIDATION,
                severity=ValidationSeverity.ERROR,
                field_path=field_path,
                message=f"Type mismatch: expected {expected_type}, got {type(value).__name__}",
                expected_value=expected_type,
                actual_value=type(value).__name__,
                suggestion=f"Convert value to {expected_type}"
            ))

        return issues

    async def _validate_array_items(
        self,
        array_value: List[Any],
        items_schema: Dict[str, Any],
        field_path: str
    ) -> List[ValidationIssue]:
        """Validate array items against item schema."""
        issues = []

        for i, item in enumerate(array_value):
            item_path = f"{field_path}[{i}]"

            if items_schema.get('type') == 'object':
                item_issues = await self._validate_schema_structure(
                    item, items_schema, item_path
                )
                issues.extend(item_issues)
            else:
                # Validate item type
                item_type_issues = await self._validate_field_type(
                    item, items_schema.get('type', 'string'), item_path
                )
                issues.extend(item_type_issues)

        return issues


class DataQualityValidator(BaseValidator):
    """Validator for data quality checks and sanity validation."""

    def __init__(self):
        """Initialize data quality validator."""
        self.quality_rules = self._get_quality_rules()

    async def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data quality and apply sanity checks."""
        issues = []

        # Check for empty or null values
        issues.extend(await self._check_empty_values(data))

        # Check for data consistency
        issues.extend(await self._check_data_consistency(data))

        # Check for suspicious patterns
        issues.extend(await self._check_suspicious_patterns(data))

        # Apply document-specific quality rules
        document_type = context.get('document_type')
        if document_type:
            issues.extend(await self._apply_document_specific_rules(data, document_type))

        return issues

    async def _check_empty_values(self, data: Dict[str, Any], path_prefix: str = "") -> List[ValidationIssue]:
        """Check for empty, null, or meaningless values."""
        issues = []

        for key, value in data.items():
            field_path = f"{path_prefix}.{key}" if path_prefix else key

            # Check for various empty states
            if value is None:
                issues.append(ValidationIssue(
                    validation_type=ValidationType.DATA_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    field_path=field_path,
                    message="Field contains null value",
                    suggestion="Verify if this field should have a value"
                ))
            elif isinstance(value, str) and not value.strip():
                issues.append(ValidationIssue(
                    validation_type=ValidationType.DATA_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    field_path=field_path,
                    message="Field contains empty string",
                    suggestion="Consider if this field should be populated"
                ))
            elif isinstance(value, list) and len(value) == 0:
                issues.append(ValidationIssue(
                    validation_type=ValidationType.DATA_QUALITY,
                    severity=ValidationSeverity.INFO,
                    field_path=field_path,
                    message="Field contains empty array"
                ))
            elif isinstance(value, dict):
                # Recursive check for nested objects
                nested_issues = await self._check_empty_values(value, field_path)
                issues.extend(nested_issues)

        return issues

    async def _check_data_consistency(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for data consistency issues."""
        issues = []

        # Check for duplicate values where they shouldn't exist
        string_values = []
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 5:  # Only check meaningful strings
                if value in string_values:
                    issues.append(ValidationIssue(
                        validation_type=ValidationType.DATA_QUALITY,
                        severity=ValidationSeverity.WARNING,
                        field_path=key,
                        message=f"Duplicate value found: '{value}'",
                        suggestion="Verify if duplicate values are expected"
                    ))
                string_values.append(value)

        # Check for inconsistent formatting
        date_fields = [k for k in data.keys() if 'date' in k.lower()]
        for field in date_fields:
            if field in data and isinstance(data[field], str):
                if not self._is_valid_date_format(data[field]):
                    issues.append(ValidationIssue(
                        validation_type=ValidationType.FORMAT_VALIDATION,
                        severity=ValidationSeverity.WARNING,
                        field_path=field,
                        message=f"Inconsistent date format: '{data[field]}'",
                        suggestion="Use standard date format (YYYY-MM-DD)"
                    ))

        return issues

    async def _check_suspicious_patterns(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for suspicious data patterns."""
        issues = []

        for key, value in data.items():
            if isinstance(value, str):
                # Check for placeholder text
                placeholder_patterns = [
                    r'^(xxx+|aaa+|bbb+|ccc+)$',
                    r'^(test|sample|placeholder|dummy)$',
                    r'^n/?a$',
                    r'^(tbd|tba)$'
                ]

                for pattern in placeholder_patterns:
                    if re.match(pattern, value.lower().strip()):
                        issues.append(ValidationIssue(
                            validation_type=ValidationType.DATA_QUALITY,
                            severity=ValidationSeverity.WARNING,
                            field_path=key,
                            message=f"Suspicious placeholder value: '{value}'",
                            suggestion="Verify if this is actual extracted data"
                        ))
                        break

                # Check for extraction errors
                error_patterns = [
                    r'error|failed|exception|null|undefined',
                    r'could\s+not\s+extract',
                    r'unable\s+to\s+process'
                ]

                for pattern in error_patterns:
                    if re.search(pattern, value.lower()):
                        issues.append(ValidationIssue(
                            validation_type=ValidationType.DATA_QUALITY,
                            severity=ValidationSeverity.ERROR,
                            field_path=key,
                            message=f"Possible extraction error in value: '{value[:50]}...'",
                            suggestion="Review extraction process for this field"
                        ))
                        break

        return issues

    async def _apply_document_specific_rules(
        self,
        data: Dict[str, Any],
        document_type: str
    ) -> List[ValidationIssue]:
        """Apply document type specific validation rules."""
        issues = []

        document_rules = self.quality_rules.get(document_type.lower(), {})

        for field_pattern, rules in document_rules.items():
            matching_fields = [k for k in data.keys() if field_pattern in k.lower()]

            for field in matching_fields:
                if field in data:
                    field_issues = await self._apply_field_rules(
                        data[field], field, rules
                    )
                    issues.extend(field_issues)

        return issues

    async def _apply_field_rules(
        self,
        value: Any,
        field_path: str,
        rules: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Apply specific validation rules to a field."""
        issues = []

        # Minimum length rule
        if 'min_length' in rules and isinstance(value, str):
            if len(value) < rules['min_length']:
                issues.append(ValidationIssue(
                    validation_type=ValidationType.BUSINESS_RULES,
                    severity=ValidationSeverity.WARNING,
                    field_path=field_path,
                    message=f"Field too short: {len(value)} chars, minimum {rules['min_length']}",
                    suggestion=f"Expected at least {rules['min_length']} characters"
                ))

        # Pattern matching rule
        if 'pattern' in rules and isinstance(value, str):
            if not re.match(rules['pattern'], value):
                issues.append(ValidationIssue(
                    validation_type=ValidationType.FORMAT_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    field_path=field_path,
                    message=f"Field does not match expected pattern",
                    suggestion=f"Value should match pattern: {rules['pattern']}"
                ))

        # Range validation for numbers
        if isinstance(value, (int, float)):
            if 'min_value' in rules and value < rules['min_value']:
                issues.append(ValidationIssue(
                    validation_type=ValidationType.BUSINESS_RULES,
                    severity=ValidationSeverity.ERROR,
                    field_path=field_path,
                    message=f"Value {value} below minimum {rules['min_value']}"
                ))

            if 'max_value' in rules and value > rules['max_value']:
                issues.append(ValidationIssue(
                    validation_type=ValidationType.BUSINESS_RULES,
                    severity=ValidationSeverity.ERROR,
                    field_path=field_path,
                    message=f"Value {value} above maximum {rules['max_value']}"
                ))

        return issues

    def _get_quality_rules(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get document-specific quality rules."""
        return {
            'invoice': {
                'total': {'min_value': 0, 'max_value': 1000000},
                'invoice_number': {'min_length': 3, 'pattern': r'^[A-Z0-9-]+$'},
                'date': {'pattern': r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}'}
            },
            'receipt': {
                'total': {'min_value': 0, 'max_value': 10000},
                'store': {'min_length': 2}
            },
            'contract': {
                'amount': {'min_value': 0},
                'party': {'min_length': 5}
            }
        }

    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if string matches common date formats."""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or M/D/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\d{2}/\d{2}/\d{2}',  # MM/DD/YY
        ]

        return any(re.match(pattern, date_str.strip()) for pattern in date_patterns)


class ExtractionResultValidator:
    """
    Main extraction result validator that orchestrates all validation checks.

    Combines schema conformance, data quality, and business rule validation
    to provide comprehensive extraction result validation with detailed reporting.
    """

    def __init__(self, baml_definition: Optional[BAMLDefinition] = None):
        """
        Initialize extraction result validator.

        Args:
            baml_definition: Optional BAML definition for schema validation
        """
        self.baml_definition = baml_definition
        self.validators = [
            SchemaConformanceValidator(baml_definition),
            DataQualityValidator()
        ]

    async def validate_extraction_result(
        self,
        extraction_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """
        Perform comprehensive validation of extraction results.

        Args:
            extraction_data: Extracted data to validate
            context: Additional context for validation (schema, document type, etc.)

        Returns:
            Comprehensive validation report with issues and scores
        """
        validation_context = context or {}
        all_issues = []

        logger.debug(f"Starting validation for {len(extraction_data)} extracted fields")

        # Run all validators
        for validator in self.validators:
            try:
                validator_issues = await validator.validate(extraction_data, validation_context)
                all_issues.extend(validator_issues)
                logger.debug(f"{validator.__class__.__name__} found {len(validator_issues)} issues")
            except Exception as e:
                logger.error(f"Validator {validator.__class__.__name__} failed: {e}")
                # Add critical issue for validator failure
                all_issues.append(ValidationIssue(
                    validation_type=ValidationType.SCHEMA_CONFORMANCE,
                    severity=ValidationSeverity.CRITICAL,
                    field_path="validator",
                    message=f"Validation failed: {str(e)}"
                ))

        # Calculate severity distribution
        issues_by_severity = {}
        for severity in ValidationSeverity:
            issues_by_severity[severity] = sum(
                1 for issue in all_issues if issue.severity == severity
            )

        # Calculate quality scores
        total_fields = len(extraction_data) if extraction_data else 1

        schema_compliance_score = self._calculate_schema_compliance_score(
            all_issues, total_fields
        )
        data_completeness_score = self._calculate_completeness_score(
            extraction_data, all_issues
        )
        type_accuracy_score = self._calculate_type_accuracy_score(
            all_issues, total_fields
        )

        # Overall quality score (weighted average)
        quality_score = (
            schema_compliance_score * 0.4 +
            data_completeness_score * 0.3 +
            type_accuracy_score * 0.3
        )

        # Determine overall validation status
        is_valid = (
            issues_by_severity.get(ValidationSeverity.CRITICAL, 0) == 0 and
            issues_by_severity.get(ValidationSeverity.ERROR, 0) == 0 and
            quality_score >= 0.7
        )

        validation_report = ValidationReport(
            is_valid=is_valid,
            validation_timestamp=datetime.now(),
            total_issues=len(all_issues),
            issues_by_severity=issues_by_severity,
            issues=all_issues,
            quality_score=quality_score,
            schema_compliance_score=schema_compliance_score,
            data_completeness_score=data_completeness_score,
            type_accuracy_score=type_accuracy_score,
            validation_metadata={
                'total_fields_validated': total_fields,
                'validators_used': [v.__class__.__name__ for v in self.validators],
                'validation_context': validation_context
            }
        )

        logger.info(
            f"Validation completed: {'PASSED' if is_valid else 'FAILED'}, "
            f"quality_score={quality_score:.2f}, "
            f"issues={len(all_issues)}"
        )

        return validation_report

    def _calculate_schema_compliance_score(
        self,
        issues: List[ValidationIssue],
        total_fields: int
    ) -> float:
        """Calculate schema compliance score (0-1)."""
        schema_issues = [
            issue for issue in issues
            if issue.validation_type in [
                ValidationType.SCHEMA_CONFORMANCE,
                ValidationType.TYPE_VALIDATION,
                ValidationType.REQUIRED_FIELDS
            ]
        ]

        if not schema_issues:
            return 1.0

        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.ERROR: 0.8,
            ValidationSeverity.WARNING: 0.4,
            ValidationSeverity.INFO: 0.1
        }

        total_penalty = sum(
            severity_weights.get(issue.severity, 0.5) for issue in schema_issues
        )

        # Normalize by total fields
        normalized_penalty = total_penalty / total_fields
        compliance_score = max(0.0, 1.0 - normalized_penalty)

        return compliance_score

    def _calculate_completeness_score(
        self,
        extraction_data: Dict[str, Any],
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate data completeness score (0-1)."""
        if not extraction_data:
            return 0.0

        # Count empty/missing value issues
        empty_value_issues = [
            issue for issue in issues
            if "empty" in issue.message.lower() or "null" in issue.message.lower()
        ]

        total_fields = len(extraction_data)
        non_empty_fields = sum(
            1 for value in extraction_data.values()
            if value is not None and str(value).strip()
        )

        base_completeness = non_empty_fields / total_fields if total_fields > 0 else 0.0

        # Apply penalty for empty value issues
        penalty = len(empty_value_issues) * 0.1
        completeness_score = max(0.0, base_completeness - penalty)

        return completeness_score

    def _calculate_type_accuracy_score(
        self,
        issues: List[ValidationIssue],
        total_fields: int
    ) -> float:
        """Calculate type accuracy score (0-1)."""
        type_issues = [
            issue for issue in issues
            if issue.validation_type == ValidationType.TYPE_VALIDATION
        ]

        if not type_issues:
            return 1.0

        # Calculate accuracy based on type validation issues
        accuracy_score = max(0.0, 1.0 - (len(type_issues) / total_fields))
        return accuracy_score

    async def validate_with_custom_rules(
        self,
        extraction_data: Dict[str, Any],
        custom_validators: List[BaseValidator],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """
        Validate extraction results with additional custom validators.

        Args:
            extraction_data: Extracted data to validate
            custom_validators: Additional validators to apply
            context: Validation context

        Returns:
            Validation report including custom validator results
        """
        # Temporarily add custom validators
        original_validators = self.validators.copy()
        self.validators.extend(custom_validators)

        try:
            # Run validation with all validators
            report = await self.validate_extraction_result(extraction_data, context)
            return report
        finally:
            # Restore original validators
            self.validators = original_validators


# Factory function for easy validator creation
def create_extraction_validator(
    baml_definition: Optional[BAMLDefinition] = None,
    document_type: Optional[str] = None
) -> ExtractionResultValidator:
    """
    Create extraction result validator with optional configuration.

    Args:
        baml_definition: BAML definition for schema validation
        document_type: Document type for specialized validation rules

    Returns:
        Configured extraction result validator
    """
    validator = ExtractionResultValidator(baml_definition)

    # Add document-specific validators if needed
    if document_type:
        # This could be extended with document-specific validators
        pass

    return validator
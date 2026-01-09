"""
Schema Validation and Compatibility - COLPALI-503 implementation.

This module provides comprehensive validation of JSON schemas for BAML compatibility
with detailed error messages, fix suggestions, and schema migration support.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import jsonschema
from packaging import version

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CompatibilityLevel(Enum):
    """BAML compatibility levels."""
    FULLY_COMPATIBLE = "fully_compatible"
    COMPATIBLE_WITH_WARNINGS = "compatible_with_warnings"
    LIMITED_COMPATIBILITY = "limited_compatibility"
    INCOMPATIBLE = "incompatible"


@dataclass
class ValidationIssue:
    """Represents a validation issue with detailed context."""
    severity: ValidationSeverity
    code: str
    message: str
    field_path: str
    suggested_fix: Optional[str] = None
    documentation_link: Optional[str] = None
    auto_fixable: bool = False
    migration_strategy: Optional[str] = None


@dataclass
class CompatibilityReport:
    """Comprehensive compatibility assessment."""
    compatibility_level: CompatibilityLevel
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    info: List[ValidationIssue]
    schema_version: Optional[str] = None
    baml_version_requirement: str = ">=0.215.2"
    migration_required: bool = False
    estimated_effort: str = "low"  # low, medium, high


@dataclass
class MigrationStep:
    """Schema migration step."""
    step_number: int
    description: str
    from_pattern: str
    to_pattern: str
    auto_applicable: bool = False
    validation_function: Optional[str] = None


class SchemaValidator:
    """
    Comprehensive JSON schema validator for BAML compatibility.

    Provides detailed validation, error analysis, fix suggestions,
    and automated schema migration capabilities.
    """

    def __init__(self, baml_version: str = "0.216.0"):
        self.baml_version = baml_version
        self.supported_types = self._initialize_supported_types()
        self.validation_rules = self._initialize_validation_rules()
        self.fix_suggestions = self._initialize_fix_suggestions()
        self.migration_strategies = self._initialize_migration_strategies()

        logger.info(f"SchemaValidator initialized for BAML {baml_version}")

    def validate_schema(self, schema: Dict[str, Any]) -> CompatibilityReport:
        """
        Perform comprehensive schema validation and compatibility assessment.

        Args:
            schema: JSON schema to validate

        Returns:
            Detailed compatibility report with issues and suggestions
        """
        try:
            logger.info("Starting comprehensive schema validation")

            issues = []
            warnings = []
            info = []

            # Basic structure validation
            self._validate_basic_structure(schema, issues)

            # Type system compatibility
            self._validate_type_system(schema, issues, warnings)

            # BAML-specific constraints
            self._validate_baml_constraints(schema, issues, warnings)

            # Performance and optimization checks
            self._validate_performance_implications(schema, warnings, info)

            # Version compatibility
            self._validate_version_compatibility(schema, warnings, info)

            # Determine overall compatibility
            compatibility_level = self._determine_compatibility_level(issues, warnings)

            # Check if migration is needed
            migration_required = self._check_migration_required(schema, issues)

            # Estimate implementation effort
            effort = self._estimate_implementation_effort(issues, warnings)

            report = CompatibilityReport(
                compatibility_level=compatibility_level,
                issues=[i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]],
                warnings=[i for i in warnings if i.severity == ValidationSeverity.WARNING],
                info=[i for i in info if i.severity == ValidationSeverity.INFO],
                schema_version=schema.get("$schema"),
                migration_required=migration_required,
                estimated_effort=effort
            )

            logger.info(f"Validation complete: {compatibility_level.value} ({len(report.issues)} issues, {len(report.warnings)} warnings)")
            return report

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            # Return critical error report
            return CompatibilityReport(
                compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    code="VALIDATION_FAILED",
                    message=f"Schema validation failed: {str(e)}",
                    field_path="$",
                    suggested_fix="Ensure schema is valid JSON and follows JSON Schema specification"
                )],
                warnings=[],
                info=[]
            )

    def suggest_fixes(self, report: CompatibilityReport) -> List[ValidationIssue]:
        """
        Generate detailed fix suggestions for validation issues.

        Args:
            report: Compatibility report with issues

        Returns:
            Enhanced issues with detailed fix suggestions
        """
        enhanced_issues = []

        for issue in report.issues + report.warnings:
            enhanced_issue = self._enhance_issue_with_fixes(issue)
            enhanced_issues.append(enhanced_issue)

        return enhanced_issues

    def generate_migration_plan(self, schema: Dict[str, Any], target_version: str = "0.216.0") -> List[MigrationStep]:
        """
        Generate automated migration plan for schema compatibility.

        Args:
            schema: Source schema to migrate
            target_version: Target BAML version

        Returns:
            Step-by-step migration plan
        """
        migration_steps = []
        step_number = 1

        # Analyze current schema for migration needs
        current_issues = self.validate_schema(schema)

        for issue in current_issues.issues:
            if issue.migration_strategy:
                strategy = self.migration_strategies.get(issue.migration_strategy)
                if strategy:
                    migration_steps.append(MigrationStep(
                        step_number=step_number,
                        description=strategy["description"],
                        from_pattern=strategy["from_pattern"],
                        to_pattern=strategy["to_pattern"],
                        auto_applicable=strategy.get("auto_applicable", False),
                        validation_function=strategy.get("validation_function")
                    ))
                    step_number += 1

        logger.info(f"Generated migration plan with {len(migration_steps)} steps")
        return migration_steps

    def auto_fix_schema(self, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Attempt to automatically fix common schema issues.

        Args:
            schema: Schema to fix

        Returns:
            Tuple of (fixed_schema, list_of_applied_fixes)
        """
        fixed_schema = schema.copy()
        applied_fixes = []

        try:
            # Auto-fix missing required fields
            if "type" not in fixed_schema and "properties" in fixed_schema:
                fixed_schema["type"] = "object"
                applied_fixes.append("Added missing 'type': 'object' field")

            # Auto-fix array items without type
            self._auto_fix_array_items(fixed_schema, applied_fixes)

            # Auto-fix deprecated type names
            self._auto_fix_deprecated_types(fixed_schema, applied_fixes)

            # Auto-fix property naming
            self._auto_fix_property_names(fixed_schema, applied_fixes)

            logger.info(f"Auto-fixed schema with {len(applied_fixes)} changes")
            return fixed_schema, applied_fixes

        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
            return schema, []

    def _validate_basic_structure(self, schema: Dict[str, Any], issues: List[ValidationIssue]) -> None:
        """Validate basic JSON schema structure."""
        if not isinstance(schema, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                code="INVALID_SCHEMA_TYPE",
                message="Schema must be a JSON object",
                field_path="$",
                suggested_fix="Ensure the schema is a valid JSON object with properties",
                auto_fixable=False
            ))
            return

        # Check required fields
        required_fields = ["type", "properties"]
        for field in required_fields:
            if field not in schema:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=f"MISSING_{field.upper()}",
                    message=f"Schema missing required field: {field}",
                    field_path="$",
                    suggested_fix=f"Add '{field}' field to schema root",
                    auto_fixable=(field == "type"),
                    migration_strategy=f"ADD_{field.upper()}_FIELD"
                ))

        # Validate schema structure
        if "properties" in schema and not isinstance(schema["properties"], dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_PROPERTIES_TYPE",
                message="Schema 'properties' field must be an object",
                field_path="$.properties",
                suggested_fix="Change 'properties' to be a JSON object with property definitions"
            ))

    def _validate_type_system(self, schema: Dict[str, Any], issues: List[ValidationIssue], warnings: List[ValidationIssue]) -> None:
        """Validate type system compatibility with BAML."""
        if "properties" not in schema:
            return

        properties = schema["properties"]
        self._validate_properties_recursive(properties, "$.properties", issues, warnings)

    def _validate_properties_recursive(self, properties: Dict, path: str, issues: List[ValidationIssue], warnings: List[ValidationIssue]) -> None:
        """Recursively validate properties and nested structures."""
        for prop_name, prop_def in properties.items():
            prop_path = f"{path}.{prop_name}"

            if not isinstance(prop_def, dict):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_PROPERTY_DEFINITION",
                    message=f"Property '{prop_name}' definition must be an object",
                    field_path=prop_path,
                    suggested_fix="Define property as object with 'type' and other schema fields"
                ))
                continue

            prop_type = prop_def.get("type")

            # Check if type is supported
            if prop_type and prop_type not in self.supported_types:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="UNSUPPORTED_TYPE",
                    message=f"Type '{prop_type}' is not supported in BAML",
                    field_path=f"{prop_path}.type",
                    suggested_fix=f"Use one of the supported types: {', '.join(self.supported_types)}",
                    migration_strategy="CONVERT_UNSUPPORTED_TYPE"
                ))

            # Validate array definitions
            if prop_type == "array":
                self._validate_array_definition(prop_def, prop_path, issues, warnings)

            # Validate object definitions
            elif prop_type == "object":
                self._validate_object_definition(prop_def, prop_path, issues, warnings)

            # Check property naming conventions
            self._validate_property_naming(prop_name, prop_path, warnings)

    def _validate_array_definition(self, array_def: Dict, path: str, issues: List[ValidationIssue], warnings: List[ValidationIssue]) -> None:
        """Validate array property definitions."""
        if "items" not in array_def:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MISSING_ARRAY_ITEMS",
                message="Array type must define 'items' property",
                field_path=f"{path}.items",
                suggested_fix="Add 'items' property with type definition for array elements",
                auto_fixable=True,
                migration_strategy="ADD_ARRAY_ITEMS"
            ))
            return

        items_def = array_def["items"]
        if isinstance(items_def, dict):
            items_type = items_def.get("type")
            if items_type == "object" and "properties" in items_def:
                # Validate nested object in array
                self._validate_properties_recursive(items_def["properties"], f"{path}.items.properties", issues, warnings)
            elif items_type and items_type not in self.supported_types:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="UNSUPPORTED_ARRAY_ITEM_TYPE",
                    message=f"Array item type '{items_type}' is not supported",
                    field_path=f"{path}.items.type",
                    suggested_fix=f"Use supported array item type: {', '.join(self.supported_types)}"
                ))

    def _validate_object_definition(self, obj_def: Dict, path: str, issues: List[ValidationIssue], warnings: List[ValidationIssue]) -> None:
        """Validate nested object definitions."""
        if "properties" in obj_def:
            nested_properties = obj_def["properties"]
            if isinstance(nested_properties, dict):
                # Recursively validate nested properties
                self._validate_properties_recursive(nested_properties, f"{path}.properties", issues, warnings)
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_NESTED_PROPERTIES",
                    message="Nested object 'properties' must be an object",
                    field_path=f"{path}.properties",
                    suggested_fix="Define properties as object with property definitions"
                ))
        else:
            warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EMPTY_OBJECT_DEFINITION",
                message="Object type has no properties defined",
                field_path=path,
                suggested_fix="Add 'properties' field with object property definitions"
            ))

    def _validate_property_naming(self, prop_name: str, path: str, warnings: List[ValidationIssue]) -> None:
        """Validate property naming conventions."""
        # Check for reserved keywords
        reserved_keywords = ["class", "function", "import", "client", "prompt"]
        if prop_name.lower() in reserved_keywords:
            warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="RESERVED_KEYWORD_PROPERTY",
                message=f"Property name '{prop_name}' conflicts with BAML reserved keyword",
                field_path=path,
                suggested_fix=f"Rename property to avoid reserved keyword (e.g., '{prop_name}_value')"
            ))

        # Check naming conventions
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', prop_name):
            warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="INVALID_PROPERTY_NAME",
                message=f"Property name '{prop_name}' doesn't follow recommended naming convention",
                field_path=path,
                suggested_fix="Use alphanumeric characters and underscores, starting with a letter"
            ))

    def _validate_baml_constraints(self, schema: Dict[str, Any], issues: List[ValidationIssue], warnings: List[ValidationIssue]) -> None:
        """Validate BAML-specific constraints and limitations."""
        # Check maximum nesting depth
        max_depth = self._calculate_nesting_depth(schema)
        if max_depth > 10:
            warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_NESTING_DEPTH",
                message=f"Schema nesting depth ({max_depth}) exceeds recommended limit of 10",
                field_path="$",
                suggested_fix="Consider flattening deeply nested structures for better performance"
            ))

        # Check for circular references (simplified check)
        if self._has_potential_circular_references(schema):
            warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="POTENTIAL_CIRCULAR_REFERENCE",
                message="Schema may contain circular references",
                field_path="$",
                suggested_fix="Ensure no circular references exist between nested objects"
            ))

        # Validate field count limitations
        total_fields = self._count_total_fields(schema)
        if total_fields > 100:
            warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_FIELD_COUNT",
                message=f"Total field count ({total_fields}) is very high",
                field_path="$",
                suggested_fix="Consider breaking large schemas into smaller, focused schemas"
            ))

    def _validate_performance_implications(self, schema: Dict[str, Any], warnings: List[ValidationIssue], info: List[ValidationIssue]) -> None:
        """Analyze performance implications of the schema."""
        # Check for large array structures
        self._check_large_arrays(schema, warnings)

        # Check for complex nested structures
        complexity_score = self._calculate_complexity_score(schema)
        if complexity_score > 50:
            info.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="HIGH_COMPLEXITY_SCHEMA",
                message=f"Schema complexity score is high ({complexity_score})",
                field_path="$",
                suggested_fix="Consider using CustomOpus4 or CustomGPT5 for better extraction accuracy"
            ))

    def _validate_version_compatibility(self, schema: Dict[str, Any], warnings: List[ValidationIssue], info: List[ValidationIssue]) -> None:
        """Check version compatibility requirements."""
        schema_version = schema.get("$schema", "")

        if schema_version:
            # Extract version from schema URL if present
            version_match = re.search(r'draft-(\d+)', schema_version)
            if version_match:
                draft_version = int(version_match.group(1))
                if draft_version > 7:  # Draft-07 is widely supported
                    warnings.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="UNSUPPORTED_SCHEMA_VERSION",
                        message=f"JSON Schema draft-{draft_version} may not be fully supported",
                        field_path="$.$schema",
                        suggested_fix="Consider using JSON Schema Draft-07 for maximum compatibility"
                    ))

        # Check for features requiring specific BAML versions
        if self._requires_advanced_features(schema):
            info.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="ADVANCED_FEATURES_DETECTED",
                message="Schema uses advanced features that may require BAML 0.216.0+",
                field_path="$",
                suggested_fix="Ensure BAML version is 0.216.0 or higher"
            ))

    def _determine_compatibility_level(self, issues: List[ValidationIssue], warnings: List[ValidationIssue]) -> CompatibilityLevel:
        """Determine overall compatibility level based on issues."""
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]

        if critical_issues:
            return CompatibilityLevel.INCOMPATIBLE
        elif error_issues:
            return CompatibilityLevel.LIMITED_COMPATIBILITY
        elif warnings:
            return CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        else:
            return CompatibilityLevel.FULLY_COMPATIBLE

    def _enhance_issue_with_fixes(self, issue: ValidationIssue) -> ValidationIssue:
        """Enhance validation issue with detailed fix suggestions."""
        fix_info = self.fix_suggestions.get(issue.code, {})

        if fix_info:
            issue.suggested_fix = fix_info.get("fix", issue.suggested_fix)
            issue.documentation_link = fix_info.get("docs")
            issue.migration_strategy = fix_info.get("migration")

        return issue

    def _auto_fix_array_items(self, schema: Dict[str, Any], applied_fixes: List[str]) -> None:
        """Auto-fix array items without proper type definitions."""
        def fix_arrays_recursive(obj, path=""):
            if isinstance(obj, dict):
                if obj.get("type") == "array" and "items" not in obj:
                    obj["items"] = {"type": "string"}
                    applied_fixes.append(f"Added default 'items' for array at {path}")

                for key, value in obj.items():
                    fix_arrays_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    fix_arrays_recursive(item, f"{path}[{i}]")

        fix_arrays_recursive(schema)

    def _auto_fix_deprecated_types(self, schema: Dict[str, Any], applied_fixes: List[str]) -> None:
        """Auto-fix deprecated type names."""
        type_mappings = {
            "integer": "int",
            "number": "float",
            "boolean": "bool"
        }

        def fix_types_recursive(obj, path=""):
            if isinstance(obj, dict):
                if "type" in obj and obj["type"] in type_mappings:
                    old_type = obj["type"]
                    new_type = type_mappings[old_type]
                    obj["type"] = new_type
                    applied_fixes.append(f"Changed type from '{old_type}' to '{new_type}' at {path}")

                for key, value in obj.items():
                    fix_types_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    fix_types_recursive(item, f"{path}[{i}]")

        fix_types_recursive(schema)

    def _auto_fix_property_names(self, schema: Dict[str, Any], applied_fixes: List[str]) -> None:
        """Auto-fix property names that conflict with reserved keywords."""
        reserved_keywords = {"class", "function", "import", "client", "prompt"}

        def fix_names_recursive(obj, path=""):
            if isinstance(obj, dict) and "properties" in obj:
                properties = obj["properties"]
                if isinstance(properties, dict):
                    keys_to_fix = []
                    for prop_name in properties:
                        if prop_name.lower() in reserved_keywords:
                            keys_to_fix.append(prop_name)

                    for old_name in keys_to_fix:
                        new_name = f"{old_name}_value"
                        properties[new_name] = properties.pop(old_name)
                        applied_fixes.append(f"Renamed property '{old_name}' to '{new_name}' at {path}")

                for key, value in obj.items():
                    fix_names_recursive(value, f"{path}.{key}")

        fix_names_recursive(schema)

    def _initialize_supported_types(self) -> Set[str]:
        """Initialize set of BAML-supported types."""
        return {
            "string", "int", "float", "bool", "array", "object"
        }

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules configuration."""
        return {
            "max_nesting_depth": 10,
            "max_field_count": 100,
            "max_array_size_hint": 1000,
            "reserved_keywords": ["class", "function", "import", "client", "prompt"]
        }

    def _initialize_fix_suggestions(self) -> Dict[str, Dict[str, str]]:
        """Initialize detailed fix suggestions for common issues."""
        return {
            "MISSING_TYPE": {
                "fix": "Add 'type' field specifying the data type (string, int, float, bool, array, object)",
                "docs": "https://json-schema.org/understanding-json-schema/reference/type.html",
                "migration": "ADD_TYPE_FIELD"
            },
            "MISSING_PROPERTIES": {
                "fix": "Add 'properties' field with object property definitions",
                "docs": "https://json-schema.org/understanding-json-schema/reference/object.html",
                "migration": "ADD_PROPERTIES_FIELD"
            },
            "UNSUPPORTED_TYPE": {
                "fix": "Replace with supported BAML type: string, int, float, bool, array, or object",
                "docs": "https://docs.boundaryml.com/docs/baml/types",
                "migration": "CONVERT_TYPE"
            },
            "MISSING_ARRAY_ITEMS": {
                "fix": "Add 'items' property defining the type of array elements",
                "docs": "https://json-schema.org/understanding-json-schema/reference/array.html",
                "migration": "ADD_ARRAY_ITEMS"
            }
        }

    def _initialize_migration_strategies(self) -> Dict[str, Dict[str, str]]:
        """Initialize schema migration strategies."""
        return {
            "ADD_TYPE_FIELD": {
                "description": "Add missing 'type' field with default value 'object'",
                "from_pattern": r'"properties":\s*\{',
                "to_pattern": r'"type": "object",\n  "properties": {',
                "auto_applicable": True
            },
            "ADD_PROPERTIES_FIELD": {
                "description": "Add missing 'properties' field with empty object",
                "from_pattern": r'"type":\s*"object"',
                "to_pattern": r'"type": "object",\n  "properties": {}',
                "auto_applicable": True
            },
            "CONVERT_UNSUPPORTED_TYPE": {
                "description": "Convert unsupported types to BAML-compatible equivalents",
                "from_pattern": r'"type":\s*"(null|integer|number|boolean)"',
                "to_pattern": r'"type": "string|int|float|bool"',
                "auto_applicable": True
            },
            "ADD_ARRAY_ITEMS": {
                "description": "Add missing 'items' field to array definition",
                "from_pattern": r'"type":\s*"array"',
                "to_pattern": r'"type": "array",\n    "items": {"type": "string"}',
                "auto_applicable": True
            }
        }

    def _calculate_nesting_depth(self, schema: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of schema."""
        max_depth = current_depth

        if isinstance(schema, dict):
            if "properties" in schema:
                for prop_def in schema["properties"].values():
                    if isinstance(prop_def, dict) and prop_def.get("type") == "object":
                        depth = self._calculate_nesting_depth(prop_def, current_depth + 1)
                        max_depth = max(max_depth, depth)

        return max_depth

    def _has_potential_circular_references(self, schema: Dict[str, Any]) -> bool:
        """Check for potential circular references (simplified heuristic)."""
        # This is a simplified check - a full implementation would need more sophisticated analysis
        def find_refs(obj, seen_refs=None):
            if seen_refs is None:
                seen_refs = set()

            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref = obj["$ref"]
                    if ref in seen_refs:
                        return True
                    seen_refs.add(ref)

                for value in obj.values():
                    if find_refs(value, seen_refs.copy()):
                        return True

            return False

        return find_refs(schema)

    def _count_total_fields(self, schema: Dict[str, Any]) -> int:
        """Count total number of fields in schema."""
        count = 0

        def count_fields(obj):
            nonlocal count
            if isinstance(obj, dict):
                if "properties" in obj and isinstance(obj["properties"], dict):
                    count += len(obj["properties"])
                    for prop_def in obj["properties"].values():
                        count_fields(prop_def)
                if "items" in obj:
                    count_fields(obj["items"])

        count_fields(schema)
        return count

    def _check_large_arrays(self, schema: Dict[str, Any], warnings: List[ValidationIssue]) -> None:
        """Check for potentially large array structures."""
        def check_arrays_recursive(obj, path="$"):
            if isinstance(obj, dict):
                if obj.get("type") == "array":
                    items = obj.get("items", {})
                    if isinstance(items, dict) and items.get("type") == "object":
                        properties = items.get("properties", {})
                        if len(properties) > 10:  # Large object in array
                            warnings.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                code="LARGE_ARRAY_OBJECTS",
                                message=f"Array contains complex objects with {len(properties)} properties",
                                field_path=path,
                                suggested_fix="Consider simplifying array object structure for better performance"
                            ))

                if "properties" in obj:
                    for prop_name, prop_def in obj["properties"].items():
                        check_arrays_recursive(prop_def, f"{path}.{prop_name}")

        check_arrays_recursive(schema)

    def _calculate_complexity_score(self, schema: Dict[str, Any]) -> int:
        """Calculate complexity score for schema."""
        score = 0

        def calculate_recursive(obj):
            nonlocal score
            if isinstance(obj, dict):
                if obj.get("type") == "object":
                    score += 2
                elif obj.get("type") == "array":
                    score += 3

                if "properties" in obj:
                    score += len(obj["properties"])
                    for prop_def in obj["properties"].values():
                        calculate_recursive(prop_def)

                if "items" in obj:
                    calculate_recursive(obj["items"])

        calculate_recursive(schema)
        return score

    def _requires_advanced_features(self, schema: Dict[str, Any]) -> bool:
        """Check if schema requires advanced BAML features."""
        # This would check for features like complex validation, custom formats, etc.
        advanced_features = ["pattern", "format", "const", "enum"]

        def check_features(obj):
            if isinstance(obj, dict):
                for feature in advanced_features:
                    if feature in obj:
                        return True
                for value in obj.values():
                    if check_features(value):
                        return True
            return False

        return check_features(schema)

    def _check_migration_required(self, schema: Dict[str, Any], issues: List[ValidationIssue]) -> bool:
        """Check if schema migration is required."""
        migration_requiring_codes = [
            "UNSUPPORTED_TYPE", "MISSING_TYPE", "MISSING_PROPERTIES",
            "MISSING_ARRAY_ITEMS", "INVALID_PROPERTY_DEFINITION"
        ]

        return any(issue.code in migration_requiring_codes for issue in issues)

    def _estimate_implementation_effort(self, issues: List[ValidationIssue], warnings: List[ValidationIssue]) -> str:
        """Estimate implementation effort based on issues."""
        critical_count = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_count = len(warnings)

        if critical_count > 0 or error_count > 5:
            return "high"
        elif error_count > 0 or warning_count > 10:
            return "medium"
        else:
            return "low"
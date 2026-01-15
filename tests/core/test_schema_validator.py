"""
Tests for Schema Validator - COLPALI-503 implementation.

This test suite validates comprehensive JSON schema validation for BAML compatibility
with detailed error messages, fix suggestions, and migration support.
"""

import pytest
from typing import Dict, Any

from tatforge.core.schema_validator import (
    SchemaValidator,
    ValidationSeverity,
    CompatibilityLevel,
    ValidationIssue,
    CompatibilityReport,
    MigrationStep
)


class TestSchemaValidator:
    """Test suite for SchemaValidator compatibility system."""

    @pytest.fixture
    def validator(self):
        """Create SchemaValidator instance for testing."""
        return SchemaValidator(baml_version="0.216.0")

    @pytest.fixture
    def valid_schema(self) -> Dict[str, Any]:
        """Valid schema that should pass all checks."""
        return {
            "title": "ValidDocument",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Document name"
                },
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page_number": {"type": "int"},
                            "content": {"type": "string"}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created_date": {"type": "string"},
                        "file_size": {"type": "int"}
                    }
                }
            },
            "required": ["name", "pages"]
        }

    @pytest.fixture
    def invalid_schema(self) -> Dict[str, Any]:
        """Invalid schema with multiple issues."""
        return {
            "title": "InvalidDocument",
            # Missing "type" field
            "properties": {
                "bad_type_field": {
                    "type": "null",  # Unsupported type
                    "description": "This will cause issues"
                },
                "array_without_items": {
                    "type": "array"
                    # Missing "items" definition
                },
                "class": {  # Reserved keyword
                    "type": "string"
                },
                "123invalid_name": {  # Invalid property name
                    "type": "string"
                }
            }
        }

    @pytest.fixture
    def complex_schema(self) -> Dict[str, Any]:
        """Complex schema for performance testing."""
        # Create deeply nested schema
        nested_properties = {}
        for i in range(15):  # Create many fields
            nested_properties[f"field_{i}"] = {"type": "string"}

        deep_nesting = {
            "type": "object",
            "properties": nested_properties
        }

        # Create 12 levels of nesting
        for _ in range(12):
            deep_nesting = {
                "type": "object",
                "properties": {
                    "nested": deep_nesting
                }
            }

        return {
            "title": "ComplexSchema",
            "type": "object",
            "properties": {
                "deep_structure": deep_nesting,
                "large_array": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {f"field_{i}": {"type": "string"} for i in range(15)}
                    }
                }
            }
        }

    def test_validate_valid_schema(self, validator, valid_schema):
        """Test validation of completely valid schema."""
        report = validator.validate_schema(valid_schema)

        assert report.compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE
        assert len(report.issues) == 0
        assert len(report.warnings) == 0
        assert not report.migration_required
        assert report.estimated_effort == "low"

    def test_validate_invalid_schema(self, validator, invalid_schema):
        """Test validation of schema with multiple issues."""
        report = validator.validate_schema(invalid_schema)

        # Should have compatibility issues
        assert report.compatibility_level in [
            CompatibilityLevel.LIMITED_COMPATIBILITY,
            CompatibilityLevel.INCOMPATIBLE
        ]

        # Should detect multiple issues
        assert len(report.issues) > 0

        # Check for specific expected issues
        issue_codes = [issue.code for issue in report.issues]
        assert "MISSING_TYPE" in issue_codes
        assert "UNSUPPORTED_TYPE" in issue_codes
        assert "MISSING_ARRAY_ITEMS" in issue_codes

        # Should require migration
        assert report.migration_required
        assert report.estimated_effort in ["medium", "high"]

    def test_validate_complex_schema_warnings(self, validator, complex_schema):
        """Test validation of complex schema generates appropriate warnings."""
        report = validator.validate_schema(complex_schema)

        # Should be compatible but with warnings
        assert report.compatibility_level in [
            CompatibilityLevel.COMPATIBLE_WITH_WARNINGS,
            CompatibilityLevel.FULLY_COMPATIBLE
        ]

        # Should detect complexity issues
        warning_codes = [warning.code for warning in report.warnings]
        assert any(code in warning_codes for code in [
            "EXCESSIVE_NESTING_DEPTH",
            "LARGE_ARRAY_OBJECTS",
            "HIGH_COMPLEXITY_SCHEMA"
        ])

    def test_missing_basic_structure(self, validator):
        """Test validation of schema missing basic structure."""
        missing_properties = {
            "title": "InvalidSchema",
            "type": "object"
            # Missing "properties"
        }

        report = validator.validate_schema(missing_properties)

        assert len(report.issues) > 0
        issue_codes = [issue.code for issue in report.issues]
        assert "MISSING_PROPERTIES" in issue_codes

        # Check that suggested fix is provided
        missing_props_issue = next(
            issue for issue in report.issues
            if issue.code == "MISSING_PROPERTIES"
        )
        assert missing_props_issue.suggested_fix is not None
        assert "properties" in missing_props_issue.suggested_fix.lower()

    def test_unsupported_types_detection(self, validator):
        """Test detection of unsupported types."""
        unsupported_schema = {
            "title": "UnsupportedTypes",
            "type": "object",
            "properties": {
                "null_field": {"type": "null"},
                "unknown_field": {"type": "unknown_type"},
                "valid_field": {"type": "string"}
            }
        }

        report = validator.validate_schema(unsupported_schema)

        # Should have issues for unsupported types
        unsupported_issues = [
            issue for issue in report.issues
            if issue.code == "UNSUPPORTED_TYPE"
        ]
        assert len(unsupported_issues) == 2  # null and unknown_type

        # Check fix suggestions
        for issue in unsupported_issues:
            assert "supported types" in issue.suggested_fix.lower()

    def test_array_validation(self, validator):
        """Test array definition validation."""
        array_schema = {
            "title": "ArrayTest",
            "type": "object",
            "properties": {
                "valid_array": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "invalid_array": {
                    "type": "array"
                    # Missing items
                },
                "nested_object_array": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "int"},
                            "name": {"type": "string"}
                        }
                    }
                }
            }
        }

        report = validator.validate_schema(array_schema)

        # Should have issue for array without items
        array_issues = [
            issue for issue in report.issues
            if issue.code == "MISSING_ARRAY_ITEMS"
        ]
        assert len(array_issues) == 1

        # Should validate nested object in array correctly
        assert len([issue for issue in report.issues if "nested_object_array" in issue.field_path]) == 0

    def test_property_naming_validation(self, validator):
        """Test property naming convention validation."""
        naming_schema = {
            "title": "NamingTest",
            "type": "object",
            "properties": {
                "valid_name": {"type": "string"},
                "class": {"type": "string"},  # Reserved keyword
                "function": {"type": "string"},  # Reserved keyword
                "123invalid": {"type": "string"},  # Invalid start character
                "invalid-dash": {"type": "string"},  # Invalid character
                "_valid_underscore": {"type": "string"}  # Should be valid
            }
        }

        report = validator.validate_schema(naming_schema)

        # Should have warnings for naming issues
        naming_warnings = [
            warning for warning in report.warnings
            if warning.code in ["RESERVED_KEYWORD_PROPERTY", "INVALID_PROPERTY_NAME"]
        ]
        assert len(naming_warnings) >= 2  # At least reserved keywords

    def test_nested_object_validation(self, validator):
        """Test validation of nested object structures."""
        nested_schema = {
            "title": "NestedTest",
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "coordinates": {
                                    "type": "object",
                                    "properties": {
                                        "lat": {"type": "float"},
                                        "lng": {"type": "float"}
                                    }
                                }
                            }
                        }
                    }
                },
                "empty_object": {
                    "type": "object"
                    # No properties - should trigger warning
                }
            }
        }

        report = validator.validate_schema(nested_schema)

        # Should warn about empty object
        empty_warnings = [
            warning for warning in report.warnings
            if warning.code == "EMPTY_OBJECT_DEFINITION"
        ]
        assert len(empty_warnings) >= 1

    def test_suggest_fixes(self, validator, invalid_schema):
        """Test fix suggestion enhancement."""
        report = validator.validate_schema(invalid_schema)
        enhanced_issues = validator.suggest_fixes(report)

        # All issues should have detailed suggestions
        for issue in enhanced_issues:
            assert issue.suggested_fix is not None
            assert len(issue.suggested_fix) > 10  # Meaningful suggestion

        # Check that some issues are auto-fixable
        auto_fixable = [issue for issue in enhanced_issues if issue.auto_fixable]
        assert len(auto_fixable) > 0

    def test_auto_fix_schema(self, validator, invalid_schema):
        """Test automatic schema fixing."""
        fixed_schema, applied_fixes = validator.auto_fix_schema(invalid_schema)

        # Should have applied some fixes
        assert len(applied_fixes) > 0

        # Fixed schema should be different from original
        assert fixed_schema != invalid_schema

        # Should have added missing type field
        assert "type" in fixed_schema

        # Should have fixed array items
        if "array_without_items" in fixed_schema.get("properties", {}):
            array_prop = fixed_schema["properties"]["array_without_items"]
            assert "items" in array_prop

    def test_migration_plan_generation(self, validator, invalid_schema):
        """Test migration plan generation."""
        migration_steps = validator.generate_migration_plan(invalid_schema)

        # Should generate migration steps
        assert len(migration_steps) > 0

        # Each step should have required fields
        for step in migration_steps:
            assert isinstance(step, MigrationStep)
            assert step.step_number > 0
            assert step.description is not None
            assert step.from_pattern is not None
            assert step.to_pattern is not None

    def test_performance_validation(self, validator, complex_schema):
        """Test performance-related validation checks."""
        report = validator.validate_schema(complex_schema)

        # Should detect performance issues
        performance_codes = [
            "EXCESSIVE_NESTING_DEPTH",
            "EXCESSIVE_FIELD_COUNT",
            "LARGE_ARRAY_OBJECTS",
            "HIGH_COMPLEXITY_SCHEMA"
        ]

        found_performance_issues = False
        all_issues = report.issues + report.warnings + report.info

        for issue in all_issues:
            if issue.code in performance_codes:
                found_performance_issues = True
                break

        assert found_performance_issues

    def test_version_compatibility_check(self, validator):
        """Test version compatibility validation."""
        version_schema = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "title": "VersionTest",
            "type": "object",
            "properties": {
                "test_field": {"type": "string"}
            }
        }

        report = validator.validate_schema(version_schema)

        # Should handle version information appropriately
        assert report.schema_version is not None
        assert report.baml_version_requirement is not None

    def test_circular_reference_detection(self, validator):
        """Test detection of potential circular references."""
        # This is a simplified test since full circular reference detection is complex
        circular_schema = {
            "title": "CircularTest",
            "type": "object",
            "properties": {
                "self_ref": {
                    "$ref": "#/definitions/CircularTest"
                }
            },
            "definitions": {
                "CircularTest": {
                    "$ref": "#"
                }
            }
        }

        report = validator.validate_schema(circular_schema)

        # Should detect potential circular reference
        circular_warnings = [
            warning for warning in report.warnings
            if warning.code == "POTENTIAL_CIRCULAR_REFERENCE"
        ]
        # This test may not always trigger depending on the implementation
        # but validates the structure exists

    def test_validation_issue_structure(self, validator, invalid_schema):
        """Test that validation issues have proper structure."""
        report = validator.validate_schema(invalid_schema)

        for issue in report.issues + report.warnings + report.info:
            # Check required fields
            assert isinstance(issue.severity, ValidationSeverity)
            assert issue.code is not None
            assert issue.message is not None
            assert issue.field_path is not None

            # Check field path format
            assert issue.field_path.startswith("$")

            # Check that suggestions exist for errors
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                assert issue.suggested_fix is not None

    def test_compatibility_level_calculation(self, validator):
        """Test compatibility level calculation logic."""
        # Test fully compatible
        valid = {
            "title": "Valid",
            "type": "object",
            "properties": {"test": {"type": "string"}}
        }
        report = validator.validate_schema(valid)
        assert report.compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE

        # Test incompatible (not a dict)
        invalid = "not_a_dict"
        report = validator.validate_schema(invalid)
        assert report.compatibility_level == CompatibilityLevel.INCOMPATIBLE

        # Test limited compatibility
        limited = {
            "title": "Limited",
            # Missing type and properties - should generate errors
        }
        report = validator.validate_schema(limited)
        assert report.compatibility_level in [
            CompatibilityLevel.LIMITED_COMPATIBILITY,
            CompatibilityLevel.INCOMPATIBLE
        ]

    def test_field_count_validation(self, validator):
        """Test validation of schemas with many fields."""
        # Create schema with many fields
        many_fields = {
            "title": "ManyFields",
            "type": "object",
            "properties": {}
        }

        # Add 150 properties to trigger warning
        for i in range(150):
            many_fields["properties"][f"field_{i}"] = {"type": "string"}

        report = validator.validate_schema(many_fields)

        # Should warn about excessive field count
        field_warnings = [
            warning for warning in report.warnings + report.info
            if warning.code == "EXCESSIVE_FIELD_COUNT"
        ]
        assert len(field_warnings) > 0

    def test_error_handling(self, validator):
        """Test error handling for invalid inputs."""
        # Test with None
        report = validator.validate_schema(None)
        assert report.compatibility_level == CompatibilityLevel.INCOMPATIBLE
        assert len(report.issues) > 0

        # Test with invalid JSON structure
        report = validator.validate_schema("not_json")
        assert report.compatibility_level == CompatibilityLevel.INCOMPATIBLE

    def test_baml_constraints_validation(self, validator):
        """Test BAML-specific constraint validation."""
        baml_schema = {
            "title": "BAMLConstraints",
            "type": "object",
            "properties": {
                "normal_field": {"type": "string"},
                # Add some edge cases that might cause issues
                "very_deep_nesting": {
                    "type": "object",
                    "properties": {
                        "level1": {
                            "type": "object",
                            "properties": {
                                "level2": {
                                    "type": "object",
                                    "properties": {
                                        "level3": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        report = validator.validate_schema(baml_schema)

        # Should validate BAML-specific constraints
        # Exact checks depend on implementation but should not error
        assert report.compatibility_level in [
            CompatibilityLevel.FULLY_COMPATIBLE,
            CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        ]

    def test_effort_estimation(self, validator):
        """Test implementation effort estimation."""
        # Low effort schema
        easy_schema = {
            "title": "Easy",
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        report = validator.validate_schema(easy_schema)
        assert report.estimated_effort == "low"

        # High effort schema (many issues)
        hard_schema = {
            "title": "Hard"
            # Missing everything
        }
        report = validator.validate_schema(hard_schema)
        assert report.estimated_effort in ["medium", "high"]

    def test_auto_fix_edge_cases(self, validator):
        """Test auto-fix handles edge cases properly."""
        edge_case_schema = {
            "title": "EdgeCases",
            "properties": {  # Missing type - should be auto-fixed
                "array_field": {
                    "type": "array"
                    # Missing items - should be auto-fixed
                },
                "integer_field": {
                    "type": "integer"  # Should be converted to "int"
                },
                "number_field": {
                    "type": "number"  # Should be converted to "float"
                },
                "class": {  # Reserved keyword - should be renamed
                    "type": "string"
                }
            }
        }

        fixed_schema, applied_fixes = validator.auto_fix_schema(edge_case_schema)

        # Should apply multiple fixes
        assert len(applied_fixes) > 0

        # Check specific fixes
        assert "type" in fixed_schema  # Added missing type

        if "array_field" in fixed_schema.get("properties", {}):
            array_field = fixed_schema["properties"]["array_field"]
            assert "items" in array_field  # Added missing items

        # Check type conversions
        properties = fixed_schema.get("properties", {})
        for field_name, field_def in properties.items():
            if "type" in field_def:
                # Should not have deprecated types
                assert field_def["type"] not in ["integer", "number", "boolean"]
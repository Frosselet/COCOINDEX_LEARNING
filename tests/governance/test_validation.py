"""
Tests for GovernanceValidator - Validation rules and compliance checking.

COLPALI-802: Governance validation implementation tests.
"""

import pytest
from datetime import datetime
from tatforge.governance.validation import (
    GovernanceValidator,
    ValidationRule,
    ValidationResult,
    ValidationSeverity,
    RuleCategory
)
from tatforge.extraction.models import (
    CanonicalData,
    ShapedData,
    TransformationRule
)


class TestValidationResult:
    """Test suite for ValidationResult."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = ValidationResult(
            rule_id="test_rule",
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Test passed",
            category=RuleCategory.CANONICAL_INTEGRITY
        )

        assert result.rule_id == "test_rule"
        assert result.passed is True
        assert result.severity == ValidationSeverity.INFO
        assert result.category == RuleCategory.CANONICAL_INTEGRITY

    def test_result_with_recommendations(self):
        """Test result with recommendations."""
        result = ValidationResult(
            rule_id="test_rule",
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Test failed",
            category=RuleCategory.DATA_DISTORTION,
            recommendations=["Fix item 1", "Fix item 2"]
        )

        assert len(result.recommendations) == 2
        assert "Fix item 1" in result.recommendations

    def test_result_serialization(self):
        """Test result to_dict serialization."""
        result = ValidationResult(
            rule_id="test_rule",
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Test message",
            category=RuleCategory.COMPLIANCE,
            details={"key": "value"}
        )

        result_dict = result.to_dict()

        assert result_dict["rule_id"] == "test_rule"
        assert result_dict["passed"] is True
        assert result_dict["severity"] == "error"
        assert result_dict["category"] == "compliance"
        assert result_dict["details"]["key"] == "value"


class TestValidationRule:
    """Test suite for ValidationRule."""

    def test_rule_creation(self):
        """Test basic rule creation."""
        rule = ValidationRule(
            rule_id="test_rule_001",
            category=RuleCategory.CANONICAL_INTEGRITY,
            name="Test Rule",
            description="Test description",
            severity=ValidationSeverity.ERROR,
            validation_fn=lambda x: True
        )

        assert rule.rule_id == "test_rule_001"
        assert rule.name == "Test Rule"
        assert rule.enabled is True

    def test_rule_validation_success(self):
        """Test rule validation success."""
        rule = ValidationRule(
            rule_id="test_rule_002",
            category=RuleCategory.CANONICAL_INTEGRITY,
            name="Always Pass",
            description="Rule that always passes",
            severity=ValidationSeverity.INFO,
            validation_fn=lambda x: True
        )

        result = rule.validate("test_data")

        assert result.passed is True
        assert result.rule_id == "test_rule_002"

    def test_rule_validation_failure(self):
        """Test rule validation failure."""
        rule = ValidationRule(
            rule_id="test_rule_003",
            category=RuleCategory.DATA_DISTORTION,
            name="Always Fail",
            description="Rule that always fails",
            severity=ValidationSeverity.ERROR,
            validation_fn=lambda x: False
        )

        result = rule.validate("test_data")

        assert result.passed is False
        assert result.severity == ValidationSeverity.ERROR

    def test_rule_with_auto_fix(self):
        """Test rule with auto-fix capability."""
        rule = ValidationRule(
            rule_id="test_rule_004",
            category=RuleCategory.TRANSFORMATION_APPROVAL,
            name="Auto-fixable Rule",
            description="Rule with auto-fix",
            severity=ValidationSeverity.WARNING,
            validation_fn=lambda x: False,
            auto_fix=True,
            fix_fn=lambda x: x
        )

        result = rule.validate("test_data")

        assert not result.passed
        assert len(result.recommendations) > 0
        assert "Auto-fix available" in result.recommendations[0]


class TestGovernanceValidator:
    """Test suite for GovernanceValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return GovernanceValidator()

    @pytest.fixture
    def sample_canonical(self):
        """Create sample canonical data."""
        return CanonicalData(
            processing_id="test_proc_001",
            extraction_data={"field1": "value1", "field2": "value2"},
            timestamp=datetime.now(),
            schema_version="1.0",
            source_metadata={"integrity_hash": "abc123def456"}
        )

    @pytest.fixture
    def sample_shaped(self):
        """Create sample shaped data."""
        return ShapedData(
            processing_id="test_proc_001_shaped",
            canonical_id="test_proc_001",
            transformed_data={"field1": "value1", "field2": "value2"},
            transformations_applied=[
                TransformationRule(
                    rule_id="rule1",
                    rule_type="normalize",
                    description="Test rule"
                )
            ],
            timestamp=datetime.now(),
            is_1nf_compliant=True
        )

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert len(validator.rules) > 0  # Default rules loaded

    def test_default_rules_loaded(self, validator):
        """Test default rules are registered."""
        # Check for canonical rules
        assert "canonical_001" in validator.rules
        assert "canonical_002" in validator.rules
        assert "canonical_003" in validator.rules

        # Check for transformation rules
        assert "transform_001" in validator.rules
        assert "transform_002" in validator.rules

        # Check for distortion rules
        assert "distortion_001" in validator.rules
        assert "distortion_002" in validator.rules

        # Check for compliance rules
        assert "compliance_001" in validator.rules
        assert "compliance_002" in validator.rules

    def test_register_rule(self, validator):
        """Test registering custom rule."""
        custom_rule = ValidationRule(
            rule_id="custom_001",
            category=RuleCategory.CANONICAL_INTEGRITY,
            name="Custom Rule",
            description="Custom validation rule",
            severity=ValidationSeverity.WARNING,
            validation_fn=lambda x: True
        )

        initial_count = len(validator.rules)
        validator.register_rule(custom_rule)

        assert len(validator.rules) == initial_count + 1
        assert "custom_001" in validator.rules

    def test_unregister_rule(self, validator):
        """Test unregistering rule."""
        # Register a custom rule
        custom_rule = ValidationRule(
            rule_id="custom_002",
            category=RuleCategory.COMPLIANCE,
            name="Temporary Rule",
            description="Rule to be removed",
            severity=ValidationSeverity.INFO,
            validation_fn=lambda x: True
        )

        validator.register_rule(custom_rule)
        assert "custom_002" in validator.rules

        validator.unregister_rule("custom_002")
        assert "custom_002" not in validator.rules

    def test_enable_disable_rule(self, validator):
        """Test enabling and disabling rules."""
        rule_id = "canonical_001"

        # Disable rule
        validator.disable_rule(rule_id)
        assert validator.rules[rule_id].enabled is False

        # Enable rule
        validator.enable_rule(rule_id)
        assert validator.rules[rule_id].enabled is True

    def test_validate_canonical(self, validator, sample_canonical):
        """Test validating canonical data."""
        results = validator.validate_canonical(sample_canonical)

        assert len(results) > 0
        # Should have at least some passing rules
        passing = [r for r in results if r.passed]
        assert len(passing) > 0

    def test_validate_canonical_none_data(self, validator):
        """Test validating canonical with None extraction_data."""
        invalid_canonical = CanonicalData(
            processing_id="test_invalid",
            extraction_data=None,
            timestamp=datetime.now()
        )

        results = validator.validate_canonical(invalid_canonical)

        # Should have failures for None data
        failures = [r for r in results if not r.passed]
        assert len(failures) > 0

        # Should have critical failure
        critical = [r for r in failures if r.severity == ValidationSeverity.CRITICAL]
        assert len(critical) > 0

    def test_validate_shaped(self, validator, sample_shaped):
        """Test validating shaped data."""
        results = validator.validate_shaped(sample_shaped)

        assert len(results) > 0
        # Most rules should pass for valid shaped data
        passing = [r for r in results if r.passed]
        assert len(passing) > 0

    def test_validate_shaped_not_1nf(self, validator):
        """Test validating shaped data that's not 1NF compliant."""
        invalid_shaped = ShapedData(
            processing_id="test_invalid_shaped",
            canonical_id="test_canonical",
            transformed_data={"field": "value"},
            transformations_applied=[],
            timestamp=datetime.now(),
            is_1nf_compliant=False  # Not compliant!
        )

        results = validator.validate_shaped(invalid_shaped)

        # Should fail 1NF compliance check
        failures = [r for r in results if not r.passed and "1NF" in r.message]
        assert len(failures) > 0

    def test_validate_transformation(self, validator, sample_canonical, sample_shaped):
        """Test validating canonical to shaped transformation."""
        results = validator.validate_transformation(sample_canonical, sample_shaped)

        assert len(results) > 0
        # Check transformation-specific rules are run
        transform_results = [
            r for r in results
            if r.category in [RuleCategory.TRANSFORMATION_APPROVAL, RuleCategory.DATA_DISTORTION]
        ]
        assert len(transform_results) > 0

    def test_get_validation_report(self, validator, sample_canonical):
        """Test generating validation report."""
        results = validator.validate_canonical(sample_canonical)
        report = validator.get_validation_report(results)

        assert "timestamp" in report
        assert "total_rules" in report
        assert "passed" in report
        assert "failed" in report
        assert "by_severity" in report
        assert "by_category" in report
        assert "critical_failures" in report
        assert "recommendations" in report

    def test_validation_report_severity_breakdown(self, validator, sample_canonical):
        """Test validation report severity breakdown."""
        results = validator.validate_canonical(sample_canonical)
        report = validator.get_validation_report(results)

        # Check severity breakdown
        for severity in ValidationSeverity:
            assert severity.value in report["by_severity"]
            severity_info = report["by_severity"][severity.value]
            assert "total" in severity_info
            assert "passed" in severity_info
            assert "failed" in severity_info

    def test_validation_report_category_breakdown(self, validator, sample_canonical):
        """Test validation report category breakdown."""
        results = validator.validate_canonical(sample_canonical)
        report = validator.get_validation_report(results)

        # Check category breakdown
        for category in RuleCategory:
            assert category.value in report["by_category"]

    def test_register_approval_workflow(self, validator):
        """Test registering approval workflow."""
        def test_workflow(rule: TransformationRule) -> bool:
            return rule.rule_type == "normalize"

        validator.register_approval_workflow("test_workflow", test_workflow)

        assert "test_workflow" in validator.approval_workflows

    def test_request_transformation_approval_with_workflow(self, validator):
        """Test requesting transformation approval with workflow."""
        # Register workflow that only approves normalize rules
        def strict_workflow(rule: TransformationRule) -> bool:
            return rule.rule_type == "normalize"

        validator.register_approval_workflow("strict", strict_workflow)

        # Test with normalize rule (should pass)
        normalize_rule = TransformationRule(
            rule_id="test_normalize",
            rule_type="normalize",
            description="Test normalization"
        )

        approved = validator.request_transformation_approval(
            normalize_rule,
            workflow_id="strict"
        )

        assert approved is True

        # Test with custom rule (should fail)
        custom_rule = TransformationRule(
            rule_id="test_custom",
            rule_type="custom",
            description="Test custom transformation"
        )

        approved = validator.request_transformation_approval(
            custom_rule,
            workflow_id="strict"
        )

        assert approved is False

    def test_request_transformation_approval_default(self, validator):
        """Test requesting transformation approval with default policy."""
        # Default policy allows normalize, filter, rename, aggregate
        allowed_types = ["normalize", "filter", "rename", "aggregate"]

        for rule_type in allowed_types:
            rule = TransformationRule(
                rule_id=f"test_{rule_type}",
                rule_type=rule_type,
                description=f"Test {rule_type}"
            )

            approved = validator.request_transformation_approval(rule)
            assert approved is True

        # Custom type should be rejected
        custom_rule = TransformationRule(
            rule_id="test_custom",
            rule_type="custom",
            description="Test custom"
        )

        approved = validator.request_transformation_approval(custom_rule)
        assert approved is False

    def test_canonical_integrity_hash_validation(self, validator):
        """Test canonical integrity hash validation."""
        # Valid canonical with hash
        valid_canonical = CanonicalData(
            processing_id="test_001",
            extraction_data={"field": "value"},
            timestamp=datetime.now(),
            source_metadata={"integrity_hash": "abc123"}
        )

        results = validator.validate_canonical(valid_canonical)
        hash_results = [r for r in results if r.rule_id == "canonical_002"]

        # Should pass hash validation
        assert len(hash_results) > 0
        assert any(r.passed for r in hash_results)

        # Invalid canonical without hash
        invalid_canonical = CanonicalData(
            processing_id="test_002",
            extraction_data={"field": "value"},
            timestamp=datetime.now(),
            source_metadata={}  # No hash!
        )

        results = validator.validate_canonical(invalid_canonical)
        hash_results = [r for r in results if r.rule_id == "canonical_002"]

        # Should fail hash validation
        assert len(hash_results) > 0
        assert any(not r.passed for r in hash_results)

    def test_metadata_preservation_validation(self, validator, sample_canonical):
        """Test metadata preservation validation."""
        results = validator.validate_canonical(sample_canonical)

        metadata_results = [r for r in results if r.rule_id == "compliance_001"]

        assert len(metadata_results) > 0
        # Should pass for valid canonical with proper metadata
        assert all(r.passed for r in metadata_results)

    def test_processing_id_consistency_validation(self, validator, sample_canonical, sample_shaped):
        """Test processing ID consistency validation."""
        # Set up shaped data with matching canonical_id
        sample_shaped.canonical_id = sample_canonical.processing_id

        validator._canonical_context = sample_canonical
        results = validator.validate_shaped(sample_shaped)

        consistency_results = [r for r in results if r.rule_id == "compliance_002"]

        assert len(consistency_results) > 0
        assert all(r.passed for r in consistency_results)

    def test_field_count_preservation_warning(self, validator):
        """Test field count preservation warning."""
        # Create canonical with many fields
        canonical = CanonicalData(
            processing_id="test_canonical",
            extraction_data={f"field{i}": f"value{i}" for i in range(20)},
            timestamp=datetime.now()
        )

        # Create shaped with significantly fewer fields
        shaped = ShapedData(
            processing_id="test_shaped",
            canonical_id="test_canonical",
            transformed_data={"field1": "value1"},  # Only 1 field!
            transformations_applied=[],
            timestamp=datetime.now(),
            is_1nf_compliant=True
        )

        validator._canonical_context = canonical
        results = validator.validate_shaped(shaped)

        # Should have warning about field reduction
        distortion_results = [r for r in results if r.category == RuleCategory.DATA_DISTORTION]
        assert len(distortion_results) > 0


class TestGovernanceIntegration:
    """Integration tests for governance system."""

    def test_complete_validation_workflow(self):
        """Test complete validation workflow from canonical to shaped."""
        validator = GovernanceValidator()

        # Create canonical data
        canonical = CanonicalData(
            processing_id="integration_test_001",
            extraction_data={
                "invoice_number": "INV-001",
                "total": 1000.00,
                "items": [
                    {"product": "Widget A", "qty": 2},
                    {"product": "Widget B", "qty": 1}
                ]
            },
            timestamp=datetime.now(),
            schema_version="1.0",
            source_metadata={"integrity_hash": "test_hash_123"}
        )

        # Validate canonical
        canonical_results = validator.validate_canonical(canonical)
        canonical_report = validator.get_validation_report(canonical_results)

        assert canonical_report["passed"] > 0
        assert canonical_report["total_rules"] > 0

        # Create shaped data
        shaped = ShapedData(
            processing_id=f"{canonical.processing_id}_shaped",
            canonical_id=canonical.processing_id,
            transformed_data={
                "invoice_number": "INV-001",
                "total": 1000.00,
                "items_0_product": "Widget A",
                "items_0_qty": 2,
                "items_1_product": "Widget B",
                "items_1_qty": 1
            },
            transformations_applied=[
                TransformationRule(
                    rule_id="flatten_items",
                    rule_type="normalize",
                    description="Flatten nested items"
                )
            ],
            timestamp=datetime.now(),
            is_1nf_compliant=True
        )

        # Validate transformation
        transform_results = validator.validate_transformation(canonical, shaped)
        transform_report = validator.get_validation_report(transform_results)

        assert transform_report["total_rules"] > 0
        assert transform_report["failed"] == 0 or transform_report["by_severity"]["critical"]["failed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

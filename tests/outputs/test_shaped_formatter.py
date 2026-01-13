"""
Tests for ShapedFormatter - 1NF enforcement and business transformations.

COLPALI-702: Shaped data formatter implementation tests.
"""

import pytest
import asyncio
from datetime import datetime
from colpali_engine.outputs.shaped import ShapedFormatter
from colpali_engine.extraction.models import CanonicalData, ShapedData, TransformationRule


class TestShapedFormatter:
    """Test suite for ShapedFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create formatter instance."""
        return ShapedFormatter()

    @pytest.fixture
    def canonical_with_nested_data(self):
        """Canonical data with nested structures."""
        return CanonicalData(
            processing_id="proc_001",
            extraction_data={
                "customer": {
                    "name": "Acme Corp",
                    "address": {
                        "street": "123 Main St",
                        "city": "Springfield"
                    }
                },
                "order_items": [
                    {"product": "Widget A", "quantity": 5},
                    {"product": "Widget B", "quantity": 3}
                ]
            },
            timestamp=datetime.now()
        )

    @pytest.fixture
    def canonical_flat_data(self):
        """Canonical data already in 1NF."""
        return CanonicalData(
            processing_id="proc_002",
            extraction_data={
                "field1": "value1",
                "field2": 123,
                "field3": 45.67
            },
            timestamp=datetime.now()
        )

    @pytest.mark.asyncio
    async def test_transform_to_1nf_basic(self, formatter, canonical_flat_data):
        """Test basic 1NF transformation with flat data."""
        shaped = await formatter.transform_to_1nf(canonical_flat_data)

        assert shaped.processing_id == f"{canonical_flat_data.processing_id}_shaped"
        assert shaped.canonical_id == canonical_flat_data.processing_id
        assert shaped.is_1nf_compliant is True
        assert isinstance(shaped.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_transform_nested_to_1nf(self, formatter, canonical_with_nested_data):
        """Test flattening nested structures to 1NF."""
        shaped = await formatter.transform_to_1nf(
            canonical_with_nested_data,
            transformation_config={"flatten_nested": True}
        )

        # Check that nested structures are flattened
        assert "customer_name" in shaped.transformed_data
        assert "customer_address_street" in shaped.transformed_data
        assert "customer_address_city" in shaped.transformed_data

        # Original nested keys should not exist
        assert "customer" not in shaped.transformed_data or \
               not isinstance(shaped.transformed_data.get("customer"), dict)

    @pytest.mark.asyncio
    async def test_transform_with_preserve_arrays(self, formatter, canonical_with_nested_data):
        """Test transformation preserving arrays."""
        shaped = await formatter.transform_to_1nf(
            canonical_with_nested_data,
            transformation_config={
                "flatten_nested": True,
                "preserve_arrays": True
            }
        )

        # With preserve_arrays, lists might be kept as-is or stringified
        assert shaped.transformed_data is not None

    @pytest.mark.asyncio
    async def test_transformation_rule_application(self, formatter, canonical_flat_data):
        """Test applying registered transformation rules."""
        # Register a transformation rule
        rule = TransformationRule(
            rule_id="test_normalize",
            rule_type="normalize",
            description="Test normalization",
            parameters={"lowercase": True}
        )
        formatter.register_transformation_rule(rule)

        # Apply transformation
        shaped = await formatter.transform_to_1nf(
            canonical_flat_data,
            transformation_config={
                "flatten_nested": False,
                "transformation_rules": ["test_normalize"]
            }
        )

        assert len(shaped.transformations_applied) >= 1
        assert shaped.transformation_count >= 1

    @pytest.mark.asyncio
    async def test_invalid_canonical_data_raises(self, formatter):
        """Test that invalid canonical data raises ValueError."""
        invalid_canonical = CanonicalData(
            processing_id="proc_003",
            extraction_data=None,
            timestamp=datetime.now()
        )

        with pytest.raises(ValueError, match="extraction_data cannot be None"):
            await formatter.transform_to_1nf(invalid_canonical)

    def test_validate_1nf_compliance_atomic_values(self, formatter):
        """Test 1NF validation for atomic values."""
        data = {
            "field1": "value1",
            "field2": 123,
            "field3": 45.67
        }

        assert formatter.validate_1nf_compliance(data) is True

    def test_validate_1nf_compliance_nested_dict_fails(self, formatter):
        """Test 1NF validation fails for nested dictionaries."""
        data = {
            "field1": "value1",
            "nested": {
                "key": "value"
            }
        }

        assert formatter.validate_1nf_compliance(data) is False

    def test_validate_1nf_compliance_nested_list_fails(self, formatter):
        """Test 1NF validation fails for nested lists."""
        data = {
            "field1": "value1",
            "list_field": [1, 2, 3]
        }

        assert formatter.validate_1nf_compliance(data) is False

    def test_validate_1nf_compliance_tabular_data(self, formatter):
        """Test 1NF validation for tabular data."""
        data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]

        assert formatter.validate_1nf_compliance(data) is True

    def test_validate_1nf_compliance_duplicate_records_fails(self, formatter):
        """Test 1NF validation fails for duplicate records."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 1, "name": "Alice"}  # Duplicate
        ]

        assert formatter.validate_1nf_compliance(data) is False

    def test_validate_1nf_compliance_mixed_types_fails(self, formatter):
        """Test 1NF validation fails for mixed column types."""
        data = [
            {"id": 1, "value": "string"},
            {"id": 2, "value": 123}  # Different type
        ]

        assert formatter.validate_1nf_compliance(data) is False

    def test_register_transformation_rule(self, formatter):
        """Test registering transformation rules."""
        rule = TransformationRule(
            rule_id="custom_rule",
            rule_type="custom",
            description="Custom transformation"
        )

        formatter.register_transformation_rule(rule)

        assert "custom_rule" in formatter.transformation_rules
        assert "custom_rule" in formatter.get_available_transformations()

    def test_apply_transformation_rule_normalize(self, formatter):
        """Test applying normalization rule."""
        rule = TransformationRule(
            rule_id="lowercase_norm",
            rule_type="normalize",
            description="Lowercase normalization",
            parameters={"lowercase": True}
        )
        formatter.register_transformation_rule(rule)

        data = {"name": "UPPERCASE", "value": "MixedCase"}
        result = formatter.apply_transformation_rule(data, "lowercase_norm")

        assert result["name"] == "uppercase"
        assert result["value"] == "mixedcase"

    def test_apply_transformation_rule_filter(self, formatter):
        """Test applying filter rule."""
        rule = TransformationRule(
            rule_id="field_filter",
            rule_type="filter",
            description="Keep only specific fields",
            parameters={"keep_fields": ["field1", "field2"]}
        )
        formatter.register_transformation_rule(rule)

        data = {"field1": "a", "field2": "b", "field3": "c"}
        result = formatter.apply_transformation_rule(data, "field_filter")

        assert "field1" in result
        assert "field2" in result
        assert "field3" not in result

    def test_apply_transformation_rule_rename(self, formatter):
        """Test applying rename rule."""
        rule = TransformationRule(
            rule_id="field_rename",
            rule_type="rename",
            description="Rename fields",
            parameters={"field_mapping": {"old_name": "new_name"}}
        )
        formatter.register_transformation_rule(rule)

        data = {"old_name": "value", "other_field": "data"}
        result = formatter.apply_transformation_rule(data, "field_rename")

        assert "new_name" in result
        assert result["new_name"] == "value"
        assert "old_name" not in result

    def test_apply_transformation_rule_not_found(self, formatter):
        """Test applying non-existent rule raises KeyError."""
        data = {"field": "value"}

        with pytest.raises(KeyError, match="Transformation rule 'nonexistent' not found"):
            formatter.apply_transformation_rule(data, "nonexistent")

    def test_flatten_dict(self, formatter):
        """Test dictionary flattening."""
        nested = {
            "level1": {
                "level2": {
                    "level3": "value"
                },
                "other": "data"
            }
        }

        flattened = formatter._flatten_dict(nested)

        assert "level1_level2_level3" in flattened
        assert flattened["level1_level2_level3"] == "value"
        assert "level1_other" in flattened

    def test_check_atomic_values_success(self, formatter):
        """Test atomic value checking with valid data."""
        data = {"field1": "string", "field2": 123, "field3": 45.67}

        assert formatter._check_atomic_values(data) is True

    def test_check_atomic_values_nested_dict_fails(self, formatter):
        """Test atomic value checking fails with nested dict."""
        data = {"field1": "value", "nested": {"key": "value"}}

        assert formatter._check_atomic_values(data) is False

    def test_check_unique_records_success(self, formatter):
        """Test unique record checking with unique records."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]

        assert formatter._check_unique_records(data) is True

    def test_check_unique_records_duplicate_fails(self, formatter):
        """Test unique record checking fails with duplicates."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 1, "name": "Alice"}
        ]

        assert formatter._check_unique_records(data) is False

    def test_check_type_consistency_success(self, formatter):
        """Test type consistency checking with consistent types."""
        data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25}
        ]

        assert formatter._check_type_consistency(data) is True

    def test_check_type_consistency_mixed_fails(self, formatter):
        """Test type consistency checking fails with mixed types."""
        data = [
            {"id": 1, "value": "string"},
            {"id": 2, "value": 123}
        ]

        assert formatter._check_type_consistency(data) is False

    def test_export_transformation_lineage(self, formatter):
        """Test exporting transformation lineage."""
        shaped = ShapedData(
            processing_id="proc_004_shaped",
            canonical_id="proc_004",
            transformed_data={"field": "value"},
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

        lineage = formatter.export_transformation_lineage(shaped)

        assert lineage["processing_id"] == "proc_004_shaped"
        assert lineage["canonical_id"] == "proc_004"
        assert lineage["transformation_count"] == 1
        assert len(lineage["transformations"]) == 1
        assert lineage["1nf_compliant"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for CanonicalFormatter - Truth layer data preservation.

COLPALI-701: Canonical data formatter implementation tests.
"""

import pytest
import asyncio
from datetime import datetime
from colpali_engine.outputs.canonical import CanonicalFormatter
from colpali_engine.extraction.models import CanonicalData


class TestCanonicalFormatter:
    """Test suite for CanonicalFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create formatter instance."""
        return CanonicalFormatter(schema_version="1.0")

    @pytest.fixture
    def sample_extraction_data(self):
        """Sample extraction data."""
        return {
            "invoice_number": "INV-2024-001",
            "total_amount": 1250.50,
            "line_items": [
                {"item": "Product A", "qty": 2, "price": 500.00},
                {"item": "Product B", "qty": 1, "price": 250.50}
            ],
            "customer_name": "Acme Corp"
        }

    @pytest.fixture
    def sample_metadata(self):
        """Sample source metadata."""
        return {
            "document_id": "doc_123",
            "source_file": "invoice.pdf",
            "page_count": 1
        }

    @pytest.mark.asyncio
    async def test_format_extraction_result_basic(self, formatter, sample_extraction_data):
        """Test basic extraction result formatting."""
        processing_id = "proc_001"

        canonical = await formatter.format_extraction_result(
            processing_id=processing_id,
            extraction_data=sample_extraction_data
        )

        assert canonical.processing_id == processing_id
        assert canonical.extraction_data == sample_extraction_data
        assert canonical.schema_version == "1.0"
        assert isinstance(canonical.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_format_with_confidence_scores(self, formatter, sample_extraction_data):
        """Test formatting with confidence scores."""
        confidence_scores = {
            "invoice_number": 0.98,
            "total_amount": 0.95,
            "customer_name": 0.92
        }

        canonical = await formatter.format_extraction_result(
            processing_id="proc_002",
            extraction_data=sample_extraction_data,
            confidence_scores=confidence_scores
        )

        assert canonical.confidence_scores == confidence_scores
        assert canonical.get_field_confidence("invoice_number") == 0.98
        assert canonical.get_field_confidence("unknown_field") is None

    @pytest.mark.asyncio
    async def test_format_with_metadata(self, formatter, sample_extraction_data, sample_metadata):
        """Test formatting with source metadata."""
        canonical = await formatter.format_extraction_result(
            processing_id="proc_003",
            extraction_data=sample_extraction_data,
            source_metadata=sample_metadata
        )

        assert canonical.source_metadata is not None
        assert canonical.source_metadata["document_id"] == "doc_123"
        assert "preserved_at" in canonical.source_metadata
        assert "integrity_hash" in canonical.source_metadata

    @pytest.mark.asyncio
    async def test_integrity_hash_generation(self, formatter, sample_extraction_data):
        """Test that integrity hash is generated."""
        canonical = await formatter.format_extraction_result(
            processing_id="proc_004",
            extraction_data=sample_extraction_data
        )

        assert canonical.source_metadata is not None
        assert "integrity_hash" in canonical.source_metadata
        assert len(canonical.source_metadata["integrity_hash"]) == 64  # SHA-256 hex digest

    @pytest.mark.asyncio
    async def test_data_immutability(self, formatter, sample_extraction_data):
        """Test that original data is not modified."""
        original_data = sample_extraction_data.copy()

        canonical = await formatter.format_extraction_result(
            processing_id="proc_005",
            extraction_data=sample_extraction_data
        )

        # Modify canonical data
        canonical.extraction_data["new_field"] = "modified"

        # Original should be unchanged
        assert sample_extraction_data == original_data
        assert "new_field" not in sample_extraction_data

    @pytest.mark.asyncio
    async def test_format_empty_extraction_data(self, formatter):
        """Test handling of empty extraction data."""
        canonical = await formatter.format_extraction_result(
            processing_id="proc_006",
            extraction_data={}
        )

        assert canonical.extraction_data == {}
        assert canonical.field_count == 0

    @pytest.mark.asyncio
    async def test_format_none_extraction_data_raises(self, formatter):
        """Test that None extraction data raises ValueError."""
        with pytest.raises(ValueError, match="extraction_data cannot be None"):
            await formatter.format_extraction_result(
                processing_id="proc_007",
                extraction_data=None
            )

    @pytest.mark.asyncio
    async def test_format_invalid_data_type_raises(self, formatter):
        """Test that invalid data type raises ValueError."""
        with pytest.raises(ValueError, match="extraction_data must be a dict"):
            await formatter.format_extraction_result(
                processing_id="proc_008",
                extraction_data="invalid"
            )

    def test_validate_data_integrity_success(self, formatter):
        """Test successful integrity validation."""
        canonical = CanonicalData(
            processing_id="proc_009",
            extraction_data={"key": "value"},
            timestamp=datetime.now(),
            source_metadata={}
        )

        # Add integrity hash
        hash_value = formatter._calculate_integrity_hash(canonical)
        canonical.source_metadata["integrity_hash"] = hash_value
        canonical.source_metadata["preserved_at"] = datetime.now().isoformat()

        assert formatter.validate_data_integrity(canonical) is True

    def test_validate_data_integrity_hash_mismatch(self, formatter):
        """Test integrity validation fails on hash mismatch."""
        canonical = CanonicalData(
            processing_id="proc_010",
            extraction_data={"key": "value"},
            timestamp=datetime.now(),
            source_metadata={"integrity_hash": "invalid_hash"}
        )

        assert formatter.validate_data_integrity(canonical) is False

    def test_validate_data_integrity_missing_processing_id(self, formatter):
        """Test validation fails with missing processing_id."""
        canonical = CanonicalData(
            processing_id="",
            extraction_data={"key": "value"},
            timestamp=datetime.now()
        )

        assert formatter.validate_data_integrity(canonical) is False

    def test_validate_data_integrity_missing_extraction_data(self, formatter):
        """Test validation fails with None extraction_data."""
        canonical = CanonicalData(
            processing_id="proc_011",
            extraction_data=None,
            timestamp=datetime.now()
        )

        assert formatter.validate_data_integrity(canonical) is False

    def test_export_for_audit(self, formatter, sample_extraction_data):
        """Test audit export functionality."""
        canonical = CanonicalData(
            processing_id="proc_012",
            extraction_data=sample_extraction_data,
            timestamp=datetime.now(),
            source_metadata={}
        )

        # Add integrity hash
        hash_value = formatter._calculate_integrity_hash(canonical)
        canonical.source_metadata["integrity_hash"] = hash_value

        audit_data = formatter.export_for_audit(canonical)

        assert "audit_metadata" in audit_data
        assert "exported_at" in audit_data["audit_metadata"]
        assert "format_version" in audit_data["audit_metadata"]
        assert "integrity_validated" in audit_data["audit_metadata"]
        assert audit_data["audit_metadata"]["integrity_validated"] is True

    def test_check_for_modifications_no_changes(self, formatter):
        """Test modification detection with no changes."""
        original = CanonicalData(
            processing_id="proc_013",
            extraction_data={"key": "value"},
            timestamp=datetime.now()
        )

        current = CanonicalData(
            processing_id="proc_013",
            extraction_data={"key": "value"},
            timestamp=original.timestamp
        )

        modifications = formatter.check_for_modifications(original, current)
        assert len(modifications) == 0

    def test_check_for_modifications_processing_id_changed(self, formatter):
        """Test modification detection when processing_id changes."""
        timestamp = datetime.now()
        original = CanonicalData(
            processing_id="proc_014",
            extraction_data={"key": "value"},
            timestamp=timestamp
        )

        current = CanonicalData(
            processing_id="proc_015",  # Changed!
            extraction_data={"key": "value"},
            timestamp=timestamp
        )

        modifications = formatter.check_for_modifications(original, current)
        assert "processing_id" in modifications

    def test_check_for_modifications_data_changed(self, formatter):
        """Test modification detection when extraction data changes."""
        timestamp = datetime.now()
        original = CanonicalData(
            processing_id="proc_016",
            extraction_data={"key": "value"},
            timestamp=timestamp
        )

        current = CanonicalData(
            processing_id="proc_016",
            extraction_data={"key": "modified"},  # Changed!
            timestamp=timestamp
        )

        modifications = formatter.check_for_modifications(original, current)
        assert any("integrity hash mismatch" in mod for mod in modifications)

    def test_field_count_property(self):
        """Test field_count property."""
        canonical = CanonicalData(
            processing_id="proc_017",
            extraction_data={"field1": "value1", "field2": "value2", "field3": "value3"},
            timestamp=datetime.now()
        )

        assert canonical.field_count == 3

    def test_to_dict_serialization(self, sample_extraction_data):
        """Test canonical data serialization to dict."""
        canonical = CanonicalData(
            processing_id="proc_018",
            extraction_data=sample_extraction_data,
            timestamp=datetime.now(),
            confidence_scores={"field": 0.95}
        )

        data_dict = canonical.to_dict()

        assert data_dict["processing_id"] == "proc_018"
        assert data_dict["extraction_data"] == sample_extraction_data
        assert "timestamp" in data_dict
        assert data_dict["confidence_scores"] == {"field": 0.95}
        assert data_dict["field_count"] == 4  # invoice_number, total_amount, line_items, customer_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

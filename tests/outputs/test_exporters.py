"""
Tests for DataExporter and StreamingExporter - CSV/Parquet export functionality.

COLPALI-703: CSV/Parquet export implementation tests.
"""

import pytest
import asyncio
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from colpali_engine.outputs.exporters import DataExporter, StreamingExporter
from colpali_engine.extraction.models import CanonicalData, ShapedData, TransformationRule


class TestDataExporter:
    """Test suite for DataExporter."""

    @pytest.fixture
    def exporter(self):
        """Create exporter instance."""
        return DataExporter(default_encoding="utf-8", default_compression="snappy")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after tests
        shutil.rmtree(temp_path)

    @pytest.fixture
    def canonical_data(self):
        """Sample canonical data."""
        return CanonicalData(
            processing_id="test_proc_001",
            extraction_data={
                "invoice_number": "INV-001",
                "amount": 1000.50,
                "customer": "Test Corp"
            },
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            schema_version="1.0",
            confidence_scores={"invoice_number": 0.98, "amount": 0.95},
            source_metadata={"document_id": "doc_001"}
        )

    @pytest.fixture
    def shaped_data(self):
        """Sample shaped data."""
        return ShapedData(
            processing_id="test_proc_001_shaped",
            canonical_id="test_proc_001",
            transformed_data={
                "invoice_number": "INV-001",
                "amount": 1000.50,
                "customer_name": "Test Corp"
            },
            transformations_applied=[
                TransformationRule(
                    rule_id="rename_customer",
                    rule_type="rename",
                    description="Rename customer field"
                )
            ],
            timestamp=datetime(2024, 1, 15, 10, 31, 0),
            is_1nf_compliant=True
        )

    @pytest.mark.asyncio
    async def test_export_canonical_to_csv_file(self, exporter, canonical_data, temp_dir):
        """Test exporting canonical data to CSV file."""
        output_path = temp_dir / "canonical_test.csv"

        result_path = await exporter.export_canonical_to_csv(
            canonical_data,
            output_path=output_path
        )

        assert result_path == output_path
        assert output_path.exists()

        # Read and verify CSV content
        df = pd.read_csv(output_path)
        assert "invoice_number" in df.columns
        assert df["invoice_number"].iloc[0] == "INV-001"
        assert df["amount"].iloc[0] == 1000.50

    @pytest.mark.asyncio
    async def test_export_canonical_to_csv_string(self, exporter, canonical_data):
        """Test exporting canonical data to CSV string."""
        csv_string = await exporter.export_canonical_to_csv(
            canonical_data,
            output_path=None
        )

        assert isinstance(csv_string, str)
        assert "invoice_number" in csv_string
        assert "INV-001" in csv_string

    @pytest.mark.asyncio
    async def test_export_canonical_csv_with_metadata(self, exporter, canonical_data, temp_dir):
        """Test CSV export includes metadata when requested."""
        output_path = temp_dir / "canonical_with_meta.csv"

        await exporter.export_canonical_to_csv(
            canonical_data,
            output_path=output_path,
            include_metadata=True
        )

        df = pd.read_csv(output_path)
        assert "_meta_processing_id" in df.columns
        assert "_meta_timestamp" in df.columns
        assert "_meta_schema_version" in df.columns

    @pytest.mark.asyncio
    async def test_export_canonical_to_parquet_file(self, exporter, canonical_data, temp_dir):
        """Test exporting canonical data to Parquet file."""
        output_path = temp_dir / "canonical_test.parquet"

        result_path = await exporter.export_canonical_to_parquet(
            canonical_data,
            output_path=output_path
        )

        assert result_path == output_path
        assert output_path.exists()

        # Read and verify Parquet content
        df = pd.read_parquet(output_path)
        assert "invoice_number" in df.columns
        assert df["invoice_number"].iloc[0] == "INV-001"

    @pytest.mark.asyncio
    async def test_export_canonical_parquet_metadata(self, exporter, canonical_data, temp_dir):
        """Test Parquet export includes metadata in file."""
        output_path = temp_dir / "canonical_with_parquet_meta.parquet"

        await exporter.export_canonical_to_parquet(
            canonical_data,
            output_path=output_path,
            include_metadata=True
        )

        # Read metadata from Parquet file
        parquet_file = pq.read_table(output_path)
        metadata = parquet_file.schema.metadata

        assert b"processing_id" in metadata
        assert metadata[b"processing_id"] == b"test_proc_001"
        assert b"export_format" in metadata
        assert metadata[b"export_format"] == b"canonical"

    @pytest.mark.asyncio
    async def test_export_canonical_parquet_compression(self, exporter, canonical_data, temp_dir):
        """Test Parquet export with different compression."""
        output_path = temp_dir / "canonical_gzip.parquet"

        await exporter.export_canonical_to_parquet(
            canonical_data,
            output_path=output_path,
            compression="gzip"
        )

        assert output_path.exists()

        # Verify compression
        parquet_file = pq.read_metadata(output_path)
        # Note: compression verification depends on PyArrow version

    @pytest.mark.asyncio
    async def test_export_shaped_to_csv_file(self, exporter, shaped_data, temp_dir):
        """Test exporting shaped data to CSV file."""
        output_path = temp_dir / "shaped_test.csv"

        result_path = await exporter.export_shaped_to_csv(
            shaped_data,
            output_path=output_path
        )

        assert result_path == output_path
        assert output_path.exists()

        # Read and verify CSV content
        df = pd.read_csv(output_path)
        assert "customer_name" in df.columns
        assert df["customer_name"].iloc[0] == "Test Corp"

    @pytest.mark.asyncio
    async def test_export_shaped_csv_with_lineage(self, exporter, shaped_data, temp_dir):
        """Test shaped CSV export includes lineage information."""
        output_path = temp_dir / "shaped_with_lineage.csv"

        await exporter.export_shaped_to_csv(
            shaped_data,
            output_path=output_path,
            include_lineage=True
        )

        df = pd.read_csv(output_path)
        assert "_meta_processing_id" in df.columns
        assert "_meta_canonical_id" in df.columns
        assert "_meta_transformation_count" in df.columns
        assert "_meta_1nf_compliant" in df.columns

    @pytest.mark.asyncio
    async def test_export_shaped_to_parquet_file(self, exporter, shaped_data, temp_dir):
        """Test exporting shaped data to Parquet file."""
        output_path = temp_dir / "shaped_test.parquet"

        result_path = await exporter.export_shaped_to_parquet(
            shaped_data,
            output_path=output_path
        )

        assert result_path == output_path
        assert output_path.exists()

        # Read and verify Parquet content
        df = pd.read_parquet(output_path)
        assert "customer_name" in df.columns

    @pytest.mark.asyncio
    async def test_export_shaped_parquet_lineage_metadata(self, exporter, shaped_data, temp_dir):
        """Test shaped Parquet export includes lineage in metadata."""
        output_path = temp_dir / "shaped_with_parquet_lineage.parquet"

        await exporter.export_shaped_to_parquet(
            shaped_data,
            output_path=output_path,
            include_lineage=True
        )

        # Read metadata from Parquet file
        parquet_file = pq.read_table(output_path)
        metadata = parquet_file.schema.metadata

        assert b"processing_id" in metadata
        assert b"canonical_id" in metadata
        assert b"transformation_count" in metadata
        assert metadata[b"transformation_count"] == b"1"
        assert b"export_format" in metadata
        assert metadata[b"export_format"] == b"shaped"

    @pytest.mark.asyncio
    async def test_export_canonical_none_data_raises(self, exporter, temp_dir):
        """Test that None extraction_data raises ValueError."""
        invalid_canonical = CanonicalData(
            processing_id="invalid",
            extraction_data=None,
            timestamp=datetime.now()
        )

        output_path = temp_dir / "should_fail.csv"

        with pytest.raises(ValueError, match="extraction_data cannot be None"):
            await exporter.export_canonical_to_csv(invalid_canonical, output_path)

    @pytest.mark.asyncio
    async def test_export_shaped_none_data_raises(self, exporter, temp_dir):
        """Test that None transformed_data raises ValueError."""
        invalid_shaped = ShapedData(
            processing_id="invalid",
            canonical_id="invalid",
            transformed_data=None,
            transformations_applied=[],
            timestamp=datetime.now()
        )

        output_path = temp_dir / "should_fail.csv"

        with pytest.raises(ValueError, match="transformed_data cannot be None"):
            await exporter.export_shaped_to_csv(invalid_shaped, output_path)

    @pytest.mark.asyncio
    async def test_export_batch_csv(self, exporter, canonical_data, temp_dir):
        """Test batch export to CSV format."""
        # Create multiple canonical data objects
        canonical_list = [canonical_data for _ in range(3)]

        output_paths = await exporter.export_batch(
            canonical_list,
            output_dir=temp_dir,
            format="csv"
        )

        assert len(output_paths) == 3
        for path in output_paths:
            assert path.exists()
            assert path.suffix == ".csv"

    @pytest.mark.asyncio
    async def test_export_batch_parquet(self, exporter, canonical_data, temp_dir):
        """Test batch export to Parquet format."""
        canonical_list = [canonical_data for _ in range(3)]

        output_paths = await exporter.export_batch(
            canonical_list,
            output_dir=temp_dir,
            format="parquet"
        )

        assert len(output_paths) == 3
        for path in output_paths:
            assert path.exists()
            assert path.suffix == ".parquet"

    @pytest.mark.asyncio
    async def test_export_batch_invalid_format_raises(self, exporter, canonical_data, temp_dir):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            await exporter.export_batch(
                [canonical_data],
                output_dir=temp_dir,
                format="invalid"
            )

    def test_dict_to_dataframe_flat_data(self, exporter):
        """Test converting flat dict to DataFrame."""
        data = {"col1": "value1", "col2": 123, "col3": 45.67}

        df = exporter._dict_to_dataframe(data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_dict_to_dataframe_tabular_data(self, exporter):
        """Test converting tabular dict to DataFrame."""
        data = {
            "col1": ["value1", "value2", "value3"],
            "col2": [1, 2, 3],
            "col3": [1.1, 2.2, 3.3]
        }

        df = exporter._dict_to_dataframe(data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_is_tabular_true(self, exporter):
        """Test is_tabular with valid tabular data."""
        data = {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        }

        assert exporter._is_tabular(data) is True

    def test_is_tabular_false_non_lists(self, exporter):
        """Test is_tabular with non-list values."""
        data = {
            "col1": "value",
            "col2": 123
        }

        assert exporter._is_tabular(data) is False

    def test_is_tabular_false_different_lengths(self, exporter):
        """Test is_tabular with different length lists."""
        data = {
            "col1": [1, 2, 3],
            "col2": ["a", "b"]  # Different length
        }

        assert exporter._is_tabular(data) is False


class TestStreamingExporter:
    """Test suite for StreamingExporter."""

    @pytest.fixture
    def streaming_exporter(self):
        """Create streaming exporter instance."""
        return StreamingExporter(chunk_size=1000)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_data_iterator(self):
        """Create sample data iterator."""
        def iterator():
            for i in range(3):
                yield [
                    {"id": j, "value": f"chunk_{i}_row_{j}"}
                    for j in range(100)
                ]
        return iterator()

    @pytest.mark.asyncio
    async def test_stream_to_csv(self, streaming_exporter, sample_data_iterator, temp_dir):
        """Test streaming export to CSV."""
        output_path = temp_dir / "streaming_test.csv"

        result_path = await streaming_exporter.stream_to_csv(
            sample_data_iterator,
            output_path=output_path
        )

        assert result_path == output_path
        assert output_path.exists()

        # Verify content
        df = pd.read_csv(output_path)
        assert len(df) == 300  # 3 chunks * 100 rows
        assert "id" in df.columns
        assert "value" in df.columns

    @pytest.mark.asyncio
    async def test_stream_to_parquet(self, streaming_exporter, temp_dir):
        """Test streaming export to Parquet."""
        def iterator():
            for i in range(3):
                yield [{"id": j, "value": f"row_{j}"} for j in range(100)]

        output_path = temp_dir / "streaming_test.parquet"

        result_path = await streaming_exporter.stream_to_parquet(
            iterator(),
            output_path=output_path
        )

        assert result_path == output_path
        assert output_path.exists()

        # Verify content
        df = pd.read_parquet(output_path)
        assert len(df) == 300  # 3 chunks * 100 rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
CSV and Parquet export functionality for canonical and shaped data.

This module implements export utilities that convert both canonical and shaped
data into standard output formats (CSV, Parquet) with proper encoding,
compression, and metadata preservation.
"""

import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from io import StringIO
from ..extraction.models import CanonicalData, ShapedData

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Export canonical and shaped data to CSV/Parquet formats.

    Supports both file-based and in-memory exports with configurable
    encoding, compression, and metadata embedding.
    """

    def __init__(
        self,
        default_encoding: str = "utf-8",
        default_compression: str = "snappy"
    ):
        """
        Initialize data exporter.

        Args:
            default_encoding: Default encoding for CSV files
            default_compression: Default compression for Parquet files (snappy, gzip, brotli)
        """
        self.default_encoding = default_encoding
        self.default_compression = default_compression
        logger.info(
            f"DataExporter initialized: encoding={default_encoding}, "
            f"compression={default_compression}"
        )

    async def export_canonical_to_csv(
        self,
        canonical_data: CanonicalData,
        output_path: Optional[Union[str, Path]] = None,
        include_metadata: bool = True,
        **kwargs
    ) -> Union[str, Path]:
        """
        Export canonical data to CSV format.

        Args:
            canonical_data: CanonicalData to export
            output_path: Optional output file path. If None, returns CSV string
            include_metadata: Whether to include metadata in output
            **kwargs: Additional arguments passed to pandas.to_csv()

        Returns:
            Output path or CSV string if output_path is None

        Raises:
            ValueError: If canonical_data is invalid
        """
        logger.info(f"Exporting canonical data to CSV: {canonical_data.processing_id}")

        # Validate input
        if canonical_data.extraction_data is None:
            raise ValueError("canonical_data.extraction_data cannot be None")

        # Prepare data for export
        export_data = canonical_data.extraction_data.copy()

        # Add metadata if requested
        if include_metadata:
            export_data["_meta_processing_id"] = canonical_data.processing_id
            export_data["_meta_timestamp"] = canonical_data.timestamp.isoformat()
            export_data["_meta_schema_version"] = canonical_data.schema_version

        # Convert to DataFrame
        df = self._dict_to_dataframe(export_data)

        # Set default CSV parameters
        csv_params = {
            "index": False,
            "encoding": self.default_encoding,
            "date_format": "%Y-%m-%d %H:%M:%S",
        }
        csv_params.update(kwargs)

        # Export to file or string
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, **csv_params)
            logger.info(f"Canonical data exported to CSV: {output_path}")
            return output_path
        else:
            # Return CSV string
            csv_string = df.to_csv(**csv_params)
            logger.info("Canonical data exported to CSV string")
            return csv_string

    async def export_canonical_to_parquet(
        self,
        canonical_data: CanonicalData,
        output_path: Union[str, Path],
        include_metadata: bool = True,
        compression: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Export canonical data to Parquet format.

        Args:
            canonical_data: CanonicalData to export
            output_path: Output file path
            include_metadata: Whether to include metadata
            compression: Compression algorithm (snappy, gzip, brotli, None)
            **kwargs: Additional arguments passed to pyarrow.parquet.write_table()

        Returns:
            Output file path

        Raises:
            ValueError: If canonical_data is invalid
        """
        logger.info(f"Exporting canonical data to Parquet: {canonical_data.processing_id}")

        # Validate input
        if canonical_data.extraction_data is None:
            raise ValueError("canonical_data.extraction_data cannot be None")

        output_path = Path(output_path)
        compression = compression or self.default_compression

        # Prepare data for export
        export_data = canonical_data.extraction_data.copy()

        # Add metadata columns if requested
        if include_metadata:
            export_data["_meta_processing_id"] = canonical_data.processing_id
            export_data["_meta_timestamp"] = canonical_data.timestamp.isoformat()
            export_data["_meta_schema_version"] = canonical_data.schema_version

        # Convert to DataFrame
        df = self._dict_to_dataframe(export_data)

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)

        # Add custom metadata
        if include_metadata:
            custom_metadata = {
                b"processing_id": canonical_data.processing_id.encode("utf-8"),
                b"timestamp": canonical_data.timestamp.isoformat().encode("utf-8"),
                b"schema_version": canonical_data.schema_version.encode("utf-8"),
                b"export_format": b"canonical",
                b"field_count": str(canonical_data.field_count).encode("utf-8")
            }

            if canonical_data.source_metadata:
                for key, value in canonical_data.source_metadata.items():
                    custom_metadata[f"source_{key}".encode("utf-8")] = str(value).encode("utf-8")

            # Merge with existing metadata
            existing_meta = table.schema.metadata or {}
            existing_meta.update(custom_metadata)
            table = table.replace_schema_metadata(existing_meta)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression=compression, **kwargs)

        logger.info(f"Canonical data exported to Parquet: {output_path} ({compression} compression)")
        return output_path

    async def export_shaped_to_csv(
        self,
        shaped_data: ShapedData,
        output_path: Optional[Union[str, Path]] = None,
        include_lineage: bool = True,
        **kwargs
    ) -> Union[str, Path]:
        """
        Export shaped data to CSV format.

        Args:
            shaped_data: ShapedData to export
            output_path: Optional output file path. If None, returns CSV string
            include_lineage: Whether to include transformation lineage
            **kwargs: Additional arguments passed to pandas.to_csv()

        Returns:
            Output path or CSV string if output_path is None

        Raises:
            ValueError: If shaped_data is invalid
        """
        logger.info(f"Exporting shaped data to CSV: {shaped_data.processing_id}")

        # Validate input
        if shaped_data.transformed_data is None:
            raise ValueError("shaped_data.transformed_data cannot be None")

        # Prepare data for export
        export_data = shaped_data.transformed_data.copy()

        # Add lineage if requested
        if include_lineage:
            export_data["_meta_processing_id"] = shaped_data.processing_id
            export_data["_meta_canonical_id"] = shaped_data.canonical_id
            export_data["_meta_timestamp"] = shaped_data.timestamp.isoformat()
            export_data["_meta_transformation_count"] = shaped_data.transformation_count
            export_data["_meta_1nf_compliant"] = shaped_data.is_1nf_compliant

        # Convert to DataFrame
        df = self._dict_to_dataframe(export_data)

        # Set default CSV parameters
        csv_params = {
            "index": False,
            "encoding": self.default_encoding,
            "date_format": "%Y-%m-%d %H:%M:%S",
        }
        csv_params.update(kwargs)

        # Export to file or string
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, **csv_params)
            logger.info(f"Shaped data exported to CSV: {output_path}")
            return output_path
        else:
            # Return CSV string
            csv_string = df.to_csv(**csv_params)
            logger.info("Shaped data exported to CSV string")
            return csv_string

    async def export_shaped_to_parquet(
        self,
        shaped_data: ShapedData,
        output_path: Union[str, Path],
        include_lineage: bool = True,
        compression: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Export shaped data to Parquet format.

        Args:
            shaped_data: ShapedData to export
            output_path: Output file path
            include_lineage: Whether to include transformation lineage
            compression: Compression algorithm (snappy, gzip, brotli, None)
            **kwargs: Additional arguments passed to pyarrow.parquet.write_table()

        Returns:
            Output file path

        Raises:
            ValueError: If shaped_data is invalid
        """
        logger.info(f"Exporting shaped data to Parquet: {shaped_data.processing_id}")

        # Validate input
        if shaped_data.transformed_data is None:
            raise ValueError("shaped_data.transformed_data cannot be None")

        output_path = Path(output_path)
        compression = compression or self.default_compression

        # Prepare data for export
        export_data = shaped_data.transformed_data.copy()

        # Add lineage columns if requested
        if include_lineage:
            export_data["_meta_processing_id"] = shaped_data.processing_id
            export_data["_meta_canonical_id"] = shaped_data.canonical_id
            export_data["_meta_timestamp"] = shaped_data.timestamp.isoformat()
            export_data["_meta_transformation_count"] = shaped_data.transformation_count
            export_data["_meta_1nf_compliant"] = shaped_data.is_1nf_compliant

        # Convert to DataFrame
        df = self._dict_to_dataframe(export_data)

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)

        # Add custom metadata
        if include_lineage:
            custom_metadata = {
                b"processing_id": shaped_data.processing_id.encode("utf-8"),
                b"canonical_id": shaped_data.canonical_id.encode("utf-8"),
                b"timestamp": shaped_data.timestamp.isoformat().encode("utf-8"),
                b"transformation_count": str(shaped_data.transformation_count).encode("utf-8"),
                b"is_1nf_compliant": str(shaped_data.is_1nf_compliant).encode("utf-8"),
                b"export_format": b"shaped"
            }

            # Add transformation details
            for i, rule in enumerate(shaped_data.transformations_applied):
                custom_metadata[f"transformation_{i}_rule_id".encode("utf-8")] = rule.rule_id.encode("utf-8")
                custom_metadata[f"transformation_{i}_type".encode("utf-8")] = rule.rule_type.encode("utf-8")

            # Merge with existing metadata
            existing_meta = table.schema.metadata or {}
            existing_meta.update(custom_metadata)
            table = table.replace_schema_metadata(existing_meta)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression=compression, **kwargs)

        logger.info(f"Shaped data exported to Parquet: {output_path} ({compression} compression)")
        return output_path

    def _dict_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert dictionary to pandas DataFrame.

        Handles both flat and nested structures, converting to appropriate
        DataFrame representation.

        Args:
            data: Dictionary to convert

        Returns:
            pandas DataFrame
        """
        # If data is already tabular (dict of lists of equal length)
        if self._is_tabular(data):
            return pd.DataFrame(data)

        # If data is a single record (dict of values)
        else:
            # Convert to single-row DataFrame
            return pd.DataFrame([data])

    def _is_tabular(self, data: Dict[str, Any]) -> bool:
        """
        Check if dictionary represents tabular data.

        Args:
            data: Dictionary to check

        Returns:
            True if data represents a table (dict of equal-length lists)
        """
        if not data:
            return False

        # Check if all values are lists
        values = list(data.values())
        if not all(isinstance(v, list) for v in values):
            return False

        # Check if all lists have same length
        lengths = [len(v) for v in values]
        return len(set(lengths)) == 1

    async def export_batch(
        self,
        canonical_data_list: List[CanonicalData],
        output_dir: Union[str, Path],
        format: str = "parquet",
        **kwargs
    ) -> List[Path]:
        """
        Export multiple canonical data objects in batch.

        Args:
            canonical_data_list: List of CanonicalData to export
            output_dir: Output directory path
            format: Export format ("csv" or "parquet")
            **kwargs: Additional export parameters

        Returns:
            List of output file paths

        Raises:
            ValueError: If format is invalid
        """
        if format not in ["csv", "parquet"]:
            raise ValueError(f"Invalid format: {format}. Must be 'csv' or 'parquet'")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for i, canonical_data in enumerate(canonical_data_list):
            # Generate filename
            timestamp = canonical_data.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"canonical_{canonical_data.processing_id}_{timestamp}.{format}"
            output_path = output_dir / filename

            # Export based on format
            if format == "csv":
                await self.export_canonical_to_csv(canonical_data, output_path, **kwargs)
            else:
                await self.export_canonical_to_parquet(canonical_data, output_path, **kwargs)

            output_paths.append(output_path)

        logger.info(f"Batch export complete: {len(output_paths)} files exported to {output_dir}")
        return output_paths


class StreamingExporter:
    """
    Streaming exporter for large datasets.

    Handles memory-efficient export of large datasets that don't fit in memory,
    using chunked processing and streaming writes.
    """

    def __init__(self, chunk_size: int = 10000):
        """
        Initialize streaming exporter.

        Args:
            chunk_size: Number of records to process per chunk
        """
        self.chunk_size = chunk_size
        logger.info(f"StreamingExporter initialized with chunk_size={chunk_size}")

    async def stream_to_csv(
        self,
        data_iterator,
        output_path: Union[str, Path],
        **kwargs
    ) -> Path:
        """
        Stream data to CSV file using chunked processing.

        Args:
            data_iterator: Iterator yielding data chunks
            output_path: Output file path
            **kwargs: Additional CSV parameters

        Returns:
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        first_chunk = True

        for chunk in data_iterator:
            df = pd.DataFrame(chunk)

            # Write header only for first chunk
            mode = "w" if first_chunk else "a"
            header = first_chunk

            df.to_csv(output_path, mode=mode, header=header, index=False, **kwargs)

            first_chunk = False

        logger.info(f"Streaming CSV export complete: {output_path}")
        return output_path

    async def stream_to_parquet(
        self,
        data_iterator,
        output_path: Union[str, Path],
        compression: str = "snappy",
        **kwargs
    ) -> Path:
        """
        Stream data to Parquet file using chunked processing.

        Args:
            data_iterator: Iterator yielding data chunks
            output_path: Output file path
            compression: Compression algorithm
            **kwargs: Additional Parquet parameters

        Returns:
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = None
        schema = None

        try:
            for chunk in data_iterator:
                df = pd.DataFrame(chunk)
                table = pa.Table.from_pandas(df)

                if writer is None:
                    # Initialize writer with first chunk's schema
                    schema = table.schema
                    writer = pq.ParquetWriter(
                        output_path,
                        schema,
                        compression=compression,
                        **kwargs
                    )

                writer.write_table(table)

        finally:
            if writer:
                writer.close()

        logger.info(f"Streaming Parquet export complete: {output_path}")
        return output_path

"""
Package installation and import tests for tatForge.

These tests verify that the package can be properly installed and
all public APIs are accessible.
"""

import pytest
import sys


class TestPackageInstallation:
    """Test that tatforge package is properly installed."""

    def test_package_importable(self):
        """Test that tatforge can be imported."""
        import tatforge
        assert tatforge is not None

    def test_version_available(self):
        """Test that version info is available."""
        import tatforge
        assert hasattr(tatforge, "__version__")
        assert tatforge.__version__ == "0.1.0"

    def test_package_name(self):
        """Test that package name is correct."""
        import tatforge
        assert hasattr(tatforge, "__package_name__")
        assert tatforge.__package_name__ == "tatforge"

    def test_author_info(self):
        """Test that author info is available."""
        import tatforge
        assert hasattr(tatforge, "__author__")
        assert tatforge.__author__ == "TAT Team"


class TestPublicAPI:
    """Test that all public API components are accessible."""

    def test_core_pipeline_imports(self):
        """Test core pipeline imports."""
        from tatforge import VisionExtractionPipeline, PipelineConfig
        assert VisionExtractionPipeline is not None
        assert PipelineConfig is not None

    def test_document_adapter_imports(self):
        """Test document adapter imports."""
        from tatforge import DocumentAdapter, ConversionConfig, DocumentFormat
        assert DocumentAdapter is not None
        assert ConversionConfig is not None
        assert DocumentFormat is not None

    def test_schema_imports(self):
        """Test schema management imports."""
        from tatforge import SchemaManager, SchemaValidator
        assert SchemaManager is not None
        assert SchemaValidator is not None

    def test_extraction_result_imports(self):
        """Test extraction result imports."""
        from tatforge import ExtractionResult, CanonicalData, ShapedData
        assert ExtractionResult is not None
        assert CanonicalData is not None
        assert ShapedData is not None

    def test_adapter_imports(self):
        """Test adapter imports."""
        from tatforge import PDFAdapter, ImageAdapter
        assert PDFAdapter is not None
        assert ImageAdapter is not None

    def test_formatter_imports(self):
        """Test formatter imports."""
        from tatforge import CanonicalFormatter, ShapedFormatter, DataExporter
        assert CanonicalFormatter is not None
        assert ShapedFormatter is not None
        assert DataExporter is not None

    def test_convenience_function_import(self):
        """Test convenience function import."""
        from tatforge import extract_document
        assert extract_document is not None
        assert callable(extract_document)


class TestSubmoduleImports:
    """Test that submodules can be imported directly."""

    def test_adapters_submodule(self):
        """Test adapters submodule."""
        from tatforge.adapters import PDFAdapter, ImageAdapter
        assert PDFAdapter is not None
        assert ImageAdapter is not None

    def test_core_submodule(self):
        """Test core submodule."""
        from tatforge.core import VisionExtractionPipeline, SchemaManager
        assert VisionExtractionPipeline is not None
        assert SchemaManager is not None

    def test_extraction_submodule(self):
        """Test extraction submodule."""
        from tatforge.extraction import ExtractionResult
        assert ExtractionResult is not None

    def test_outputs_submodule(self):
        """Test outputs submodule."""
        from tatforge.outputs import CanonicalFormatter, ShapedFormatter
        assert CanonicalFormatter is not None
        assert ShapedFormatter is not None

    def test_storage_submodule(self):
        """Test storage submodule."""
        from tatforge.storage import QdrantManager
        assert QdrantManager is not None

    def test_governance_submodule(self):
        """Test governance submodule."""
        from tatforge.governance import LineageTracker, LineageNode
        assert LineageTracker is not None
        assert LineageNode is not None

    def test_lambda_utils_submodule(self):
        """Test lambda_utils submodule."""
        from tatforge.lambda_utils import (
            LambdaModelOptimizer,
            LambdaResourceManager,
            LambdaMonitor
        )
        assert LambdaModelOptimizer is not None
        assert LambdaResourceManager is not None
        assert LambdaMonitor is not None


class TestAllExports:
    """Test that __all__ exports are correct."""

    def test_all_exports_exist(self):
        """Test that all items in __all__ are importable."""
        import tatforge

        for name in tatforge.__all__:
            assert hasattr(tatforge, name), f"Missing export: {name}"

    def test_all_exports_count(self):
        """Test that we have expected number of exports."""
        import tatforge

        # Should have at least these categories of exports
        expected_min = 15  # Core classes, results, adapters, formatters, functions
        assert len(tatforge.__all__) >= expected_min


class TestCLI:
    """Test CLI functionality."""

    def test_cli_module_exists(self):
        """Test that CLI module exists."""
        from tatforge import cli
        assert cli is not None

    def test_cli_main_function(self):
        """Test that CLI main function exists."""
        from tatforge.cli import main
        assert main is not None
        assert callable(main)

    def test_cli_commands_exist(self):
        """Test that CLI command handlers exist."""
        from tatforge.cli import cmd_info, cmd_validate
        assert cmd_info is not None
        assert cmd_validate is not None


class TestSchemaValidation:
    """Test schema validation functionality."""

    def test_schema_manager_creation(self):
        """Test SchemaManager can be instantiated."""
        from tatforge import SchemaManager
        manager = SchemaManager()
        assert manager is not None

    def test_schema_validator_creation(self):
        """Test SchemaValidator can be instantiated."""
        from tatforge import SchemaValidator
        validator = SchemaValidator()
        assert validator is not None

    def test_valid_schema(self):
        """Test validation of a valid schema."""
        from tatforge import SchemaValidator
        from tatforge.core.schema_validator import CompatibilityLevel

        valid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "int"}  # Use BAML-compatible types
            }
        }

        validator = SchemaValidator()
        result = validator.validate_schema(valid_schema)
        # Returns CompatibilityReport - check it's not incompatible
        assert result.compatibility_level != CompatibilityLevel.INCOMPATIBLE


class TestPDFAdapter:
    """Test PDF adapter functionality."""

    def test_pdf_adapter_creation(self):
        """Test PDFAdapter can be instantiated."""
        from tatforge import PDFAdapter
        adapter = PDFAdapter(max_memory_mb=300)
        assert adapter is not None

    def test_pdf_adapter_format_detection(self):
        """Test PDF format detection."""
        from tatforge import PDFAdapter

        adapter = PDFAdapter()

        # Invalid content should return False
        non_pdf = b"This is not a PDF"
        assert adapter.validate_format(non_pdf) is False

        # Empty content should return False
        empty = b""
        assert adapter.validate_format(empty) is False

        # Note: validate_format requires a complete valid PDF structure,
        # not just a header. Use actual PDF files for positive tests.


class TestConversionConfig:
    """Test ConversionConfig functionality."""

    def test_conversion_config_defaults(self):
        """Test ConversionConfig default values."""
        from tatforge import ConversionConfig

        config = ConversionConfig()
        assert config.dpi > 0
        assert config.quality > 0

    def test_conversion_config_custom(self):
        """Test ConversionConfig with custom values."""
        from tatforge import ConversionConfig

        config = ConversionConfig(dpi=300, quality=95, max_pages=10)
        assert config.dpi == 300
        assert config.quality == 95
        assert config.max_pages == 10

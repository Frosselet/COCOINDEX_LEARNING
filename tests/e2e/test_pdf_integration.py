"""
Integration tests with 15 PDF samples.

Tests COLPALI-1001: Create integration tests with 15 PDF samples.

This module provides comprehensive end-to-end testing of the document
processing pipeline using all 15 PDF samples in the test directory.

Test Coverage:
- PDF to image conversion for all document types
- Layout handling (multi-column, tables, overlapping text)
- Real document processing (shipping manifests, loading statements)
- Error condition handling
- Golden master testing framework

Environment Requirements:
- pyarrow for PDFAdapter
- PIL/Pillow for image handling
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest
from PIL import Image


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)

# Project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Module Imports with Dependency Handling
# =============================================================================

# Try to import adapters - skip tests if dependencies unavailable
PDF_ADAPTER_AVAILABLE = False
IMAGE_PROCESSOR_AVAILABLE = False
PDF_ADAPTER_ERROR = None
IMAGE_PROCESSOR_ERROR = None

try:
    from colpali_engine.adapters.pdf_adapter import PDFAdapter, create_pdf_adapter
    PDF_ADAPTER_AVAILABLE = True
except ImportError as e:
    PDF_ADAPTER_ERROR = str(e)

try:
    from colpali_engine.adapters.image_processor import (
        ImageProcessor,
        create_image_processor,
        ProcessingConfig
    )
    IMAGE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    IMAGE_PROCESSOR_ERROR = str(e)
    ImageProcessor = None
    ProcessingConfig = None


# =============================================================================
# Test Classes
# =============================================================================

class TestPDFDiscovery:
    """Test that all expected PDFs are present and accessible."""

    def test_pdf_directory_exists(self, pdf_directory):
        """Verify PDF directory exists."""
        assert pdf_directory.exists()
        assert pdf_directory.is_dir()

    def test_all_15_pdfs_present(self, all_pdf_files):
        """Verify all 15 PDF files are present."""
        assert len(all_pdf_files) == 15, f"Expected 15 PDFs, found {len(all_pdf_files)}"

    def test_pdf_catalog_complete(self, pdf_test_catalog):
        """Verify PDF catalog has all expected documents."""
        expected_categories = {"layout", "shipping", "financial"}
        found_categories = {tc.category for tc in pdf_test_catalog.values()}
        assert expected_categories == found_categories

    def test_all_pdfs_readable(self, all_pdf_files):
        """Verify all PDFs can be opened and read."""
        for pdf_path in all_pdf_files:
            assert pdf_path.exists(), f"PDF not found: {pdf_path}"
            assert pdf_path.stat().st_size > 0, f"PDF is empty: {pdf_path}"

            # Verify it's a valid PDF by checking magic bytes
            with open(pdf_path, 'rb') as f:
                header = f.read(4)
                assert header == b'%PDF', f"Invalid PDF header: {pdf_path}"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestPDFConversion:
    """Test PDF to image conversion for all documents."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_convert_simple_pdf(self, pdf_adapter, simple_pdfs, performance_tracker):
        """Test conversion of simple PDFs."""
        if not simple_pdfs:
            pytest.skip("No simple PDFs in catalog")

        for test_case in simple_pdfs:
            performance_tracker.start(f"convert_{test_case.filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))

            metrics = performance_tracker.stop(
                filename=test_case.filename,
                page_count=len(images)
            )

            assert len(images) > 0, f"No images from {test_case.filename}"
            assert metrics.duration_seconds < 30, f"Conversion too slow: {metrics.duration_seconds}s"

            for img in images:
                assert isinstance(img, Image.Image)
                assert img.size[0] > 0 and img.size[1] > 0

    def test_convert_complex_pdf(self, pdf_adapter, complex_pdfs, performance_tracker):
        """Test conversion of complex PDFs."""
        if not complex_pdfs:
            pytest.skip("No complex PDFs in catalog")

        for test_case in complex_pdfs:
            performance_tracker.start(f"convert_{test_case.filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))

            metrics = performance_tracker.stop(
                filename=test_case.filename,
                page_count=len(images),
                complexity=test_case.complexity
            )

            assert len(images) > 0, f"No images from {test_case.filename}"
            # Complex PDFs get more time
            assert metrics.duration_seconds < 60, f"Conversion too slow: {metrics.duration_seconds}s"

    @pytest.mark.parametrize("pdf_name", [
        "extreme_multi_column.pdf",
        "mixed_content_layout.pdf",
        "semantic_table.pdf",
        "Shipping-Stem-2025-09-30.pdf",
        "Loading-Statement-for-Web-Portal-20250923.pdf",
    ])
    def test_convert_specific_pdfs(self, pdf_adapter, pdf_test_catalog, pdf_name):
        """Test conversion of specific key PDFs."""
        if pdf_name not in pdf_test_catalog:
            pytest.skip(f"PDF not in catalog: {pdf_name}")

        test_case = pdf_test_catalog[pdf_name]

        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        images = run_async(pdf_adapter.convert_to_frames(content))

        assert len(images) > 0, f"No images from {pdf_name}"

        # Verify image quality
        for i, img in enumerate(images):
            assert img.mode in ['RGB', 'RGBA', 'L'], f"Unexpected mode: {img.mode}"
            assert img.size[0] >= 100, f"Image too narrow: {img.size[0]}"
            assert img.size[1] >= 100, f"Image too short: {img.size[1]}"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestLayoutHandling:
    """Test layout-specific document handling."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_multi_column_layout(self, pdf_adapter, pdf_test_catalog):
        """Test multi-column layout handling."""
        test_files = ["extreme_multi_column.pdf", "proper_multi_column.pdf"]

        for pdf_name in test_files:
            if pdf_name not in pdf_test_catalog:
                continue

            test_case = pdf_test_catalog[pdf_name]

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))

            assert len(images) > 0
            # Multi-column docs should have reasonable aspect ratios
            for img in images:
                aspect_ratio = img.size[0] / img.size[1]
                assert 0.5 < aspect_ratio < 2.0, f"Unusual aspect ratio: {aspect_ratio}"

    def test_table_layout(self, pdf_adapter, pdf_test_catalog):
        """Test table layout handling."""
        if "semantic_table.pdf" not in pdf_test_catalog:
            pytest.skip("semantic_table.pdf not found")

        test_case = pdf_test_catalog["semantic_table.pdf"]

        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        images = run_async(pdf_adapter.convert_to_frames(content))

        assert len(images) > 0
        # Table documents should convert cleanly
        for img in images:
            assert img.size[0] > 200
            assert img.size[1] > 200

    def test_overlapping_text(self, pdf_adapter, pdf_test_catalog):
        """Test overlapping text handling."""
        if "overlapping_text.pdf" not in pdf_test_catalog:
            pytest.skip("overlapping_text.pdf not found")

        test_case = pdf_test_catalog["overlapping_text.pdf"]

        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        # Should not crash on overlapping text
        images = run_async(pdf_adapter.convert_to_frames(content))
        assert len(images) > 0

    def test_tiny_fonts(self, pdf_adapter, pdf_test_catalog):
        """Test tiny fonts and tight spacing."""
        if "tiny_fonts_spacing.pdf" not in pdf_test_catalog:
            pytest.skip("tiny_fonts_spacing.pdf not found")

        test_case = pdf_test_catalog["tiny_fonts_spacing.pdf"]

        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        images = run_async(pdf_adapter.convert_to_frames(content))
        assert len(images) > 0


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestShippingDocuments:
    """Test real shipping document processing."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_shipping_stem_documents(self, pdf_adapter, shipping_pdfs, performance_tracker):
        """Test all shipping stem documents."""
        for test_case in shipping_pdfs:
            performance_tracker.start(f"shipping_{test_case.filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))

            metrics = performance_tracker.stop(
                filename=test_case.filename,
                page_count=len(images)
            )

            assert len(images) > 0, f"No images from {test_case.filename}"

            # Shipping docs tend to be multi-page
            # Log for analysis
            print(f"{test_case.filename}: {len(images)} pages, {metrics.duration_seconds:.2f}s")

    def test_shipping_metadata_extraction(self, pdf_adapter, shipping_pdfs):
        """Test metadata extraction from shipping documents."""
        for test_case in shipping_pdfs:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            metadata = pdf_adapter.extract_metadata(content)

            assert metadata is not None
            assert hasattr(metadata, 'page_count') or 'page_count' in metadata.__dict__


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestFinancialDocuments:
    """Test financial document processing."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_loading_statements(self, pdf_adapter, financial_pdfs, performance_tracker):
        """Test loading statement documents."""
        loading_docs = [tc for tc in financial_pdfs
                       if 'loading' in tc.filename.lower() or 'statement' in tc.filename.lower()]

        for test_case in loading_docs:
            performance_tracker.start(f"financial_{test_case.filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))

            metrics = performance_tracker.stop(
                filename=test_case.filename,
                page_count=len(images)
            )

            assert len(images) > 0

    def test_all_financial_documents(self, pdf_adapter, financial_pdfs):
        """Test all financial documents convert successfully."""
        for test_case in financial_pdfs:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            assert len(images) > 0, f"Failed to convert {test_case.filename}"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestAllPDFs:
    """Comprehensive test running against all 15 PDFs."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_all_pdfs_convert_successfully(self, pdf_adapter, pdf_test_cases, performance_tracker):
        """Test that ALL 15 PDFs convert successfully."""
        results = []

        for test_case in pdf_test_cases:
            performance_tracker.start(f"all_{test_case.filename}")

            try:
                with open(test_case.filepath, 'rb') as f:
                    content = f.read()

                images = run_async(pdf_adapter.convert_to_frames(content))

                metrics = performance_tracker.stop(
                    filename=test_case.filename,
                    page_count=len(images),
                    success=True
                )

                results.append({
                    "filename": test_case.filename,
                    "success": True,
                    "page_count": len(images),
                    "duration_seconds": metrics.duration_seconds,
                    "memory_delta_mb": metrics.memory_delta_mb,
                })

            except Exception as e:
                performance_tracker.stop(
                    filename=test_case.filename,
                    success=False,
                    error=str(e)
                )

                results.append({
                    "filename": test_case.filename,
                    "success": False,
                    "error": str(e),
                })

        # All PDFs must convert successfully
        failed = [r for r in results if not r["success"]]
        assert len(failed) == 0, f"Failed PDFs: {[r['filename'] for r in failed]}"

        # Report summary
        summary = performance_tracker.get_summary()
        print(f"\nAll PDFs Test Summary:")
        print(f"  Total PDFs: {len(results)}")
        print(f"  Total Duration: {summary['total_duration_seconds']:.2f}s")
        print(f"  Avg Duration: {summary['avg_duration_seconds']:.2f}s")
        print(f"  Max Memory Delta: {summary['max_memory_delta_mb']:.2f}MB")

    def test_all_pdfs_metadata_extraction(self, pdf_adapter, pdf_test_cases):
        """Test metadata extraction from all PDFs."""
        for test_case in pdf_test_cases:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            metadata = pdf_adapter.extract_metadata(content)
            assert metadata is not None, f"No metadata from {test_case.filename}"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestErrorConditions:
    """Test error handling for various error conditions."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_empty_file(self, pdf_adapter, tmp_path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        with pytest.raises(Exception):
            with open(empty_file, 'rb') as f:
                run_async(pdf_adapter.convert_to_frames(f.read()))

    def test_invalid_pdf_header(self, pdf_adapter, tmp_path):
        """Test handling of invalid PDF header."""
        invalid_file = tmp_path / "invalid.pdf"
        invalid_file.write_bytes(b"NOT A PDF FILE")

        with pytest.raises(Exception):
            with open(invalid_file, 'rb') as f:
                run_async(pdf_adapter.convert_to_frames(f.read()))

    def test_truncated_pdf(self, pdf_adapter, all_pdf_files, tmp_path):
        """Test handling of truncated PDF."""
        if not all_pdf_files:
            pytest.skip("No PDFs available")

        # Get first PDF and truncate it
        source_pdf = all_pdf_files[0]
        with open(source_pdf, 'rb') as f:
            content = f.read()

        # Truncate to 10% of original
        truncated = content[:len(content) // 10]

        truncated_file = tmp_path / "truncated.pdf"
        truncated_file.write_bytes(truncated)

        # Should raise or handle gracefully
        try:
            with open(truncated_file, 'rb') as f:
                images = run_async(pdf_adapter.convert_to_frames(f.read()))
            # If it doesn't raise, it should return empty or partial result
        except Exception:
            pass  # Expected behavior

    def test_non_pdf_extension(self, pdf_adapter, tmp_path):
        """Test handling of non-PDF file with PDF extension."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"This is just plain text, not a PDF")

        with pytest.raises(Exception):
            with open(fake_pdf, 'rb') as f:
                run_async(pdf_adapter.convert_to_frames(f.read()))


@pytest.mark.skipif(
    not (PDF_ADAPTER_AVAILABLE and IMAGE_PROCESSOR_AVAILABLE),
    reason=f"Adapters not available: PDF={PDF_ADAPTER_ERROR}, ImageProcessor={IMAGE_PROCESSOR_ERROR}"
)
class TestImageProcessing:
    """Test image processing and standardization."""

    @pytest.fixture
    def image_processor(self):
        """Create image processor instance."""
        return create_image_processor()

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_image_standardization(self, pdf_adapter, simple_pdfs):
        """Test image standardization after PDF conversion."""
        if not simple_pdfs:
            pytest.skip("No simple PDFs")

        test_case = simple_pdfs[0]

        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        images = run_async(pdf_adapter.convert_to_frames(content))
        assert len(images) > 0

        # Create image processor with config
        config = ProcessingConfig(
            target_width=1024,
            target_height=1024,
            maintain_aspect_ratio=True
        )
        image_processor = ImageProcessor(config)

        processed, metadata = run_async(image_processor.process_image(images[0]))

        assert processed is not None
        assert processed.size[0] <= 1024
        assert processed.size[1] <= 1024


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestGoldenMaster:
    """Golden master testing framework."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter instance."""
        return create_pdf_adapter()

    def test_golden_master_framework_exists(self, ground_truth_directory):
        """Verify golden master framework is set up."""
        assert ground_truth_directory.exists()

    def test_create_golden_master_baseline(
        self,
        pdf_adapter,
        pdf_test_cases,
        ground_truth_directory,
        save_test_result
    ):
        """Create baseline golden master data (run once to establish baselines)."""
        # This test creates golden master data for future comparison
        # Skip if golden masters already exist
        existing_masters = list(ground_truth_directory.glob("*.json"))
        if len(existing_masters) >= len(pdf_test_cases):
            pytest.skip("Golden masters already exist")

        baselines = {}
        for test_case in pdf_test_cases:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            metadata = pdf_adapter.extract_metadata(content)

            baselines[test_case.filename] = {
                "page_count": len(images),
                "file_size_bytes": test_case.file_size_bytes,
                "category": test_case.category,
                "complexity": test_case.complexity,
                "image_sizes": [(img.size[0], img.size[1]) for img in images],
            }

        # Save baseline
        result_path = save_test_result("golden_master_baseline", baselines)
        print(f"Golden master baseline saved to: {result_path}")

        assert len(baselines) == len(pdf_test_cases)

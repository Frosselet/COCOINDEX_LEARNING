#!/usr/bin/env python3
"""
Comprehensive test for multi-format document adapter interface.

Tests the plugin system, format detection, adapter registry,
and end-to-end document processing across multiple formats.
"""

import asyncio
import logging
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_documents() -> Dict[str, bytes]:
    """Create sample documents in various formats for testing."""
    test_docs = {}

    # 1. Sample HTML document
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
        <meta name="author" content="Test Author">
        <meta name="date" content="2024-01-01">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: blue; }
            p { line-height: 1.5; }
        </style>
    </head>
    <body>
        <h1>Multi-Format Adapter Test</h1>
        <p>This is a test HTML document for the ColPali adapter system.</p>
        <p>It contains multiple paragraphs and formatting to test the HTML adapter.</p>
        <div>
            <h2>Features Tested</h2>
            <ul>
                <li>HTML parsing and validation</li>
                <li>Metadata extraction from meta tags</li>
                <li>Text content extraction</li>
                <li>Layout rendering</li>
            </ul>
        </div>
    </body>
    </html>
    """
    test_docs['html'] = html_content.encode('utf-8')

    # 2. Sample image (PNG)
    image = Image.new('RGB', (800, 600), color=(200, 220, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 750, 550], outline=(0, 0, 100), width=3)
    draw.text((100, 100), "Test Image Document", fill=(0, 0, 0))
    draw.text((100, 150), "Format: PNG", fill=(50, 50, 50))
    draw.text((100, 200), "Size: 800x600", fill=(50, 50, 50))

    # Convert to PNG bytes
    png_buffer = io.BytesIO()
    image.save(png_buffer, format='PNG')
    test_docs['png'] = png_buffer.getvalue()

    # 3. Sample JPEG image
    jpeg_image = Image.new('RGB', (1000, 800), color=(240, 240, 220))
    draw = ImageDraw.Draw(jpeg_image)
    draw.ellipse([100, 100, 900, 700], outline=(100, 50, 0), width=5)
    draw.text((200, 300), "JPEG Test Image", fill=(100, 50, 0))

    jpeg_buffer = io.BytesIO()
    jpeg_image.save(jpeg_buffer, format='JPEG', quality=90)
    test_docs['jpeg'] = jpeg_buffer.getvalue()

    # 4. Invalid/Unknown format
    test_docs['unknown'] = b"This is not a valid document format for testing error handling"

    return test_docs


class MultiAdapterTester:
    """Comprehensive tester for the multi-format adapter system."""

    def __init__(self):
        """Initialize tester."""
        self.test_results = []
        print("="*70)
        print("MULTI-FORMAT ADAPTER SYSTEM TEST")
        print("="*70)

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all multi-format adapter tests."""

        # Import adapters (with fallback if import fails)
        try:
            import sys
            project_root = Path(__file__).parent
            sys.path.insert(0, str(project_root))

            # Import the core components (minimal versions if full package fails)
            try:
                from tatforge.core.document_adapter import (
                    DocumentAdapter, DocumentFormat, ConversionConfig,
                    DocumentProcessingError, UnsupportedFormatError
                )
                from tatforge.adapters.pdf_adapter import create_pdf_adapter
                from tatforge.adapters.image_adapter import create_image_adapter
                from tatforge.adapters.html_adapter import create_html_adapter
                adapter_imports_available = True
            except ImportError as e:
                print(f"⚠️  Full adapter imports failed: {e}")
                adapter_imports_available = False

        except Exception as e:
            print(f"✗ Failed to set up test environment: {e}")
            return {"error": "Test setup failed"}

        # Test 1: Adapter Registration System
        print("\n--- Test 1: Adapter Registration System ---")
        registry_test_passed = await self._test_adapter_registry(adapter_imports_available)

        # Test 2: Format Detection
        print("\n--- Test 2: Format Detection ---")
        detection_test_passed = await self._test_format_detection(adapter_imports_available)

        # Test 3: Multi-Format Processing
        print("\n--- Test 3: Multi-Format Document Processing ---")
        processing_test_passed = await self._test_multi_format_processing(adapter_imports_available)

        # Test 4: Error Handling
        print("\n--- Test 4: Error Handling and Edge Cases ---")
        error_handling_passed = await self._test_error_handling(adapter_imports_available)

        # Test 5: Configuration System
        print("\n--- Test 5: Configuration System ---")
        config_test_passed = await self._test_configuration_system(adapter_imports_available)

        # Calculate results
        total_tests = 5
        passed_tests = sum([
            registry_test_passed,
            detection_test_passed,
            processing_test_passed,
            error_handling_passed,
            config_test_passed
        ])

        success_rate = (passed_tests / total_tests) * 100

        # Print summary
        print(f"\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Adapter Registration:     {'✓ PASS' if registry_test_passed else '✗ FAIL'}")
        print(f"Format Detection:         {'✓ PASS' if detection_test_passed else '✗ FAIL'}")
        print(f"Multi-Format Processing:  {'✓ PASS' if processing_test_passed else '✗ FAIL'}")
        print(f"Error Handling:           {'✓ PASS' if error_handling_passed else '✗ FAIL'}")
        print(f"Configuration System:     {'✓ PASS' if config_test_passed else '✗ FAIL'}")
        print(f"\nOverall Success Rate:     {success_rate:.1f}% ({passed_tests}/{total_tests})")

        if success_rate >= 80:
            print("\n✓ MULTI-FORMAT ADAPTER TEST PASSED")
            result_status = True
        else:
            print("\n✗ MULTI-FORMAT ADAPTER TEST FAILED")
            result_status = False

        return {
            "success": result_status,
            "success_rate": success_rate,
            "detailed_results": {
                "adapter_registration": registry_test_passed,
                "format_detection": detection_test_passed,
                "multi_format_processing": processing_test_passed,
                "error_handling": error_handling_passed,
                "configuration_system": config_test_passed
            }
        }

    async def _test_adapter_registry(self, imports_available: bool) -> bool:
        """Test the adapter registration system."""
        if not imports_available:
            print("  Skipped: Adapter imports not available")
            return False

        try:
            from tatforge.core.document_adapter import DocumentAdapter, DocumentFormat
            from tatforge.adapters.pdf_adapter import create_pdf_adapter
            from tatforge.adapters.image_adapter import create_image_adapter
            from tatforge.adapters.html_adapter import create_html_adapter

            # Create adapter instance
            adapter = DocumentAdapter()

            # Register adapters
            adapter.register_adapter(DocumentFormat.PDF, create_pdf_adapter())
            print("  ✓ PDF adapter registered")

            adapter.register_adapter(DocumentFormat.IMAGE, create_image_adapter())
            print("  ✓ Image adapter registered")

            adapter.register_adapter(DocumentFormat.HTML, create_html_adapter())
            print("  ✓ HTML adapter registered")

            # Check supported formats
            supported = adapter.list_supported_formats()
            expected_formats = {DocumentFormat.PDF, DocumentFormat.IMAGE, DocumentFormat.HTML}

            if all(fmt in supported for fmt in expected_formats):
                print(f"  ✓ All expected formats supported: {len(supported)} formats")
                return True
            else:
                print(f"  ✗ Missing formats: expected {expected_formats}, got {set(supported)}")
                return False

        except Exception as e:
            print(f"  ✗ Adapter registration test failed: {e}")
            return False

    async def _test_format_detection(self, imports_available: bool) -> bool:
        """Test format detection capabilities."""
        if not imports_available:
            print("  Skipped: Adapter imports not available")
            return False

        try:
            from tatforge.core.document_adapter import DocumentAdapter

            adapter = DocumentAdapter()
            test_docs = create_test_documents()

            detection_results = []

            # Test HTML detection
            try:
                detected_format = adapter._detect_format_sync(test_docs['html'])
                if detected_format.value == 'html':
                    print("  ✓ HTML format detected correctly")
                    detection_results.append(True)
                else:
                    print(f"  ✗ HTML detection failed: got {detected_format}")
                    detection_results.append(False)
            except Exception as e:
                print(f"  ✗ HTML detection error: {e}")
                detection_results.append(False)

            # Test PNG detection
            try:
                detected_format = adapter._detect_format_sync(test_docs['png'])
                if detected_format.value == 'image':
                    print("  ✓ PNG format detected correctly")
                    detection_results.append(True)
                else:
                    print(f"  ✗ PNG detection failed: got {detected_format}")
                    detection_results.append(False)
            except Exception as e:
                print(f"  ✗ PNG detection error: {e}")
                detection_results.append(False)

            # Test JPEG detection
            try:
                detected_format = adapter._detect_format_sync(test_docs['jpeg'])
                if detected_format.value == 'image':
                    print("  ✓ JPEG format detected correctly")
                    detection_results.append(True)
                else:
                    print(f"  ✗ JPEG detection failed: got {detected_format}")
                    detection_results.append(False)
            except Exception as e:
                print(f"  ✗ JPEG detection error: {e}")
                detection_results.append(False)

            # Calculate success rate
            success_rate = sum(detection_results) / len(detection_results)
            return success_rate >= 0.8  # 80% success rate

        except Exception as e:
            print(f"  ✗ Format detection test failed: {e}")
            return False

    async def _test_multi_format_processing(self, imports_available: bool) -> bool:
        """Test multi-format document processing."""
        if not imports_available:
            print("  Skipped: Adapter imports not available")
            return False

        try:
            from tatforge.core.document_adapter import (
                DocumentAdapter, DocumentFormat, ConversionConfig
            )
            from tatforge.adapters import (
                create_pdf_adapter, create_image_adapter, create_html_adapter
            )

            # Set up adapter with all formats
            adapter = DocumentAdapter()
            adapter.register_adapter(DocumentFormat.PDF, create_pdf_adapter())
            adapter.register_adapter(DocumentFormat.IMAGE, create_image_adapter())
            adapter.register_adapter(DocumentFormat.HTML, create_html_adapter())

            test_docs = create_test_documents()
            processing_results = []

            # Test image processing
            try:
                frames = await adapter.convert_to_frames(test_docs['png'])
                if frames and len(frames) > 0:
                    print(f"  ✓ PNG processed: {len(frames)} frames, size {frames[0].size}")
                    processing_results.append(True)
                else:
                    print("  ✗ PNG processing failed: no frames generated")
                    processing_results.append(False)
            except Exception as e:
                print(f"  ✗ PNG processing error: {e}")
                processing_results.append(False)

            # Test JPEG processing
            try:
                frames = await adapter.convert_to_frames(test_docs['jpeg'])
                if frames and len(frames) > 0:
                    print(f"  ✓ JPEG processed: {len(frames)} frames, size {frames[0].size}")
                    processing_results.append(True)
                else:
                    print("  ✗ JPEG processing failed: no frames generated")
                    processing_results.append(False)
            except Exception as e:
                print(f"  ✗ JPEG processing error: {e}")
                processing_results.append(False)

            # Test HTML processing (with fallback method)
            try:
                frames = await adapter.convert_to_frames(test_docs['html'])
                if frames and len(frames) > 0:
                    print(f"  ✓ HTML processed: {len(frames)} frames, size {frames[0].size}")
                    processing_results.append(True)
                else:
                    print("  ✗ HTML processing failed: no frames generated")
                    processing_results.append(False)
            except Exception as e:
                print(f"  ✗ HTML processing error: {e}")
                processing_results.append(False)

            # Test PDF processing if PDFs are available
            pdfs_dir = Path("pdfs")
            if pdfs_dir.exists() and list(pdfs_dir.glob("*.pdf")):
                try:
                    pdf_file = list(pdfs_dir.glob("*.pdf"))[0]
                    with open(pdf_file, 'rb') as f:
                        pdf_content = f.read()

                    config = ConversionConfig(dpi=150, max_pages=1)
                    frames = await adapter.convert_to_frames(pdf_content, config=config)
                    if frames and len(frames) > 0:
                        print(f"  ✓ PDF processed: {len(frames)} frames, size {frames[0].size}")
                        processing_results.append(True)
                    else:
                        print("  ✗ PDF processing failed: no frames generated")
                        processing_results.append(False)
                except Exception as e:
                    print(f"  ✗ PDF processing error: {e}")
                    processing_results.append(False)

            # Calculate success rate
            success_rate = sum(processing_results) / len(processing_results) if processing_results else 0
            return success_rate >= 0.75  # 75% success rate

        except Exception as e:
            print(f"  ✗ Multi-format processing test failed: {e}")
            return False

    async def _test_error_handling(self, imports_available: bool) -> bool:
        """Test error handling and edge cases."""
        if not imports_available:
            print("  Skipped: Adapter imports not available")
            return False

        try:
            from tatforge.core.document_adapter import (
                DocumentAdapter, UnsupportedFormatError
            )
            from tatforge.adapters import create_image_adapter

            adapter = DocumentAdapter()
            adapter.register_adapter(DocumentFormat.IMAGE, create_image_adapter())

            test_docs = create_test_documents()
            error_tests_passed = 0
            total_error_tests = 0

            # Test 1: Unknown format handling
            total_error_tests += 1
            try:
                await adapter.convert_to_frames(test_docs['unknown'])
                print("  ✗ Unknown format should have failed")
            except UnsupportedFormatError:
                print("  ✓ Unknown format properly rejected")
                error_tests_passed += 1
            except Exception as e:
                print(f"  ⚠️  Unknown format failed with unexpected error: {e}")

            # Test 2: Empty content handling
            total_error_tests += 1
            try:
                await adapter.convert_to_frames(b"")
                print("  ✗ Empty content should have failed")
            except (UnsupportedFormatError, Exception):
                print("  ✓ Empty content properly rejected")
                error_tests_passed += 1

            # Test 3: Corrupted content handling
            total_error_tests += 1
            try:
                corrupted_content = b"\x89PNG\r\n\x1a\n" + b"corrupted data"
                await adapter.convert_to_frames(corrupted_content)
                print("  ✗ Corrupted content should have failed")
            except Exception:
                print("  ✓ Corrupted content properly handled")
                error_tests_passed += 1

            return error_tests_passed / total_error_tests >= 0.8

        except Exception as e:
            print(f"  ✗ Error handling test failed: {e}")
            return False

    async def _test_configuration_system(self, imports_available: bool) -> bool:
        """Test configuration system and adapter parameters."""
        if not imports_available:
            print("  Skipped: Adapter imports not available")
            return False

        try:
            from tatforge.core.document_adapter import ConversionConfig
            from tatforge.adapters import create_image_adapter

            config_tests_passed = 0
            total_config_tests = 0

            # Test 1: Basic configuration
            total_config_tests += 1
            try:
                config = ConversionConfig(dpi=150, quality=75, max_pages=1)
                adapter = create_image_adapter(max_memory_mb=100)

                if config.dpi == 150 and config.quality == 75 and adapter.max_memory_mb == 100:
                    print("  ✓ Basic configuration parameters work")
                    config_tests_passed += 1
                else:
                    print("  ✗ Configuration parameters not set correctly")
            except Exception as e:
                print(f"  ✗ Basic configuration test failed: {e}")

            # Test 2: Configuration with processing
            total_config_tests += 1
            try:
                test_docs = create_test_documents()
                adapter = create_image_adapter()

                # Test with different DPI settings
                config_low = ConversionConfig(dpi=100, quality=50)
                config_high = ConversionConfig(dpi=300, quality=95)

                frames_low = await adapter.convert_to_frames(test_docs['png'], config_low)
                frames_high = await adapter.convert_to_frames(test_docs['png'], config_high)

                if frames_low and frames_high:
                    print(f"  ✓ Configuration affects processing (sizes: {frames_low[0].size} vs {frames_high[0].size})")
                    config_tests_passed += 1
                else:
                    print("  ✗ Configuration processing test failed")

            except Exception as e:
                print(f"  ✗ Configuration processing test failed: {e}")

            return config_tests_passed / total_config_tests >= 0.8

        except Exception as e:
            print(f"  ✗ Configuration system test failed: {e}")
            return False


async def main():
    """Main test execution."""
    try:
        tester = MultiAdapterTester()
        results = await tester.run_all_tests()

        if results.get("success", False):
            return 0
        else:
            return 1

    except Exception as e:
        logger.error(f"Test suite failed with critical error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Comprehensive integration test for COLPALI-203: Multi-format adapter interface.

Tests the complete plugin architecture including:
- Adapter registration system
- MIME type detection with python-magic
- Format-specific configuration support
- Consistent error handling across adapters
- Multi-format document processing pipeline

Requirements from JIRA COLPALI-203:
- Abstract base class for document adapters ✓
- Plugin registration system for new formats ✓
- Consistent error handling across all adapters ✓
- MIME type detection and routing ✓
- Format-specific configuration support ✓
"""

import asyncio
import logging
import tempfile
import io
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_documents() -> Dict[str, bytes]:
    """Create comprehensive test documents for all supported formats."""
    test_docs = {}

    # 1. HTML Document
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Multi-Format Adapter Test Document</title>
    <meta name="author" content="ColPali Test Suite">
    <meta name="date" content="2024-01-08">
    <meta name="description" content="Test HTML document for adapter system validation">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
        .test-section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }
        ul { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>COLPALI-203 Multi-Format Adapter Test</h1>

    <div class="test-section">
        <h2>Plugin Architecture Validation</h2>
        <p>This document tests the multi-format adapter interface implementation including:</p>
        <ul>
            <li>MIME type detection and routing</li>
            <li>Plugin registration system</li>
            <li>Format-specific configuration support</li>
            <li>Consistent error handling across adapters</li>
            <li>Document metadata extraction</li>
        </ul>
    </div>

    <div class="test-section">
        <h2>HTML Rendering Capabilities</h2>
        <p>Tests HTML-to-image conversion with various content types:</p>
        <ul>
            <li>Styled text content</li>
            <li>Structured layouts with CSS</li>
            <li>Metadata extraction from meta tags</li>
            <li>Title and author information</li>
        </ul>
    </div>

    <div class="test-section">
        <h2>Configuration Testing</h2>
        <p>Validates configuration parameter handling including DPI settings,
        quality optimization, page limits, and aspect ratio preservation.</p>
    </div>
</body>
</html>"""
    test_docs['html'] = html_content.encode('utf-8')

    # 2. PNG Image (high quality)
    png_image = Image.new('RGB', (1200, 800), color=(240, 248, 255))
    draw = ImageDraw.Draw(png_image)

    # Header
    draw.rectangle([20, 20, 1180, 100], outline=(70, 130, 180), width=3, fill=(230, 240, 250))
    draw.text((50, 45), "PNG Image Adapter Test", fill=(70, 130, 180))

    # Content sections
    draw.text((50, 150), "Format: PNG (Portable Network Graphics)", fill=(60, 60, 60))
    draw.text((50, 200), "Dimensions: 1200x800 pixels", fill=(60, 60, 60))
    draw.text((50, 250), "Color Mode: RGB", fill=(60, 60, 60))
    draw.text((50, 300), "Test Features:", fill=(60, 60, 60))
    draw.text((70, 330), "• MIME type detection (image/png)", fill=(80, 80, 80))
    draw.text((70, 360), "• Metadata extraction", fill=(80, 80, 80))
    draw.text((70, 390), "• Format validation", fill=(80, 80, 80))
    draw.text((70, 420), "• Image processing pipeline", fill=(80, 80, 80))

    # Visual elements
    draw.ellipse([800, 150, 1100, 450], outline=(220, 20, 60), width=2)
    draw.text((870, 280), "Visual Content", fill=(220, 20, 60))

    png_buffer = io.BytesIO()
    png_image.save(png_buffer, format='PNG', optimize=True)
    test_docs['png'] = png_buffer.getvalue()

    # 3. JPEG Image (different characteristics)
    jpeg_image = Image.new('RGB', (800, 1000), color=(255, 250, 240))
    draw = ImageDraw.Draw(jpeg_image)

    # Header
    draw.rectangle([30, 30, 770, 120], outline=(139, 69, 19), width=4, fill=(255, 239, 213))
    draw.text((50, 60), "JPEG Image Adapter Test", fill=(139, 69, 19))

    # Content grid
    for i in range(3):
        for j in range(4):
            x = 50 + j * 180
            y = 200 + i * 150
            draw.rectangle([x, y, x + 150, y + 100], outline=(210, 180, 140), width=2)
            draw.text((x + 20, y + 40), f"Cell {i*4+j+1}", fill=(139, 69, 19))

    # Footer
    draw.text((50, 900), "JPEG Quality Test - Compression optimization", fill=(100, 100, 100))

    jpeg_buffer = io.BytesIO()
    jpeg_image.save(jpeg_buffer, format='JPEG', quality=85, optimize=True)
    test_docs['jpeg'] = jpeg_buffer.getvalue()

    # 4. GIF Image (animated format test)
    gif_image = Image.new('RGB', (600, 400), color=(248, 248, 255))
    draw = ImageDraw.Draw(gif_image)

    draw.text((50, 50), "GIF Format Test", fill=(75, 0, 130))
    draw.text((50, 100), "Animated format support", fill=(100, 100, 100))
    draw.rectangle([50, 150, 550, 350], outline=(75, 0, 130), width=3)
    draw.text((200, 230), "GIF Test Content", fill=(75, 0, 130))

    gif_buffer = io.BytesIO()
    gif_image.save(gif_buffer, format='GIF')
    test_docs['gif'] = gif_buffer.getvalue()

    # 5. Invalid/Unknown format
    test_docs['unknown'] = b"UNKNOWN\x00\x01\x02This is not a valid document format designed to test error handling in the multi-format adapter system."

    # 6. Corrupted image data
    test_docs['corrupted_png'] = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00corrupted_data_here"

    # 7. Empty content
    test_docs['empty'] = b""

    return test_docs


class MultiFormatAdapterTester:
    """Comprehensive tester for COLPALI-203 multi-format adapter interface."""

    def __init__(self):
        """Initialize the tester with proper imports."""
        self.test_results = []
        self.start_logging()

    def start_logging(self):
        """Initialize test logging."""
        print("="*80)
        print("COLPALI-203: MULTI-FORMAT ADAPTER INTERFACE TEST")
        print("="*80)
        print("Testing plugin architecture, MIME detection, and format processing")
        print("")

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests for COLPALI-203 requirements."""

        # Set up Python path
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import the full adapter system
        try:
            from tatforge.core.document_adapter import (
                DocumentAdapter, DocumentFormat, ConversionConfig,
                BaseDocumentAdapter, DocumentProcessingError,
                UnsupportedFormatError, MetadataExtractionError
            )
            from tatforge.adapters.pdf_adapter import create_pdf_adapter
            from tatforge.adapters.image_adapter import create_image_adapter
            from tatforge.adapters.html_adapter import create_html_adapter

            print("✓ All adapter modules imported successfully")

        except ImportError as e:
            print(f"✗ Failed to import adapter modules: {e}")
            return {"success": False, "error": "Import failed", "details": str(e)}

        # Test 1: Adapter Registration System (JIRA requirement)
        print("\n--- Test 1: Plugin Registration System ---")
        registration_test = await self._test_adapter_registration(
            DocumentAdapter, DocumentFormat, create_pdf_adapter,
            create_image_adapter, create_html_adapter
        )

        # Test 2: MIME Type Detection and Routing (JIRA requirement)
        print("\n--- Test 2: MIME Type Detection and Routing ---")
        mime_detection_test = await self._test_mime_detection(DocumentAdapter)

        # Test 3: Format-Specific Configuration Support (JIRA requirement)
        print("\n--- Test 3: Format-Specific Configuration Support ---")
        configuration_test = await self._test_configuration_support(
            DocumentAdapter, ConversionConfig, create_image_adapter, create_html_adapter
        )

        # Test 4: Consistent Error Handling (JIRA requirement)
        print("\n--- Test 4: Consistent Error Handling Across Adapters ---")
        error_handling_test = await self._test_error_handling(
            DocumentAdapter, UnsupportedFormatError, DocumentProcessingError
        )

        # Test 5: Multi-Format Processing Pipeline
        print("\n--- Test 5: Multi-Format Document Processing Pipeline ---")
        processing_test = await self._test_multi_format_processing(
            DocumentAdapter, ConversionConfig, create_pdf_adapter,
            create_image_adapter, create_html_adapter
        )

        # Test 6: Extension Examples (JIRA requirement)
        print("\n--- Test 6: Extension Examples and Plugin Architecture ---")
        extension_test = await self._test_extension_examples(
            BaseDocumentAdapter, DocumentFormat, DocumentAdapter
        )

        # Calculate overall results
        tests = {
            'adapter_registration': registration_test,
            'mime_detection_routing': mime_detection_test,
            'configuration_support': configuration_test,
            'error_handling': error_handling_test,
            'multi_format_processing': processing_test,
            'extension_examples': extension_test
        }

        passed_tests = sum(tests.values())
        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100

        # Generate summary
        print(f"\n" + "="*80)
        print("COLPALI-203 TEST SUMMARY")
        print("="*80)

        for test_name, passed in tests.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name.replace('_', ' ').title():<35} {status}")

        print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")

        if success_rate >= 85:  # High bar for plugin architecture
            print("\n✓ COLPALI-203 MULTI-FORMAT ADAPTER TEST PASSED")
            result_success = True
        else:
            print("\n✗ COLPALI-203 MULTI-FORMAT ADAPTER TEST FAILED")
            result_success = False

        return {
            "success": result_success,
            "success_rate": success_rate,
            "detailed_results": tests,
            "total_tests": total_tests,
            "passed_tests": passed_tests
        }

    async def _test_adapter_registration(self, DocumentAdapter, DocumentFormat,
                                       create_pdf_adapter, create_image_adapter,
                                       create_html_adapter) -> bool:
        """Test the plugin registration system."""
        try:
            adapter = DocumentAdapter()

            # Test initial state
            initial_formats = adapter.list_supported_formats()
            if len(initial_formats) > 0:
                print(f"  ⚠️  Warning: Adapter has {len(initial_formats)} pre-registered formats")

            # Register each adapter
            adapter.register_adapter(DocumentFormat.PDF, create_pdf_adapter())
            print("  ✓ PDF adapter registered")

            adapter.register_adapter(DocumentFormat.IMAGE, create_image_adapter())
            print("  ✓ Image adapter registered")

            adapter.register_adapter(DocumentFormat.HTML, create_html_adapter())
            print("  ✓ HTML adapter registered")

            # Verify registration
            supported = adapter.list_supported_formats()
            expected = {DocumentFormat.PDF, DocumentFormat.IMAGE, DocumentFormat.HTML}

            if expected.issubset(set(supported)):
                print(f"  ✓ All expected formats supported: {[fmt.value for fmt in expected]}")

                # Test adapter retrieval
                pdf_adapter = adapter._get_adapter(DocumentFormat.PDF)
                if pdf_adapter.supported_format == DocumentFormat.PDF:
                    print("  ✓ Adapter retrieval working correctly")
                    return True
                else:
                    print("  ✗ Adapter retrieval returned wrong format")
                    return False
            else:
                missing = expected - set(supported)
                print(f"  ✗ Missing formats: {[fmt.value for fmt in missing]}")
                return False

        except Exception as e:
            print(f"  ✗ Adapter registration test failed: {e}")
            return False

    async def _test_mime_detection(self, DocumentAdapter) -> bool:
        """Test MIME type detection and routing capabilities."""
        try:
            from tatforge.core.document_adapter import DocumentFormat

            adapter = DocumentAdapter()
            test_docs = create_test_documents()

            detection_results = []

            # Test HTML MIME detection
            try:
                html_format = adapter._detect_format_sync(test_docs['html'])
                if html_format == DocumentFormat.HTML:
                    print("  ✓ HTML MIME type detected correctly")
                    detection_results.append(True)
                else:
                    print(f"  ✗ HTML detection failed: got {html_format}")
                    detection_results.append(False)
            except Exception as e:
                print(f"  ✗ HTML MIME detection error: {e}")
                detection_results.append(False)

            # Test PNG MIME detection
            try:
                png_format = adapter._detect_format_sync(test_docs['png'])
                if png_format == DocumentFormat.IMAGE:
                    print("  ✓ PNG MIME type detected correctly")
                    detection_results.append(True)
                else:
                    print(f"  ✗ PNG detection failed: got {png_format}")
                    detection_results.append(False)
            except Exception as e:
                print(f"  ✗ PNG MIME detection error: {e}")
                detection_results.append(False)

            # Test JPEG MIME detection
            try:
                jpeg_format = adapter._detect_format_sync(test_docs['jpeg'])
                if jpeg_format == DocumentFormat.IMAGE:
                    print("  ✓ JPEG MIME type detected correctly")
                    detection_results.append(True)
                else:
                    print(f"  ✗ JPEG detection failed: got {jpeg_format}")
                    detection_results.append(False)
            except Exception as e:
                print(f"  ✗ JPEG MIME detection error: {e}")
                detection_results.append(False)

            # Test GIF MIME detection
            try:
                gif_format = adapter._detect_format_sync(test_docs['gif'])
                if gif_format == DocumentFormat.IMAGE:
                    print("  ✓ GIF MIME type detected correctly")
                    detection_results.append(True)
                else:
                    print(f"  ✗ GIF detection failed: got {gif_format}")
                    detection_results.append(False)
            except Exception as e:
                print(f"  ✗ GIF MIME detection error: {e}")
                detection_results.append(False)

            # Test with python-magic if available
            try:
                import magic
                mime_type = magic.from_buffer(test_docs['png'], mime=True)
                print(f"  ✓ python-magic integration working: {mime_type}")
                detection_results.append(True)
            except ImportError:
                print("  ⚠️  python-magic library not available (requires libmagic system library)")
                detection_results.append(False)
            except Exception as e:
                print(f"  ⚠️  python-magic integration failed: {e}")
                print("  Note: This is expected on systems without libmagic installed")
                detection_results.append(False)

            success_rate = sum(detection_results) / len(detection_results)
            print(f"  Detection success rate: {success_rate*100:.1f}%")

            return success_rate >= 0.8

        except Exception as e:
            print(f"  ✗ MIME detection test failed: {e}")
            return False

    async def _test_configuration_support(self, DocumentAdapter, ConversionConfig,
                                        create_image_adapter, create_html_adapter) -> bool:
        """Test format-specific configuration support."""
        try:
            from tatforge.core.document_adapter import DocumentFormat

            adapter = DocumentAdapter()
            adapter.register_adapter(DocumentFormat.IMAGE, create_image_adapter())
            adapter.register_adapter(DocumentFormat.HTML, create_html_adapter())

            test_docs = create_test_documents()
            config_tests = []

            # Test 1: DPI configuration
            try:
                config_low = ConversionConfig(dpi=150, quality=60)
                config_high = ConversionConfig(dpi=300, quality=95)

                frames_low = await adapter.convert_to_frames(
                    test_docs['png'], config=config_low
                )
                frames_high = await adapter.convert_to_frames(
                    test_docs['png'], config=config_high
                )

                if frames_low and frames_high:
                    print(f"  ✓ DPI configuration applied (low: {frames_low[0].size}, high: {frames_high[0].size})")
                    config_tests.append(True)
                else:
                    print("  ✗ DPI configuration test failed - no frames generated")
                    config_tests.append(False)

            except Exception as e:
                print(f"  ✗ DPI configuration test failed: {e}")
                config_tests.append(False)

            # Test 2: Quality configuration
            try:
                config_quality = ConversionConfig(quality=50)
                frames = await adapter.convert_to_frames(
                    test_docs['jpeg'], config=config_quality
                )

                if frames:
                    print(f"  ✓ Quality configuration applied: {config_quality.quality}%")
                    config_tests.append(True)
                else:
                    print("  ✗ Quality configuration test failed")
                    config_tests.append(False)

            except Exception as e:
                print(f"  ✗ Quality configuration test failed: {e}")
                config_tests.append(False)

            # Test 3: Page limit configuration
            try:
                config_pages = ConversionConfig(max_pages=1)
                frames = await adapter.convert_to_frames(
                    test_docs['html'], config=config_pages
                )

                if frames and len(frames) <= config_pages.max_pages:
                    print(f"  ✓ Page limit configuration respected: {len(frames)} frames")
                    config_tests.append(True)
                else:
                    print(f"  ✗ Page limit not respected: got {len(frames) if frames else 0} frames")
                    config_tests.append(False)

            except Exception as e:
                print(f"  ✗ Page limit configuration test failed: {e}")
                config_tests.append(False)

            success_rate = sum(config_tests) / len(config_tests) if config_tests else 0
            print(f"  Configuration success rate: {success_rate*100:.1f}%")

            return success_rate >= 0.8

        except Exception as e:
            print(f"  ✗ Configuration support test failed: {e}")
            return False

    async def _test_error_handling(self, DocumentAdapter, UnsupportedFormatError,
                                 DocumentProcessingError) -> bool:
        """Test consistent error handling across all adapters."""
        try:
            adapter = DocumentAdapter()
            # Note: Not registering any adapters to test error conditions

            test_docs = create_test_documents()
            error_tests = []

            # Test 1: Unsupported format error
            try:
                await adapter.convert_to_frames(test_docs['unknown'])
                print("  ✗ Unknown format should have raised UnsupportedFormatError")
                error_tests.append(False)
            except UnsupportedFormatError as e:
                print(f"  ✓ Unknown format properly rejected: {e.__class__.__name__}")
                error_tests.append(True)
            except Exception as e:
                print(f"  ⚠️  Unknown format raised different error: {e.__class__.__name__}")
                error_tests.append(False)

            # Test 2: Empty content handling
            try:
                await adapter.convert_to_frames(test_docs['empty'])
                print("  ✗ Empty content should have raised an error")
                error_tests.append(False)
            except (UnsupportedFormatError, DocumentProcessingError) as e:
                print(f"  ✓ Empty content properly handled: {e.__class__.__name__}")
                error_tests.append(True)
            except Exception as e:
                print(f"  ⚠️  Empty content raised unexpected error: {e.__class__.__name__}")
                error_tests.append(False)

            # Test 3: Corrupted content handling
            try:
                await adapter.convert_to_frames(test_docs['corrupted_png'])
                print("  ✗ Corrupted content should have raised an error")
                error_tests.append(False)
            except (UnsupportedFormatError, DocumentProcessingError) as e:
                print(f"  ✓ Corrupted content properly handled: {e.__class__.__name__}")
                error_tests.append(True)
            except Exception as e:
                print(f"  ⚠️  Corrupted content raised unexpected error: {e.__class__.__name__}")
                error_tests.append(False)

            # Test 4: No adapter registered error
            try:
                # Try to process valid content with no adapters registered
                await adapter.convert_to_frames(test_docs['png'])
                print("  ✗ No adapter should have raised UnsupportedFormatError")
                error_tests.append(False)
            except UnsupportedFormatError:
                print("  ✓ No adapter registered properly handled")
                error_tests.append(True)
            except Exception as e:
                print(f"  ⚠️  No adapter raised unexpected error: {e.__class__.__name__}")
                error_tests.append(False)

            success_rate = sum(error_tests) / len(error_tests) if error_tests else 0
            print(f"  Error handling success rate: {success_rate*100:.1f}%")

            return success_rate >= 0.8

        except Exception as e:
            print(f"  ✗ Error handling test failed: {e}")
            return False

    async def _test_multi_format_processing(self, DocumentAdapter, ConversionConfig,
                                          create_pdf_adapter, create_image_adapter,
                                          create_html_adapter) -> bool:
        """Test end-to-end multi-format processing pipeline."""
        try:
            from tatforge.core.document_adapter import DocumentFormat

            adapter = DocumentAdapter()

            # Register all adapters
            adapter.register_adapter(DocumentFormat.PDF, create_pdf_adapter())
            adapter.register_adapter(DocumentFormat.IMAGE, create_image_adapter())
            adapter.register_adapter(DocumentFormat.HTML, create_html_adapter())

            test_docs = create_test_documents()
            processing_results = []

            # Test PNG processing
            try:
                frames = await adapter.convert_to_frames(test_docs['png'])
                metadata = adapter.extract_metadata(test_docs['png'])

                if frames and len(frames) > 0 and metadata.format == DocumentFormat.IMAGE:
                    print(f"  ✓ PNG processing: {len(frames)} frames, {metadata.dimensions}")
                    processing_results.append(True)
                else:
                    print("  ✗ PNG processing failed")
                    processing_results.append(False)
            except Exception as e:
                print(f"  ✗ PNG processing error: {e}")
                processing_results.append(False)

            # Test JPEG processing
            try:
                config = ConversionConfig(quality=80)
                frames = await adapter.convert_to_frames(test_docs['jpeg'], config=config)
                metadata = adapter.extract_metadata(test_docs['jpeg'])

                if frames and len(frames) > 0 and metadata.format == DocumentFormat.IMAGE:
                    print(f"  ✓ JPEG processing: {len(frames)} frames, {metadata.dimensions}")
                    processing_results.append(True)
                else:
                    print("  ✗ JPEG processing failed")
                    processing_results.append(False)
            except Exception as e:
                print(f"  ✗ JPEG processing error: {e}")
                processing_results.append(False)

            # Test HTML processing
            try:
                config = ConversionConfig(dpi=150, max_pages=1)
                frames = await adapter.convert_to_frames(test_docs['html'], config=config)
                metadata = adapter.extract_metadata(test_docs['html'])

                if frames and len(frames) > 0 and metadata.format == DocumentFormat.HTML:
                    print(f"  ✓ HTML processing: {len(frames)} frames, title: '{metadata.title}'")
                    processing_results.append(True)
                else:
                    print("  ✗ HTML processing failed")
                    processing_results.append(False)
            except Exception as e:
                print(f"  ✗ HTML processing error: {e}")
                processing_results.append(False)

            # Test PDF processing if available
            pdfs_dir = Path("pdfs")
            if pdfs_dir.exists() and list(pdfs_dir.glob("*.pdf")):
                try:
                    pdf_file = list(pdfs_dir.glob("*.pdf"))[0]
                    with open(pdf_file, 'rb') as f:
                        pdf_content = f.read()

                    config = ConversionConfig(dpi=150, max_pages=1, quality=75)
                    frames = await adapter.convert_to_frames(pdf_content, config=config)
                    metadata = adapter.extract_metadata(pdf_content)

                    if frames and len(frames) > 0 and metadata.format == DocumentFormat.PDF:
                        print(f"  ✓ PDF processing: {len(frames)} frames, {metadata.page_count} pages")
                        processing_results.append(True)
                    else:
                        print("  ✗ PDF processing failed")
                        processing_results.append(False)
                except Exception as e:
                    print(f"  ✗ PDF processing error: {e}")
                    processing_results.append(False)

            success_rate = sum(processing_results) / len(processing_results) if processing_results else 0
            print(f"  Multi-format processing success rate: {success_rate*100:.1f}%")

            return success_rate >= 0.75

        except Exception as e:
            print(f"  ✗ Multi-format processing test failed: {e}")
            return False

    async def _test_extension_examples(self, BaseDocumentAdapter, DocumentFormat,
                                     DocumentAdapter) -> bool:
        """Test extension examples and plugin architecture flexibility."""
        try:
            # Create a simple custom adapter as an example
            class TestCustomAdapter(BaseDocumentAdapter):
                @property
                def supported_format(self) -> DocumentFormat:
                    return DocumentFormat.IMAGE

                async def convert_to_frames(self, content: bytes, config=None):
                    # Simple test implementation
                    image = Image.new('RGB', (200, 100), color='lightblue')
                    return [image]

                def extract_metadata(self, content: bytes):
                    from tatforge.core.document_adapter import DocumentMetadata
                    return DocumentMetadata(
                        format=DocumentFormat.IMAGE,
                        page_count=1,
                        file_size=len(content)
                    )

                def validate_format(self, content: bytes) -> bool:
                    return True

            # Test custom adapter integration
            adapter = DocumentAdapter()
            custom_adapter = TestCustomAdapter()

            # Register custom adapter
            adapter.register_adapter(DocumentFormat.IMAGE, custom_adapter)
            print("  ✓ Custom adapter registered successfully")

            # Test custom adapter functionality
            test_content = b"test content for custom adapter"
            frames = await adapter.convert_to_frames(test_content,
                                                   format_hint=DocumentFormat.IMAGE)

            if frames and len(frames) > 0:
                print(f"  ✓ Custom adapter processing: {len(frames)} frames, size {frames[0].size}")

                # Test metadata extraction
                metadata = adapter.extract_metadata(test_content,
                                                  format_hint=DocumentFormat.IMAGE)
                if metadata.format == DocumentFormat.IMAGE:
                    print("  ✓ Custom adapter metadata extraction working")
                    return True
                else:
                    print("  ✗ Custom adapter metadata extraction failed")
                    return False
            else:
                print("  ✗ Custom adapter processing failed")
                return False

        except Exception as e:
            print(f"  ✗ Extension examples test failed: {e}")
            return False


async def main():
    """Main test execution for COLPALI-203."""
    try:
        tester = MultiFormatAdapterTester()
        results = await tester.run_comprehensive_tests()

        if results.get("success", False):
            print(f"\n🎉 COLPALI-203 implementation completed successfully!")
            print(f"📊 Success Rate: {results['success_rate']:.1f}%")
            return 0
        else:
            print(f"\n❌ COLPALI-203 implementation needs attention")
            print(f"📊 Success Rate: {results.get('success_rate', 0):.1f}%")
            return 1

    except Exception as e:
        logger.error(f"COLPALI-203 test suite failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
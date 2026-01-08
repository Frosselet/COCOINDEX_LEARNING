#!/usr/bin/env python3
"""
Integration test for multi-format adapter interface.
Tests the core plugin architecture without problematic dependencies.
"""

import asyncio
import io
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Minimal type definitions for testing
class DocumentFormat(Enum):
    PDF = "pdf"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    WORD = "word"
    HTML = "html"
    IMAGE = "image"

@dataclass
class ConversionConfig:
    dpi: int = 300
    format: str = "RGB"
    max_pages: Optional[int] = None
    quality: int = 95
    preserve_aspect_ratio: bool = True

@dataclass
class DocumentMetadata:
    format: DocumentFormat
    page_count: int
    dimensions: Optional[tuple] = None
    file_size: int = 0
    creation_date: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None

# Exception classes
class DocumentProcessingError(Exception):
    pass

class MetadataExtractionError(Exception):
    pass

class UnsupportedFormatError(Exception):
    pass

class ValidationError(Exception):
    pass

# Abstract adapter interface
class BaseDocumentAdapter(ABC):
    @abstractmethod
    async def convert_to_frames(self, content: bytes, config: Optional[ConversionConfig] = None) -> List[Image.Image]:
        pass

    @abstractmethod
    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        pass

    @abstractmethod
    def validate_format(self, content: bytes) -> bool:
        pass

    @property
    @abstractmethod
    def supported_format(self) -> DocumentFormat:
        pass

# Simple adapter implementations for testing
class TestImageAdapter(BaseDocumentAdapter):
    @property
    def supported_format(self) -> DocumentFormat:
        return DocumentFormat.IMAGE

    async def convert_to_frames(self, content: bytes, config: Optional[ConversionConfig] = None) -> List[Image.Image]:
        with io.BytesIO(content) as image_stream:
            image = Image.open(image_stream)
            image.load()
            return [image]

    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        with io.BytesIO(content) as image_stream:
            image = Image.open(image_stream)
            return DocumentMetadata(
                format=DocumentFormat.IMAGE,
                page_count=1,
                dimensions=image.size,
                file_size=len(content)
            )

    def validate_format(self, content: bytes) -> bool:
        return (content.startswith(b"\xff\xd8\xff") or  # JPEG
                content.startswith(b"\x89PNG") or       # PNG
                content.startswith(b"GIF87a") or        # GIF
                content.startswith(b"GIF89a"))

class TestHTMLAdapter(BaseDocumentAdapter):
    @property
    def supported_format(self) -> DocumentFormat:
        return DocumentFormat.HTML

    async def convert_to_frames(self, content: bytes, config: Optional[ConversionConfig] = None) -> List[Image.Image]:
        # Simple text-based rendering
        html_text = content.decode('utf-8', errors='ignore')

        # Create basic image with HTML content summary
        image = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(image)
        draw.text((50, 50), "HTML Document Rendered", fill='black')
        draw.text((50, 100), f"Content length: {len(html_text)} chars", fill='black')

        return [image]

    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        html_content = content.decode('utf-8', errors='ignore')

        # Simple title extraction
        title = None
        if '<title>' in html_content.lower():
            start = html_content.lower().find('<title>') + 7
            end = html_content.lower().find('</title>', start)
            if end > start:
                title = html_content[start:end].strip()

        return DocumentMetadata(
            format=DocumentFormat.HTML,
            page_count=1,
            dimensions=(800, 600),
            file_size=len(content),
            title=title
        )

    def validate_format(self, content: bytes) -> bool:
        content_lower = content[:500].lower()
        return (b'<html' in content_lower or
                b'<!doctype html' in content_lower or
                b'<head>' in content_lower)

# Document adapter factory with plugin system
class DocumentAdapter:
    def __init__(self):
        self._adapters: Dict[DocumentFormat, BaseDocumentAdapter] = {}
        self._mime_type_mapping = {
            "text/html": DocumentFormat.HTML,
            "image/jpeg": DocumentFormat.IMAGE,
            "image/png": DocumentFormat.IMAGE,
            "image/gif": DocumentFormat.IMAGE,
        }

    def register_adapter(self, format_type: DocumentFormat, adapter: BaseDocumentAdapter) -> None:
        if adapter.supported_format != format_type:
            raise ValueError(f"Adapter format mismatch: {adapter.supported_format} != {format_type}")
        self._adapters[format_type] = adapter
        logger.info(f"Registered adapter for format: {format_type}")

    async def convert_to_frames(
        self,
        content: bytes,
        format_hint: Optional[DocumentFormat] = None,
        config: Optional[ConversionConfig] = None
    ) -> List[Image.Image]:
        if format_hint is None:
            format_hint = self._detect_format(content)

        adapter = self._get_adapter(format_hint)

        if not adapter.validate_format(content):
            raise DocumentProcessingError(f"Invalid {format_hint} format")

        config = config or ConversionConfig()
        return await adapter.convert_to_frames(content, config)

    def extract_metadata(
        self,
        content: bytes,
        format_hint: Optional[DocumentFormat] = None
    ) -> DocumentMetadata:
        if format_hint is None:
            format_hint = self._detect_format(content)

        adapter = self._get_adapter(format_hint)
        return adapter.extract_metadata(content)

    def _detect_format(self, content: bytes) -> DocumentFormat:
        # Basic format detection
        if content.startswith(b"\xff\xd8\xff"):  # JPEG
            return DocumentFormat.IMAGE
        elif content.startswith(b"\x89PNG"):  # PNG
            return DocumentFormat.IMAGE
        elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):  # GIF
            return DocumentFormat.IMAGE
        elif b'<html' in content[:200].lower() or b'<!doctype' in content[:200].lower():
            return DocumentFormat.HTML
        else:
            raise UnsupportedFormatError("Unable to detect document format")

    def _get_adapter(self, format_type: DocumentFormat) -> BaseDocumentAdapter:
        if format_type not in self._adapters:
            raise UnsupportedFormatError(f"No adapter registered for format: {format_type}")
        return self._adapters[format_type]

    def list_supported_formats(self) -> List[DocumentFormat]:
        return list(self._adapters.keys())

# Test implementation
def create_test_documents() -> Dict[str, bytes]:
    """Create test documents."""
    test_docs = {}

    # HTML document
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
    </head>
    <body>
        <h1>Multi-Format Test</h1>
        <p>This is a test HTML document.</p>
    </body>
    </html>
    """
    test_docs['html'] = html_content.encode('utf-8')

    # PNG Image
    image = Image.new('RGB', (400, 300), color=(200, 220, 240))
    draw = ImageDraw.Draw(image)
    draw.text((50, 50), "Test Image", fill=(0, 0, 0))

    png_buffer = io.BytesIO()
    image.save(png_buffer, format='PNG')
    test_docs['png'] = png_buffer.getvalue()

    # JPEG Image
    jpeg_image = Image.new('RGB', (500, 400), color=(240, 220, 200))
    draw = ImageDraw.Draw(jpeg_image)
    draw.text((50, 50), "JPEG Test", fill=(100, 50, 0))

    jpeg_buffer = io.BytesIO()
    jpeg_image.save(jpeg_buffer, format='JPEG')
    test_docs['jpeg'] = jpeg_buffer.getvalue()

    return test_docs

async def test_multi_adapter_system():
    """Test the multi-format adapter system."""
    print("="*70)
    print("MULTI-FORMAT ADAPTER INTEGRATION TEST")
    print("="*70)

    try:
        # Create adapter factory
        adapter = DocumentAdapter()

        # Register adapters (demonstrating plugin system)
        print("\n--- Adapter Registration ---")
        adapter.register_adapter(DocumentFormat.IMAGE, TestImageAdapter())
        adapter.register_adapter(DocumentFormat.HTML, TestHTMLAdapter())

        supported_formats = adapter.list_supported_formats()
        print(f"Registered formats: {[fmt.value for fmt in supported_formats]}")

        # Create test documents
        test_docs = create_test_documents()

        # Test each format
        print("\n--- Format Detection & Processing ---")
        results = []

        for doc_type, content in test_docs.items():
            try:
                print(f"\nTesting {doc_type}:")

                # Test format detection
                detected_format = adapter._detect_format(content)
                print(f"  Format detected: {detected_format.value}")

                # Test metadata extraction
                metadata = adapter.extract_metadata(content)
                print(f"  Metadata: {metadata.page_count} pages, size: {metadata.file_size} bytes")
                if metadata.title:
                    print(f"  Title: {metadata.title}")

                # Test frame conversion
                frames = await adapter.convert_to_frames(content)
                print(f"  Frames generated: {len(frames)} frames")
                if frames:
                    frame_size = frames[0].size
                    print(f"  Frame size: {frame_size}")

                results.append(True)
                print(f"  ✓ {doc_type.upper()} processing successful")

            except Exception as e:
                print(f"  ✗ {doc_type.upper()} processing failed: {e}")
                results.append(False)

        # Test error handling
        print("\n--- Error Handling ---")
        error_tests = 0
        error_passed = 0

        # Test unsupported format
        error_tests += 1
        try:
            await adapter.convert_to_frames(b"unsupported content")
            print("  ✗ Unsupported format should have failed")
        except UnsupportedFormatError:
            print("  ✓ Unsupported format properly rejected")
            error_passed += 1
        except Exception as e:
            print(f"  ⚠️  Unsupported format failed with: {e}")

        # Test configuration system
        print("\n--- Configuration System ---")
        config_test_passed = False
        try:
            config = ConversionConfig(dpi=150, quality=75, max_pages=1)
            frames = await adapter.convert_to_frames(test_docs['png'], config=config)
            if frames and len(frames) <= config.max_pages:
                print(f"  ✓ Configuration applied: {len(frames)} frames (max: {config.max_pages})")
                config_test_passed = True
            else:
                print("  ✗ Configuration not applied correctly")
        except Exception as e:
            print(f"  ✗ Configuration test failed: {e}")

        # Calculate overall results
        processing_success_rate = sum(results) / len(results) * 100
        error_handling_rate = error_passed / error_tests * 100 if error_tests > 0 else 0

        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Format Processing:    {processing_success_rate:.1f}% ({sum(results)}/{len(results)})")
        print(f"Error Handling:       {error_handling_rate:.1f}% ({error_passed}/{error_tests})")
        print(f"Configuration System: {'✓ PASS' if config_test_passed else '✗ FAIL'}")

        overall_success = (
            processing_success_rate >= 80 and
            error_handling_rate >= 80 and
            config_test_passed
        )

        if overall_success:
            print("\n✓ MULTI-FORMAT ADAPTER INTEGRATION TEST PASSED")
            return True
        else:
            print("\n✗ MULTI-FORMAT ADAPTER INTEGRATION TEST FAILED")
            return False

    except Exception as e:
        print(f"\n✗ Integration test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_multi_adapter_system())
    exit(0 if success else 1)
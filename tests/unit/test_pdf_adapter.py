#!/usr/bin/env python3
"""
Minimal test for PDF adapter functionality.
Tests the PDF adapter directly without package dependencies.
"""

import asyncio
import sys
import io
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Minimal type definitions (normally from document_adapter.py)
class DocumentFormat(Enum):
    PDF = "pdf"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    WORD = "word"
    HTML = "html"

@dataclass
class DocumentMetadata:
    format: DocumentFormat
    page_count: int
    dimensions: Optional[tuple] = None
    file_size: Optional[int] = None
    creation_date: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None

@dataclass
class ConversionConfig:
    dpi: int = 200
    quality: int = 95
    format: str = "RGB"
    max_pages: Optional[int] = None

# Custom exceptions
class DocumentProcessingError(Exception):
    pass

class MetadataExtractionError(Exception):
    pass

# Minimal PDF adapter implementation (adapted from pdf_adapter.py)
import PyPDF2
from pdf2image import convert_from_bytes
from PIL import Image

class PDFAdapter:
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        logger.info(f"PDFAdapter initialized with max_memory={max_memory_mb}MB")

    def validate_format(self, content: bytes) -> bool:
        try:
            if not content.startswith(b'%PDF-'):
                return False
            with io.BytesIO(content) as pdf_stream:
                reader = PyPDF2.PdfReader(pdf_stream)
                _ = len(reader.pages)
                if reader.is_encrypted:
                    logger.warning("PDF is password protected")
                    return False
                return True
        except Exception as e:
            logger.debug(f"PDF validation failed: {e}")
            return False

    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        try:
            with io.BytesIO(content) as pdf_stream:
                reader = PyPDF2.PdfReader(pdf_stream)
                page_count = len(reader.pages)
                file_size = len(content)

                metadata_dict = {}
                if reader.metadata:
                    metadata_dict = {
                        'title': reader.metadata.get('/Title'),
                        'author': reader.metadata.get('/Author'),
                        'creator': reader.metadata.get('/Creator'),
                        'producer': reader.metadata.get('/Producer'),
                        'creation_date': reader.metadata.get('/CreationDate'),
                        'modification_date': reader.metadata.get('/ModDate')
                    }

                dimensions = None
                if page_count > 0:
                    first_page = reader.pages[0]
                    page_box = first_page.mediabox
                    dimensions = (float(page_box.width), float(page_box.height))

                return DocumentMetadata(
                    format=DocumentFormat.PDF,
                    page_count=page_count,
                    dimensions=dimensions,
                    file_size=file_size,
                    creation_date=metadata_dict.get('creation_date'),
                    title=metadata_dict.get('title'),
                    author=metadata_dict.get('author')
                )

        except Exception as e:
            logger.error(f"PDF metadata extraction failed: {e}")
            raise MetadataExtractionError(f"Failed to extract PDF metadata: {e}")

    async def convert_to_frames(self, content: bytes, config: Optional[ConversionConfig] = None) -> List[Image.Image]:
        config = config or ConversionConfig()

        try:
            if not self.validate_format(content):
                raise DocumentProcessingError("Invalid PDF format")

            metadata = self.extract_metadata(content)
            total_pages = metadata.page_count

            if config.max_pages:
                total_pages = min(total_pages, config.max_pages)

            logger.info(f"Converting PDF: {total_pages} pages at {config.dpi} DPI")

            # Simple conversion for testing
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                None,
                self._pdf2image_convert,
                content,
                total_pages,
                config
            )

            logger.info(f"Successfully converted PDF to {len(images)} frames")
            return images

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise DocumentProcessingError(f"Failed to convert PDF: {e}")

    def _pdf2image_convert(self, content: bytes, total_pages: int, config: ConversionConfig) -> List[Image.Image]:
        try:
            kwargs = {
                'dpi': config.dpi,
                'fmt': 'RGB' if config.format == 'RGB' else 'L',
                'last_page': total_pages,
                'poppler_path': None,
                'use_cropbox': False,
                'strict': False
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            logger.debug(f"Converting with pdf2image: {kwargs}")
            images = convert_from_bytes(content, **kwargs)

            # Apply quality settings if needed
            if config.quality < 100:
                optimized_images = []
                for img in images:
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=config.quality, optimize=True)
                    buffer.seek(0)
                    optimized_img = Image.open(buffer)
                    optimized_images.append(optimized_img)
                images = optimized_images

            logger.debug(f"Converted {len(images)} pages successfully")
            return images

        except Exception as e:
            logger.error(f"pdf2image conversion failed: {e}")
            raise DocumentProcessingError(f"pdf2image failed: {e}")

async def test_pdf_adapter():
    """Test PDF adapter with all sample PDFs."""
    print("="*70)
    print("PDF ADAPTER TEST SUITE")
    print("="*70)

    adapter = PDFAdapter(max_memory_mb=300)
    pdfs_dir = Path("pdfs")

    if not pdfs_dir.exists():
        print(f"✗ PDFs directory not found: {pdfs_dir}")
        return False

    pdf_files = list(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"✗ No PDF files found in {pdfs_dir}")
        return False

    print(f"Found {len(pdf_files)} PDF files to test\n")

    results = []

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i:2d}/{len(pdf_files)}] Testing: {pdf_file.name}")

        try:
            with open(pdf_file, 'rb') as f:
                content = f.read()

            # Test format validation
            is_valid = adapter.validate_format(content)
            if not is_valid:
                print(f"        ✗ Format validation failed")
                results.append(False)
                continue

            # Test metadata extraction
            metadata = adapter.extract_metadata(content)
            print(f"        ✓ {metadata.page_count} pages, {len(content)//1024}KB")

            # Test conversion with small config
            config = ConversionConfig(dpi=150, quality=75, max_pages=1)
            frames = await adapter.convert_to_frames(content, config)

            if frames:
                frame = frames[0]
                print(f"        ✓ Converted to {frame.width}x{frame.height} image")
                results.append(True)
            else:
                print(f"        ✗ No frames generated")
                results.append(False)

        except Exception as e:
            print(f"        ✗ Failed: {str(e)[:50]}...")
            results.append(False)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_tests = len(results)
    successful = sum(results)
    failed = total_tests - successful
    success_rate = (successful / total_tests * 100) if total_tests > 0 else 0

    print(f"Total Files:   {total_tests}")
    print(f"Successful:    {successful} ✓")
    print(f"Failed:        {failed} ✗")
    print(f"Success Rate:  {success_rate:.1f}%")

    if success_rate >= 80:
        print(f"\n✓ PDF ADAPTER TEST PASSED")
        return True
    else:
        print(f"\n✗ PDF ADAPTER TEST FAILED")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pdf_adapter())
    sys.exit(0 if success else 1)
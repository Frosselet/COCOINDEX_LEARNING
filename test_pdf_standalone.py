#!/usr/bin/env python3
"""
Standalone test for PDF adapter functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import required components directly
from colpali_engine.core.document_adapter import ConversionConfig, DocumentFormat, DocumentMetadata
from colpali_engine.adapters.pdf_adapter import PDFAdapter

async def test_pdf_adapter():
    """Test PDF adapter with basic functionality."""
    print("Testing PDF adapter functionality...")

    # Initialize adapter
    adapter = PDFAdapter(max_memory_mb=300)
    print(f"✓ PDFAdapter initialized with max_memory=300MB")

    # Check if PDFs directory exists
    pdfs_dir = Path("pdfs")
    if not pdfs_dir.exists():
        print(f"✗ PDFs directory not found: {pdfs_dir}")
        return False

    # Get test PDFs
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"✗ No PDF files found in {pdfs_dir}")
        return False

    print(f"✓ Found {len(pdf_files)} PDF files to test")

    # Test with the first PDF
    test_pdf = pdf_files[0]
    print(f"\nTesting with: {test_pdf.name}")

    try:
        # Read file content
        with open(test_pdf, 'rb') as f:
            content = f.read()

        print(f"✓ Read PDF file: {len(content)} bytes")

        # Test format validation
        is_valid = adapter.validate_format(content)
        print(f"✓ Format validation: {'Valid' if is_valid else 'Invalid'}")

        if not is_valid:
            print(f"✗ PDF format validation failed")
            return False

        # Test metadata extraction
        metadata = adapter.extract_metadata(content)
        print(f"✓ Metadata extracted: {metadata.page_count} pages")
        print(f"  - File size: {metadata.file_size} bytes")
        print(f"  - Dimensions: {metadata.dimensions}")
        print(f"  - Title: {metadata.title}")

        # Test conversion with small config for speed
        config = ConversionConfig(dpi=150, quality=75, max_pages=1)
        frames = await adapter.convert_to_frames(content, config)

        print(f"✓ Conversion successful: {len(frames)} frames")

        if frames:
            frame = frames[0]
            print(f"  - Frame size: {frame.width}x{frame.height}")
            print(f"  - Mode: {frame.mode}")
            print(f"  - Format: {frame.format}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pdf_adapter())
    sys.exit(0 if success else 1)
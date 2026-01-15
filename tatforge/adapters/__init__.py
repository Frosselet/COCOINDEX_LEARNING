"""
Document adapter implementations for various format types.

This module contains concrete implementations of document adapters
for PDF, Excel, PowerPoint, Word, HTML, and Image formats.

The adapter system provides a plugin architecture where new format
adapters can be easily registered and used through the DocumentAdapter
factory interface.
"""

from .pdf_adapter import PDFAdapter, create_pdf_adapter
from .image_adapter import ImageAdapter, create_image_adapter
from .html_adapter import HTMLAdapter, create_html_adapter
from .image_processor import ImageProcessor, ProcessingConfig, create_image_processor

__all__ = [
    # Core adapters
    "PDFAdapter",
    "ImageAdapter",
    "HTMLAdapter",
    "ImageProcessor",

    # Factory functions
    "create_pdf_adapter",
    "create_image_adapter",
    "create_html_adapter",
    "create_image_processor",

    # Configuration classes
    "ProcessingConfig"
]
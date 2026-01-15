"""
PDF document adapter using pdf2image for high-fidelity conversion.

This adapter converts PDF documents to high-resolution image frames
using the pdf2image library with poppler backend, optimized for
ColPali vision processing.
"""

import asyncio
import io
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

import PyPDF2
from pdf2image import convert_from_bytes
from PIL import Image

from ..core.document_adapter import (
    BaseDocumentAdapter,
    DocumentFormat,
    DocumentMetadata,
    ConversionConfig,
    DocumentProcessingError,
    MetadataExtractionError,
    ValidationError
)

logger = logging.getLogger(__name__)


class PDFAdapter(BaseDocumentAdapter):
    """
    PDF document adapter using pdf2image for conversion.

    Handles PDF to image conversion with high fidelity, preserving
    spatial relationships and visual quality for ColPali processing.
    Includes memory optimization for large documents.
    """

    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize PDF adapter.

        Args:
            max_memory_mb: Maximum memory usage per conversion (MB)
        """
        self.max_memory_mb = max_memory_mb
        logger.info(f"PDFAdapter initialized with max_memory={max_memory_mb}MB")

    @property
    def supported_format(self) -> DocumentFormat:
        """The document format this adapter supports."""
        return DocumentFormat.PDF

    async def convert_to_frames(
        self,
        content: Union[bytes, Path, str],
        config: Optional[ConversionConfig] = None
    ) -> List[Image.Image]:
        """
        Convert PDF content to image frames.

        Args:
            content: Raw PDF bytes, or path to PDF file (Path or str)
            config: Optional conversion configuration

        Returns:
            List of PIL Image objects representing PDF pages

        Raises:
            DocumentProcessingError: If conversion fails
        """
        config = config or ConversionConfig()

        try:
            # Handle file path input - convert to bytes
            if isinstance(content, (str, Path)):
                file_path = Path(content)
                if not file_path.exists():
                    raise DocumentProcessingError(f"PDF file not found: {file_path}")
                logger.info(f"Reading PDF from file: {file_path}")
                content = file_path.read_bytes()

            # Validate PDF format first
            if not self.validate_format(content):
                raise DocumentProcessingError("Invalid PDF format")

            # Get page count for memory optimization
            metadata = self.extract_metadata(content)
            total_pages = metadata.page_count

            if config.max_pages:
                total_pages = min(total_pages, config.max_pages)

            logger.info(f"Converting PDF: {total_pages} pages at {config.dpi} DPI")

            # Process in batches for memory efficiency
            batch_size = self._calculate_batch_size(config.dpi, total_pages)
            frames = []

            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                batch_pages = list(range(batch_start + 1, batch_end + 1))  # pdf2image uses 1-based indexing

                logger.debug(f"Processing batch: pages {batch_start + 1}-{batch_end}")

                # Convert batch to images
                batch_frames = await self._convert_batch(
                    content, batch_pages, config
                )

                frames.extend(batch_frames)

                # Memory cleanup between batches
                if len(batch_frames) > 0:
                    del batch_frames

            logger.info(f"Successfully converted PDF to {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise DocumentProcessingError(f"Failed to convert PDF: {e}")

    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        """
        Extract metadata from PDF without full conversion.

        Args:
            content: Raw PDF bytes

        Returns:
            DocumentMetadata with PDF properties

        Raises:
            MetadataExtractionError: If metadata extraction fails
        """
        try:
            # Use PyPDF2 for fast metadata extraction
            with io.BytesIO(content) as pdf_stream:
                reader = PyPDF2.PdfReader(pdf_stream)

                # Basic metadata
                page_count = len(reader.pages)
                file_size = len(content)

                # Extract PDF info if available
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

                # Get page dimensions from first page
                dimensions = None
                if page_count > 0:
                    first_page = reader.pages[0]
                    page_box = first_page.mediabox
                    dimensions = (
                        float(page_box.width),
                        float(page_box.height)
                    )

                # Format creation date if available
                creation_date = None
                if metadata_dict.get('creation_date'):
                    creation_date = self._parse_pdf_date(metadata_dict['creation_date'])

                logger.debug(f"Extracted metadata: {page_count} pages, {file_size} bytes")

                return DocumentMetadata(
                    format=DocumentFormat.PDF,
                    page_count=page_count,
                    dimensions=dimensions,
                    file_size=file_size,
                    creation_date=creation_date,
                    title=metadata_dict.get('title'),
                    author=metadata_dict.get('author')
                )

        except Exception as e:
            logger.error(f"PDF metadata extraction failed: {e}")
            raise MetadataExtractionError(f"Failed to extract PDF metadata: {e}")

    def validate_format(self, content: bytes) -> bool:
        """
        Validate that the content is a valid PDF.

        Args:
            content: Raw document bytes

        Returns:
            True if format is valid PDF

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check PDF header
            if not content.startswith(b'%PDF-'):
                return False

            # Try to read with PyPDF2 for deeper validation
            with io.BytesIO(content) as pdf_stream:
                reader = PyPDF2.PdfReader(pdf_stream)

                # Check if we can read the document structure
                _ = len(reader.pages)

                # Check for encryption/password protection
                if reader.is_encrypted:
                    logger.warning("PDF is password protected")
                    # For now, we don't support encrypted PDFs
                    return False

                return True

        except Exception as e:
            logger.debug(f"PDF validation failed: {e}")
            return False

    async def _convert_batch(
        self,
        content: bytes,
        page_numbers: List[int],
        config: ConversionConfig
    ) -> List[Image.Image]:
        """
        Convert a batch of PDF pages to images.

        Args:
            content: PDF content bytes
            page_numbers: List of page numbers to convert (1-based)
            config: Conversion configuration

        Returns:
            List of PIL Images for the batch
        """
        try:
            # Run pdf2image conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            images = await loop.run_in_executor(
                None,
                self._pdf2image_convert,
                content,
                page_numbers,
                config
            )

            return images

        except Exception as e:
            logger.error(f"Batch conversion failed for pages {page_numbers}: {e}")
            raise DocumentProcessingError(f"Batch conversion failed: {e}")

    def _pdf2image_convert(
        self,
        content: bytes,
        page_numbers: List[int],
        config: ConversionConfig
    ) -> List[Image.Image]:
        """
        Synchronous pdf2image conversion (runs in executor).

        Args:
            content: PDF content bytes
            page_numbers: Pages to convert (1-based indexing)
            config: Conversion configuration

        Returns:
            List of PIL Images
        """
        try:
            # Configure pdf2image parameters
            kwargs = {
                'dpi': config.dpi,
                'fmt': 'RGB' if config.format == 'RGB' else 'L',
                'first_page': min(page_numbers) if page_numbers else None,
                'last_page': max(page_numbers) if page_numbers else None,
                'poppler_path': None,  # Use system poppler
                'use_cropbox': False,
                'strict': False  # Don't fail on warnings
            }

            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            logger.debug(f"Converting with pdf2image: {kwargs}")

            # Convert using pdf2image
            images = convert_from_bytes(content, **kwargs)

            # Apply quality settings if needed
            if config.quality < 100:
                optimized_images = []
                for img in images:
                    # Apply JPEG compression to reduce quality/size
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

    def _calculate_batch_size(self, dpi: int, total_pages: int) -> int:
        """
        Calculate optimal batch size based on DPI and memory constraints.

        Args:
            dpi: Image DPI
            total_pages: Total number of pages

        Returns:
            Optimal batch size for memory efficiency
        """
        # Estimate memory per page (rough calculation)
        # A4 at 300 DPI ≈ 2480x3508 pixels ≈ 26MB (RGB)
        pixels_per_page = (8.27 * dpi) * (11.69 * dpi)  # A4 dimensions in inches
        mb_per_page = (pixels_per_page * 3) / (1024 * 1024)  # RGB bytes to MB

        # Calculate safe batch size
        max_pages_per_batch = max(1, int(self.max_memory_mb / mb_per_page))

        # Don't exceed total pages
        batch_size = min(max_pages_per_batch, total_pages)

        logger.debug(f"Calculated batch size: {batch_size} pages "
                    f"(estimated {mb_per_page:.1f}MB per page)")

        return batch_size

    def _parse_pdf_date(self, pdf_date_string: str) -> Optional[str]:
        """
        Parse PDF date format to ISO string.

        Args:
            pdf_date_string: PDF date in format D:YYYYMMDDHHmmSSOHH'mm'

        Returns:
            ISO format date string or None if parsing fails
        """
        try:
            # Remove 'D:' prefix if present
            if pdf_date_string.startswith('D:'):
                pdf_date_string = pdf_date_string[2:]

            # Parse basic format YYYYMMDDHHMMSS
            if len(pdf_date_string) >= 14:
                year = pdf_date_string[0:4]
                month = pdf_date_string[4:6]
                day = pdf_date_string[6:8]
                hour = pdf_date_string[8:10]
                minute = pdf_date_string[10:12]
                second = pdf_date_string[12:14]

                # Create ISO string
                iso_string = f"{year}-{month}-{day}T{hour}:{minute}:{second}"

                # Validate by parsing
                datetime.fromisoformat(iso_string)

                return iso_string

            return None

        except Exception as e:
            logger.debug(f"Failed to parse PDF date '{pdf_date_string}': {e}")
            return None


# Factory function for easy instantiation
def create_pdf_adapter(max_memory_mb: int = 500) -> PDFAdapter:
    """
    Create a configured PDF adapter.

    Args:
        max_memory_mb: Maximum memory usage per conversion

    Returns:
        Configured PDFAdapter instance
    """
    return PDFAdapter(max_memory_mb=max_memory_mb)
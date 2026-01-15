"""
Document adapter interfaces and factory for format-agnostic processing.

This module provides the abstract interfaces and factory pattern for converting
various document formats (PDF, Excel, PowerPoint, Word, HTML) into canonical
image frames for vision processing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass
from PIL import Image
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentFormat(Enum):
    """Supported document formats."""
    PDF = "pdf"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    WORD = "word"
    HTML = "html"
    IMAGE = "image"


@dataclass
class ConversionConfig:
    """Configuration for document to image conversion."""
    dpi: int = 300
    format: str = "RGB"
    max_pages: Optional[int] = None
    quality: int = 95
    preserve_aspect_ratio: bool = True


@dataclass
class DocumentMetadata:
    """Metadata extracted during document processing."""
    format: DocumentFormat
    page_count: int
    dimensions: Optional[tuple] = None
    file_size: int = 0
    creation_date: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None


class BaseDocumentAdapter(ABC):
    """
    Abstract base class for document adapters.

    Each adapter is responsible for converting a specific document format
    into standardized image frames while preserving visual fidelity and
    spatial relationships.
    """

    @abstractmethod
    async def convert_to_frames(
        self,
        content: bytes,
        config: Optional[ConversionConfig] = None
    ) -> List[Image.Image]:
        """
        Convert document content to image frames.

        Args:
            content: Raw document bytes
            config: Optional conversion configuration

        Returns:
            List of PIL Image objects representing document pages

        Raises:
            DocumentProcessingError: If conversion fails
        """
        pass

    @abstractmethod
    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        """
        Extract metadata from document without full conversion.

        Args:
            content: Raw document bytes

        Returns:
            DocumentMetadata with document properties

        Raises:
            MetadataExtractionError: If metadata extraction fails
        """
        pass

    @abstractmethod
    def validate_format(self, content: bytes) -> bool:
        """
        Validate that the content matches the expected format.

        Args:
            content: Raw document bytes

        Returns:
            True if format is valid and supported

        Raises:
            ValidationError: If validation fails
        """
        pass

    @property
    @abstractmethod
    def supported_format(self) -> DocumentFormat:
        """The document format this adapter supports."""
        pass


class DocumentAdapterProtocol(Protocol):
    """Protocol definition for document adapters (for type checking)."""

    async def convert_to_frames(
        self,
        content: bytes,
        config: Optional[ConversionConfig] = None
    ) -> List[Image.Image]:
        ...

    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        ...

    def validate_format(self, content: bytes) -> bool:
        ...

    @property
    def supported_format(self) -> DocumentFormat:
        ...


class DocumentAdapter:
    """
    Main document adapter factory and orchestrator.

    This class provides the main interface for document processing,
    automatically detecting document formats and routing to appropriate
    adapters using a plugin system.
    """

    def __init__(self):
        self._adapters: Dict[DocumentFormat, BaseDocumentAdapter] = {}
        self._mime_type_mapping = {
            "application/pdf": DocumentFormat.PDF,
            "application/vnd.ms-excel": DocumentFormat.EXCEL,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentFormat.EXCEL,
            "application/vnd.ms-powerpoint": DocumentFormat.POWERPOINT,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentFormat.POWERPOINT,
            "application/msword": DocumentFormat.WORD,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFormat.WORD,
            "text/html": DocumentFormat.HTML,
            "image/jpeg": DocumentFormat.IMAGE,
            "image/png": DocumentFormat.IMAGE,
            "image/tiff": DocumentFormat.IMAGE
        }

        logger.info("DocumentAdapter initialized")

    def register_adapter(
        self,
        format_type: DocumentFormat,
        adapter: BaseDocumentAdapter
    ) -> None:
        """
        Register a document adapter for a specific format.

        Args:
            format_type: The document format this adapter handles
            adapter: The adapter instance

        Raises:
            ValueError: If adapter format doesn't match registration format
        """
        if adapter.supported_format != format_type:
            raise ValueError(
                f"Adapter format mismatch: {adapter.supported_format} != {format_type}"
            )

        self._adapters[format_type] = adapter
        logger.info(f"Registered adapter for format: {format_type}")

    async def convert_to_frames(
        self,
        content: bytes,
        format_hint: Optional[DocumentFormat] = None,
        config: Optional[ConversionConfig] = None
    ) -> List[Image.Image]:
        """
        Convert document to image frames using appropriate adapter.

        Args:
            content: Raw document bytes
            format_hint: Optional format hint to skip detection
            config: Optional conversion configuration

        Returns:
            List of PIL Image objects

        Raises:
            UnsupportedFormatError: If no adapter available for format
            DocumentProcessingError: If conversion fails
        """
        # Detect format if not provided
        if format_hint is None:
            format_hint = await self._detect_format(content)

        # Get appropriate adapter
        adapter = self._get_adapter(format_hint)

        # Validate format
        if not adapter.validate_format(content):
            raise DocumentProcessingError(f"Invalid {format_hint} format")

        # Convert to frames
        logger.info(f"Converting {format_hint} document to image frames")
        config = config or ConversionConfig()

        try:
            frames = await adapter.convert_to_frames(content, config)
            logger.info(f"Successfully converted to {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"Document conversion failed: {e}")
            raise DocumentProcessingError(f"Failed to convert {format_hint}: {e}")

    def extract_metadata(
        self,
        content: bytes,
        format_hint: Optional[DocumentFormat] = None
    ) -> DocumentMetadata:
        """
        Extract metadata from document using appropriate adapter.

        Args:
            content: Raw document bytes
            format_hint: Optional format hint

        Returns:
            DocumentMetadata object

        Raises:
            UnsupportedFormatError: If no adapter available
            MetadataExtractionError: If extraction fails
        """
        # Detect format if not provided
        if format_hint is None:
            format_hint = self._detect_format_sync(content)

        # Get appropriate adapter
        adapter = self._get_adapter(format_hint)

        try:
            metadata = adapter.extract_metadata(content)
            logger.info(f"Extracted metadata for {format_hint} document")
            return metadata

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise MetadataExtractionError(f"Failed to extract metadata: {e}")

    async def _detect_format(self, content: bytes) -> DocumentFormat:
        """
        Detect document format from content using MIME type detection.

        Uses python-magic for robust MIME type detection with fallback
        to signature-based detection for reliability.
        """
        # Try MIME type detection first
        try:
            import magic
            mime_type = magic.from_buffer(content, mime=True)

            if mime_type in self._mime_type_mapping:
                logger.debug(f"Detected format via MIME type: {mime_type}")
                return self._mime_type_mapping[mime_type]

            logger.debug(f"Unknown MIME type: {mime_type}, falling back to signature detection")
        except ImportError:
            logger.debug("python-magic not available, using signature detection")
        except Exception as e:
            logger.debug(f"MIME detection failed: {e}, using signature detection")

        # Fallback to file signature detection
        return self._detect_by_signature(content)

    def _detect_by_signature(self, content: bytes) -> DocumentFormat:
        """Detect format using file signatures (magic numbers)."""
        # PDF format
        if content.startswith(b"%PDF"):
            return DocumentFormat.PDF

        # ZIP-based formats (Office documents)
        elif content.startswith(b"PK"):
            # Look for Office format indicators within ZIP structure
            content_str = content[:2048].decode('latin-1', errors='ignore')

            if any(indicator in content_str for indicator in ['xl/', 'worksheets/', 'sharedStrings']):
                return DocumentFormat.EXCEL
            elif any(indicator in content_str for indicator in ['ppt/', 'slides/', 'presentation']):
                return DocumentFormat.POWERPOINT
            elif any(indicator in content_str for indicator in ['word/', 'document.xml']):
                return DocumentFormat.WORD
            else:
                # Default to Excel for unknown ZIP formats
                return DocumentFormat.EXCEL

        # HTML format
        elif (content.startswith(b"<!DOCTYPE html") or
              content.startswith(b"<html") or
              content.startswith(b"<HTML")):
            return DocumentFormat.HTML

        # Image formats
        elif content.startswith(b"\xff\xd8\xff"):  # JPEG
            return DocumentFormat.IMAGE
        elif content.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            return DocumentFormat.IMAGE
        elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):  # GIF
            return DocumentFormat.IMAGE
        elif content.startswith(b"II*\x00") or content.startswith(b"MM\x00*"):  # TIFF
            return DocumentFormat.IMAGE
        elif content.startswith(b"BM"):  # BMP
            return DocumentFormat.IMAGE
        elif content.startswith(b"RIFF") and b"WEBP" in content[:12]:  # WebP
            return DocumentFormat.IMAGE

        # RTF format (Rich Text Format)
        elif content.startswith(b"{\\rtf"):
            return DocumentFormat.WORD  # Treat RTF as Word-compatible

        # Legacy Office formats
        elif content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):  # OLE2/CFB
            # This is a legacy Office format, need to check subtype
            return DocumentFormat.EXCEL  # Default assumption

        else:
            raise UnsupportedFormatError(
                f"Unable to detect document format from signature: {content[:16].hex()}"
            )

    def _detect_format_sync(self, content: bytes) -> DocumentFormat:
        """
        Synchronous format detection using MIME type detection.

        Same logic as async version but without async/await.
        """
        # Try MIME type detection first
        try:
            import magic
            mime_type = magic.from_buffer(content, mime=True)

            if mime_type in self._mime_type_mapping:
                logger.debug(f"Detected format via MIME type: {mime_type}")
                return self._mime_type_mapping[mime_type]

            logger.debug(f"Unknown MIME type: {mime_type}, falling back to signature detection")
        except ImportError:
            logger.debug("python-magic not available, using signature detection")
        except Exception as e:
            logger.debug(f"MIME detection failed: {e}, using signature detection")

        # Fallback to file signature detection
        return self._detect_by_signature(content)

    def _get_adapter(self, format_type: DocumentFormat) -> BaseDocumentAdapter:
        """Get adapter for the specified format."""
        if format_type not in self._adapters:
            raise UnsupportedFormatError(f"No adapter registered for format: {format_type}")

        return self._adapters[format_type]

    def list_supported_formats(self) -> List[DocumentFormat]:
        """Get list of currently supported document formats."""
        return list(self._adapters.keys())


# Custom exceptions
class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class MetadataExtractionError(Exception):
    """Raised when metadata extraction fails."""
    pass


class UnsupportedFormatError(Exception):
    """Raised when document format is not supported."""
    pass


class ValidationError(Exception):
    """Raised when document validation fails."""
    pass
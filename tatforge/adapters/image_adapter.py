"""
Image document adapter for direct image processing.

This adapter handles image files (JPEG, PNG, TIFF, etc.) that don't require
document-to-image conversion, but do need standardization for ColPali processing.
"""

import logging
import io
import asyncio
from typing import List, Optional
from datetime import datetime

from PIL import Image, ExifTags

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


class ImageAdapter(BaseDocumentAdapter):
    """
    Image document adapter for direct image processing.

    Handles image formats (JPEG, PNG, TIFF, GIF, BMP, WebP) by loading them
    directly and applying standardization processing without document conversion.
    """

    def __init__(self, max_memory_mb: int = 200):
        """
        Initialize image adapter.

        Args:
            max_memory_mb: Maximum memory usage for image processing (MB)
        """
        self.max_memory_mb = max_memory_mb
        self.supported_formats = {
            'JPEG', 'JPG', 'PNG', 'TIFF', 'TIF', 'GIF', 'BMP', 'WEBP'
        }
        logger.info(f"ImageAdapter initialized with max_memory={max_memory_mb}MB")

    @property
    def supported_format(self) -> DocumentFormat:
        """The document format this adapter supports."""
        return DocumentFormat.IMAGE

    async def convert_to_frames(
        self,
        content: bytes,
        config: Optional[ConversionConfig] = None
    ) -> List[Image.Image]:
        """
        Convert image content to standardized frames.

        For image files, this means loading the image and optionally
        applying standardization processing.

        Args:
            content: Raw image bytes
            config: Optional conversion configuration

        Returns:
            List containing the processed image (single frame)

        Raises:
            DocumentProcessingError: If image processing fails
        """
        config = config or ConversionConfig()

        try:
            # Validate image format first
            if not self.validate_format(content):
                raise DocumentProcessingError("Invalid image format")

            logger.info("Processing image file")

            # Load image
            image = await self._load_image(content)

            # Apply any requested processing
            if config.max_pages and config.max_pages < 1:
                # If max_pages is 0, return empty list
                return []

            # Handle animated formats (like GIF)
            frames = await self._extract_frames(image, config)

            logger.info(f"Successfully processed image: {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise DocumentProcessingError(f"Failed to process image: {e}")

    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        """
        Extract metadata from image file.

        Args:
            content: Raw image bytes

        Returns:
            DocumentMetadata with image properties

        Raises:
            MetadataExtractionError: If metadata extraction fails
        """
        try:
            with io.BytesIO(content) as image_stream:
                image = Image.open(image_stream)

                # Basic metadata
                width, height = image.size
                file_size = len(content)
                page_count = 1

                # Handle animated images (GIF)
                if hasattr(image, 'is_animated') and image.is_animated:
                    try:
                        page_count = image.n_frames
                    except Exception:
                        page_count = 1

                # Extract EXIF data if available
                title = None
                author = None
                creation_date = None

                if hasattr(image, '_getexif') and image._getexif():
                    try:
                        exif_dict = image._getexif()
                        if exif_dict:
                            # Look for common EXIF fields
                            for tag_id, value in exif_dict.items():
                                tag = ExifTags.TAGS.get(tag_id, tag_id)

                                if tag == 'ImageDescription':
                                    title = str(value)[:200]  # Limit length
                                elif tag == 'Artist':
                                    author = str(value)[:100]
                                elif tag == 'DateTime':
                                    creation_date = self._parse_exif_datetime(str(value))

                    except Exception as e:
                        logger.debug(f"EXIF parsing failed: {e}")

                # Format creation date if we got one
                if not creation_date and hasattr(image, 'info'):
                    # Try to get creation date from image info
                    date_info = image.info.get('date', None)
                    if date_info:
                        creation_date = str(date_info)

                logger.debug(f"Extracted image metadata: {width}x{height}, {file_size} bytes")

                return DocumentMetadata(
                    format=DocumentFormat.IMAGE,
                    page_count=page_count,
                    dimensions=(width, height),
                    file_size=file_size,
                    creation_date=creation_date,
                    title=title,
                    author=author
                )

        except Exception as e:
            logger.error(f"Image metadata extraction failed: {e}")
            raise MetadataExtractionError(f"Failed to extract image metadata: {e}")

    def validate_format(self, content: bytes) -> bool:
        """
        Validate that the content is a supported image format.

        Args:
            content: Raw image bytes

        Returns:
            True if format is valid and supported

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check magic numbers for common image formats
            if content.startswith(b"\xff\xd8\xff"):  # JPEG
                return True
            elif content.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
                return True
            elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):  # GIF
                return True
            elif content.startswith(b"II*\x00") or content.startswith(b"MM\x00*"):  # TIFF
                return True
            elif content.startswith(b"BM"):  # BMP
                return True
            elif content.startswith(b"RIFF") and b"WEBP" in content[:12]:  # WebP
                return True

            # Try to open with PIL as final validation
            try:
                with io.BytesIO(content) as image_stream:
                    image = Image.open(image_stream)
                    image.verify()  # Verify it's a valid image
                    return True
            except Exception:
                return False

        except Exception as e:
            logger.debug(f"Image validation failed: {e}")
            return False

    async def _load_image(self, content: bytes) -> Image.Image:
        """Load image from bytes asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                self._load_image_sync,
                content
            )
            return image
        except Exception as e:
            raise DocumentProcessingError(f"Failed to load image: {e}")

    def _load_image_sync(self, content: bytes) -> Image.Image:
        """Synchronous image loading."""
        with io.BytesIO(content) as image_stream:
            image = Image.open(image_stream)
            # Load the image data to avoid lazy loading issues
            image.load()
            return image

    async def _extract_frames(
        self,
        image: Image.Image,
        config: ConversionConfig
    ) -> List[Image.Image]:
        """
        Extract frames from image (handles animated images).

        Args:
            image: PIL Image object
            config: Conversion configuration

        Returns:
            List of image frames
        """
        frames = []

        try:
            # Check if it's an animated image
            if hasattr(image, 'is_animated') and image.is_animated:
                # Process animated image frames
                n_frames = getattr(image, 'n_frames', 1)

                # Limit frames if configured
                if config.max_pages:
                    n_frames = min(n_frames, config.max_pages)

                logger.debug(f"Processing animated image: {n_frames} frames")

                for i in range(n_frames):
                    try:
                        image.seek(i)
                        # Convert frame to consistent format
                        frame = image.copy()
                        if frame.mode not in ['RGB', 'RGBA']:
                            frame = frame.convert('RGB')
                        frames.append(frame)
                    except Exception as e:
                        logger.debug(f"Failed to extract frame {i}: {e}")
                        break

            else:
                # Single frame image
                frame = image.copy()
                if frame.mode not in ['RGB', 'RGBA']:
                    frame = frame.convert('RGB')
                frames.append(frame)

            return frames

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            # Fallback: return single frame
            frame = image.copy()
            if frame.mode not in ['RGB', 'RGBA']:
                frame = frame.convert('RGB')
            return [frame]

    def _parse_exif_datetime(self, datetime_str: str) -> Optional[str]:
        """
        Parse EXIF datetime to ISO format.

        Args:
            datetime_str: EXIF datetime string (YYYY:MM:DD HH:MM:SS)

        Returns:
            ISO format datetime string or None if parsing fails
        """
        try:
            # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
            if len(datetime_str) == 19 and datetime_str[4] == ':' and datetime_str[7] == ':':
                # Convert to ISO format
                iso_string = datetime_str.replace(':', '-', 2).replace(' ', 'T')

                # Validate by parsing
                datetime.fromisoformat(iso_string)

                return iso_string

            return None

        except Exception as e:
            logger.debug(f"Failed to parse EXIF datetime '{datetime_str}': {e}")
            return None


# Factory function for easy instantiation
def create_image_adapter(max_memory_mb: int = 200) -> ImageAdapter:
    """
    Create a configured image adapter.

    Args:
        max_memory_mb: Maximum memory usage for processing

    Returns:
        Configured ImageAdapter instance
    """
    return ImageAdapter(max_memory_mb=max_memory_mb)
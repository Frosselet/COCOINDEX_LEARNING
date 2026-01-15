"""
Image standardization processor for ColPali vision processing.

This module provides image standardization utilities that ensure consistent
image dimensions, color spaces, and quality across different document sources
for optimal ColPali model performance.
"""

import logging
import io
import asyncio
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ExifTags
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for processed images."""
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    color_mode: str
    file_size_bytes: int
    format: Optional[str] = None
    dpi: Optional[Tuple[int, int]] = None
    quality_score: Optional[float] = None
    exif_data: Optional[Dict[str, Any]] = None
    processing_notes: Optional[List[str]] = None


@dataclass
class ProcessingConfig:
    """Configuration for image processing operations."""
    target_width: int = 1024
    target_height: int = 1024
    maintain_aspect_ratio: bool = True
    color_mode: str = "RGB"  # RGB, L (grayscale), or "auto"
    quality: int = 90
    resample_filter: str = "LANCZOS"  # LANCZOS, BICUBIC, BILINEAR
    preserve_exif: bool = True
    enable_enhancement: bool = True
    max_file_size_mb: Optional[float] = None
    background_color: Tuple[int, int, int] = (255, 255, 255)  # White background for padding


class ImageProcessor:
    """
    High-performance image processor for ColPali vision pipeline.

    Provides standardized image processing with consistent dimensions,
    color space normalization, quality optimization, and metadata preservation.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize image processor with configuration.

        Args:
            config: Processing configuration, defaults to standard ColPali settings
        """
        self.config = config or ProcessingConfig()
        self.processing_stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'average_quality_score': 0.0
        }
        logger.info(f"ImageProcessor initialized with target size: "
                   f"{self.config.target_width}x{self.config.target_height}")

    async def process_image(self, image: Image.Image, metadata: Optional[Dict] = None) -> Tuple[Image.Image, ImageMetadata]:
        """
        Process a single image to ColPali standards.

        Args:
            image: PIL Image to process
            metadata: Optional metadata for lineage tracking

        Returns:
            Tuple of processed image and processing metadata
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_image_sync, image, metadata
        )

    async def process_batch(
        self,
        images: List[Image.Image],
        metadata: Optional[List[Dict]] = None
    ) -> List[Tuple[Image.Image, ImageMetadata]]:
        """
        Process multiple images concurrently.

        Args:
            images: List of PIL Images to process
            metadata: Optional metadata list for lineage tracking

        Returns:
            List of tuples containing processed images and their metadata
        """
        if metadata is None:
            metadata = [None] * len(images)

        logger.info(f"Processing batch of {len(images)} images")

        # Process images concurrently
        tasks = [
            self.process_image(img, meta)
            for img, meta in zip(images, metadata)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process image {i}: {result}")
                # Return original image with error metadata
                error_metadata = ImageMetadata(
                    original_size=images[i].size,
                    processed_size=images[i].size,
                    color_mode=images[i].mode,
                    file_size_bytes=0,
                    processing_notes=[f"Processing failed: {result}"]
                )
                processed_results.append((images[i], error_metadata))
            else:
                processed_results.append(result)

        logger.info(f"Batch processing complete: {len(processed_results)} images")
        return processed_results

    def _process_image_sync(self, image: Image.Image, metadata: Optional[Dict] = None) -> Tuple[Image.Image, ImageMetadata]:
        """
        Synchronous image processing implementation.

        Args:
            image: PIL Image to process
            metadata: Optional source metadata

        Returns:
            Tuple of processed image and processing metadata
        """
        try:
            processing_notes = []
            original_size = image.size
            original_mode = image.mode

            # Extract EXIF data if present
            exif_data = None
            if self.config.preserve_exif and hasattr(image, '_getexif'):
                try:
                    exif_dict = image._getexif()
                    if exif_dict:
                        exif_data = {
                            ExifTags.TAGS.get(k, k): v
                            for k, v in exif_dict.items()
                            if k in ExifTags.TAGS
                        }
                except Exception as e:
                    logger.debug(f"Failed to extract EXIF: {e}")

            # Convert color mode if needed
            processed_image = self._normalize_color_space(image, processing_notes)

            # Resize image to target dimensions
            processed_image = self._resize_image(processed_image, processing_notes)

            # Apply image enhancements if enabled
            if self.config.enable_enhancement:
                processed_image = self._enhance_image(processed_image, processing_notes)

            # Calculate quality score
            quality_score = self._calculate_quality_score(processed_image, image)

            # Apply compression if max file size is set
            if self.config.max_file_size_mb:
                processed_image = self._optimize_file_size(processed_image, processing_notes)

            # Get DPI information
            dpi = None
            try:
                dpi = processed_image.info.get('dpi', None)
            except Exception:
                pass

            # Calculate file size estimation
            buffer = io.BytesIO()
            processed_image.save(buffer, format='JPEG', quality=self.config.quality)
            estimated_file_size = len(buffer.getvalue())

            # Create metadata
            image_metadata = ImageMetadata(
                original_size=original_size,
                processed_size=processed_image.size,
                color_mode=processed_image.mode,
                file_size_bytes=estimated_file_size,
                format="JPEG",
                dpi=dpi,
                quality_score=quality_score,
                exif_data=exif_data,
                processing_notes=processing_notes if processing_notes else None
            )

            # Update processing stats
            self.processing_stats['images_processed'] += 1
            if quality_score:
                current_avg = self.processing_stats['average_quality_score']
                total_processed = self.processing_stats['images_processed']
                self.processing_stats['average_quality_score'] = (
                    (current_avg * (total_processed - 1) + quality_score) / total_processed
                )

            logger.debug(f"Processed image: {original_size} -> {processed_image.size}, "
                        f"mode: {original_mode} -> {processed_image.mode}, "
                        f"quality: {quality_score:.2f}")

            return processed_image, image_metadata

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            # Return original image with error metadata
            error_metadata = ImageMetadata(
                original_size=image.size,
                processed_size=image.size,
                color_mode=image.mode,
                file_size_bytes=0,
                processing_notes=[f"Processing failed: {str(e)}"]
            )
            return image, error_metadata

    def _normalize_color_space(self, image: Image.Image, notes: List[str]) -> Image.Image:
        """Normalize image color space according to configuration."""
        target_mode = self.config.color_mode

        if target_mode == "auto":
            # Automatically determine best color mode
            if image.mode in ['RGBA', 'LA']:
                # Convert images with alpha to RGB with background
                background = Image.new('RGB', image.size, self.config.background_color)
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
                notes.append("Converted alpha channel to RGB with background")
            elif image.mode not in ['RGB', 'L']:
                # Convert other modes to RGB
                image = image.convert('RGB')
                notes.append(f"Converted to RGB from {image.mode}")
        else:
            if image.mode != target_mode:
                if target_mode == 'RGB' and image.mode in ['RGBA', 'LA']:
                    # Handle alpha channel conversion
                    background = Image.new('RGB', image.size, self.config.background_color)
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[3])
                    else:
                        background.paste(image, mask=image.split()[1])
                    image = background
                    notes.append(f"Converted {image.mode} to RGB with white background")
                else:
                    image = image.convert(target_mode)
                    notes.append(f"Converted color mode to {target_mode}")

        return image

    def _resize_image(self, image: Image.Image, notes: List[str]) -> Image.Image:
        """Resize image to target dimensions while maintaining aspect ratio if configured."""
        original_size = image.size
        target_size = (self.config.target_width, self.config.target_height)

        if original_size == target_size:
            return image

        if not self.config.maintain_aspect_ratio:
            # Direct resize without maintaining aspect ratio
            resample = getattr(Image.Resampling, self.config.resample_filter, Image.Resampling.LANCZOS)
            image = image.resize(target_size, resample)
            notes.append(f"Resized to {target_size} (aspect ratio not maintained)")
        else:
            # Calculate new size maintaining aspect ratio
            aspect_ratio = original_size[0] / original_size[1]
            target_aspect_ratio = target_size[0] / target_size[1]

            if aspect_ratio > target_aspect_ratio:
                # Image is wider than target
                new_width = target_size[0]
                new_height = int(target_size[0] / aspect_ratio)
            else:
                # Image is taller than target
                new_width = int(target_size[1] * aspect_ratio)
                new_height = target_size[1]

            # Resize image
            resample = getattr(Image.Resampling, self.config.resample_filter, Image.Resampling.LANCZOS)
            image = image.resize((new_width, new_height), resample)

            # Add padding if needed to reach exact target size
            if (new_width, new_height) != target_size:
                # Create background with padding
                padded_image = Image.new(image.mode, target_size, self.config.background_color)

                # Calculate centering offset
                x_offset = (target_size[0] - new_width) // 2
                y_offset = (target_size[1] - new_height) // 2

                # Paste resized image onto padded background
                padded_image.paste(image, (x_offset, y_offset))
                image = padded_image

                notes.append(f"Resized to {new_width}x{new_height} and padded to {target_size}")
            else:
                notes.append(f"Resized to {target_size} (aspect ratio maintained)")

        return image

    def _enhance_image(self, image: Image.Image, notes: List[str]) -> Image.Image:
        """Apply subtle image enhancements for better ColPali processing."""
        try:
            # Subtle sharpening for text clarity
            image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=10, threshold=1))

            # Slight contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)

            notes.append("Applied sharpening and contrast enhancement")
        except Exception as e:
            logger.debug(f"Enhancement failed: {e}")
            notes.append("Enhancement skipped due to error")

        return image

    def _calculate_quality_score(self, processed_image: Image.Image, original_image: Image.Image) -> float:
        """
        Calculate a quality score for the processed image.

        Returns a score between 0-100 based on resolution preservation,
        color information, and estimated clarity.
        """
        try:
            # Resolution score (based on how much detail is preserved)
            original_pixels = original_image.size[0] * original_image.size[1]
            processed_pixels = processed_image.size[0] * processed_image.size[1]
            resolution_score = min(100, (processed_pixels / original_pixels) * 100) if original_pixels > 0 else 50

            # Color information score
            color_score = 100 if processed_image.mode == 'RGB' else (70 if processed_image.mode == 'L' else 50)

            # Clarity estimation (based on image variance - higher variance usually means more detail)
            try:
                img_array = np.array(processed_image.convert('L'))  # Convert to grayscale for analysis
                clarity_score = min(100, np.var(img_array) / 10)  # Normalize variance to 0-100 scale
            except Exception:
                clarity_score = 75  # Default score if calculation fails

            # Weighted average
            final_score = (resolution_score * 0.4 + color_score * 0.3 + clarity_score * 0.3)
            return round(final_score, 2)

        except Exception as e:
            logger.debug(f"Quality score calculation failed: {e}")
            return 75.0  # Default score

    def _optimize_file_size(self, image: Image.Image, notes: List[str]) -> Image.Image:
        """Optimize image file size if it exceeds the configured maximum."""
        if not self.config.max_file_size_mb:
            return image

        max_bytes = int(self.config.max_file_size_mb * 1024 * 1024)
        quality = self.config.quality

        # Test current size
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        current_size = len(buffer.getvalue())

        if current_size <= max_bytes:
            return image

        # Reduce quality iteratively to meet size requirement
        while quality > 20 and current_size > max_bytes:
            quality -= 5
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            current_size = len(buffer.getvalue())

        # Load the optimized image
        buffer.seek(0)
        optimized_image = Image.open(buffer)

        notes.append(f"Optimized file size: quality reduced to {quality}")
        return optimized_image

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'average_quality_score': 0.0
        }


# Factory function for easy instantiation
def create_image_processor(
    target_size: Tuple[int, int] = (1024, 1024),
    color_mode: str = "RGB",
    quality: int = 90,
    maintain_aspect_ratio: bool = True
) -> ImageProcessor:
    """
    Create a configured image processor for ColPali processing.

    Args:
        target_size: Target image dimensions (width, height)
        color_mode: Target color mode ('RGB', 'L', or 'auto')
        quality: JPEG quality (1-100)
        maintain_aspect_ratio: Whether to preserve aspect ratios

    Returns:
        Configured ImageProcessor instance
    """
    config = ProcessingConfig(
        target_width=target_size[0],
        target_height=target_size[1],
        color_mode=color_mode,
        quality=quality,
        maintain_aspect_ratio=maintain_aspect_ratio
    )
    return ImageProcessor(config)
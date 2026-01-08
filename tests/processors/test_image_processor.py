#!/usr/bin/env python3
"""
Standalone test for image standardization processor.
Includes the ImageProcessor implementation directly to avoid import issues.
"""

import asyncio
import logging
import time
import io
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ExifTags

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    color_mode: str = "RGB"
    quality: int = 90
    resample_filter: str = "LANCZOS"
    preserve_exif: bool = True
    enable_enhancement: bool = True
    max_file_size_mb: Optional[float] = None
    background_color: Tuple[int, int, int] = (255, 255, 255)


class ImageProcessor:
    """Image processor for ColPali vision pipeline."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.processing_stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'average_quality_score': 0.0
        }
        logger.info(f"ImageProcessor initialized with target size: "
                   f"{self.config.target_width}x{self.config.target_height}")

    async def process_image(self, image: Image.Image, metadata: Optional[Dict] = None) -> Tuple[Image.Image, ImageMetadata]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_image_sync, image, metadata
        )

    async def process_batch(self, images: List[Image.Image], metadata: Optional[List[Dict]] = None) -> List[Tuple[Image.Image, ImageMetadata]]:
        if metadata is None:
            metadata = [None] * len(images)

        logger.info(f"Processing batch of {len(images)} images")
        tasks = [self.process_image(img, meta) for img, meta in zip(images, metadata)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process image {i}: {result}")
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

        return processed_results

    def _process_image_sync(self, image: Image.Image, metadata: Optional[Dict] = None) -> Tuple[Image.Image, ImageMetadata]:
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
                        exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif_dict.items() if k in ExifTags.TAGS}
                except Exception:
                    pass

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

            return processed_image, image_metadata

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            error_metadata = ImageMetadata(
                original_size=image.size,
                processed_size=image.size,
                color_mode=image.mode,
                file_size_bytes=0,
                processing_notes=[f"Processing failed: {str(e)}"]
            )
            return image, error_metadata

    def _normalize_color_space(self, image: Image.Image, notes: List[str]) -> Image.Image:
        target_mode = self.config.color_mode

        if target_mode == "auto":
            if image.mode in ['RGBA', 'LA']:
                background = Image.new('RGB', image.size, self.config.background_color)
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
                notes.append("Converted alpha channel to RGB with background")
            elif image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
                notes.append(f"Converted to RGB from {image.mode}")
        else:
            if image.mode != target_mode:
                if target_mode == 'RGB' and image.mode in ['RGBA', 'LA']:
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
        original_size = image.size
        target_size = (self.config.target_width, self.config.target_height)

        if original_size == target_size:
            return image

        if not self.config.maintain_aspect_ratio:
            resample = getattr(Image.Resampling, self.config.resample_filter, Image.Resampling.LANCZOS)
            image = image.resize(target_size, resample)
            notes.append(f"Resized to {target_size} (aspect ratio not maintained)")
        else:
            # Calculate new size maintaining aspect ratio
            aspect_ratio = original_size[0] / original_size[1]
            target_aspect_ratio = target_size[0] / target_size[1]

            if aspect_ratio > target_aspect_ratio:
                new_width = target_size[0]
                new_height = int(target_size[0] / aspect_ratio)
            else:
                new_width = int(target_size[1] * aspect_ratio)
                new_height = target_size[1]

            # Resize image
            resample = getattr(Image.Resampling, self.config.resample_filter, Image.Resampling.LANCZOS)
            image = image.resize((new_width, new_height), resample)

            # Add padding if needed to reach exact target size
            if (new_width, new_height) != target_size:
                padded_image = Image.new(image.mode, target_size, self.config.background_color)
                x_offset = (target_size[0] - new_width) // 2
                y_offset = (target_size[1] - new_height) // 2
                padded_image.paste(image, (x_offset, y_offset))
                image = padded_image
                notes.append(f"Resized to {new_width}x{new_height} and padded to {target_size}")
            else:
                notes.append(f"Resized to {target_size} (aspect ratio maintained)")

        return image

    def _enhance_image(self, image: Image.Image, notes: List[str]) -> Image.Image:
        try:
            image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=10, threshold=1))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            notes.append("Applied sharpening and contrast enhancement")
        except Exception as e:
            logger.debug(f"Enhancement failed: {e}")
            notes.append("Enhancement skipped due to error")
        return image

    def _calculate_quality_score(self, processed_image: Image.Image, original_image: Image.Image) -> float:
        try:
            original_pixels = original_image.size[0] * original_image.size[1]
            processed_pixels = processed_image.size[0] * processed_image.size[1]
            resolution_score = min(100, (processed_pixels / original_pixels) * 100) if original_pixels > 0 else 50

            color_score = 100 if processed_image.mode == 'RGB' else (70 if processed_image.mode == 'L' else 50)

            try:
                img_array = np.array(processed_image.convert('L'))
                clarity_score = min(100, np.var(img_array) / 10)
            except Exception:
                clarity_score = 75

            final_score = (resolution_score * 0.4 + color_score * 0.3 + clarity_score * 0.3)
            return round(final_score, 2)

        except Exception:
            return 75.0

    def _optimize_file_size(self, image: Image.Image, notes: List[str]) -> Image.Image:
        if not self.config.max_file_size_mb:
            return image

        max_bytes = int(self.config.max_file_size_mb * 1024 * 1024)
        quality = self.config.quality

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        current_size = len(buffer.getvalue())

        if current_size <= max_bytes:
            return image

        while quality > 20 and current_size > max_bytes:
            quality -= 5
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            current_size = len(buffer.getvalue())

        buffer.seek(0)
        optimized_image = Image.open(buffer)
        notes.append(f"Optimized file size: quality reduced to {quality}")
        return optimized_image


def create_test_images() -> List[Tuple[str, Image.Image]]:
    """Create various test images with different characteristics."""
    test_images = []

    # 1. Large RGB image
    img1 = Image.new('RGB', (2000, 2800), (255, 255, 255))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([100, 100, 1900, 2700], outline=(0, 0, 0), width=5)
    draw.text((200, 200), "Large RGB Document", fill=(0, 0, 0))
    test_images.append(('large_rgb', img1))

    # 2. Small grayscale image
    img2 = Image.new('L', (400, 600), 128)
    draw = ImageDraw.Draw(img2)
    draw.ellipse([50, 50, 350, 550], outline=0, width=3)
    test_images.append(('small_grayscale', img2))

    # 3. RGBA image with transparency
    img3 = Image.new('RGBA', (800, 600), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img3)
    draw.polygon([(100, 100), (700, 100), (400, 500)], fill=(255, 0, 0, 128))
    test_images.append(('rgba_transparent', img3))

    # 4. Wide aspect ratio
    img4 = Image.new('RGB', (3000, 500), (200, 200, 255))
    draw = ImageDraw.Draw(img4)
    for i in range(0, 3000, 100):
        draw.line([(i, 0), (i, 500)], fill=(0, 0, 100), width=2)
    test_images.append(('wide_aspect', img4))

    # 5. Tall aspect ratio
    img5 = Image.new('RGB', (500, 3000), (255, 200, 200))
    draw = ImageDraw.Draw(img5)
    for i in range(0, 3000, 100):
        draw.line([(0, i), (500, i)], fill=(100, 0, 0), width=2)
    test_images.append(('tall_aspect', img5))

    # 6. Square image (target size)
    img6 = Image.new('RGB', (1024, 1024), (200, 255, 200))
    draw = ImageDraw.Draw(img6)
    draw.rectangle([100, 100, 924, 924], outline=(0, 100, 0), width=10)
    test_images.append(('square_target', img6))

    return test_images


async def test_image_processor():
    """Test image processor with various configurations."""
    print("="*70)
    print("IMAGE PROCESSOR TEST SUITE")
    print("="*70)

    # Create test images
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images")

    # Test configurations
    test_configs = [
        ('standard_1024', ProcessingConfig(target_width=1024, target_height=1024)),
        ('high_res_2048', ProcessingConfig(target_width=2048, target_height=2048)),
        ('grayscale_mode', ProcessingConfig(target_width=1024, target_height=1024, color_mode='L')),
        ('no_aspect_ratio', ProcessingConfig(target_width=1024, target_height=1024, maintain_aspect_ratio=False)),
        ('low_quality', ProcessingConfig(target_width=1024, target_height=1024, quality=50)),
        ('size_limited', ProcessingConfig(target_width=1024, target_height=1024, max_file_size_mb=0.5))
    ]

    results = []

    for config_name, config in test_configs:
        print(f"\n--- Testing Configuration: {config_name} ---")
        processor = ImageProcessor(config)

        config_results = []

        for img_name, img in test_images:
            try:
                start_time = time.time()
                processed_img, metadata = await processor.process_image(img)
                processing_time = time.time() - start_time

                # Validate results
                checks = []

                # Size check
                expected_size = (config.target_width, config.target_height)
                if processed_img.size == expected_size:
                    checks.append("✓ Size correct")
                else:
                    checks.append(f"✗ Size mismatch: got {processed_img.size}")

                # Color mode check
                if config.color_mode == "auto" or processed_img.mode == config.color_mode:
                    checks.append("✓ Color mode correct")
                else:
                    checks.append(f"✗ Color mode: got {processed_img.mode}")

                # Quality score check
                if metadata.quality_score is not None and metadata.quality_score > 0:
                    checks.append(f"✓ Quality: {metadata.quality_score:.1f}")
                else:
                    checks.append("✗ Quality score missing")

                # File size check
                if config.max_file_size_mb:
                    max_bytes = config.max_file_size_mb * 1024 * 1024
                    if metadata.file_size_bytes <= max_bytes:
                        checks.append(f"✓ Size OK: {metadata.file_size_bytes//1024}KB")
                    else:
                        checks.append(f"✗ Size exceeded: {metadata.file_size_bytes//1024}KB")

                success = all("✓" in check for check in checks)
                config_results.append(success)

                print(f"  {img_name:15} ({processing_time*1000:.1f}ms): " + " | ".join(checks))

            except Exception as e:
                print(f"  {img_name:15} ✗ Failed: {str(e)[:50]}...")
                config_results.append(False)

        # Calculate success rate for this configuration
        success_rate = sum(config_results) / len(config_results) * 100
        print(f"  Configuration Success Rate: {success_rate:.1f}%")
        results.append((config_name, success_rate))

    # Test batch processing
    print(f"\n--- Testing Batch Processing ---")
    processor = ImageProcessor()
    images_only = [img for _, img in test_images]

    try:
        start_time = time.time()
        batch_results = await processor.process_batch(images_only)
        batch_time = time.time() - start_time

        if len(batch_results) == len(images_only):
            print(f"✓ Batch processing: {len(batch_results)} images in {batch_time:.2f}s")
            print(f"✓ Average time per image: {batch_time/len(batch_results)*1000:.1f}ms")

            successful_batch = sum(1 for img, meta in batch_results
                                 if meta.processing_notes is None or
                                 not any("failed" in note.lower() for note in meta.processing_notes))
            batch_success_rate = successful_batch / len(batch_results) * 100
            print(f"✓ Batch success rate: {batch_success_rate:.1f}%")
        else:
            print(f"✗ Batch processing failed: got {len(batch_results)} results")

    except Exception as e:
        print(f"✗ Batch processing failed: {e}")

    # Print final summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    overall_success = sum(rate for _, rate in results) / len(results) if results else 0

    print(f"Configuration Test Results:")
    for config_name, success_rate in results:
        status = "✓" if success_rate >= 80 else "✗"
        print(f"  {status} {config_name:20} {success_rate:6.1f}%")

    print(f"\nOverall Success Rate: {overall_success:.1f}%")

    if overall_success >= 80:
        print(f"\n✓ IMAGE PROCESSOR TEST PASSED")
        return True
    else:
        print(f"\n✗ IMAGE PROCESSOR TEST FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_image_processor())
    exit(0 if success else 1)
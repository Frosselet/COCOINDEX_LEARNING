#!/usr/bin/env python3
"""
Test suite for image standardization processor.
Tests the ImageProcessor with various configurations and validates
standardization capabilities for ColPali processing.
"""

import asyncio
import logging
import time
import io
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_images() -> List[Image.Image]:
    """Create various test images with different characteristics."""
    test_images = []

    # 1. Large RGB image (simulates high-res document scan)
    img1 = Image.new('RGB', (2000, 2800), (255, 255, 255))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([100, 100, 1900, 2700], outline=(0, 0, 0), width=5)
    draw.text((200, 200), "Large RGB Document", fill=(0, 0, 0))
    test_images.append(('large_rgb', img1))

    # 2. Small grayscale image
    img2 = Image.new('L', (400, 600), 128)
    draw = ImageDraw.Draw(img2)
    draw.ellipse([50, 50, 350, 550], outline=0, width=3)
    draw.text((150, 300), "Grayscale", fill=0)
    test_images.append(('small_grayscale', img2))

    # 3. RGBA image with transparency
    img3 = Image.new('RGBA', (800, 600), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img3)
    draw.polygon([(100, 100), (700, 100), (400, 500)], fill=(255, 0, 0, 128))
    test_images.append(('rgba_transparent', img3))

    # 4. Very wide image (aspect ratio challenge)
    img4 = Image.new('RGB', (3000, 500), (200, 200, 255))
    draw = ImageDraw.Draw(img4)
    for i in range(0, 3000, 100):
        draw.line([(i, 0), (i, 500)], fill=(0, 0, 100), width=2)
    test_images.append(('wide_aspect', img4))

    # 5. Very tall image (aspect ratio challenge)
    img5 = Image.new('RGB', (500, 3000), (255, 200, 200))
    draw = ImageDraw.Draw(img5)
    for i in range(0, 3000, 100):
        draw.line([(0, i), (500, i)], fill=(100, 0, 0), width=2)
    test_images.append(('tall_aspect', img5))

    # 6. Square image (no resize needed)
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

    # Import here to avoid dependency issues during testing
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))

        from tatforge.adapters.image_processor import (
            ImageProcessor, ProcessingConfig, create_image_processor
        )
    except ImportError as e:
        print(f"✗ Failed to import ImageProcessor: {e}")
        return False

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
                    checks.append(f"✗ Size mismatch: got {processed_img.size}, expected {expected_size}")

                # Color mode check
                if config.color_mode == "auto" or processed_img.mode == config.color_mode:
                    checks.append("✓ Color mode correct")
                else:
                    checks.append(f"✗ Color mode: got {processed_img.mode}, expected {config.color_mode}")

                # Metadata check
                if metadata.quality_score is not None and metadata.quality_score > 0:
                    checks.append(f"✓ Quality score: {metadata.quality_score:.1f}")
                else:
                    checks.append("✗ Quality score missing")

                # File size check (if configured)
                if config.max_file_size_mb:
                    max_bytes = config.max_file_size_mb * 1024 * 1024
                    if metadata.file_size_bytes <= max_bytes:
                        checks.append(f"✓ File size OK: {metadata.file_size_bytes//1024}KB")
                    else:
                        checks.append(f"✗ File size exceeded: {metadata.file_size_bytes//1024}KB")

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
    processor = create_image_processor()
    images_only = [img for _, img in test_images]

    try:
        start_time = time.time()
        batch_results = await processor.process_batch(images_only)
        batch_time = time.time() - start_time

        if len(batch_results) == len(images_only):
            print(f"✓ Batch processing: {len(batch_results)} images in {batch_time:.2f}s")
            print(f"✓ Average time per image: {batch_time/len(batch_results)*1000:.1f}ms")

            # Check if all images were processed successfully
            successful_batch = sum(1 for img, meta in batch_results if meta.processing_notes is None or not any("failed" in note.lower() for note in meta.processing_notes))
            batch_success_rate = successful_batch / len(batch_results) * 100
            print(f"✓ Batch success rate: {batch_success_rate:.1f}%")
        else:
            print(f"✗ Batch processing failed: got {len(batch_results)} results for {len(images_only)} inputs")

    except Exception as e:
        print(f"✗ Batch processing failed: {e}")

    # Test with real PDF images (if available)
    print(f"\n--- Testing with PDF-Generated Images ---")
    pdfs_dir = Path("pdfs")
    if pdfs_dir.exists():
        try:
            # Use the minimal PDF adapter to generate test images
            from test_pdf_minimal import PDFAdapter
            from tatforge.adapters.image_processor import ConversionConfig as PDFConfig

            pdf_files = list(pdfs_dir.glob("*.pdf"))[:3]  # Test with first 3 PDFs
            processor = create_image_processor()
            pdf_adapter = PDFAdapter()

            for pdf_file in pdf_files:
                try:
                    with open(pdf_file, 'rb') as f:
                        content = f.read()

                    # Generate image from PDF
                    pdf_config = PDFConfig(dpi=150, quality=85, max_pages=1)
                    frames = await pdf_adapter.convert_to_frames(content, pdf_config)

                    if frames:
                        # Process the first frame
                        processed_img, metadata = await processor.process_image(frames[0])
                        print(f"  {pdf_file.name[:20]:20} ✓ PDF->Image: {frames[0].size} -> {processed_img.size}, "
                              f"Quality: {metadata.quality_score:.1f}")
                    else:
                        print(f"  {pdf_file.name[:20]:20} ✗ No frames generated")

                except Exception as e:
                    print(f"  {pdf_file.name[:20]:20} ✗ Failed: {str(e)[:30]}...")

        except ImportError:
            print("  Skipped: PDF adapter not available for testing")
    else:
        print("  Skipped: No PDFs directory found")

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
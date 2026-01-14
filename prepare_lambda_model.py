#!/usr/bin/env python3
"""
Pre-download and optimize ColPali model for AWS Lambda deployment.

This script runs during Docker build to prepare the model with:
- INT8 dynamic quantization for 50% memory reduction
- CPU-optimized inference configuration
- Pre-warmed model cache for faster cold starts

Implements COLPALI-901: Optimize Lambda container for 3B model deployment.
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.environ.get("COLPALI_MODEL", "vidore/colqwen2-v0.1")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
QUANTIZE_MODEL = os.environ.get("QUANTIZE_MODEL", "true").lower() == "true"


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def calculate_model_size(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def download_model():
    """Download ColPali model from HuggingFace."""
    logger.info(f"Downloading ColPali model: {MODEL_NAME}")
    start_time = time.time()
    initial_memory = get_memory_usage_mb()

    try:
        from transformers import AutoModel, AutoProcessor

        # Download processor first (lightweight)
        logger.info("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Download model
        logger.info("Downloading model...")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32  # Use float32 for CPU
        )

        download_time = time.time() - start_time
        model_size = calculate_model_size(model)
        memory_after = get_memory_usage_mb()

        logger.info(f"Model downloaded successfully in {download_time:.2f}s")
        logger.info(f"Model size: {model_size:.2f}MB")
        logger.info(f"Memory usage: {memory_after:.2f}MB (+{memory_after - initial_memory:.2f}MB)")

        return model, processor

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def quantize_model(model: nn.Module) -> tuple:
    """Apply INT8 dynamic quantization to model."""
    logger.info("Starting INT8 dynamic quantization...")
    start_time = time.time()

    original_size = calculate_model_size(model)
    logger.info(f"Original model size: {original_size:.2f}MB")

    try:
        # Set model to evaluation mode
        model.eval()

        # Apply dynamic quantization to linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )

        # Disable gradients
        for param in quantized_model.parameters():
            param.requires_grad = False

        quantized_size = calculate_model_size(quantized_model)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        quantization_time = time.time() - start_time

        logger.info(f"Quantization completed in {quantization_time:.2f}s")
        logger.info(f"Quantized model size: {quantized_size:.2f}MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")

        quantization_info = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio,
            "quantization_time_seconds": quantization_time,
            "dtype": "qint8"
        }

        return quantized_model, quantization_info

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        logger.warning("Falling back to unquantized model")
        return model, {"quantized": False, "error": str(e)}


def save_model(model: nn.Module, processor, quantization_info: dict):
    """Save optimized model and processor."""
    logger.info(f"Saving model to {MODEL_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Save model state dict
        model_path = MODEL_DIR / "colpali_optimized.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': MODEL_NAME,
            'quantization_info': quantization_info,
            'torch_version': torch.__version__
        }, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save processor
        processor_path = MODEL_DIR / "processor"
        processor.save_pretrained(str(processor_path))
        logger.info(f"Processor saved to {processor_path}")

        # Save metadata
        metadata = {
            "model_name": MODEL_NAME,
            "quantization_info": quantization_info,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "files": {
                "model": str(model_path),
                "processor": str(processor_path)
            }
        }

        metadata_path = MODEL_DIR / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")

        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file()
        ) / (1024 * 1024)
        logger.info(f"Total saved size: {total_size:.2f}MB")

        return {
            "model_path": str(model_path),
            "processor_path": str(processor_path),
            "metadata_path": str(metadata_path),
            "total_size_mb": total_size
        }

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def create_placeholder_files():
    """Create placeholder files if download fails (allows build to continue)."""
    logger.warning("Creating placeholder files for build to continue")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Create placeholder model file
    model_path = MODEL_DIR / "colpali_optimized.pth"
    torch.save({
        'placeholder': True,
        'error': 'Model download failed during build'
    }, model_path)

    # Create placeholder processor directory
    processor_path = MODEL_DIR / "processor"
    processor_path.mkdir(exist_ok=True)

    # Create placeholder config
    config_path = processor_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump({"placeholder": True}, f)

    # Create error metadata
    metadata = {
        "placeholder": True,
        "error": "Model download failed during build",
        "note": "Model will be downloaded at runtime"
    }

    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.warning("Placeholder files created - model will be downloaded at runtime")


def main():
    """Main function to prepare model for Lambda deployment."""
    logger.info("=" * 60)
    logger.info("ColPali Model Preparation for AWS Lambda")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Output directory: {MODEL_DIR}")
    logger.info(f"Quantization enabled: {QUANTIZE_MODEL}")

    start_time = time.time()
    initial_memory = get_memory_usage_mb()
    logger.info(f"Initial memory usage: {initial_memory:.2f}MB")

    try:
        # Step 1: Download model
        model, processor = download_model()

        # Step 2: Quantize model (optional)
        if QUANTIZE_MODEL:
            model, quantization_info = quantize_model(model)
        else:
            quantization_info = {"quantized": False}
            logger.info("Skipping quantization as per configuration")

        # Step 3: Save model
        save_info = save_model(model, processor, quantization_info)

        # Cleanup
        del model
        del processor
        gc.collect()

        # Final summary
        total_time = time.time() - start_time
        final_memory = get_memory_usage_mb()

        logger.info("=" * 60)
        logger.info("Model Preparation Complete")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Final memory usage: {final_memory:.2f}MB")
        logger.info(f"Model saved to: {save_info['model_path']}")
        logger.info(f"Total size: {save_info['total_size_mb']:.2f}MB")

        if quantization_info.get('compression_ratio'):
            logger.info(f"Compression ratio: {quantization_info['compression_ratio']:.2f}x")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Model preparation failed: {e}")
        logger.error("Creating placeholder files...")
        create_placeholder_files()
        logger.warning("Build will continue but model will be downloaded at runtime")


if __name__ == "__main__":
    main()

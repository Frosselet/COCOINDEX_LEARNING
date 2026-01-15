"""
Model optimization utilities for AWS Lambda deployment.

Implements COLPALI-901: Optimize Lambda container for 3B model deployment.
Provides INT8 quantization, model pruning, and memory optimization strategies.
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""

    # Quantization settings
    enable_int8_quantization: bool = True
    quantization_dtype: torch.dtype = torch.qint8
    quantize_linear_only: bool = True

    # Memory optimization
    enable_memory_mapping: bool = True
    enable_gradient_checkpointing: bool = False
    max_memory_gb: float = 8.0  # Target memory limit

    # Model pruning
    enable_pruning: bool = False
    pruning_ratio: float = 0.1  # 10% pruning

    # CPU optimization
    enable_cpu_optimization: bool = True
    num_threads: int = 4

    # Caching
    model_cache_dir: str = "/tmp/model_cache"
    enable_model_caching: bool = True


@dataclass
class OptimizationMetrics:
    """Metrics from model optimization."""

    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    quantization_time_seconds: float = 0.0
    load_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_threads: int = 0
    optimizations_applied: list = field(default_factory=list)


class LambdaModelOptimizer:
    """
    Optimizes models for AWS Lambda deployment.

    Implements COLPALI-901 requirements:
    - Model quantization (INT8) reducing memory footprint by 50%
    - Container image under 10GB total size
    - Model loading time under 5 seconds
    - CPU inference optimization for Lambda
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the model optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.metrics = OptimizationMetrics()

        # Set CPU optimization threads
        if self.config.enable_cpu_optimization:
            self._configure_cpu_threads()

        logger.info(f"LambdaModelOptimizer initialized with config: {self.config}")

    def _configure_cpu_threads(self) -> None:
        """Configure optimal CPU thread settings for Lambda."""
        num_threads = self.config.num_threads

        # Set PyTorch thread settings (may fail if already set)
        try:
            torch.set_num_threads(num_threads)
        except RuntimeError:
            pass  # Already set, ignore

        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass  # Already set or parallel work started, ignore

        # Set environment variables
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        self.metrics.cpu_threads = num_threads
        logger.info(f"CPU threads configured: {num_threads}")

    def quantize_model(
        self,
        model: nn.Module,
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply INT8 dynamic quantization to model.

        Args:
            model: PyTorch model to quantize
            dtype: Quantization dtype (default: qint8)

        Returns:
            Tuple of (quantized model, quantization info)
        """
        if not self.config.enable_int8_quantization:
            logger.info("INT8 quantization disabled, skipping")
            return model, {"quantized": False}

        start_time = time.time()
        dtype = dtype or self.config.quantization_dtype

        logger.info("Starting INT8 dynamic quantization...")

        # Calculate original model size
        original_size = self._calculate_model_size(model)
        self.metrics.original_size_mb = original_size

        try:
            # Determine layers to quantize
            if self.config.quantize_linear_only:
                layers_to_quantize = {nn.Linear}
            else:
                layers_to_quantize = {nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU}

            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                layers_to_quantize,
                dtype=dtype
            )

            # Calculate quantized model size
            quantized_size = self._calculate_model_size(quantized_model)
            self.metrics.optimized_size_mb = quantized_size
            self.metrics.compression_ratio = original_size / quantized_size if quantized_size > 0 else 0

            quantization_time = time.time() - start_time
            self.metrics.quantization_time_seconds = quantization_time
            self.metrics.optimizations_applied.append("int8_quantization")

            logger.info(f"Quantization completed in {quantization_time:.2f}s")
            logger.info(f"Size reduction: {original_size:.2f}MB -> {quantized_size:.2f}MB "
                       f"({self.metrics.compression_ratio:.2f}x compression)")

            return quantized_model, {
                "quantized": True,
                "original_size_mb": original_size,
                "quantized_size_mb": quantized_size,
                "compression_ratio": self.metrics.compression_ratio,
                "time_seconds": quantization_time
            }

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model, {"quantized": False, "error": str(e)}

    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Apply inference optimizations to model.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        logger.info("Applying inference optimizations...")

        # Set to evaluation mode
        model.eval()
        self.metrics.optimizations_applied.append("eval_mode")

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        self.metrics.optimizations_applied.append("gradients_disabled")

        # Enable memory-efficient attention if available
        if hasattr(model, 'enable_memory_efficient_attention'):
            try:
                model.enable_memory_efficient_attention()
                self.metrics.optimizations_applied.append("memory_efficient_attention")
                logger.info("Memory-efficient attention enabled")
            except Exception as e:
                logger.warning(f"Could not enable memory-efficient attention: {e}")

        # Try to compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.enable_cpu_optimization:
            try:
                model = torch.compile(model, mode='reduce-overhead', backend='inductor')
                self.metrics.optimizations_applied.append("torch_compile")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")

        logger.info(f"Inference optimizations applied: {self.metrics.optimizations_applied}")
        return model

    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: Optional[float] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply magnitude-based pruning to model.

        Args:
            model: PyTorch model to prune
            pruning_ratio: Ratio of weights to prune (0-1)

        Returns:
            Tuple of (pruned model, pruning info)
        """
        if not self.config.enable_pruning:
            logger.info("Pruning disabled, skipping")
            return model, {"pruned": False}

        pruning_ratio = pruning_ratio or self.config.pruning_ratio
        logger.info(f"Starting magnitude-based pruning with ratio {pruning_ratio}...")

        try:
            import torch.nn.utils.prune as prune

            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, 'weight'))

            if not parameters_to_prune:
                logger.warning("No linear layers found for pruning")
                return model, {"pruned": False, "reason": "no_layers"}

            # Apply global unstructured pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio
            )

            # Make pruning permanent
            for module, name in parameters_to_prune:
                prune.remove(module, name)

            self.metrics.optimizations_applied.append(f"pruning_{pruning_ratio}")

            logger.info(f"Pruning completed: {len(parameters_to_prune)} layers pruned")

            return model, {
                "pruned": True,
                "pruning_ratio": pruning_ratio,
                "layers_pruned": len(parameters_to_prune)
            }

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model, {"pruned": False, "error": str(e)}

    def save_optimized_model(
        self,
        model: nn.Module,
        processor: Any,
        save_path: str
    ) -> Dict[str, Any]:
        """
        Save optimized model for Lambda deployment.

        Args:
            model: Optimized model
            processor: Model processor
            save_path: Directory to save model

        Returns:
            Dictionary with save information
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving optimized model to {save_path}...")
        start_time = time.time()

        try:
            # Save model state dict (quantized weights)
            model_path = save_dir / "model_optimized.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimization_config': self.config.__dict__,
                'metrics': {
                    'original_size_mb': self.metrics.original_size_mb,
                    'optimized_size_mb': self.metrics.optimized_size_mb,
                    'compression_ratio': self.metrics.compression_ratio,
                    'optimizations_applied': self.metrics.optimizations_applied
                }
            }, model_path)

            # Save processor
            processor_path = save_dir / "processor"
            if hasattr(processor, 'save_pretrained'):
                processor.save_pretrained(str(processor_path))

            save_time = time.time() - start_time
            model_size = model_path.stat().st_size / (1024 * 1024)

            logger.info(f"Model saved in {save_time:.2f}s, size: {model_size:.2f}MB")

            return {
                "model_path": str(model_path),
                "processor_path": str(processor_path),
                "model_size_mb": model_size,
                "save_time_seconds": save_time
            }

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Model save failed: {e}")

    def load_optimized_model(
        self,
        model_class: type,
        model_path: str,
        processor_class: Optional[type] = None
    ) -> Tuple[nn.Module, Any]:
        """
        Load pre-optimized model for Lambda.

        Args:
            model_class: Model class to instantiate
            model_path: Path to saved model
            processor_class: Optional processor class

        Returns:
            Tuple of (model, processor)
        """
        start_time = time.time()
        model_dir = Path(model_path)

        logger.info(f"Loading optimized model from {model_path}...")

        try:
            # Load model
            model_file = model_dir / "model_optimized.pth"
            checkpoint = torch.load(model_file, map_location='cpu')

            # Instantiate and load model
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load processor
            processor = None
            processor_path = model_dir / "processor"
            if processor_class and processor_path.exists():
                processor = processor_class.from_pretrained(str(processor_path))

            load_time = time.time() - start_time
            self.metrics.load_time_seconds = load_time

            logger.info(f"Model loaded in {load_time:.2f}s")

            return model, processor

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model load failed: {e}")

    def estimate_memory_requirements(self, model: nn.Module) -> Dict[str, float]:
        """
        Estimate memory requirements for model inference.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with memory estimates
        """
        model_size = self._calculate_model_size(model)

        # Estimate inference memory (model + activations + buffers)
        # Rule of thumb: inference needs ~2-3x model size
        inference_multiplier = 2.5
        estimated_inference = model_size * inference_multiplier

        # Current system memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        total_gb = memory.total / (1024 ** 3)

        estimates = {
            "model_size_mb": model_size,
            "estimated_inference_mb": estimated_inference,
            "estimated_inference_gb": estimated_inference / 1024,
            "available_memory_gb": available_gb,
            "total_memory_gb": total_gb,
            "fits_in_memory": estimated_inference / 1024 < available_gb,
            "recommended_batch_size": self._recommend_batch_size(
                model_size, available_gb * 1024
            )
        }

        logger.info(f"Memory estimates: {estimates}")
        return estimates

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() for b in model.buffers()
        )
        return (param_size + buffer_size) / (1024 * 1024)

    def _recommend_batch_size(
        self,
        model_size_mb: float,
        available_memory_mb: float
    ) -> int:
        """Recommend batch size based on memory constraints."""
        # Estimate memory per sample (conservative: 100MB for 1024x1024 image)
        memory_per_sample = 100
        safety_factor = 0.7

        usable_memory = (available_memory_mb - model_size_mb * 2.5) * safety_factor
        recommended = max(1, int(usable_memory / memory_per_sample))

        return min(recommended, 16)  # Cap at 16 for stability

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics summary."""
        return {
            "original_size_mb": self.metrics.original_size_mb,
            "optimized_size_mb": self.metrics.optimized_size_mb,
            "compression_ratio": self.metrics.compression_ratio,
            "quantization_time_seconds": self.metrics.quantization_time_seconds,
            "load_time_seconds": self.metrics.load_time_seconds,
            "memory_usage_mb": self._get_current_memory(),
            "cpu_threads": self.metrics.cpu_threads,
            "optimizations_applied": self.metrics.optimizations_applied
        }

    def _get_current_memory(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def cleanup(self) -> None:
        """Clean up resources and force garbage collection."""
        logger.info("Running cleanup...")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.metrics.memory_usage_mb = self._get_current_memory()
        logger.info(f"Cleanup complete. Memory usage: {self.metrics.memory_usage_mb:.2f}MB")

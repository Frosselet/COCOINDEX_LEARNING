"""
ColPali/ColQwen2 model client for vision-based document processing.

This module provides the interface to ColPali and ColQwen2 vision models for generating
patch-level embeddings from document images with memory optimization for
AWS Lambda deployment.

Supported model families:
- ColQwen2: vidore/colqwen2-v0.1, vidore/colqwen2-v1.0, etc.
- ColPali: vidore/colpali-v1.2, vidore/colpali, etc.
"""

import asyncio
import gc
import logging
import os
import time
import psutil
from typing import List, Optional, Dict, Any, Tuple
import torch
from PIL import Image
import warnings

# ColPali/ColQwen2 imports - support both model families
try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from colpali_engine.utils.torch_utils import get_torch_device
    COLQWEN2_AVAILABLE = True
except ImportError:
    ColQwen2 = None
    ColQwen2Processor = None
    COLQWEN2_AVAILABLE = False

try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    ColPali = None
    ColPaliProcessor = None
    COLPALI_AVAILABLE = False

try:
    from colpali_engine.utils.torch_utils import get_torch_device
except ImportError:
    get_torch_device = None

if not COLQWEN2_AVAILABLE and not COLPALI_AVAILABLE:
    logger = logging.getLogger(__name__)
    logger.warning("ColPali engine not available. Install with: pip install colpali-engine")

logger = logging.getLogger(__name__)


class ColPaliClient:
    """
    Client interface for ColPali/ColQwen2 vision models.

    Handles model loading, memory optimization, and batch processing
    for generating semantic embeddings from document images.
    Automatically detects the appropriate model class (ColPali or ColQwen2)
    based on the model name.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v0.1",
        device: str = "auto",
        memory_limit_gb: Optional[int] = None,
        enable_prewarming: bool = True,
        lazy_loading: bool = True
    ):
        """
        Initialize ColPali client.

        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ('cpu', 'cuda', 'auto')
            memory_limit_gb: Optional memory limit for Lambda deployment
            enable_prewarming: Enable model prewarming for cold start optimization
            lazy_loading: Enable lazy loading of model components
        """
        self.model_name = model_name
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self.enable_prewarming = enable_prewarming
        self.lazy_loading = lazy_loading

        # Model components (to be loaded lazily)
        self.model = None
        self.processor = None
        self.is_loaded = False

        # Cold start optimization tracking (COLPALI-304)
        self._cold_start_metrics = {
            "load_time": None,
            "warmup_time": None,
            "first_inference_time": None,
            "is_prewarmed": False
        }

        logger.info(f"ColPali client initialized for model: {model_name}")
        logger.info(f"Lambda optimizations: prewarming={enable_prewarming}, lazy_loading={lazy_loading}")

    async def load_model(self) -> None:
        """
        Load ColPali model with memory optimization.

        Implements COLPALI-301: Model loading with quantization and memory management.
        """
        if self.is_loaded:
            logger.info("ColPali model already loaded")
            return

        start_time = time.time()
        initial_memory = self._get_memory_usage()

        logger.info(f"Loading ColPali model: {self.model_name}")
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        try:
            # Determine device
            device = self._determine_device()
            logger.info(f"Using device: {device}")

            # Check available memory
            available_memory = self._get_available_memory_gb()
            logger.info(f"Available memory: {available_memory:.2f} GB")

            if self.memory_limit_gb and available_memory < self.memory_limit_gb:
                logger.warning(f"Available memory ({available_memory:.2f} GB) below limit ({self.memory_limit_gb} GB)")

            # Load model with optimizations
            await self._load_model_components(device)

            # Run model warmup
            await self._warmup_model()

            # Final memory check
            final_memory = self._get_memory_usage()
            memory_increase = final_memory - initial_memory
            load_time = time.time() - start_time

            # Track cold start metrics (COLPALI-304)
            self._cold_start_metrics["load_time"] = load_time
            self._cold_start_metrics["is_prewarmed"] = True

            logger.info(f"ColPali model loaded successfully in {load_time:.2f}s")
            logger.info(f"Memory increase: {memory_increase:.2f} MB (total: {final_memory:.2f} MB)")

            self.is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load ColPali model: {e}")
            # Cleanup on failure
            await self._cleanup_model()
            raise RuntimeError(f"ColPali model loading failed: {e}")

    async def embed_frames(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[torch.Tensor]:
        """
        Generate patch-level embeddings for image frames.

        Implements COLPALI-302: Batch processing with dynamic sizing.
        Implements COLPALI-303: Patch-level embedding generation.

        Args:
            images: List of PIL Image objects
            batch_size: Optional batch size override
            progress_callback: Optional callback for progress tracking

        Returns:
            List of embedding tensors for each image (one tensor per image)

        Raises:
            RuntimeError: If model is not loaded or processing fails
        """
        if not self.is_loaded:
            raise RuntimeError("ColPali model not loaded. Call load_model() first.")

        if not images:
            return []

        logger.info(f"Generating embeddings for {len(images)} images")
        start_time = time.time()

        try:
            # Determine optimal batch size
            if batch_size is None:
                available_memory = self._get_available_memory_gb()
                batch_size = self.calculate_optimal_batch_size(int(available_memory))
                logger.info(f"Using calculated batch size: {batch_size}")

            # Process images in batches
            all_embeddings = []
            total_batches = (len(images) + batch_size - 1) // batch_size

            for batch_idx in range(0, len(images), batch_size):
                batch_images = images[batch_idx:batch_idx + batch_size]
                current_batch = (batch_idx // batch_size) + 1

                logger.debug(f"Processing batch {current_batch}/{total_batches} ({len(batch_images)} images)")

                # Process batch
                batch_embeddings = await self._process_batch(batch_images)
                all_embeddings.extend(batch_embeddings)

                # Update progress
                if progress_callback:
                    progress = current_batch / total_batches
                    await asyncio.create_task(asyncio.to_thread(progress_callback, progress))

                # Memory cleanup between batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            processing_time = time.time() - start_time
            logger.info(f"Embedding generation completed in {processing_time:.2f}s")
            logger.info(f"Generated {len(all_embeddings)} embeddings ({processing_time/len(images):.3f}s per image)")

            return all_embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def calculate_optimal_batch_size(self, available_memory_gb: int) -> int:
        """
        Calculate optimal batch size based on available memory.

        Implements COLPALI-302: Dynamic batch sizing.

        Args:
            available_memory_gb: Available memory in GB

        Returns:
            Optimal batch size for processing
        """
        # Conservative estimates for ColQwen2-v0.1 model
        # Each image (1024x1024) processes roughly:
        # - Input tensors: ~12MB per image
        # - Model forward pass: ~50MB additional memory per image
        # - Patch embeddings output: ~8MB per image
        # Total: ~70MB per image in batch

        memory_per_image_mb = 70
        safety_margin = 0.7  # Use 70% of available memory for safety

        # Calculate based on available memory
        usable_memory_mb = available_memory_gb * 1024 * safety_margin
        theoretical_batch_size = int(usable_memory_mb / memory_per_image_mb)

        # Apply constraints
        min_batch_size = 1
        max_batch_size = 16  # Practical limit for stability

        # Apply memory-based constraints
        if self.memory_limit_gb:
            if self.memory_limit_gb < 2:  # Lambda constraint
                max_batch_size = 1
            elif self.memory_limit_gb < 4:
                max_batch_size = 2
            elif self.memory_limit_gb < 8:
                max_batch_size = 4

        optimal_batch_size = max(min_batch_size, min(theoretical_batch_size, max_batch_size))

        logger.info(f"Calculated optimal batch size: {optimal_batch_size} "
                   f"(available: {available_memory_gb:.2f} GB, per-image: {memory_per_image_mb} MB)")

        return optimal_batch_size

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "memory_limit_gb": self.memory_limit_gb,
            "patch_size": (32, 32),
            "embedding_dimension": 128
        }

        if self.is_loaded and self.model:
            info.update({
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024,
                "current_memory_mb": self._get_memory_usage()
            })

        return info

    # Private helper methods for COLPALI-301

    def _get_model_classes(self) -> Tuple[Any, Any]:
        """
        Get the appropriate model and processor classes based on model name.

        Returns:
            Tuple of (ModelClass, ProcessorClass)

        Raises:
            ImportError: If required model classes are not available
        """
        model_name_lower = self.model_name.lower()

        # ColQwen2 models (e.g., vidore/colqwen2-v0.1)
        if "colqwen2" in model_name_lower or "colqwen" in model_name_lower:
            if not COLQWEN2_AVAILABLE:
                raise ImportError(
                    f"Model {self.model_name} requires ColQwen2 classes. "
                    "Install with: pip install colpali-engine"
                )
            logger.info(f"Using ColQwen2 model classes for {self.model_name}")
            return ColQwen2, ColQwen2Processor

        # ColPali models (e.g., vidore/colpali-v1.2)
        if "colpali" in model_name_lower:
            if not COLPALI_AVAILABLE:
                raise ImportError(
                    f"Model {self.model_name} requires ColPali classes. "
                    "Install with: pip install colpali-engine"
                )
            logger.info(f"Using ColPali model classes for {self.model_name}")
            return ColPali, ColPaliProcessor

        # Default: try ColQwen2 first (newer), then ColPali
        if COLQWEN2_AVAILABLE:
            logger.info(f"Defaulting to ColQwen2 model classes for {self.model_name}")
            return ColQwen2, ColQwen2Processor
        elif COLPALI_AVAILABLE:
            logger.info(f"Defaulting to ColPali model classes for {self.model_name}")
            return ColPali, ColPaliProcessor
        else:
            raise ImportError(
                "ColPali engine not available. Install with: pip install colpali-engine"
            )

    def _determine_device(self) -> str:
        """
        Determine the best available device for inference.

        Returns:
            Device string ('cuda', 'cpu')
        """
        if self.device != "auto":
            return self.device

        # Use ColPali's device detection if available
        if get_torch_device:
            device = get_torch_device()
            logger.info(f"ColPali device detection: {device}")
            return str(device)

        # Fallback device detection
        if torch.cuda.is_available():
            # Check CUDA memory for Lambda constraints
            if self.memory_limit_gb:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if gpu_memory_gb < self.memory_limit_gb:
                    logger.info(f"GPU memory ({gpu_memory_gb:.2f} GB) below limit, using CPU")
                    return "cpu"
            return "cuda"
        else:
            return "cpu"

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Current memory usage in megabytes
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Unable to get memory usage: {e}")
            return 0.0

    def _get_available_memory_gb(self) -> float:
        """
        Get available system memory in GB.

        Returns:
            Available memory in gigabytes
        """
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Unable to get available memory: {e}")
            return 8.0  # Conservative fallback

    async def _load_model_components(self, device: str) -> None:
        """
        Load ColPali/ColQwen2 model and processor components with optimization.

        Args:
            device: Target device for inference
        """
        # Get the appropriate model classes based on model name
        ModelClass, ProcessorClass = self._get_model_classes()
        model_type = ModelClass.__name__

        logger.info(f"Loading {model_type} model components...")

        # Load in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Load processor (lightweight, load first)
        logger.info(f"Loading {model_type} processor...")
        self.processor = await loop.run_in_executor(
            None, ProcessorClass.from_pretrained, self.model_name
        )

        # Load model with memory optimizations
        logger.info(f"Loading {model_type} model...")
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
        }

        # Apply quantization for memory optimization
        if self.memory_limit_gb and self.memory_limit_gb < 6:
            logger.info("Applying INT8 quantization for memory optimization")
            model_kwargs["load_in_8bit"] = True

        self.model = await loop.run_in_executor(
            None, lambda: ModelClass.from_pretrained(self.model_name, **model_kwargs)
        )

        # Move to target device if needed
        if device == "cpu" and not model_kwargs.get("device_map"):
            self.model = self.model.to(device)

        # Set to evaluation mode
        self.model.eval()

        # Enable memory-efficient attention if available
        if hasattr(self.model, "enable_memory_efficient_attention"):
            self.model.enable_memory_efficient_attention()

        logger.info("ColPali model components loaded successfully")

    async def _warmup_model(self) -> None:
        """
        Warm up the model with a dummy forward pass.

        This reduces cold start latency for subsequent inference calls.
        """
        logger.info("Warming up ColPali model...")

        try:
            # Create dummy input image
            dummy_image = Image.new('RGB', (224, 224), color='white')

            # Run dummy inference
            with torch.no_grad():
                # Use appropriate processor method based on processor type
                if hasattr(self.processor, 'process_images'):
                    # ColQwen2 processor
                    inputs = self.processor.process_images([dummy_image])
                else:
                    # Standard ColPali processor
                    inputs = self.processor([dummy_image], return_tensors="pt")

                # Move to device
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                # Forward pass
                _ = self.model(**inputs)

            # Cleanup
            del inputs
            del dummy_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model warmup completed")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    async def _cleanup_model(self) -> None:
        """
        Clean up model resources.
        """
        logger.info("Cleaning up ColPali model resources...")

        if self.model:
            del self.model
            self.model = None

        if self.processor:
            del self.processor
            self.processor = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info("Model cleanup completed")

    async def _process_batch(self, batch_images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Process a batch of images to generate embeddings.

        Implements COLPALI-303: Patch-level embedding generation.

        Args:
            batch_images: List of PIL Images to process

        Returns:
            List of embedding tensors (one per image)
        """
        if not batch_images:
            return []

        try:
            # Preprocess images for ColPali/ColQwen2
            logger.debug(f"Processing batch of {len(batch_images)} images")

            # ColQwen2Processor requires both images and text
            # Use process_images method if available (ColQwen2), otherwise use __call__ (ColPali)
            if hasattr(self.processor, 'process_images'):
                # ColQwen2 processor - use process_images method for image-only processing
                inputs = self.processor.process_images(batch_images)
            else:
                # Standard ColPali processor
                inputs = self.processor(batch_images, return_tensors="pt")

            # Move inputs to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate embeddings with no gradient computation
            with torch.no_grad():
                # Forward pass through ColPali model
                outputs = self.model(**inputs)

                # Extract patch embeddings
                # ColPali generates patch-level embeddings automatically
                if hasattr(outputs, 'multi_vector'):
                    # Multi-vector output (patch embeddings)
                    embeddings = outputs.multi_vector
                elif hasattr(outputs, 'last_hidden_state'):
                    # Fallback: use last hidden state
                    embeddings = outputs.last_hidden_state
                else:
                    # Fallback: use raw outputs
                    embeddings = outputs

                # Process embeddings to separate by image
                batch_embeddings = self._extract_patch_embeddings(embeddings, len(batch_images))

            # Clean up intermediate tensors
            del inputs
            if 'outputs' in locals():
                del outputs
            gc.collect()

            logger.debug(f"Generated embeddings for batch: {[emb.shape for emb in batch_embeddings]}")
            return batch_embeddings

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Ensure cleanup on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Failed to process batch: {e}")

    def _extract_patch_embeddings(self, embeddings: torch.Tensor, num_images: int) -> List[torch.Tensor]:
        """
        Extract and organize patch embeddings by image.

        Implements COLPALI-303: Spatial coordinate preservation and patch extraction.

        Args:
            embeddings: Raw embedding tensor from model
            num_images: Number of images in the batch

        Returns:
            List of embedding tensors, one per image
        """
        try:
            # ColPali generates embeddings in shape [batch_size, num_patches, hidden_size]
            # where num_patches corresponds to 32x32 image regions
            batch_size, num_patches, hidden_size = embeddings.shape

            if batch_size != num_images:
                logger.warning(f"Batch size mismatch: expected {num_images}, got {batch_size}")

            # Split embeddings by image
            image_embeddings = []
            for i in range(min(batch_size, num_images)):
                # Extract embeddings for single image
                image_emb = embeddings[i]  # Shape: [num_patches, hidden_size]

                # Add spatial metadata as a property (for downstream use)
                # ColPali model handles spatial relationships internally
                # The patch organization follows row-major order (left-to-right, top-to-bottom)
                image_emb = image_emb.clone().detach()

                # Store spatial metadata as tensor attributes if possible
                if hasattr(image_emb, 'spatial_info'):
                    # Calculate patch grid dimensions (assuming square patches)
                    patches_per_side = int(num_patches ** 0.5)
                    image_emb.spatial_info = {
                        'patch_size': (32, 32),
                        'grid_size': (patches_per_side, patches_per_side),
                        'num_patches': num_patches,
                        'embedding_dim': hidden_size
                    }

                image_embeddings.append(image_emb)

            logger.debug(f"Extracted {len(image_embeddings)} image embeddings, "
                        f"each with {num_patches} patches of {hidden_size}D")

            return image_embeddings

        except Exception as e:
            logger.error(f"Failed to extract patch embeddings: {e}")
            raise RuntimeError(f"Patch embedding extraction failed: {e}")

    # COLPALI-304: Lambda Cold Start Optimization Methods

    async def prewarm_for_lambda(self, cache_path: Optional[str] = None) -> Dict[str, float]:
        """
        Prewarm model for Lambda deployment with cold start optimization.

        Implements COLPALI-304: Lambda cold start optimization.

        Args:
            cache_path: Optional path for model caching (e.g., EFS mount)

        Returns:
            Dictionary with prewarming metrics
        """
        logger.info("Prewarming ColPali model for Lambda deployment...")
        prewarm_start = time.time()

        try:
            # Set environment variables for optimized loading
            os.environ["TRANSFORMERS_CACHE"] = cache_path or "/tmp/transformers_cache"
            os.environ["HF_HOME"] = cache_path or "/tmp/hf_cache"

            # Load model with Lambda-specific optimizations
            if not self.is_loaded:
                # Use smaller memory limit for Lambda
                original_limit = self.memory_limit_gb
                self.memory_limit_gb = min(self.memory_limit_gb or 10, 3)  # 3GB Lambda limit

                await self.load_model()

                self.memory_limit_gb = original_limit  # Restore original

            # Extended warmup for Lambda
            await self._extended_warmup_for_lambda()

            prewarm_time = time.time() - prewarm_start
            self._cold_start_metrics["warmup_time"] = prewarm_time

            logger.info(f"Lambda prewarming completed in {prewarm_time:.2f}s")

            return {
                "prewarm_time": prewarm_time,
                "load_time": self._cold_start_metrics["load_time"],
                "total_cold_start": prewarm_time,
                "memory_usage_mb": self._get_memory_usage()
            }

        except Exception as e:
            logger.error(f"Lambda prewarming failed: {e}")
            raise RuntimeError(f"Failed to prewarm for Lambda: {e}")

    async def _extended_warmup_for_lambda(self) -> None:
        """
        Extended warmup specifically for Lambda cold start optimization.
        """
        logger.info("Running extended Lambda warmup...")

        try:
            # Multiple warmup passes with different input sizes
            warmup_sizes = [(224, 224), (512, 512), (1024, 1024)]

            for size in warmup_sizes:
                logger.debug(f"Warmup pass for size {size}")

                # Create dummy image of specific size
                dummy_image = Image.new('RGB', size, color='white')

                with torch.no_grad():
                    # Use appropriate processor method based on processor type
                    if hasattr(self.processor, 'process_images'):
                        inputs = self.processor.process_images([dummy_image])
                    else:
                        inputs = self.processor([dummy_image], return_tensors="pt")

                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                    # Forward pass
                    _ = self.model(**inputs)

                # Cleanup between passes
                del inputs, dummy_image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Compile model for faster subsequent calls (if using PyTorch 2.0+)
            if hasattr(torch, 'compile') and hasattr(self.model, 'forward'):
                logger.info("Compiling model for optimized inference...")
                try:
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")

            logger.info("Extended Lambda warmup completed")

        except Exception as e:
            logger.warning(f"Extended warmup failed: {e}")

    def get_cold_start_metrics(self) -> Dict[str, Any]:
        """
        Get cold start performance metrics.

        Returns:
            Dictionary with cold start metrics
        """
        return {
            **self._cold_start_metrics,
            "current_memory_mb": self._get_memory_usage(),
            "model_info": self.get_model_info(),
            "optimization_settings": {
                "enable_prewarming": self.enable_prewarming,
                "lazy_loading": self.lazy_loading,
                "memory_limit_gb": self.memory_limit_gb
            }
        }

    async def benchmark_inference_speed(self, num_images: int = 5) -> Dict[str, float]:
        """
        Benchmark inference speed for performance monitoring.

        Args:
            num_images: Number of test images to benchmark

        Returns:
            Dictionary with benchmark results
        """
        if not self.is_loaded:
            await self.load_model()

        logger.info(f"Benchmarking inference speed with {num_images} images...")

        try:
            # Create test images
            test_images = [
                Image.new('RGB', (1024, 1024), color=f'#{i*40:02x}{i*60:02x}{i*80:02x}')
                for i in range(num_images)
            ]

            # Benchmark cold inference (first call)
            cold_start = time.time()
            first_embeddings = await self.embed_frames([test_images[0]])
            cold_time = time.time() - cold_start

            # Track first inference time
            if self._cold_start_metrics["first_inference_time"] is None:
                self._cold_start_metrics["first_inference_time"] = cold_time

            # Benchmark warm inference (subsequent calls)
            warm_times = []
            for i in range(1, num_images):
                warm_start = time.time()
                _ = await self.embed_frames([test_images[i]])
                warm_times.append(time.time() - warm_start)

            # Benchmark batch processing
            batch_start = time.time()
            batch_embeddings = await self.embed_frames(test_images)
            batch_time = time.time() - batch_start

            avg_warm_time = sum(warm_times) / len(warm_times) if warm_times else 0

            results = {
                "cold_inference_time": cold_time,
                "avg_warm_inference_time": avg_warm_time,
                "batch_processing_time": batch_time,
                "throughput_images_per_sec": len(test_images) / batch_time,
                "speedup_batch_vs_single": (cold_time + sum(warm_times)) / batch_time if batch_time > 0 else 0,
                "memory_usage_mb": self._get_memory_usage()
            }

            logger.info(f"Benchmark results: {results}")
            return results

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            raise RuntimeError(f"Failed to benchmark inference: {e}")
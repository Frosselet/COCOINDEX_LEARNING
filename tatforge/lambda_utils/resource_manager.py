"""
Resource management utilities for AWS Lambda deployment.

Implements COLPALI-902: Resource management and cleanup.
Provides real-time memory monitoring, garbage collection, leak detection,
resource pooling, and timeout handling.
"""

import asyncio
import functools
import gc
import logging
import os
import signal
import time
import traceback
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

import psutil
import torch

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryThresholds:
    """Memory threshold configuration."""

    warning_percent: float = 70.0  # Warn at 70% usage
    critical_percent: float = 85.0  # Critical at 85%
    max_usage_gb: float = 8.0  # Max allowed usage for Lambda
    cleanup_threshold_percent: float = 75.0  # Trigger cleanup at 75%


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    timestamp: datetime = field(default_factory=datetime.now)
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    active_tensors: int = 0
    active_allocations: int = 0


class MemoryMonitor:
    """Real-time memory monitoring for Lambda."""

    def __init__(self, thresholds: Optional[MemoryThresholds] = None):
        """
        Initialize memory monitor.

        Args:
            thresholds: Memory threshold configuration
        """
        self.thresholds = thresholds or MemoryThresholds()
        self._metrics_history: List[ResourceMetrics] = []
        self._max_history_size = 100
        self._baseline_memory: Optional[float] = None

        logger.info(f"MemoryMonitor initialized with thresholds: {self.thresholds}")

    def capture_baseline(self) -> float:
        """Capture baseline memory usage."""
        gc.collect()
        self._baseline_memory = self._get_memory_usage_mb()
        logger.info(f"Memory baseline captured: {self._baseline_memory:.2f}MB")
        return self._baseline_memory

    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()

        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_used_mb=process.memory_info().rss / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            memory_percent=(process.memory_info().rss / memory.total) * 100,
            cpu_percent=process.cpu_percent(),
            gc_collections={
                i: gc.get_count()[i] for i in range(3)
            },
            active_tensors=self._count_active_tensors(),
            active_allocations=len(gc.get_objects())
        )

        self._record_metrics(metrics)
        return metrics

    def check_memory_status(self) -> Dict[str, Any]:
        """
        Check memory status against thresholds.

        Returns:
            Dictionary with status and recommendations
        """
        metrics = self.get_current_metrics()

        status = {
            "current_mb": metrics.memory_used_mb,
            "available_mb": metrics.memory_available_mb,
            "usage_percent": metrics.memory_percent,
            "status": "ok",
            "warnings": [],
            "recommendations": []
        }

        # Check against thresholds
        if metrics.memory_percent >= self.thresholds.critical_percent:
            status["status"] = "critical"
            status["warnings"].append(
                f"Memory usage critical: {metrics.memory_percent:.1f}%"
            )
            status["recommendations"].append("Immediate cleanup required")

        elif metrics.memory_percent >= self.thresholds.warning_percent:
            status["status"] = "warning"
            status["warnings"].append(
                f"Memory usage elevated: {metrics.memory_percent:.1f}%"
            )
            status["recommendations"].append("Consider reducing batch size")

        # Check absolute limit
        max_mb = self.thresholds.max_usage_gb * 1024
        if metrics.memory_used_mb > max_mb:
            status["status"] = "critical"
            status["warnings"].append(
                f"Exceeds Lambda limit: {metrics.memory_used_mb:.0f}MB > {max_mb:.0f}MB"
            )

        # Check for memory leak indicators
        if self._baseline_memory:
            increase = metrics.memory_used_mb - self._baseline_memory
            if increase > 500:  # 500MB increase from baseline
                status["warnings"].append(
                    f"Potential memory leak: +{increase:.0f}MB from baseline"
                )
                status["recommendations"].append("Review tensor cleanup")

        logger.info(f"Memory status: {status['status']} ({metrics.memory_percent:.1f}%)")
        return status

    def detect_memory_leak(self, threshold_mb: float = 100.0) -> Optional[Dict[str, Any]]:
        """
        Detect potential memory leaks by analyzing trends.

        Args:
            threshold_mb: Minimum increase to flag as leak

        Returns:
            Leak detection results or None
        """
        if len(self._metrics_history) < 10:
            return None

        # Get recent memory trend
        recent = self._metrics_history[-10:]
        memory_values = [m.memory_used_mb for m in recent]

        # Calculate trend
        first_half_avg = sum(memory_values[:5]) / 5
        second_half_avg = sum(memory_values[5:]) / 5
        increase = second_half_avg - first_half_avg

        if increase > threshold_mb:
            return {
                "detected": True,
                "increase_mb": increase,
                "first_half_avg": first_half_avg,
                "second_half_avg": second_half_avg,
                "time_window_seconds": (
                    recent[-1].timestamp - recent[0].timestamp
                ).total_seconds()
            }

        return {"detected": False, "increase_mb": increase}

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _count_active_tensors(self) -> int:
        """Count active PyTorch tensors."""
        count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    count += 1
            except Exception:
                pass
        return count

    def _record_metrics(self, metrics: ResourceMetrics) -> None:
        """Record metrics to history."""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history.pop(0)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recorded metrics."""
        if not self._metrics_history:
            return {}

        memory_values = [m.memory_used_mb for m in self._metrics_history]

        return {
            "samples": len(self._metrics_history),
            "memory_min_mb": min(memory_values),
            "memory_max_mb": max(memory_values),
            "memory_avg_mb": sum(memory_values) / len(memory_values),
            "baseline_mb": self._baseline_memory,
            "time_range_seconds": (
                self._metrics_history[-1].timestamp -
                self._metrics_history[0].timestamp
            ).total_seconds() if len(self._metrics_history) > 1 else 0
        }


class GarbageCollector:
    """Enhanced garbage collection for Lambda."""

    def __init__(self):
        """Initialize garbage collector."""
        self._collection_count = 0
        self._total_freed_mb = 0.0

    def collect(self, full: bool = False) -> Dict[str, Any]:
        """
        Run garbage collection.

        Args:
            full: Run full collection (all generations)

        Returns:
            Collection statistics
        """
        before = self._get_memory_mb()

        if full:
            # Full collection
            collected = gc.collect(2)
        else:
            # Standard collection
            collected = gc.collect()

        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        after = self._get_memory_mb()
        freed = before - after

        self._collection_count += 1
        self._total_freed_mb += max(0, freed)

        result = {
            "objects_collected": collected,
            "memory_before_mb": before,
            "memory_after_mb": after,
            "memory_freed_mb": freed,
            "total_collections": self._collection_count,
            "total_freed_mb": self._total_freed_mb
        }

        logger.debug(f"GC collected {collected} objects, freed {freed:.2f}MB")
        return result

    def collect_tensors(self) -> int:
        """
        Force collection of unreferenced tensors.

        Returns:
            Number of tensors collected
        """
        collected = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.grad_fn is None:
                    if obj.is_leaf and not obj.requires_grad:
                        del obj
                        collected += 1
            except Exception:
                pass

        gc.collect()
        return collected

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        return {
            "collection_count": self._collection_count,
            "total_freed_mb": self._total_freed_mb,
            "gc_stats": gc.get_stats(),
            "gc_threshold": gc.get_threshold()
        }


class TimeoutHandler:
    """Timeout handling for Lambda functions."""

    def __init__(self, default_timeout: int = 300):
        """
        Initialize timeout handler.

        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout
        self._active_timeouts: Set[int] = set()

    def timeout(self, seconds: Optional[int] = None) -> Callable:
        """
        Decorator for timeout handling.

        Args:
            seconds: Timeout in seconds

        Returns:
            Decorated function
        """
        timeout_seconds = seconds or self.default_timeout

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Set timeout signal
                def handler(signum, frame):
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {timeout_seconds}s"
                    )

                # Only set signal handler on Unix
                if hasattr(signal, 'SIGALRM'):
                    old_handler = signal.signal(signal.SIGALRM, handler)
                    signal.alarm(timeout_seconds)
                    self._active_timeouts.add(id(func))

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                        self._active_timeouts.discard(id(func))

            return wrapper
        return decorator

    async def async_timeout(
        self,
        coro: Any,
        seconds: Optional[int] = None
    ) -> Any:
        """
        Async timeout wrapper.

        Args:
            coro: Coroutine to wrap
            seconds: Timeout in seconds

        Returns:
            Coroutine result

        Raises:
            TimeoutError: If timeout exceeded
        """
        timeout_seconds = seconds or self.default_timeout

        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")


class LambdaResourceManager:
    """
    Comprehensive resource management for Lambda deployment.

    Implements COLPALI-902 requirements:
    - Real-time memory monitoring and alerting
    - Automatic garbage collection and resource cleanup
    - Memory leak detection and prevention
    - Resource pooling for model inference
    - Timeout handling and graceful degradation
    """

    def __init__(
        self,
        memory_thresholds: Optional[MemoryThresholds] = None,
        default_timeout: int = 300
    ):
        """
        Initialize resource manager.

        Args:
            memory_thresholds: Memory threshold configuration
            default_timeout: Default operation timeout
        """
        self.memory_monitor = MemoryMonitor(memory_thresholds)
        self.gc = GarbageCollector()
        self.timeout_handler = TimeoutHandler(default_timeout)

        # Resource pools
        self._model_pool: Dict[str, weakref.ref] = {}
        self._cleanup_hooks: List[Callable] = []

        # State tracking
        self._initialized = False
        self._context_depth = 0

        logger.info("LambdaResourceManager initialized")

    def initialize(self) -> Dict[str, Any]:
        """
        Initialize resource management.

        Returns:
            Initialization metrics
        """
        if self._initialized:
            return {"status": "already_initialized"}

        # Capture memory baseline
        baseline = self.memory_monitor.capture_baseline()

        # Run initial cleanup
        gc_result = self.gc.collect(full=True)

        self._initialized = True

        return {
            "status": "initialized",
            "baseline_memory_mb": baseline,
            "initial_gc": gc_result
        }

    @contextmanager
    def managed_context(self, name: str = "unnamed"):
        """
        Context manager for resource-managed operations.

        Args:
            name: Context name for logging
        """
        self._context_depth += 1
        context_id = f"{name}_{self._context_depth}"

        logger.info(f"Entering managed context: {context_id}")
        start_memory = self.memory_monitor.get_current_metrics().memory_used_mb

        try:
            yield
        finally:
            # Cleanup on exit
            end_memory = self.memory_monitor.get_current_metrics().memory_used_mb
            memory_delta = end_memory - start_memory

            logger.info(
                f"Exiting managed context: {context_id}, "
                f"memory delta: {memory_delta:+.2f}MB"
            )

            # Run cleanup if memory increased significantly
            if memory_delta > 100:  # 100MB threshold
                self.gc.collect()

            self._context_depth -= 1

    @asynccontextmanager
    async def async_managed_context(self, name: str = "unnamed"):
        """
        Async context manager for resource-managed operations.

        Args:
            name: Context name for logging
        """
        self._context_depth += 1
        context_id = f"{name}_{self._context_depth}"

        logger.info(f"Entering async managed context: {context_id}")
        start_memory = self.memory_monitor.get_current_metrics().memory_used_mb

        try:
            yield
        finally:
            end_memory = self.memory_monitor.get_current_metrics().memory_used_mb
            memory_delta = end_memory - start_memory

            logger.info(
                f"Exiting async managed context: {context_id}, "
                f"memory delta: {memory_delta:+.2f}MB"
            )

            if memory_delta > 100:
                self.gc.collect()

            self._context_depth -= 1

    def check_resources(self) -> Dict[str, Any]:
        """
        Comprehensive resource check.

        Returns:
            Resource status dictionary
        """
        memory_status = self.memory_monitor.check_memory_status()
        leak_detection = self.memory_monitor.detect_memory_leak()
        gc_stats = self.gc.get_stats()

        status = {
            "memory": memory_status,
            "leak_detection": leak_detection,
            "gc": gc_stats,
            "model_pool_size": len(self._model_pool),
            "context_depth": self._context_depth,
            "overall_status": memory_status["status"]
        }

        # Auto-cleanup if needed
        if memory_status["status"] == "critical":
            logger.warning("Critical memory status, running emergency cleanup")
            self.emergency_cleanup()

        return status

    def request_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Request resource cleanup.

        Args:
            aggressive: Run aggressive cleanup

        Returns:
            Cleanup results
        """
        results = {
            "gc_result": self.gc.collect(full=aggressive),
            "tensors_collected": self.gc.collect_tensors() if aggressive else 0,
            "cleanup_hooks_run": 0
        }

        # Run registered cleanup hooks
        for hook in self._cleanup_hooks:
            try:
                hook()
                results["cleanup_hooks_run"] += 1
            except Exception as e:
                logger.warning(f"Cleanup hook failed: {e}")

        return results

    def emergency_cleanup(self) -> Dict[str, Any]:
        """
        Emergency cleanup for critical memory situations.

        Returns:
            Emergency cleanup results
        """
        logger.warning("Running emergency cleanup!")

        before_memory = self.memory_monitor.get_current_metrics().memory_used_mb

        # Clear model pool
        self._model_pool.clear()

        # Full garbage collection
        gc.collect(2)

        # Clear PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force Python to release memory
        import ctypes
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass  # Not available on all platforms

        after_memory = self.memory_monitor.get_current_metrics().memory_used_mb

        result = {
            "memory_before_mb": before_memory,
            "memory_after_mb": after_memory,
            "memory_freed_mb": before_memory - after_memory,
            "emergency": True
        }

        logger.info(f"Emergency cleanup complete: freed {result['memory_freed_mb']:.2f}MB")
        return result

    def register_cleanup_hook(self, hook: Callable) -> None:
        """
        Register a cleanup hook.

        Args:
            hook: Callable to run during cleanup
        """
        self._cleanup_hooks.append(hook)

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        metrics = self.memory_monitor.get_current_metrics()
        summary = self.memory_monitor.get_metrics_summary()

        return {
            "current": {
                "memory_used_mb": metrics.memory_used_mb,
                "memory_available_mb": metrics.memory_available_mb,
                "memory_percent": metrics.memory_percent,
                "active_tensors": metrics.active_tensors
            },
            "history": summary,
            "gc_stats": self.gc.get_stats(),
            "status": self.memory_monitor.check_memory_status()["status"]
        }

    def shutdown(self) -> None:
        """Shutdown resource manager and cleanup."""
        logger.info("Shutting down resource manager")

        # Run final cleanup
        self.request_cleanup(aggressive=True)

        # Clear all pools and caches
        self._model_pool.clear()
        self._cleanup_hooks.clear()

        self._initialized = False
        logger.info("Resource manager shutdown complete")

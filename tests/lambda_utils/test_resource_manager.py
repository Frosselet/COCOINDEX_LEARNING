"""
Tests for Lambda resource management utilities.

Tests COLPALI-902: Resource management and cleanup.
"""

import gc
import sys
import time
from pathlib import Path
import importlib.util

import pytest
from unittest.mock import Mock, patch

# Load module directly to avoid colpali_engine package dependencies
def load_module_directly(module_name: str, file_path: Path):
    """Load a module directly from file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get paths
project_root = Path(__file__).parent.parent.parent
lambda_utils_path = project_root / "colpali_engine" / "lambda_utils"

# Load module
resource_manager = load_module_directly(
    "resource_manager_test",
    lambda_utils_path / "resource_manager.py"
)

# Get classes
MemoryThresholds = resource_manager.MemoryThresholds
ResourceMetrics = resource_manager.ResourceMetrics
MemoryMonitor = resource_manager.MemoryMonitor
GarbageCollector = resource_manager.GarbageCollector
TimeoutHandler = resource_manager.TimeoutHandler
LambdaResourceManager = resource_manager.LambdaResourceManager


class TestMemoryThresholds:
    """Test MemoryThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        thresholds = MemoryThresholds()

        assert thresholds.warning_percent == 70.0
        assert thresholds.critical_percent == 85.0
        assert thresholds.max_usage_gb == 8.0
        assert thresholds.cleanup_threshold_percent == 75.0

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = MemoryThresholds(
            warning_percent=60.0,
            critical_percent=80.0,
            max_usage_gb=10.0
        )

        assert thresholds.warning_percent == 60.0
        assert thresholds.critical_percent == 80.0
        assert thresholds.max_usage_gb == 10.0


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = MemoryMonitor()

        assert monitor.thresholds is not None
        assert monitor._baseline_memory is None
        assert len(monitor._metrics_history) == 0

    def test_capture_baseline(self):
        """Test baseline memory capture."""
        monitor = MemoryMonitor()
        baseline = monitor.capture_baseline()

        assert baseline > 0
        assert monitor._baseline_memory == baseline

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = MemoryMonitor()
        metrics = monitor.get_current_metrics()

        assert isinstance(metrics, ResourceMetrics)
        assert metrics.memory_used_mb > 0
        assert metrics.memory_available_mb > 0
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.timestamp is not None

    def test_check_memory_status_ok(self):
        """Test memory status check when OK."""
        # Use high thresholds so status is always OK
        thresholds = MemoryThresholds(
            warning_percent=99.0,
            critical_percent=99.9
        )
        monitor = MemoryMonitor(thresholds=thresholds)
        status = monitor.check_memory_status()

        assert status["status"] == "ok"
        assert "current_mb" in status
        assert "available_mb" in status

    def test_check_memory_status_returns_recommendations(self):
        """Test that status check includes recommendations."""
        monitor = MemoryMonitor()
        status = monitor.check_memory_status()

        assert "recommendations" in status
        assert "warnings" in status
        assert isinstance(status["recommendations"], list)

    def test_metrics_history_recording(self):
        """Test that metrics are recorded to history."""
        monitor = MemoryMonitor()

        # Get metrics multiple times
        for _ in range(5):
            monitor.get_current_metrics()

        assert len(monitor._metrics_history) == 5

    def test_metrics_history_limit(self):
        """Test metrics history size limit."""
        monitor = MemoryMonitor()
        monitor._max_history_size = 10

        # Get metrics more than limit
        for _ in range(15):
            monitor.get_current_metrics()

        assert len(monitor._metrics_history) == 10

    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        monitor = MemoryMonitor()

        # Generate some metrics
        for _ in range(5):
            monitor.get_current_metrics()

        summary = monitor.get_metrics_summary()

        assert summary["samples"] == 5
        assert "memory_min_mb" in summary
        assert "memory_max_mb" in summary
        assert "memory_avg_mb" in summary

    def test_detect_memory_leak_insufficient_data(self):
        """Test leak detection with insufficient data."""
        monitor = MemoryMonitor()

        # Only a few samples
        for _ in range(5):
            monitor.get_current_metrics()

        result = monitor.detect_memory_leak()
        assert result is None  # Not enough data

    def test_detect_memory_leak_with_data(self):
        """Test leak detection with sufficient data."""
        monitor = MemoryMonitor()

        # Generate enough metrics
        for _ in range(15):
            monitor.get_current_metrics()

        result = monitor.detect_memory_leak()

        assert result is not None
        assert "detected" in result
        assert "increase_mb" in result


class TestGarbageCollector:
    """Test GarbageCollector class."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = GarbageCollector()

        assert collector._collection_count == 0
        assert collector._total_freed_mb == 0.0

    def test_collect(self):
        """Test garbage collection."""
        collector = GarbageCollector()
        result = collector.collect()

        assert "objects_collected" in result
        assert "memory_before_mb" in result
        assert "memory_after_mb" in result
        assert collector._collection_count == 1

    def test_collect_full(self):
        """Test full garbage collection."""
        collector = GarbageCollector()
        result = collector.collect(full=True)

        assert result["objects_collected"] >= 0
        assert collector._collection_count == 1

    def test_get_stats(self):
        """Test getting GC statistics."""
        collector = GarbageCollector()
        collector.collect()

        stats = collector.get_stats()

        assert stats["collection_count"] == 1
        assert "gc_stats" in stats
        assert "gc_threshold" in stats


class TestTimeoutHandler:
    """Test TimeoutHandler class."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = TimeoutHandler(default_timeout=60)

        assert handler.default_timeout == 60

    def test_timeout_decorator_success(self):
        """Test timeout decorator with successful function."""
        handler = TimeoutHandler(default_timeout=10)

        @handler.timeout(5)
        def quick_function():
            return "success"

        result = quick_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_timeout_success(self):
        """Test async timeout with successful coroutine."""
        handler = TimeoutHandler(default_timeout=10)

        async def quick_coro():
            return "success"

        result = await handler.async_timeout(quick_coro(), 5)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_timeout_exceeded(self):
        """Test async timeout when exceeded."""
        import asyncio

        handler = TimeoutHandler(default_timeout=1)

        async def slow_coro():
            await asyncio.sleep(5)
            return "should not reach"

        with pytest.raises(TimeoutError):
            await handler.async_timeout(slow_coro(), 0.1)


class TestLambdaResourceManager:
    """Test LambdaResourceManager class."""

    def test_initialization(self):
        """Test resource manager initialization."""
        manager = LambdaResourceManager()

        assert manager.memory_monitor is not None
        assert manager.gc is not None
        assert manager.timeout_handler is not None
        assert not manager._initialized

    def test_initialize(self):
        """Test initialization."""
        manager = LambdaResourceManager()
        result = manager.initialize()

        assert result["status"] == "initialized"
        assert "baseline_memory_mb" in result
        assert manager._initialized

    def test_initialize_idempotent(self):
        """Test that initialization is idempotent."""
        manager = LambdaResourceManager()
        manager.initialize()
        result = manager.initialize()

        assert result["status"] == "already_initialized"

    def test_managed_context(self):
        """Test managed context."""
        manager = LambdaResourceManager()
        manager.initialize()

        with manager.managed_context("test"):
            # Do some work
            _ = [i for i in range(1000)]

        # Context should have exited cleanly
        assert manager._context_depth == 0

    @pytest.mark.asyncio
    async def test_async_managed_context(self):
        """Test async managed context."""
        manager = LambdaResourceManager()
        manager.initialize()

        async with manager.async_managed_context("test"):
            # Do some work
            _ = [i for i in range(1000)]

        assert manager._context_depth == 0

    def test_check_resources(self):
        """Test resource checking."""
        manager = LambdaResourceManager()
        manager.initialize()

        status = manager.check_resources()

        assert "memory" in status
        assert "gc" in status
        assert "overall_status" in status

    def test_request_cleanup(self):
        """Test cleanup request."""
        manager = LambdaResourceManager()
        manager.initialize()

        result = manager.request_cleanup()

        assert "gc_result" in result
        assert "cleanup_hooks_run" in result

    def test_request_cleanup_aggressive(self):
        """Test aggressive cleanup."""
        manager = LambdaResourceManager()
        manager.initialize()

        result = manager.request_cleanup(aggressive=True)

        assert "gc_result" in result
        assert "tensors_collected" in result

    def test_emergency_cleanup(self):
        """Test emergency cleanup."""
        manager = LambdaResourceManager()
        manager.initialize()

        result = manager.emergency_cleanup()

        assert result["emergency"] is True
        assert "memory_before_mb" in result
        assert "memory_after_mb" in result

    def test_register_cleanup_hook(self):
        """Test cleanup hook registration."""
        manager = LambdaResourceManager()
        hook_called = []

        def test_hook():
            hook_called.append(True)

        manager.register_cleanup_hook(test_hook)
        manager.request_cleanup()

        assert len(hook_called) == 1

    def test_get_resource_summary(self):
        """Test resource summary."""
        manager = LambdaResourceManager()
        manager.initialize()

        summary = manager.get_resource_summary()

        assert "current" in summary
        assert "history" in summary
        assert "gc_stats" in summary
        assert "status" in summary

    def test_shutdown(self):
        """Test shutdown."""
        manager = LambdaResourceManager()
        manager.initialize()

        manager.shutdown()

        assert not manager._initialized
        assert len(manager._cleanup_hooks) == 0

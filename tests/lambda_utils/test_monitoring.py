"""
Tests for Lambda monitoring utilities.

Tests COLPALI-904: Set up monitoring and logging.
"""

import sys
from pathlib import Path
import importlib.util

import pytest
from datetime import datetime, timedelta
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
monitoring = load_module_directly(
    "monitoring_test",
    lambda_utils_path / "monitoring.py"
)

# Get classes
RequestContext = monitoring.RequestContext
PerformanceMetrics = monitoring.PerformanceMetrics
StructuredLogger = monitoring.StructuredLogger
LambdaMonitor = monitoring.LambdaMonitor
track_performance = monitoring.track_performance


class TestRequestContext:
    """Test RequestContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        context = RequestContext()

        assert context.correlation_id is not None
        assert len(context.correlation_id) == 36  # UUID format
        assert context.request_id is None
        assert context.timestamp is not None

    def test_custom_values(self):
        """Test custom context values."""
        context = RequestContext(
            correlation_id="test-123",
            request_id="req-456",
            function_name="test-function"
        )

        assert context.correlation_id == "test-123"
        assert context.request_id == "req-456"
        assert context.function_name == "test-function"

    def test_extra_fields(self):
        """Test extra fields."""
        context = RequestContext(
            extra={"custom_field": "value"}
        )

        assert context.extra["custom_field"] == "value"


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_default_values(self):
        """Test default metrics values."""
        metrics = PerformanceMetrics(
            correlation_id="test-123",
            operation="test_op",
            start_time=datetime.utcnow()
        )

        assert metrics.correlation_id == "test-123"
        assert metrics.operation == "test_op"
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.duration_ms == 0.0

    def test_with_all_values(self):
        """Test metrics with all values."""
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)

        metrics = PerformanceMetrics(
            correlation_id="test-123",
            operation="test_op",
            start_time=start,
            end_time=end,
            duration_ms=1000.0,
            memory_start_mb=100.0,
            memory_end_mb=150.0,
            memory_delta_mb=50.0,
            success=True,
            metadata={"key": "value"}
        )

        assert metrics.duration_ms == 1000.0
        assert metrics.memory_delta_mb == 50.0
        assert metrics.metadata["key"] == "value"


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger(service_name="test-service")

        assert logger.service_name == "test-service"
        assert logger._context is None

    def test_set_context(self):
        """Test setting context."""
        logger = StructuredLogger()
        context = RequestContext(correlation_id="test-123")

        logger.set_context(context)

        assert logger.get_context() == context
        assert logger.get_context().correlation_id == "test-123"

    def test_request_context_manager(self):
        """Test request context manager."""
        logger = StructuredLogger()

        with logger.request_context(correlation_id="test-123") as ctx:
            assert ctx.correlation_id == "test-123"
            assert logger.get_context() is not None

        # Context should be cleared after exit
        assert logger.get_context() is None

    def test_request_context_auto_id(self):
        """Test auto-generated correlation ID."""
        logger = StructuredLogger()

        with logger.request_context() as ctx:
            assert ctx.correlation_id is not None
            assert len(ctx.correlation_id) == 36

    def test_log_levels(self):
        """Test different log levels."""
        logger = StructuredLogger()

        # These should not raise
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

    def test_metric_logging(self):
        """Test metric logging."""
        logger = StructuredLogger()

        # Should not raise
        logger.metric(
            "TestMetric",
            100.0,
            "Milliseconds",
            operation="test"
        )


class TestLambdaMonitor:
    """Test LambdaMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = LambdaMonitor(
            service_name="test-service",
            version="1.0.0"
        )

        assert monitor.service_name == "test-service"
        assert monitor.version == "1.0.0"
        assert monitor._request_count == 0
        assert monitor._error_count == 0

    def test_trace_operation_success(self):
        """Test successful operation tracing."""
        monitor = LambdaMonitor()

        with monitor.trace_operation("test_op") as metrics:
            # Do some work
            _ = sum(range(1000))

        assert metrics.success is True
        assert metrics.error is None
        assert metrics.duration_ms > 0
        assert monitor._request_count == 1
        assert monitor._error_count == 0

    def test_trace_operation_failure(self):
        """Test failed operation tracing."""
        monitor = LambdaMonitor()

        with pytest.raises(ValueError):
            with monitor.trace_operation("test_op") as metrics:
                raise ValueError("Test error")

        assert metrics.success is False
        assert metrics.error == "Test error"
        assert monitor._error_count == 1

    def test_trace_operation_correlation_id(self):
        """Test correlation ID in tracing."""
        monitor = LambdaMonitor()

        with monitor.trace_operation(
            "test_op",
            correlation_id="test-123"
        ) as metrics:
            pass

        assert metrics.correlation_id == "test-123"

    def test_record_metric(self):
        """Test custom metric recording."""
        monitor = LambdaMonitor()

        # Should not raise
        monitor.record_metric(
            "CustomMetric",
            100.0,
            "Count",
            dimension1="value1"
        )

    def test_health_check(self):
        """Test health check."""
        monitor = LambdaMonitor()
        health = monitor.health_check()

        assert health["status"] in ["healthy", "degraded", "warning", "critical"]
        assert health["service"] == "colpali-baml-engine"
        assert "metrics" in health
        assert "system" in health
        assert "timestamp" in health

    def test_health_check_error_rate(self):
        """Test health check with errors."""
        monitor = LambdaMonitor()

        # Generate some errors
        for _ in range(5):
            try:
                with monitor.trace_operation("test"):
                    raise Exception("error")
            except Exception:
                pass

        health = monitor.health_check()

        assert health["metrics"]["error_count"] == 5

    def test_get_metrics_summary_empty(self):
        """Test metrics summary with no data."""
        monitor = LambdaMonitor()
        summary = monitor.get_metrics_summary()

        assert summary["total_operations"] == 0

    def test_get_metrics_summary_with_data(self):
        """Test metrics summary with data."""
        monitor = LambdaMonitor()

        # Generate some operations
        for _ in range(5):
            with monitor.trace_operation("test"):
                pass

        summary = monitor.get_metrics_summary()

        assert summary["total_operations"] == 5
        assert summary["success_count"] == 5
        assert "latency" in summary
        assert "memory" in summary

    def test_get_recent_errors(self):
        """Test getting recent errors."""
        monitor = LambdaMonitor()

        # Generate some errors
        for i in range(3):
            try:
                with monitor.trace_operation(f"test_{i}"):
                    raise Exception(f"error_{i}")
            except Exception:
                pass

        errors = monitor.get_recent_errors(limit=2)

        assert len(errors) == 2
        assert all("error" in e for e in errors)

    def test_cloudwatch_dashboard_config(self):
        """Test CloudWatch dashboard configuration."""
        monitor = LambdaMonitor()
        config = monitor.create_cloudwatch_dashboard_config()

        assert "widgets" in config
        assert len(config["widgets"]) > 0

        # Check widget structure
        for widget in config["widgets"]:
            assert "type" in widget
            assert "properties" in widget


class TestTrackPerformanceDecorator:
    """Test track_performance decorator."""

    def test_decorator_basic(self):
        """Test basic decorator usage."""
        @track_performance()
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_decorator_with_operation_name(self):
        """Test decorator with custom operation name."""
        @track_performance(operation="custom_operation")
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_decorator_with_exception(self):
        """Test decorator with exception."""
        @track_performance()
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function name."""
        @track_performance()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_with_arguments(self):
        """Test decorator with function arguments."""
        @track_performance()
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""
        @track_performance()
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"

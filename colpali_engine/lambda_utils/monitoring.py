"""
Monitoring and logging utilities for AWS Lambda deployment.

Implements COLPALI-904: Set up monitoring and logging.
Provides structured logging with correlation IDs, CloudWatch integration,
health check endpoints, and performance metrics.
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

import psutil

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RequestContext:
    """Request context with correlation tracking."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    function_name: Optional[str] = None
    function_version: Optional[str] = None
    memory_limit_mb: Optional[int] = None
    remaining_time_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a request."""

    correlation_id: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_delta_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """
    Structured JSON logging for Lambda with CloudWatch integration.

    Provides consistent log format with correlation IDs and metadata
    for easy CloudWatch Insights querying.
    """

    def __init__(
        self,
        service_name: str = "colpali-baml-engine",
        log_level: int = logging.INFO
    ):
        """
        Initialize structured logger.

        Args:
            service_name: Name of the service for logs
            log_level: Logging level
        """
        self.service_name = service_name
        self.log_level = log_level
        self._context: Optional[RequestContext] = None

        # Configure root logger for structured output
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configure logging for structured JSON output."""
        # Custom formatter for CloudWatch
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "service": "colpali-baml-engine"
                }

                # Add exception info if present
                if record.exc_info:
                    log_record["exception"] = self.formatException(record.exc_info)

                # Add extra fields
                if hasattr(record, 'extra_fields'):
                    log_record.update(record.extra_fields)

                return json.dumps(log_record)

        # Apply formatter to handler
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers = [handler]
        root_logger.setLevel(self.log_level)

    def set_context(self, context: RequestContext) -> None:
        """Set the current request context."""
        self._context = context

    def get_context(self) -> Optional[RequestContext]:
        """Get the current request context."""
        return self._context

    @contextmanager
    def request_context(self, correlation_id: Optional[str] = None, **extra):
        """
        Context manager for request scoped logging.

        Args:
            correlation_id: Optional correlation ID (auto-generated if not provided)
            **extra: Additional context fields
        """
        context = RequestContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            extra=extra
        )
        self._context = context

        self.info("Request started", operation="request_start")

        try:
            yield context
        finally:
            self.info("Request completed", operation="request_end")
            self._context = None

    def _create_log_record(
        self,
        message: str,
        level: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a structured log record."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": self.service_name,
            "message": message
        }

        # Add correlation context
        if self._context:
            record["correlation_id"] = self._context.correlation_id
            if self._context.request_id:
                record["request_id"] = self._context.request_id

        # Add extra fields
        record.update(kwargs)

        return record

    def log(self, level: int, message: str, **kwargs) -> None:
        """
        Log a structured message.

        Args:
            level: Logging level
            message: Log message
            **kwargs: Additional fields
        """
        record = self._create_log_record(
            message, logging.getLevelName(level), **kwargs
        )

        # Use standard logging with extra context
        extra_record = logging.LogRecord(
            name=self.service_name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        extra_record.extra_fields = record

        logging.getLogger().handle(extra_record)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log(logging.CRITICAL, message, **kwargs)

    def metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "None",
        **dimensions
    ) -> None:
        """
        Log a CloudWatch-compatible metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: CloudWatch unit (Seconds, Milliseconds, Count, etc.)
            **dimensions: Metric dimensions
        """
        self.info(
            f"METRIC {metric_name}",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            dimensions=dimensions,
            _aws_cloudwatch_metric=True
        )


class LambdaMonitor:
    """
    Comprehensive monitoring for Lambda deployments.

    Implements COLPALI-904 requirements:
    - Structured logging with correlation IDs
    - Performance metrics (latency, memory, accuracy)
    - Error tracking and alerting
    - Health check endpoints
    - CloudWatch integration and dashboards
    """

    def __init__(
        self,
        service_name: str = "colpali-baml-engine",
        version: str = "0.1.0"
    ):
        """
        Initialize Lambda monitor.

        Args:
            service_name: Service name for metrics
            version: Service version
        """
        self.service_name = service_name
        self.version = version
        self.logger = StructuredLogger(service_name)

        # Metrics storage
        self._metrics: List[PerformanceMetrics] = []
        self._max_metrics = 1000
        self._error_count = 0
        self._request_count = 0

        # Start time for uptime tracking
        self._start_time = datetime.utcnow()

        logger.info(f"LambdaMonitor initialized: {service_name} v{version}")

    @contextmanager
    def trace_operation(
        self,
        operation: str,
        correlation_id: Optional[str] = None,
        **metadata
    ):
        """
        Context manager for tracing operations.

        Args:
            operation: Name of the operation
            correlation_id: Optional correlation ID
            **metadata: Additional metadata

        Yields:
            PerformanceMetrics object
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = datetime.utcnow()
        start_memory = self._get_memory_mb()

        metrics = PerformanceMetrics(
            correlation_id=correlation_id,
            operation=operation,
            start_time=start_time,
            memory_start_mb=start_memory,
            metadata=metadata
        )

        self.logger.info(
            f"Starting operation: {operation}",
            operation=operation,
            correlation_id=correlation_id
        )

        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            self._error_count += 1
            self.logger.error(
                f"Operation failed: {operation}",
                operation=operation,
                error=str(e),
                correlation_id=correlation_id
            )
            raise
        finally:
            end_time = datetime.utcnow()
            end_memory = self._get_memory_mb()

            metrics.end_time = end_time
            metrics.duration_ms = (end_time - start_time).total_seconds() * 1000
            metrics.memory_end_mb = end_memory
            metrics.memory_delta_mb = end_memory - start_memory

            self._record_metrics(metrics)
            self._request_count += 1

            self.logger.info(
                f"Completed operation: {operation}",
                operation=operation,
                duration_ms=metrics.duration_ms,
                memory_delta_mb=metrics.memory_delta_mb,
                success=metrics.success,
                correlation_id=correlation_id
            )

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "None",
        **dimensions
    ) -> None:
        """
        Record a custom metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            **dimensions: Metric dimensions
        """
        self.logger.metric(name, value, unit, **dimensions)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health check response
        """
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())

        health = {
            "status": "healthy",
            "service": self.service_name,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": uptime,
            "metrics": {
                "total_requests": self._request_count,
                "error_count": self._error_count,
                "error_rate": (
                    self._error_count / self._request_count
                    if self._request_count > 0 else 0
                )
            },
            "system": {
                "memory_used_mb": process.memory_info().rss / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_percent": memory.percent,
                "cpu_percent": process.cpu_percent()
            }
        }

        # Determine overall health status
        if health["metrics"]["error_rate"] > 0.5:
            health["status"] = "degraded"
        if health["system"]["memory_percent"] > 90:
            health["status"] = "warning"
        if health["system"]["memory_percent"] > 95:
            health["status"] = "critical"

        return health

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of recorded metrics.

        Returns:
            Metrics summary
        """
        if not self._metrics:
            return {
                "total_operations": 0,
                "summary": "No metrics recorded"
            }

        durations = [m.duration_ms for m in self._metrics]
        memory_deltas = [m.memory_delta_mb for m in self._metrics]
        success_count = sum(1 for m in self._metrics if m.success)

        return {
            "total_operations": len(self._metrics),
            "success_count": success_count,
            "error_count": len(self._metrics) - success_count,
            "success_rate": success_count / len(self._metrics),
            "latency": {
                "min_ms": min(durations),
                "max_ms": max(durations),
                "avg_ms": sum(durations) / len(durations),
                "p50_ms": sorted(durations)[len(durations) // 2],
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
            },
            "memory": {
                "min_delta_mb": min(memory_deltas),
                "max_delta_mb": max(memory_deltas),
                "avg_delta_mb": sum(memory_deltas) / len(memory_deltas)
            },
            "time_range": {
                "start": self._metrics[0].start_time.isoformat(),
                "end": self._metrics[-1].start_time.isoformat()
            }
        }

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error records
        """
        errors = [
            {
                "correlation_id": m.correlation_id,
                "operation": m.operation,
                "error": m.error,
                "timestamp": m.start_time.isoformat(),
                "duration_ms": m.duration_ms
            }
            for m in self._metrics if not m.success
        ]

        return errors[-limit:]

    def create_cloudwatch_dashboard_config(self) -> Dict[str, Any]:
        """
        Generate CloudWatch dashboard configuration.

        Returns:
            Dashboard configuration JSON
        """
        return {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "title": "Request Latency",
                        "metrics": [
                            [self.service_name, "Latency", "Operation", "process_document"]
                        ],
                        "period": 60,
                        "stat": "Average"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Error Rate",
                        "metrics": [
                            [self.service_name, "Errors", "Operation", "process_document"]
                        ],
                        "period": 60,
                        "stat": "Sum"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Memory Usage",
                        "metrics": [
                            [self.service_name, "MemoryUsage"]
                        ],
                        "period": 60,
                        "stat": "Average"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Cold Starts",
                        "metrics": [
                            [self.service_name, "ColdStart"]
                        ],
                        "period": 300,
                        "stat": "Sum"
                    }
                }
            ]
        }

    def _record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record metrics to storage."""
        self._metrics.append(metrics)
        if len(self._metrics) > self._max_metrics:
            self._metrics.pop(0)

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


def track_performance(operation: str = None):
    """
    Decorator for tracking function performance.

    Args:
        operation: Operation name (defaults to function name)

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            start_memory = _get_memory_mb()
            correlation_id = str(uuid.uuid4())

            logger.info(
                f"Starting {op_name}",
                extra={
                    "operation": op_name,
                    "correlation_id": correlation_id
                }
            )

            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                memory_delta = _get_memory_mb() - start_memory

                logger.info(
                    f"Completed {op_name}",
                    extra={
                        "operation": op_name,
                        "correlation_id": correlation_id,
                        "duration_ms": duration_ms,
                        "memory_delta_mb": memory_delta,
                        "success": success,
                        "error": error
                    }
                )

            return result

        return wrapper
    return decorator


def _get_memory_mb() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

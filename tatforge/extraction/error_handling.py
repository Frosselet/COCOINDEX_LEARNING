"""
Error Handling and Retry Logic System - COLPALI-603.

This module provides robust error handling with intelligent retry mechanisms
for model failures, network issues, and extraction quality problems. Includes
exponential backoff, circuit breaker pattern, error classification, graceful
degradation, and comprehensive alerting capabilities.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import traceback

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors that can occur during extraction."""
    TRANSIENT_NETWORK = "transient_network"
    MODEL_FAILURE = "model_failure"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_PROCESSING = "data_processing"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Different retry strategies available."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


@dataclass
class ErrorClassification:
    """Classification of an error with metadata."""
    category: ErrorCategory
    severity: ErrorSeverity
    is_retryable: bool
    suggested_action: str
    recovery_strategy: Optional[str] = None
    estimated_recovery_time: Optional[float] = None  # seconds
    error_code: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout_per_attempt: Optional[float] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3
    success_threshold: int = 2  # Successes needed to close circuit


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ErrorMetrics:
    """Metrics tracking for error handling."""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    retry_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    circuit_breaker_trips: int = 0
    last_error_time: Optional[datetime] = None
    error_rate_1m: float = 0.0
    error_rate_5m: float = 0.0


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""
    action_type: str
    description: str
    execute_at: datetime
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classifies errors into categories with appropriate handling strategies."""

    def __init__(self):
        """Initialize error classifier with predefined rules."""
        self.classification_rules = self._build_classification_rules()

    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorClassification:
        """
        Classify an error and determine handling strategy.

        Args:
            error: Exception that occurred
            context: Additional context about the error

        Returns:
            Error classification with handling recommendations
        """
        error_message = str(error).lower()
        error_type = type(error).__name__

        # Check classification rules
        for pattern, classification in self.classification_rules.items():
            if self._matches_pattern(pattern, error_message, error_type, context or {}):
                return classification

        # Default classification for unknown errors
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            suggested_action="Log error and retry with caution",
            recovery_strategy="exponential_backoff"
        )

    def _build_classification_rules(self) -> Dict[str, ErrorClassification]:
        """Build the error classification rules."""
        return {
            "timeout": ErrorClassification(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                suggested_action="Retry with increased timeout",
                recovery_strategy="exponential_backoff",
                estimated_recovery_time=30.0
            ),
            "network|connection|dns": ErrorClassification(
                category=ErrorCategory.TRANSIENT_NETWORK,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                suggested_action="Retry after brief delay",
                recovery_strategy="exponential_backoff",
                estimated_recovery_time=10.0
            ),
            "rate.limit|too.many.requests|429": ErrorClassification(
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.HIGH,
                is_retryable=True,
                suggested_action="Back off and retry with longer delay",
                recovery_strategy="linear_backoff",
                estimated_recovery_time=60.0
            ),
            "auth|unauthorized|forbidden|401|403": ErrorClassification(
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.CRITICAL,
                is_retryable=False,
                suggested_action="Check authentication credentials",
                recovery_strategy=None
            ),
            "out.of.memory|memory|oom": ErrorClassification(
                category=ErrorCategory.RESOURCE_EXHAUSTED,
                severity=ErrorSeverity.HIGH,
                is_retryable=True,
                suggested_action="Reduce batch size and retry",
                recovery_strategy="graceful_degradation",
                estimated_recovery_time=120.0
            ),
            "model.failed|inference.error|vision.error": ErrorClassification(
                category=ErrorCategory.MODEL_FAILURE,
                severity=ErrorSeverity.HIGH,
                is_retryable=True,
                suggested_action="Try alternative model or fallback strategy",
                recovery_strategy="model_fallback",
                estimated_recovery_time=15.0
            ),
            "validation|schema|type.error": ErrorClassification(
                category=ErrorCategory.VALIDATION_ERROR,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=False,
                suggested_action="Check input data and schema definitions",
                recovery_strategy="data_correction"
            ),
            "config|configuration|setup": ErrorClassification(
                category=ErrorCategory.CONFIGURATION_ERROR,
                severity=ErrorSeverity.HIGH,
                is_retryable=False,
                suggested_action="Review and fix configuration",
                recovery_strategy=None
            )
        }

    def _matches_pattern(
        self,
        pattern: str,
        error_message: str,
        error_type: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if error matches a classification pattern."""
        # Simple pattern matching - could be enhanced with regex
        pattern_terms = pattern.split("|")
        return any(
            term in error_message or term in error_type.lower()
            for term in pattern_terms
        )


class CircuitBreaker:
    """Circuit breaker implementation to prevent cascading failures."""

    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        """Initialize circuit breaker with configuration."""
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result if successful

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next attempt allowed in {self._time_until_reset():.1f}s"
                )

        elif self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' in HALF_OPEN state exceeded max calls"
                )

        try:
            # Execute the function
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Handle successful execution
            self._on_success()
            return result

        except Exception as e:
            # Handle failed execution
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout

    def _time_until_reset(self) -> float:
        """Calculate time until reset attempt is allowed."""
        if not self.last_failure_time:
            return 0.0

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return max(0.0, self.config.recovery_timeout - elapsed)

    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")

    def _on_success(self):
        """Handle successful function execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()

    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")

    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "time_until_reset": self._time_until_reset() if self.state == CircuitState.OPEN else 0.0
        }


class RetryManager:
    """Manages retry logic with different strategies and backoff algorithms."""

    def __init__(self, config: RetryConfig):
        """Initialize retry manager with configuration."""
        self.config = config

    async def execute_with_retry(
        self,
        func: Callable,
        error_classifier: ErrorClassifier,
        context: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        """
        Execute function with retry logic based on error classification.

        Args:
            func: Function to execute
            error_classifier: Error classifier for retry decisions
            context: Additional context for error classification
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result if successful

        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        attempt = 0

        while attempt < self.config.max_attempts:
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                attempt += 1

                # Classify the error
                classification = error_classifier.classify_error(e, context)

                logger.warning(
                    f"Attempt {attempt}/{self.config.max_attempts} failed: {e} "
                    f"(category: {classification.category.value}, "
                    f"retryable: {classification.is_retryable})"
                )

                # Check if error is retryable
                if not classification.is_retryable:
                    logger.error(f"Non-retryable error: {classification.suggested_action}")
                    raise

                # Check if we've exhausted all attempts
                if attempt >= self.config.max_attempts:
                    logger.error(f"All {self.config.max_attempts} retry attempts exhausted")
                    break

                # Calculate delay and wait
                delay = self._calculate_delay(attempt, classification)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        # All retries failed, raise the last exception
        raise last_exception

    def _calculate_delay(self, attempt: int, classification: ErrorClassification) -> float:
        """Calculate delay before next retry attempt."""
        if self.config.strategy == RetryStrategy.NO_RETRY:
            return 0.0

        elif self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt

        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))

        else:
            delay = self.config.base_delay

        # Apply jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        # Respect maximum delay
        delay = min(delay, self.config.max_delay)

        # Consider error-specific delay adjustments
        if classification.estimated_recovery_time:
            delay = max(delay, classification.estimated_recovery_time * 0.5)

        return max(0.0, delay)


class GracefulDegradationManager:
    """Manages graceful degradation strategies for partial failures."""

    def __init__(self):
        """Initialize graceful degradation manager."""
        self.degradation_strategies = self._build_degradation_strategies()

    async def apply_degradation(
        self,
        original_func: Callable,
        error: Exception,
        classification: ErrorClassification,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply graceful degradation strategy for a failed operation.

        Args:
            original_func: Original function that failed
            error: Exception that caused the failure
            classification: Error classification
            context: Additional context

        Returns:
            Degraded result or partial success
        """
        strategy_name = classification.recovery_strategy
        if strategy_name in self.degradation_strategies:
            strategy_func = self.degradation_strategies[strategy_name]
            return await strategy_func(original_func, error, context or {})

        # Default degradation: return partial results
        return await self._default_degradation(error, context or {})

    def _build_degradation_strategies(self) -> Dict[str, Callable]:
        """Build available degradation strategies."""
        return {
            "graceful_degradation": self._graceful_degradation,
            "model_fallback": self._model_fallback,
            "data_correction": self._data_correction,
            "partial_processing": self._partial_processing
        }

    async def _graceful_degradation(
        self,
        original_func: Callable,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply graceful degradation with reduced functionality."""
        logger.info("Applying graceful degradation strategy")

        return {
            "success": False,
            "partial_result": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "degradation_applied": "graceful_degradation",
            "extracted_data": {},
            "metadata": {
                "degraded": True,
                "original_error": str(error),
                "degradation_timestamp": datetime.now().isoformat()
            }
        }

    async def _model_fallback(
        self,
        original_func: Callable,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply model fallback strategy."""
        logger.info("Applying model fallback strategy")

        # Try to extract any partial information from error context
        partial_data = context.get("partial_extraction", {})

        return {
            "success": True,
            "partial_result": True,
            "extracted_data": partial_data,
            "degradation_applied": "model_fallback",
            "metadata": {
                "fallback_used": True,
                "original_error": str(error),
                "fallback_timestamp": datetime.now().isoformat()
            }
        }

    async def _data_correction(
        self,
        original_func: Callable,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply data correction strategy."""
        logger.info("Applying data correction strategy")

        # Attempt to correct common data issues
        corrected_data = {}
        if "input_data" in context:
            corrected_data = self._attempt_data_correction(context["input_data"])

        return {
            "success": True,
            "partial_result": True,
            "extracted_data": corrected_data,
            "degradation_applied": "data_correction",
            "metadata": {
                "data_corrected": True,
                "original_error": str(error)
            }
        }

    async def _partial_processing(
        self,
        original_func: Callable,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply partial processing strategy."""
        logger.info("Applying partial processing strategy")

        # Process available data even if some parts failed
        available_data = context.get("available_data", {})

        return {
            "success": True,
            "partial_result": True,
            "extracted_data": available_data,
            "degradation_applied": "partial_processing",
            "metadata": {
                "partial_processing": True,
                "original_error": str(error)
            }
        }

    async def _default_degradation(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default degradation strategy."""
        return {
            "success": False,
            "partial_result": True,
            "error": str(error),
            "degradation_applied": "default",
            "extracted_data": {},
            "metadata": {
                "degraded": True,
                "error_type": type(error).__name__
            }
        }

    def _attempt_data_correction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to correct common data issues."""
        corrected = {}

        for key, value in input_data.items():
            try:
                if isinstance(value, str):
                    # Basic string cleaning
                    corrected[key] = value.strip()
                elif isinstance(value, (int, float)):
                    corrected[key] = value
                else:
                    corrected[key] = str(value) if value is not None else ""
            except Exception:
                corrected[key] = ""

        return corrected


class ErrorHandler:
    """
    Main error handling orchestrator that combines all error handling strategies.

    Provides unified error handling with retry logic, circuit breaker protection,
    graceful degradation, and comprehensive error tracking and alerting.
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_alerting: bool = True
    ):
        """
        Initialize error handler with configuration.

        Args:
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            enable_alerting: Whether to enable error alerting
        """
        self.error_classifier = ErrorClassifier()
        self.retry_manager = RetryManager(retry_config or RetryConfig())
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig(),
            name="extraction_handler"
        )
        self.degradation_manager = GracefulDegradationManager()

        self.enable_alerting = enable_alerting
        self.error_metrics = ErrorMetrics()
        self.recovery_actions: List[RecoveryAction] = []

        logger.info("Error handler initialized")

    async def execute_with_error_handling(
        self,
        func: Callable,
        context: Optional[Dict[str, Any]] = None,
        enable_circuit_breaker: bool = True,
        enable_graceful_degradation: bool = True,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute function with comprehensive error handling.

        Args:
            func: Function to execute
            context: Execution context
            enable_circuit_breaker: Whether to use circuit breaker
            enable_graceful_degradation: Whether to apply graceful degradation
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or degraded result
        """
        execution_context = context or {}

        try:
            if enable_circuit_breaker:
                # Execute with circuit breaker and retry protection
                result = await self.circuit_breaker.call(
                    self.retry_manager.execute_with_retry,
                    func,
                    self.error_classifier,
                    execution_context,
                    *args,
                    **kwargs
                )
            else:
                # Execute with retry protection only
                result = await self.retry_manager.execute_with_retry(
                    func,
                    self.error_classifier,
                    execution_context,
                    *args,
                    **kwargs
                )

            # Update success metrics
            self._update_success_metrics()

            return result if isinstance(result, dict) else {"result": result, "success": True}

        except Exception as e:
            # Classify the error
            classification = self.error_classifier.classify_error(e, execution_context)

            # Update error metrics
            self._update_error_metrics(classification, e)

            # Apply graceful degradation if enabled
            if enable_graceful_degradation:
                try:
                    degraded_result = await self.degradation_manager.apply_degradation(
                        func, e, classification, execution_context
                    )
                    logger.info(f"Graceful degradation applied: {degraded_result.get('degradation_applied')}")
                    return degraded_result
                except Exception as degradation_error:
                    logger.error(f"Graceful degradation failed: {degradation_error}")

            # If graceful degradation is disabled or fails, return error result
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "error_category": classification.category.value,
                "error_severity": classification.severity.value,
                "suggested_action": classification.suggested_action,
                "timestamp": datetime.now().isoformat()
            }

    def _update_error_metrics(self, classification: ErrorClassification, error: Exception):
        """Update error tracking metrics."""
        self.error_metrics.total_errors += 1
        self.error_metrics.last_error_time = datetime.now()

        # Update category and severity counts
        category = classification.category
        severity = classification.severity

        if category not in self.error_metrics.errors_by_category:
            self.error_metrics.errors_by_category[category] = 0
        self.error_metrics.errors_by_category[category] += 1

        if severity not in self.error_metrics.errors_by_severity:
            self.error_metrics.errors_by_severity[severity] = 0
        self.error_metrics.errors_by_severity[severity] += 1

        # Schedule recovery action if needed
        if classification.recovery_strategy:
            self._schedule_recovery_action(classification, error)

        # Alert if enabled
        if self.enable_alerting:
            self._send_alert(classification, error)

    def _update_success_metrics(self):
        """Update success metrics."""
        # Reset error rates on success
        self.error_metrics.error_rate_1m *= 0.9  # Decay error rate

    def _schedule_recovery_action(self, classification: ErrorClassification, error: Exception):
        """Schedule a recovery action based on error classification."""
        if not classification.recovery_strategy:
            return

        execute_at = datetime.now()
        if classification.estimated_recovery_time:
            execute_at += timedelta(seconds=classification.estimated_recovery_time)

        recovery_action = RecoveryAction(
            action_type=classification.recovery_strategy,
            description=f"Recover from {classification.category.value}: {classification.suggested_action}",
            execute_at=execute_at,
            priority=2 if classification.severity == ErrorSeverity.CRITICAL else 1,
            parameters={
                "error_message": str(error),
                "error_category": classification.category.value,
                "error_severity": classification.severity.value
            }
        )

        self.recovery_actions.append(recovery_action)
        logger.info(f"Scheduled recovery action: {recovery_action.action_type} at {execute_at}")

    def _send_alert(self, classification: ErrorClassification, error: Exception):
        """Send alert for error (placeholder implementation)."""
        if classification.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error(
                f"ALERT: {classification.severity.value.upper()} error detected - "
                f"{classification.category.value}: {str(error)}"
            )

    def get_error_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error metrics."""
        return {
            "total_errors": self.error_metrics.total_errors,
            "errors_by_category": {
                cat.value: count for cat, count in self.error_metrics.errors_by_category.items()
            },
            "errors_by_severity": {
                sev.value: count for sev, count in self.error_metrics.errors_by_severity.items()
            },
            "retry_attempts": self.error_metrics.retry_attempts,
            "successful_retries": self.error_metrics.successful_retries,
            "failed_retries": self.error_metrics.failed_retries,
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "last_error_time": self.error_metrics.last_error_time.isoformat() if self.error_metrics.last_error_time else None,
            "pending_recovery_actions": len(self.recovery_actions)
        }

    async def execute_pending_recovery_actions(self):
        """Execute any pending recovery actions."""
        current_time = datetime.now()
        executed_actions = []

        for action in self.recovery_actions:
            if current_time >= action.execute_at:
                try:
                    await self._execute_recovery_action(action)
                    executed_actions.append(action)
                    logger.info(f"Executed recovery action: {action.action_type}")
                except Exception as e:
                    logger.error(f"Recovery action failed: {action.action_type} - {e}")

        # Remove executed actions
        for action in executed_actions:
            self.recovery_actions.remove(action)

    async def _execute_recovery_action(self, action: RecoveryAction):
        """Execute a specific recovery action."""
        # Placeholder implementation - would be extended based on action type
        logger.info(f"Executing recovery action: {action.description}")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Factory functions for easy setup
def create_error_handler(
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_circuit_breaker: bool = True,
    enable_alerting: bool = True
) -> ErrorHandler:
    """
    Create error handler with standard configuration.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        enable_circuit_breaker: Whether to enable circuit breaker
        enable_alerting: Whether to enable error alerting

    Returns:
        Configured error handler
    """
    retry_config = RetryConfig(
        max_attempts=max_retries + 1,  # +1 for initial attempt
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    )

    circuit_breaker_config = CircuitBreakerConfig() if enable_circuit_breaker else None

    return ErrorHandler(
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        enable_alerting=enable_alerting
    )
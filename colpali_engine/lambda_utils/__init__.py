"""
Lambda utilities for ColPali-BAML Vision Processing Engine.

This module provides Lambda-specific utilities for model optimization,
resource management, monitoring, and deployment.
"""

from .model_optimizer import LambdaModelOptimizer
from .resource_manager import LambdaResourceManager
from .monitoring import LambdaMonitor, StructuredLogger

__all__ = [
    "LambdaModelOptimizer",
    "LambdaResourceManager",
    "LambdaMonitor",
    "StructuredLogger"
]

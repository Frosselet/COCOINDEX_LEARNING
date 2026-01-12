"""
Extraction and validation components.

This module handles BAML execution, result validation, and quality metrics
for structured data extraction from document images.
"""

from .models import ExtractionResult, CanonicalData, ShapedData, ProcessingMetadata
from .baml_interface import BAMLExecutionInterface
from .validation import ExtractionResultValidator
from .error_handling import ErrorHandler
from .quality_metrics import ExtractionQualityManager

__all__ = [
    "ExtractionResult",
    "CanonicalData",
    "ShapedData",
    "ProcessingMetadata",
    "BAMLExecutionInterface",
    "ExtractionResultValidator",
    "ErrorHandler",
    "ExtractionQualityManager",
]
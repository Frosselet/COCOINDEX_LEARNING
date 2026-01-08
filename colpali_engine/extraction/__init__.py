"""
Extraction and validation components.

This module handles BAML execution, result validation, and quality metrics
for structured data extraction from document images.
"""

from .models import ExtractionResult, CanonicalData, ShapedData, ProcessingMetadata
from .baml_interface import BAMLInterface

__all__ = [
    "ExtractionResult",
    "CanonicalData",
    "ShapedData",
    "ProcessingMetadata",
    "BAMLInterface",
]
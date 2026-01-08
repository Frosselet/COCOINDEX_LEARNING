"""
ColPali-BAML Vision Processing Engine

A containerized Python package for vision-native document processing using
ColPali embeddings, Qdrant vector storage, CocoIndex orchestration, and
BAML type-safe extraction.

Architecture:
- Vision-First: Documents → canonical image frames (no OCR)
- Schema-Governed: JSON schemas → BAML classes for type-safe extraction
- Dual Output: Canonical truth layer + shaped business transformations
- Container-Native: Docker-first deployment for all environments
"""

__version__ = "0.1.0"
__author__ = "ColPali-BAML Team"

from .core.pipeline import VisionExtractionPipeline
from .core.document_adapter import DocumentAdapter
from .core.schema_manager import SchemaManager

# Main processing interface
from .extraction.models import ExtractionResult, CanonicalData, ShapedData

# Public API exports
__all__ = [
    "VisionExtractionPipeline",
    "DocumentAdapter",
    "SchemaManager",
    "ExtractionResult",
    "CanonicalData",
    "ShapedData",
]
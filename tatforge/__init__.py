"""
tatForge - AI-powered document extraction using vision AI.

Forge structured data from unstructured documents using ColPali vision embeddings,
Qdrant vector storage, and BAML type-safe extraction.

Architecture:
- Vision-First: Documents -> canonical image frames (no OCR)
- Schema-Governed: JSON schemas -> BAML classes for type-safe extraction
- Dual Output: Canonical truth layer + shaped business transformations
- Container-Native: Docker-first deployment for all environments

Example:
    >>> from tatforge import VisionExtractionPipeline, SchemaManager
    >>> pipeline = VisionExtractionPipeline()
    >>> result = await pipeline.process_document(document_blob, schema)
"""

__version__ = "0.1.0"
__author__ = "TAT Team"
__package_name__ = "tatforge"

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
"""
tatForge - AI-powered document extraction using vision AI.

Forge structured data from unstructured documents using ColPali vision embeddings,
Qdrant vector storage, and BAML type-safe extraction.

Quick Start:
    >>> from tatforge import extract_document
    >>> result = await extract_document("invoice.pdf", schema)

Full Pipeline:
    >>> from tatforge import VisionExtractionPipeline, SchemaManager
    >>> pipeline = VisionExtractionPipeline()
    >>> result = await pipeline.process_document(document_blob, schema)

Architecture:
- Vision-First: Documents -> canonical image frames (no OCR)
- Schema-Governed: JSON schemas -> BAML classes for type-safe extraction
- Dual Output: Canonical truth layer + shaped business transformations
- Container-Native: Docker-first deployment for all environments

Components:
- tatforge.adapters: Document format adapters (PDF, images, HTML)
- tatforge.vision: ColPali vision model integration
- tatforge.extraction: BAML-based structured extraction
- tatforge.storage: Qdrant vector database integration
- tatforge.outputs: Output formatting (canonical, shaped, exports)
- tatforge.governance: Data lineage and validation
- tatforge.lambda_utils: AWS Lambda deployment utilities
"""

__version__ = "0.1.0"
__author__ = "TAT Team"
__package_name__ = "tatforge"

# Core Pipeline
from .core.pipeline import VisionExtractionPipeline, PipelineConfig
from .core.document_adapter import DocumentAdapter, ConversionConfig, DocumentFormat
from .core.schema_manager import SchemaManager
from .core.schema_validator import SchemaValidator

# Extraction Results
from .extraction.models import ExtractionResult, CanonicalData, ShapedData

# Adapters
from .adapters.pdf_adapter import PDFAdapter
from .adapters.image_adapter import ImageAdapter

# Output Formatters
from .outputs.canonical import CanonicalFormatter
from .outputs.shaped import ShapedFormatter
from .outputs.exporters import DataExporter


async def extract_document(
    document_path: str,
    schema: dict,
    output_format: str = "dict"
) -> dict:
    """
    High-level convenience function for document extraction.

    Args:
        document_path: Path to document file (PDF, PNG, JPG, etc.)
        schema: JSON schema defining extraction structure
        output_format: Output format ('dict', 'json', 'dataframe')

    Returns:
        Extracted data as dictionary

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "invoice_number": {"type": "string"},
        ...         "total_amount": {"type": "number"}
        ...     }
        ... }
        >>> result = await extract_document("invoice.pdf", schema)
        >>> print(result["invoice_number"])
    """
    from pathlib import Path

    # Read document
    doc_path = Path(document_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    with open(doc_path, "rb") as f:
        document_blob = f.read()

    # Create pipeline and process
    pipeline = VisionExtractionPipeline()
    result = await pipeline.process_document(
        document_blob=document_blob,
        schema_json=schema
    )

    # Format output
    if result.canonical:
        return result.canonical.extraction_data
    return {}


# Public API exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__package_name__",
    # Core classes
    "VisionExtractionPipeline",
    "PipelineConfig",
    "DocumentAdapter",
    "ConversionConfig",
    "DocumentFormat",
    "SchemaManager",
    "SchemaValidator",
    # Results
    "ExtractionResult",
    "CanonicalData",
    "ShapedData",
    # Adapters
    "PDFAdapter",
    "ImageAdapter",
    # Formatters
    "CanonicalFormatter",
    "ShapedFormatter",
    "DataExporter",
    # Convenience functions
    "extract_document",
]

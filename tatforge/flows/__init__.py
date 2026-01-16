"""
CocoIndex flows for tatforge document processing.

This module contains module-level CocoIndex flow definitions following
the official pattern from cocoindex examples:
- multi_format_indexing: ColPali + Qdrant indexing
- patient_intake_extraction_baml: BAML extraction with caching

Usage:
    # Import flows (this registers them with CocoIndex at import time)
    from tatforge.flows import (
        document_indexing_flow,
        query_to_colpali_embedding,
        extract_with_baml,
    )

    # Initialize CocoIndex (required before using flows)
    import cocoindex
    cocoindex.init()

    # Run indexing via CLI
    # $ cocoindex setup
    # $ cocoindex update

    # Search documents
    query_embedding = query_to_colpali_embedding.eval("search query")

Architecture:
    INDEXING (cocoindex setup/update):
      pdfs/ -> LocalFile -> file_to_pages -> ColPaliEmbedImage -> Qdrant

    SEARCH + EXTRACT:
      Query -> ColPaliEmbedQuery -> Qdrant Search -> Page Images
            -> extract_with_baml (cached) -> Structured Output
"""

# Operations (module-level @cocoindex.op.function())
from .ops import file_to_pages, Page, COLPALI_MODEL

# Indexing flow (module-level @cocoindex.flow_def())
from .indexing import (
    document_indexing_flow,
    query_to_colpali_embedding,
    qdrant_connection,
    QDRANT_GRPC_URL,
    QDRANT_COLLECTION,
    PDF_PATH,
)

# Extraction functions (module-level @cocoindex.op.function(cache=True))
from .extraction import extract_with_baml, extract_with_schema

__all__ = [
    # Operations
    "file_to_pages",
    "Page",
    "COLPALI_MODEL",
    # Indexing
    "document_indexing_flow",
    "query_to_colpali_embedding",
    "qdrant_connection",
    "QDRANT_GRPC_URL",
    "QDRANT_COLLECTION",
    "PDF_PATH",
    # Extraction
    "extract_with_baml",
    "extract_with_schema",
]

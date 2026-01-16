"""
CocoIndex indexing flow for document processing.

FLOW 1: PDF -> pdf2image -> ColPali Embeddings -> Qdrant

All @cocoindex.flow_def() and @cocoindex.transform_flow() decorators
MUST be at module level per CocoIndex design pattern.

This follows the official pattern from:
https://github.com/cocoindex-io/cocoindex/blob/main/examples/multi_format_indexing/main.py
"""

import os
import logging

import cocoindex

from .ops import file_to_pages, COLPALI_MODEL

logger = logging.getLogger(__name__)

# Environment configuration
QDRANT_GRPC_URL = os.getenv("QDRANT_GRPC_URL", "http://localhost:6334")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "DocumentEmbeddings")
PDF_PATH = os.getenv("PDF_PATH", "pdfs")


# Module-level Qdrant connection registration
# This MUST be at module level, not inside a function
qdrant_connection = cocoindex.add_auth_entry(
    "qdrant_connection",
    cocoindex.targets.QdrantConnection(grpc_url=QDRANT_GRPC_URL),
)


@cocoindex.flow_def(name="DocumentIndexingFlow")
def document_indexing_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope,
) -> None:
    """
    Index documents with ColPali embeddings in Qdrant.

    Flow:
    1. Load PDF/image files from local directory (binary mode)
    2. Convert PDFs to page images at 300 DPI
    3. Generate ColPali embeddings for each page
    4. Store embeddings in Qdrant with metadata

    This flow is executed via `cocoindex setup` and `cocoindex update`.
    """
    # Load documents from local file source (binary mode for PDFs)
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path=PDF_PATH, binary=True)
    )

    # Collector for page embeddings
    output_embeddings = data_scope.add_collector()

    # Process each document
    with data_scope["documents"].row() as doc:
        # Convert file to pages (PDF->images or pass through images)
        doc["pages"] = flow_builder.transform(
            file_to_pages,
            filename=doc["filename"],
            content=doc["content"],
        )

        # Process each page
        with doc["pages"].row() as page:
            # Generate ColPali embedding for the page image
            page["embedding"] = page["image"].transform(
                cocoindex.functions.ColPaliEmbedImage(model=COLPALI_MODEL)
            )

            # Collect with metadata
            output_embeddings.collect(
                id=cocoindex.GeneratedField.UUID,
                filename=doc["filename"],
                page=page["page_number"],
                embedding=page["embedding"],
            )

    # Export to Qdrant
    output_embeddings.export(
        "document_embeddings",
        cocoindex.targets.Qdrant(
            connection=qdrant_connection,
            collection_name=QDRANT_COLLECTION,
        ),
        primary_key_fields=["id"],
    )

    logger.info(
        f"DocumentIndexingFlow configured: {PDF_PATH} -> {QDRANT_COLLECTION}"
    )


@cocoindex.transform_flow()
def query_to_colpali_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[list[list[float]]]:
    """
    Convert text query to ColPali multi-vector embedding.

    ColPali uses multi-vector embeddings (list of vectors) for
    late interaction retrieval, providing spatial awareness.

    This transform flow can be evaluated directly:
        query_embedding = query_to_colpali_embedding.eval("search query")

    Args:
        text: Text query to embed

    Returns:
        ColPali multi-vector embedding
    """
    return text.transform(
        cocoindex.functions.ColPaliEmbedQuery(model=COLPALI_MODEL)
    )

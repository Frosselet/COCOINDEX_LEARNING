"""
CocoIndex flow for document extraction and embedding.

This module defines the data flow orchestrated by CocoIndex:
1. Load PDF documents from local files (binary mode)
2. Extract structured data using BAML (native PDF support)
3. Generate embeddings for vector search
4. Store embeddings in Qdrant

Based on official CocoIndex patterns:
- https://github.com/cocoindex-io/cocoindex/tree/main/examples/patient_intake_extraction_baml
- https://github.com/cocoindex-io/cocoindex/tree/main/examples/text_embedding_qdrant
"""
import os
import base64
import functools
from typing import List

import baml_py
import cocoindex
from qdrant_client import QdrantClient

from baml_client import b
from baml_client.types import DocumentExtractionResult

# Configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "document_embeddings")
PDF_PATH = os.getenv("PDF_PATH", "pdfs")


@cocoindex.op.function(cache=True, behavior_version=1)
async def extract_document_fields(content: bytes, extraction_prompt: str) -> str:
    """
    Extract structured fields from a PDF document using BAML.

    BAML natively supports PDF inputs via baml_py.Pdf.from_base64().
    The extraction prompt defines what fields to extract.

    Args:
        content: Raw PDF bytes
        extraction_prompt: Prompt describing what to extract

    Returns:
        JSON string with extracted data
    """
    pdf_base64 = base64.b64encode(content).decode("utf-8")
    pdf = baml_py.Pdf.from_base64(pdf_base64)
    return await b.ExtractDocumentFieldsFromPDF(document=pdf, extraction_prompt=extraction_prompt)


@cocoindex.op.function(cache=True, behavior_version=1)
async def extract_with_schema(
    content: bytes,
    schema_description: str,
    field_names: List[str]
) -> DocumentExtractionResult:
    """
    Extract structured data from a PDF document using BAML with a defined schema.

    Args:
        content: Raw PDF bytes
        schema_description: Description of the document schema
        field_names: List of field names to extract

    Returns:
        DocumentExtractionResult with extracted fields, document type, and notes
    """
    pdf_base64 = base64.b64encode(content).decode("utf-8")
    pdf = baml_py.Pdf.from_base64(pdf_base64)
    return await b.ExtractFromPDF(
        document=pdf,
        schema_description=schema_description,
        field_names=field_names
    )


@cocoindex.transform_flow()
def text_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[list[float]]:
    """
    Embed text using SentenceTransformer model.

    This is shared between indexing and querying for consistency.
    Uses the all-MiniLM-L6-v2 model for efficient embeddings.
    """
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


@cocoindex.flow_def(name="DocumentExtractionWithQdrant")
def document_extraction_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope
) -> None:
    """
    Define a flow that extracts document data and stores embeddings in Qdrant.

    This flow:
    1. Reads PDF documents as binary from the configured path
    2. Extracts structured data using BAML's native PDF support
    3. Creates text embeddings from extracted content
    4. Stores results in Qdrant for vector search
    """
    # Load documents from local file source (binary mode for PDFs)
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path=PDF_PATH,
            binary=True
        )
    )

    # Create collector for document embeddings
    doc_embeddings = data_scope.add_collector()

    # Process each document
    with data_scope["documents"].row() as doc:
        # Extract key fields using BAML with a generic extraction prompt
        doc["extracted_text"] = doc["content"].transform(
            extract_document_fields,
            extraction_prompt="""
            Extract all text content from this document.
            Return as JSON with fields:
            - title: Document title if present
            - content: Main text content
            - metadata: Any relevant metadata (dates, authors, identifiers)
            """
        )

        # Create embedding from extracted text
        doc["embedding"] = text_to_embedding(doc["extracted_text"])

        # Collect document with embedding
        doc_embeddings.collect(
            id=cocoindex.GeneratedField.UUID,
            filename=doc["filename"],
            extracted_text=doc["extracted_text"],
            text_embedding=doc["embedding"],
        )

    # Export to Qdrant
    doc_embeddings.export(
        "doc_embeddings",
        cocoindex.targets.Qdrant(collection_name=QDRANT_COLLECTION),
        primary_key_fields=["id"],
    )


@functools.cache
def get_qdrant_client() -> QdrantClient:
    """Get cached Qdrant client for query operations."""
    return QdrantClient(url=QDRANT_URL)


@document_extraction_flow.query_handler(
    result_fields=cocoindex.QueryHandlerResultFields(
        embedding=["embedding"],
        score="score",
    ),
)
def search_documents(query: str) -> cocoindex.QueryOutput:
    """
    Search documents by semantic similarity.

    Args:
        query: Natural language search query

    Returns:
        QueryOutput with matching documents ranked by similarity
    """
    client = get_qdrant_client()

    # Get embedding for the query using same model as indexing
    query_embedding = text_to_embedding.eval(query)

    search_results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=("text_embedding", query_embedding),
        limit=10,
    )

    return cocoindex.QueryOutput(
        results=[
            {
                "filename": result.payload["filename"],
                "extracted_text": result.payload["extracted_text"],
                "embedding": result.vector,
                "score": result.score,
            }
            for result in search_results
            if result.payload is not None
        ],
        query_info=cocoindex.QueryInfo(
            embedding=query_embedding,
            similarity_metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
        ),
    )


def main() -> None:
    """Run the document extraction flow interactively."""
    from dotenv import load_dotenv
    load_dotenv()

    cocoindex.init()

    print("CocoIndex Document Extraction Flow")
    print("=" * 40)
    print(f"PDF Path: {PDF_PATH}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {QDRANT_COLLECTION}")
    print()

    # Interactive search loop
    while True:
        query = input("Enter search query (or 'quit' to exit): ")
        if query.lower() in ('quit', 'exit', 'q', ''):
            break

        try:
            results = search_documents(query)
            print(f"\nFound {len(results.results)} results:")
            for i, result in enumerate(results.results, 1):
                print(f"\n{i}. [{result['score']:.3f}] {result['filename']}")
                # Show first 200 chars of extracted text
                text = result.get('extracted_text', '')
                text_preview = text[:200] + "..." if len(text) > 200 else text
                print(f"   {text_preview}")
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()

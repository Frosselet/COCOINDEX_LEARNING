#!/usr/bin/env python3
"""
tatforge - CocoIndex-powered document extraction.

Entry point that initializes CocoIndex and provides CLI access.

Usage:
    # Interactive search
    python main.py

    # CocoIndex commands
    cocoindex setup    # Create Qdrant collection
    cocoindex update   # Index all PDFs

Architecture:
    INDEXING (cocoindex setup/update):
      pdfs/ -> LocalFile -> file_to_pages -> ColPaliEmbedImage -> Qdrant

    SEARCH + EXTRACT:
      Query -> ColPaliEmbedQuery -> Qdrant Search -> Page Images
            -> extract_with_baml (cached) -> Structured Output
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
import cocoindex
from qdrant_client import QdrantClient

# Import flows (this registers them with CocoIndex at import time)
from tatforge.flows import (
    document_indexing_flow,
    query_to_colpali_embedding,
    extract_with_baml,
    QDRANT_GRPC_URL,
    QDRANT_COLLECTION,
    PDF_PATH,
)

logger = logging.getLogger(__name__)


def search_documents(query: str, limit: int = 5) -> list[dict]:
    """
    Search indexed documents using ColPali embeddings.

    Args:
        query: Natural language search query
        limit: Maximum number of results to return

    Returns:
        List of search results with score, filename, and page number
    """
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_GRPC_URL, prefer_grpc=True)

    # Convert query to ColPali multi-vector embedding
    query_embedding = query_to_colpali_embedding.eval(query)

    # Search Qdrant
    search_results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        using="embedding",
        limit=limit,
        with_payload=True,
    )

    results = []
    for result in search_results.points:
        if result.payload is None:
            continue
        results.append({
            "score": result.score,
            "filename": result.payload.get("filename"),
            "page": result.payload.get("page"),
        })

    return results


async def extract_from_search_results(
    query: str,
    extraction_prompt: str,
    limit: int = 3,
) -> list[dict]:
    """
    Search for relevant pages and extract structured data.

    Args:
        query: Search query to find relevant pages
        extraction_prompt: Prompt describing what to extract
        limit: Number of pages to process

    Returns:
        List of extraction results
    """
    # Search for relevant pages
    search_results = search_documents(query, limit=limit)

    if not search_results:
        return []

    # Load page images and extract
    extractions = []
    for result in search_results:
        # TODO: Load actual page image from filename
        # This requires reading the PDF and extracting the specific page
        extractions.append({
            "source": result,
            "extraction": {"status": "requires_page_loading"},
        })

    return extractions


def _main() -> None:
    """Interactive query loop for document search."""
    print("=" * 60)
    print("tatforge - CocoIndex Document Search")
    print("=" * 60)
    print(f"Qdrant: {QDRANT_GRPC_URL}")
    print(f"Collection: {QDRANT_COLLECTION}")
    print(f"PDF Path: {PDF_PATH}")
    print("-" * 60)
    print("Commands:")
    print("  - Enter a search query to find documents")
    print("  - Type 'quit' or 'q' to exit")
    print("-" * 60)

    while True:
        try:
            query = input("\nSearch query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            if not query:
                continue

            print(f"\nSearching for: '{query}'...")
            results = search_documents(query, limit=5)

            if results:
                print(f"\nFound {len(results)} results:")
                for i, r in enumerate(results, 1):
                    page_str = f"page {r['page']}" if r["page"] else "single page"
                    print(f"  {i}. [{r['score']:.4f}] {r['filename']} ({page_str})")
            else:
                print("\nNo results found.")
                print("Tip: Run 'cocoindex update' to index documents first.")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception("Search error")


def main() -> int:
    """
    Main entry point for tatforge CocoIndex integration.

    Returns:
        Exit code (0 for success)
    """
    # Load environment variables
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize CocoIndex - REQUIRED before using any flows
    cocoindex.init()

    logger.info("CocoIndex initialized")
    logger.info(f"PDF Path: {PDF_PATH}")
    logger.info(f"Qdrant: {QDRANT_GRPC_URL}")
    logger.info(f"Collection: {QDRANT_COLLECTION}")

    # Run interactive search
    _main()

    return 0


if __name__ == "__main__":
    sys.exit(main())

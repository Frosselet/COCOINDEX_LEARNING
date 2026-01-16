"""
tatForge CLI - Command-line interface for AI-powered document extraction.

This module provides the main entry point for the tatforge command-line tool.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .core.pipeline import VisionExtractionPipeline, PipelineConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for CLI operations."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="tatforge",
        description="tatForge - AI-powered document extraction using vision AI",
        epilog="For more information, visit: https://github.com/Frosselet/COCOINDEX_LEARNING",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version="tatforge 0.1.0",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract structured data from documents",
    )
    extract_parser.add_argument(
        "input",
        type=Path,
        help="Input document path (PDF or image)",
    )
    extract_parser.add_argument(
        "-s", "--schema",
        type=Path,
        required=True,
        help="JSON schema file defining extraction structure",
    )
    extract_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    extract_parser.add_argument(
        "--format",
        choices=["json", "csv", "parquet"],
        default="json",
        help="Output format (default: json)",
    )
    extract_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing (default: auto)",
    )
    extract_parser.add_argument(
        "--memory-limit",
        type=int,
        help="Memory limit in GB",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display package and model information",
    )
    info_parser.add_argument(
        "--models",
        action="store_true",
        help="Show available models",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a schema file",
    )
    validate_parser.add_argument(
        "schema",
        type=Path,
        help="Schema file to validate",
    )

    # CocoIndex command
    cocoindex_parser = subparsers.add_parser(
        "cocoindex",
        help="Run CocoIndex operations (setup, update, server)",
    )
    cocoindex_parser.add_argument(
        "action",
        choices=["setup", "update", "server", "search"],
        help="CocoIndex action to perform",
    )
    cocoindex_parser.add_argument(
        "-q", "--query",
        type=str,
        help="Search query (for search action)",
    )

    return parser.parse_args()


async def cmd_extract(args: argparse.Namespace) -> int:
    """Execute the extract command."""
    logger = logging.getLogger(__name__)

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load schema
    if not args.schema.exists():
        logger.error(f"Schema file not found: {args.schema}")
        return 1

    try:
        with open(args.schema) as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON schema: {e}")
        return 1

    # Read document
    try:
        with open(args.input, "rb") as f:
            document_blob = f.read()
    except IOError as e:
        logger.error(f"Failed to read input file: {e}")
        return 1

    logger.info(f"Processing document: {args.input}")
    logger.info(f"Using schema: {args.schema}")

    # Create pipeline configuration
    config = PipelineConfig(
        memory_limit_gb=args.memory_limit,
        batch_size=args.batch_size or "auto",
    )

    try:
        # Initialize pipeline
        pipeline = VisionExtractionPipeline(config=config)

        # Process document
        result = await pipeline.process_document(
            document_blob=document_blob,
            schema_json=schema,
        )

        # Format output
        output_data = result.canonical.extraction_data if result.canonical else {}

        if args.format == "json":
            output_str = json.dumps(output_data, indent=2, default=str)
        else:
            # CSV/Parquet would require pandas
            logger.warning(f"Format {args.format} not yet implemented, using JSON")
            output_str = json.dumps(output_data, indent=2, default=str)

        # Write output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_str)
            logger.info(f"Output written to: {args.output}")
        else:
            print(output_str)

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    info = {
        "package": "tatforge",
        "version": "0.1.0",
        "description": "AI-powered document extraction using vision AI",
        "components": {
            "vision": "ColPali (ColQwen2-v0.1, 3B parameters)",
            "extraction": "BAML type-safe schemas",
            "storage": "Qdrant vector database",
        },
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"],
    }

    if args.models:
        info["available_models"] = [
            {
                "name": "vidore/colqwen2-v0.1",
                "parameters": "3B",
                "embedding_dim": 128,
                "patch_size": "32x32",
            }
        ]

    print(json.dumps(info, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    logger = logging.getLogger(__name__)

    if not args.schema.exists():
        logger.error(f"Schema file not found: {args.schema}")
        return 1

    try:
        with open(args.schema) as f:
            schema = json.load(f)

        # Basic schema validation
        required_keys = ["type", "properties"]
        missing = [k for k in required_keys if k not in schema]

        if missing:
            logger.error(f"Schema missing required keys: {missing}")
            return 1

        if schema.get("type") != "object":
            logger.error("Schema type must be 'object'")
            return 1

        properties = schema.get("properties", {})
        if not properties:
            logger.warning("Schema has no properties defined")

        print(f"Schema is valid: {len(properties)} properties defined")
        return 0

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return 1


def cmd_cocoindex(args: argparse.Namespace) -> int:
    """Execute CocoIndex operations."""
    logger = logging.getLogger(__name__)

    try:
        from dotenv import load_dotenv
        import cocoindex

        # Load environment variables
        load_dotenv()

        # Initialize CocoIndex
        cocoindex.init()

        # Import flows (registers them at import time)
        from tatforge.flows import (
            document_indexing_flow,
            query_to_colpali_embedding,
            QDRANT_GRPC_URL,
            QDRANT_COLLECTION,
            PDF_PATH,
        )

        logger.info(f"CocoIndex initialized")
        logger.info(f"  PDF Path: {PDF_PATH}")
        logger.info(f"  Qdrant: {QDRANT_GRPC_URL}")
        logger.info(f"  Collection: {QDRANT_COLLECTION}")

        if args.action == "search":
            if not args.query:
                logger.error("Search requires --query argument")
                return 1

            from qdrant_client import QdrantClient

            print(f"\nSearching for: '{args.query}'...")
            query_embedding = query_to_colpali_embedding.eval(args.query)

            client = QdrantClient(url=QDRANT_GRPC_URL, prefer_grpc=True)
            search_results = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_embedding,
                using="embedding",
                limit=5,
                with_payload=True,
            )

            if search_results.points:
                print(f"\nFound {len(search_results.points)} results:")
                for i, result in enumerate(search_results.points, 1):
                    if result.payload:
                        page = result.payload.get("page", "?")
                        filename = result.payload.get("filename", "unknown")
                        print(f"  {i}. [{result.score:.4f}] {filename} (page {page})")
            else:
                print("\nNo results found.")
                print("Tip: Run 'tatforge cocoindex update' to index documents first.")
            return 0

        else:
            # Delegate to cocoindex CLI for setup, update, server
            print(f"\nRunning: cocoindex {args.action}")
            print("-" * 40)
            # CocoIndex CLI handles the rest
            import sys
            sys.argv = ["cocoindex", args.action]
            cocoindex.cli_main()
            return 0

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install cocoindex python-dotenv")
        return 1
    except Exception as e:
        logger.error(f"CocoIndex error: {e}")
        return 1


def main() -> int:
    """Main entry point for tatforge CLI."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    # Dispatch command
    if args.command == "extract":
        return asyncio.run(cmd_extract(args))
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "cocoindex":
        return cmd_cocoindex(args)
    else:
        # No command provided, show help
        print("tatForge - AI-powered document extraction")
        print()
        print("Usage: tatforge <command> [options]")
        print()
        print("Commands:")
        print("  extract    Extract structured data from documents")
        print("  info       Display package and model information")
        print("  validate   Validate a schema file")
        print("  cocoindex  Run CocoIndex operations (setup, update, server, search)")
        print()
        print("CocoIndex commands:")
        print("  tatforge cocoindex setup     Create Qdrant collection")
        print("  tatforge cocoindex update    Index all PDFs with ColPali")
        print("  tatforge cocoindex search -q 'query'  Search documents")
        print()
        print("Run 'tatforge <command> --help' for command-specific help.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

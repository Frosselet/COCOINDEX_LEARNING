"""
Vector database and storage components.

This module handles Qdrant vector database operations for storing and
querying ColPali embeddings with spatial metadata and document lineage.
"""

from .qdrant_client import QdrantManager

__all__ = [
    "QdrantManager",
]
"""
Vision processing components.

This module handles ColPali model integration, image processing utilities,
and patch-level embedding generation for semantic document retrieval.
"""

from .colpali_client import ColPaliClient
from .image_processor import ImageProcessor

__all__ = [
    "ColPaliClient",
    "ImageProcessor",
]
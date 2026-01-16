"""
CocoIndex operations for document processing.

All @cocoindex.op.function() decorators MUST be at module level.
This follows the official pattern from:
https://github.com/cocoindex-io/cocoindex/blob/main/examples/multi_format_indexing/main.py
"""

import os
import mimetypes
import logging
from dataclasses import dataclass
from io import BytesIO

import cocoindex
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)

# Environment configuration
COLPALI_MODEL = os.getenv("COLPALI_MODEL", "vidore/colqwen2-v0.1")


@dataclass
class Page:
    """Represents a single page from a document for ColPali embedding."""

    page_number: int | None
    image: bytes


@cocoindex.op.function()
def file_to_pages(filename: str, content: bytes) -> list[Page]:
    """
    Convert document files to a list of page images.

    - PDFs are converted to PNG images at 300 DPI
    - Image files are passed through directly
    - Other file types return empty list

    Args:
        filename: Name of the file (used for MIME type detection)
        content: Raw bytes of the file content

    Returns:
        List of Page objects with page_number and image bytes
    """
    mime_type, _ = mimetypes.guess_type(filename)

    if mime_type == "application/pdf":
        logger.debug(f"Converting PDF to images: {filename}")
        images = convert_from_bytes(content, dpi=300)
        pages = []
        for i, image in enumerate(images):
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                pages.append(Page(page_number=i + 1, image=buffer.getvalue()))
        logger.info(f"Converted {filename} to {len(pages)} page images")
        return pages
    elif mime_type and mime_type.startswith("image/"):
        logger.debug(f"Passing through image file: {filename}")
        return [Page(page_number=None, image=content)]
    else:
        logger.warning(f"Unsupported file type for {filename}: {mime_type}")
        return []

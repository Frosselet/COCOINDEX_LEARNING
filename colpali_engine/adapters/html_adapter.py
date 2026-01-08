"""
HTML document adapter for web content processing.

This adapter converts HTML documents to image frames using a headless browser
approach, preserving visual layout and styling for ColPali processing.
"""

import logging
import asyncio
import io
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import re

from PIL import Image

from ..core.document_adapter import (
    BaseDocumentAdapter,
    DocumentFormat,
    DocumentMetadata,
    ConversionConfig,
    DocumentProcessingError,
    MetadataExtractionError,
    ValidationError
)

logger = logging.getLogger(__name__)


class HTMLAdapter(BaseDocumentAdapter):
    """
    HTML document adapter for web content processing.

    Converts HTML documents to image frames using either:
    1. wkhtmltopdf (if available) for high-quality rendering
    2. Playwright/Selenium for browser-based rendering
    3. Basic HTML parsing with layout approximation (fallback)

    Preserves visual layout, CSS styling, and responsive design.
    """

    def __init__(self, viewport_width: int = 1200, viewport_height: int = 800):
        """
        Initialize HTML adapter.

        Args:
            viewport_width: Browser viewport width for rendering
            viewport_height: Browser viewport height for rendering
        """
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.rendering_method = self._detect_rendering_method()

        logger.info(f"HTMLAdapter initialized with viewport {viewport_width}x{viewport_height}, "
                   f"rendering method: {self.rendering_method}")

    @property
    def supported_format(self) -> DocumentFormat:
        """The document format this adapter supports."""
        return DocumentFormat.HTML

    async def convert_to_frames(
        self,
        content: bytes,
        config: Optional[ConversionConfig] = None
    ) -> List[Image.Image]:
        """
        Convert HTML content to image frames.

        Args:
            content: Raw HTML bytes
            config: Optional conversion configuration

        Returns:
            List of PIL Image objects representing rendered HTML pages

        Raises:
            DocumentProcessingError: If conversion fails
        """
        config = config or ConversionConfig()

        try:
            # Validate HTML format first
            if not self.validate_format(content):
                raise DocumentProcessingError("Invalid HTML format")

            html_content = content.decode('utf-8', errors='ignore')
            logger.info(f"Converting HTML content using {self.rendering_method}")

            # Choose rendering method
            if self.rendering_method == "wkhtmltopdf":
                frames = await self._render_with_wkhtmltopdf(html_content, config)
            elif self.rendering_method == "playwright":
                frames = await self._render_with_playwright(html_content, config)
            else:
                # Fallback to basic text extraction and layout approximation
                frames = await self._render_with_fallback(html_content, config)

            # Limit frames if configured
            if config.max_pages and len(frames) > config.max_pages:
                frames = frames[:config.max_pages]

            logger.info(f"Successfully converted HTML to {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"HTML conversion failed: {e}")
            raise DocumentProcessingError(f"Failed to convert HTML: {e}")

    def extract_metadata(self, content: bytes) -> DocumentMetadata:
        """
        Extract metadata from HTML content.

        Args:
            content: Raw HTML bytes

        Returns:
            DocumentMetadata with HTML properties

        Raises:
            MetadataExtractionError: If metadata extraction fails
        """
        try:
            html_content = content.decode('utf-8', errors='ignore')

            # Initialize metadata
            file_size = len(content)
            page_count = 1  # HTML is typically single page
            title = None
            author = None
            creation_date = None

            # Parse HTML for metadata
            try:
                # Extract title
                title_match = re.search(r'<title[^>]*>([^<]*)</title>', html_content, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()[:200]  # Limit length

                # Extract meta tags
                meta_tags = re.findall(r'<meta[^>]+>', html_content, re.IGNORECASE)

                for meta_tag in meta_tags:
                    # Author
                    if 'name="author"' in meta_tag.lower() or 'property="author"' in meta_tag.lower():
                        content_match = re.search(r'content=["\']([^"\']*)["\']', meta_tag, re.IGNORECASE)
                        if content_match and not author:
                            author = content_match.group(1).strip()[:100]

                    # Creation date
                    elif any(date_field in meta_tag.lower() for date_field in [
                        'name="date"', 'name="created"', 'property="article:published_time"'
                    ]):
                        content_match = re.search(r'content=["\']([^"\']*)["\']', meta_tag, re.IGNORECASE)
                        if content_match and not creation_date:
                            date_str = content_match.group(1).strip()
                            creation_date = self._parse_html_date(date_str)

                # Estimate page count based on content length (very rough)
                content_length = len(re.sub(r'<[^>]*>', '', html_content).strip())
                if content_length > 5000:  # Arbitrary threshold for "long" content
                    page_count = max(1, content_length // 3000)  # ~3000 chars per "page"

                # Try to detect dimensions from viewport meta tag
                dimensions = None
                viewport_match = re.search(r'<meta[^>]*name=["\']viewport["\'][^>]*content=["\']([^"\']*)["\']',
                                         html_content, re.IGNORECASE)
                if viewport_match:
                    viewport_content = viewport_match.group(1)
                    width_match = re.search(r'width=(\d+)', viewport_content)
                    if width_match:
                        width = int(width_match.group(1))
                        dimensions = (width, self.viewport_height)

                if not dimensions:
                    dimensions = (self.viewport_width, self.viewport_height)

            except Exception as e:
                logger.debug(f"HTML metadata parsing failed: {e}")

            logger.debug(f"Extracted HTML metadata: {file_size} bytes, title='{title}'")

            return DocumentMetadata(
                format=DocumentFormat.HTML,
                page_count=page_count,
                dimensions=dimensions,
                file_size=file_size,
                creation_date=creation_date,
                title=title,
                author=author
            )

        except Exception as e:
            logger.error(f"HTML metadata extraction failed: {e}")
            raise MetadataExtractionError(f"Failed to extract HTML metadata: {e}")

    def validate_format(self, content: bytes) -> bool:
        """
        Validate that the content is valid HTML.

        Args:
            content: Raw HTML bytes

        Returns:
            True if format is valid HTML
        """
        try:
            # Check for HTML indicators
            content_lower = content[:1000].lower()

            # Look for common HTML patterns
            html_indicators = [
                b'<!doctype html',
                b'<html',
                b'<head>',
                b'<body>',
                b'<title>',
                b'<meta',
                b'<div',
                b'<p>',
                b'<h1>',
                b'<script'
            ]

            # Check if content contains HTML-like structure
            if any(indicator in content_lower for indicator in html_indicators):
                # Additional validation: try to decode as text
                try:
                    content.decode('utf-8', errors='strict')
                    return True
                except UnicodeDecodeError:
                    # Try with error handling
                    try:
                        content.decode('utf-8', errors='ignore')
                        return True
                    except Exception:
                        return False

            return False

        except Exception as e:
            logger.debug(f"HTML validation failed: {e}")
            return False

    def _detect_rendering_method(self) -> str:
        """Detect available rendering method."""
        # Check for wkhtmltopdf
        try:
            import subprocess
            subprocess.run(['wkhtmltopdf', '--version'], capture_output=True, timeout=5)
            return "wkhtmltopdf"
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        # Check for playwright
        try:
            import playwright
            return "playwright"
        except ImportError:
            pass

        # Check for selenium
        try:
            import selenium
            return "selenium"
        except ImportError:
            pass

        # Fallback to basic rendering
        logger.warning("No advanced HTML rendering tools available, using fallback method")
        return "fallback"

    async def _render_with_wkhtmltopdf(self, html_content: str, config: ConversionConfig) -> List[Image.Image]:
        """Render HTML using wkhtmltopdf."""
        try:
            import subprocess
            from pdf2image import convert_from_bytes

            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as html_file:
                html_file.write(html_content)
                html_file_path = html_file.name

            try:
                # Convert HTML to PDF using wkhtmltopdf
                cmd = [
                    'wkhtmltopdf',
                    '--page-size', 'A4',
                    '--viewport-size', f'{self.viewport_width}x{self.viewport_height}',
                    '--disable-smart-shrinking',
                    '--print-media-type',
                    html_file_path,
                    '-'  # Output to stdout
                ]

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(cmd, capture_output=True, timeout=30)
                )

                if result.returncode != 0:
                    raise DocumentProcessingError(f"wkhtmltopdf failed: {result.stderr.decode()}")

                # Convert PDF bytes to images
                images = convert_from_bytes(result.stdout, dpi=config.dpi)
                return images

            finally:
                # Clean up temporary file
                Path(html_file_path).unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"wkhtmltopdf rendering failed: {e}")
            # Fall back to basic rendering
            return await self._render_with_fallback(html_content, config)

    async def _render_with_playwright(self, html_content: str, config: ConversionConfig) -> List[Image.Image]:
        """Render HTML using Playwright headless browser."""
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    viewport={'width': self.viewport_width, 'height': self.viewport_height}
                )

                # Set content and wait for load
                await page.set_content(html_content, wait_until='networkidle')

                # Take screenshot
                screenshot_bytes = await page.screenshot(full_page=True, type='png')

                await browser.close()

                # Convert screenshot to PIL Image
                with io.BytesIO(screenshot_bytes) as image_stream:
                    image = Image.open(image_stream)
                    image.load()

                return [image]

        except Exception as e:
            logger.warning(f"Playwright rendering failed: {e}")
            # Fall back to basic rendering
            return await self._render_with_fallback(html_content, config)

    async def _render_with_fallback(self, html_content: str, config: ConversionConfig) -> List[Image.Image]:
        """Basic fallback rendering using text extraction and layout approximation."""
        try:
            from PIL import ImageDraw, ImageFont

            # Extract text content from HTML
            text_content = self._extract_text_from_html(html_content)

            # Create a basic image with the text
            img_width = min(self.viewport_width, config.dpi * 8)  # ~8 inches at specified DPI
            img_height = min(self.viewport_height, config.dpi * 11)  # ~11 inches

            # Create white background image
            image = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(image)

            # Try to load a decent font
            try:
                # Try system fonts
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=24)
            except Exception:
                try:
                    font = ImageFont.truetype("arial.ttf", size=24)
                except Exception:
                    font = ImageFont.load_default()

            # Draw text with basic wrapping
            y_position = 50
            line_height = 30
            margin = 50

            for line in text_content.split('\n'):
                if y_position > img_height - margin:
                    break  # Don't exceed image height

                # Basic text wrapping
                words = line.split()
                current_line = ""

                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    text_width = bbox[2] - bbox[0]

                    if text_width <= img_width - (margin * 2):
                        current_line = test_line
                    else:
                        if current_line:
                            draw.text((margin, y_position), current_line, fill='black', font=font)
                            y_position += line_height
                            current_line = word
                        else:
                            # Word is too long, draw it anyway
                            draw.text((margin, y_position), word, fill='black', font=font)
                            y_position += line_height

                # Draw remaining text in current_line
                if current_line:
                    draw.text((margin, y_position), current_line, fill='black', font=font)
                    y_position += line_height

            logger.info("Generated fallback HTML rendering")
            return [image]

        except Exception as e:
            logger.error(f"Fallback rendering failed: {e}")
            # Create a minimal error image
            error_image = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(error_image)
            draw.text((50, 50), "HTML content could not be rendered", fill='black')
            return [error_image]

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract readable text from HTML content."""
        try:
            # Remove scripts and styles
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags but keep structure
            html_content = re.sub(r'<br[^>]*>', '\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<p[^>]*>', '\n\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<h[1-6][^>]*>', '\n\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<div[^>]*>', '\n', html_content, flags=re.IGNORECASE)

            # Remove all remaining HTML tags
            text = re.sub(r'<[^>]+>', '', html_content)

            # Clean up whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space

            return text.strip()

        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            return "HTML content extraction failed"

    def _parse_html_date(self, date_str: str) -> Optional[str]:
        """
        Parse various HTML date formats to ISO format.

        Args:
            date_str: Date string from HTML meta tags

        Returns:
            ISO format date string or None if parsing fails
        """
        try:
            # Try common date formats
            for fmt in [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y",
                "%d/%m/%Y",
            ]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue

            return None

        except Exception as e:
            logger.debug(f"Failed to parse HTML date '{date_str}': {e}")
            return None


# Factory function for easy instantiation
def create_html_adapter(viewport_width: int = 1200, viewport_height: int = 800) -> HTMLAdapter:
    """
    Create a configured HTML adapter.

    Args:
        viewport_width: Browser viewport width for rendering
        viewport_height: Browser viewport height for rendering

    Returns:
        Configured HTMLAdapter instance
    """
    return HTMLAdapter(viewport_width=viewport_width, viewport_height=viewport_height)
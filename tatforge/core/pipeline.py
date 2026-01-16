"""
Main orchestration pipeline for vision-based document processing.

This module implements the CocoIndex-orchestrated pipeline that coordinates
document processing, ColPali vision processing, BAML extraction, and output
generation with complete lineage tracking.
"""

import asyncio
import io
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from pdf2image import convert_from_bytes
import torch

from ..extraction.models import ExtractionResult, CanonicalData, ShapedData, ProcessingMetadata, ProcessingStatus
from ..vision.colpali_client import ColPaliClient
from ..storage.qdrant_client import QdrantManager
from ..extraction.baml_interface import BAMLExecutionInterface
from ..outputs.canonical import CanonicalFormatter
from ..outputs.shaped import ShapedFormatter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the vision extraction pipeline."""

    # Document processing
    image_dpi: int = 300
    image_format: str = "RGB"

    # ColPali processing
    batch_size: str = "auto"  # or specific number
    memory_limit_gb: int = 8

    # BAML processing
    retry_attempts: int = 3
    timeout_seconds: int = 300

    # Output configuration
    enable_shaped_output: bool = True
    enforce_1nf: bool = True

    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24


class VisionExtractionPipeline:
    """
    Main orchestration pipeline for vision-based document processing.

    This pipeline coordinates the entire flow from document input to structured
    output, using CocoIndex for workflow orchestration and dependency management.
    """

    def __init__(
        self,
        colpali_client: ColPaliClient,
        qdrant_manager: QdrantManager,
        baml_interface: BAMLExecutionInterface,
        canonical_formatter: CanonicalFormatter,
        shaped_formatter: Optional[ShapedFormatter] = None,
        config: Optional[PipelineConfig] = None
    ):
        self.colpali_client = colpali_client
        self.qdrant_manager = qdrant_manager
        self.baml_interface = baml_interface
        self.canonical_formatter = canonical_formatter
        self.shaped_formatter = shaped_formatter
        self.config = config or PipelineConfig()

        # Processing state
        self.processing_id: Optional[str] = None
        self.lineage_steps: List[Dict] = []

        logger.info("VisionExtractionPipeline initialized")

    async def process_document(
        self,
        document_blob: Union[bytes, List[Image.Image]],
        schema_json: Dict,
        document_metadata: Optional[Dict] = None
    ) -> ExtractionResult:
        """
        Process a document through the complete vision extraction pipeline.

        Args:
            document_blob: Raw document bytes (PDF) OR list of pre-converted PIL Images
            schema_json: JSON schema defining desired output structure
            document_metadata: Optional metadata about the document

        Returns:
            ExtractionResult with canonical and optional shaped data
        """
        # Initialize processing session
        self.processing_id = str(uuid.uuid4())
        self.lineage_steps = []
        start_time = datetime.now()

        logger.info(f"Starting document processing: {self.processing_id}")

        try:
            # Stage 1: Document to Image Conversion
            image_frames = await self._stage_document_processing(document_blob)

            # Stage 2: ColPali Vision Processing
            embeddings = await self._stage_vision_processing(image_frames)

            # Stage 3: Vector Storage (parallel with schema processing)
            storage_task = asyncio.create_task(
                self._stage_vector_storage(embeddings, document_metadata)
            )

            # Stage 4: Schema Processing (parallel with storage)
            baml_definition = await self._stage_schema_processing(schema_json)

            # Wait for storage to complete
            await storage_task

            # Stage 5: Semantic Retrieval and Extraction
            extraction_data = await self._stage_extraction(schema_json, baml_definition)

            # Stage 6: Output Generation
            canonical_data = await self._stage_canonical_output(extraction_data)
            shaped_data = None

            if self.config.enable_shaped_output and self.shaped_formatter:
                shaped_data = await self._stage_shaped_output(canonical_data)

            # Create processing metadata
            end_time = datetime.now()
            processing_metadata = ProcessingMetadata(
                processing_id=self.processing_id,
                start_time=start_time,
                end_time=end_time,
                processing_time_seconds=(end_time - start_time).total_seconds(),
                lineage_steps=self.lineage_steps,
                config=self.config.__dict__
            )

            result = ExtractionResult(
                canonical=canonical_data,
                shaped=shaped_data,
                metadata=processing_metadata
            )

            logger.info(f"Document processing completed: {self.processing_id}")
            return result

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            await self._cleanup_resources()
            raise

    async def _stage_document_processing(self, document_blob: Union[bytes, List[Image.Image]]) -> List[Image.Image]:
        """Stage 1: Convert document to canonical image frames."""
        step_start = datetime.now()

        logger.info("Stage 1: Document processing")

        # If already converted to images, use them directly
        if isinstance(document_blob, list) and all(isinstance(img, Image.Image) for img in document_blob):
            logger.info(f"Using {len(document_blob)} pre-converted image frames")
            image_frames = document_blob
            input_size = sum(img.size[0] * img.size[1] * 3 for img in document_blob)  # Approximate size
        else:
            # Convert PDF bytes to images using pdf2image
            logger.info("Converting PDF bytes to image frames...")
            try:
                loop = asyncio.get_event_loop()
                image_frames = await loop.run_in_executor(
                    None,
                    lambda: convert_from_bytes(
                        document_blob,
                        dpi=self.config.image_dpi,
                        fmt='RGB'
                    )
                )
                input_size = len(document_blob)
                logger.info(f"Converted PDF to {len(image_frames)} image frames")
            except Exception as e:
                logger.error(f"PDF conversion failed: {e}")
                raise RuntimeError(f"Failed to convert document to images: {e}")

        # Ensure all images are in RGB mode
        processed_frames = []
        for img in image_frames:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_frames.append(img)

        self.lineage_steps.append({
            "stage": "document_processing",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "input_size_bytes": input_size,
            "output_frames": len(processed_frames)
        })

        return processed_frames

    async def _stage_vision_processing(self, image_frames: List[Image.Image]) -> List[torch.Tensor]:
        """Stage 2: Generate ColPali embeddings from image frames."""
        step_start = datetime.now()

        logger.info(f"Stage 2: ColPali processing {len(image_frames)} frames")

        if not image_frames:
            logger.warning("No image frames to process")
            return []

        try:
            # Ensure ColPali model is loaded
            if not self.colpali_client.is_loaded:
                logger.info("Loading ColPali model...")
                await self.colpali_client.load_model()

            # Calculate batch size
            batch_size = None
            if self.config.batch_size != "auto":
                batch_size = int(self.config.batch_size)

            # Generate embeddings for all image frames
            embeddings = await self.colpali_client.embed_frames(
                images=image_frames,
                batch_size=batch_size
            )

            logger.info(f"Generated embeddings for {len(embeddings)} frames")

            # Store image frames for later extraction
            self._current_image_frames = image_frames

        except Exception as e:
            logger.error(f"ColPali embedding generation failed: {e}")
            raise RuntimeError(f"Vision processing failed: {e}")

        self.lineage_steps.append({
            "stage": "vision_processing",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "input_frames": len(image_frames),
            "output_embeddings": len(embeddings),
            "batch_size": str(self.config.batch_size)
        })

        return embeddings

    async def _stage_vector_storage(self, embeddings: List[torch.Tensor], metadata: Optional[Dict]) -> None:
        """Stage 3: Store embeddings in Qdrant vector database."""
        step_start = datetime.now()

        logger.info(f"Stage 3: Storing {len(embeddings)} embeddings in Qdrant")

        if not embeddings:
            logger.warning("No embeddings to store")
            self.lineage_steps.append({
                "stage": "vector_storage",
                "start_time": step_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "embeddings_stored": 0
            })
            return

        try:
            # Prepare embeddings for storage
            # Each embedding tensor is [num_patches, embedding_dim]
            all_vectors = []
            patch_metadata = []

            for page_idx, page_embedding in enumerate(embeddings):
                # Convert tensor to list of vectors
                if isinstance(page_embedding, torch.Tensor):
                    page_vectors = page_embedding.cpu().numpy().tolist()
                else:
                    page_vectors = page_embedding

                for patch_idx, vector in enumerate(page_vectors):
                    all_vectors.append(vector)
                    patch_metadata.append({
                        "page_number": page_idx,
                        "patch_index": patch_idx
                    })

            # Build storage metadata with all required fields
            storage_metadata = {
                "document_id": self.processing_id,
                "page_number": 0,  # Default to page 0 for single-page documents
                "document_type": metadata.get("document_type", "pdf") if metadata else "pdf",
                "processing_timestamp": datetime.now().isoformat(),
                "patch_coordinates": [(m["page_number"], m["patch_index"]) for m in patch_metadata],
                **(metadata or {})
            }

            # Store in Qdrant
            result = await self.qdrant_manager.store_embeddings(
                embeddings=all_vectors,
                metadata=storage_metadata
            )

            logger.info(f"Stored {len(all_vectors)} patch embeddings in Qdrant")

        except Exception as e:
            logger.error(f"Qdrant storage failed: {e}")
            # Don't fail the pipeline for storage errors - extraction can still proceed
            logger.warning("Continuing pipeline despite storage error")

        self.lineage_steps.append({
            "stage": "vector_storage",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "embeddings_stored": len(all_vectors) if 'all_vectors' in locals() else 0
        })

    async def _stage_schema_processing(self, schema_json: Dict) -> Dict:
        """Stage 4: Convert JSON schema to BAML classes and functions."""
        step_start = datetime.now()

        logger.info("Stage 4: Schema processing and BAML generation")

        try:
            from .schema_manager import SchemaManager

            schema_manager = SchemaManager()
            baml_definition = schema_manager.generate_baml_classes(schema_json)

            # Store for extraction stage
            self._current_baml_definition = baml_definition
            self._current_schema = schema_json

            logger.info(f"Generated BAML definition with {len(baml_definition.classes)} classes")

        except Exception as e:
            logger.warning(f"BAML generation failed: {e}, will use direct extraction")
            self._current_baml_definition = None
            self._current_schema = schema_json
            baml_definition = {"schema": schema_json}

        self.lineage_steps.append({
            "stage": "schema_processing",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "schema_fields": len(schema_json.get("properties", {}))
        })

        return baml_definition

    async def _stage_extraction(self, schema_json: Dict, baml_definition: Dict) -> Dict:
        """Stage 5: Execute BAML extraction with semantic retrieval."""
        step_start = datetime.now()

        logger.info("Stage 5: Semantic retrieval and BAML extraction")

        extraction_data = {}

        try:
            # Check if we have images and BAML interface
            if not hasattr(self, '_current_image_frames') or not self._current_image_frames:
                logger.warning("No image frames available for extraction")
                return {}

            images = self._current_image_frames
            baml_def = getattr(self, '_current_baml_definition', None)

            # Try BAML interface extraction if available
            if self.baml_interface and baml_def and baml_def.functions:
                from ..extraction.baml_interface import ExtractionRequest, ExtractionContext

                # Create extraction request
                context = ExtractionContext(
                    document_id=self.processing_id,
                    document_type="pdf",
                    processing_config=self.config.__dict__
                )

                request = ExtractionRequest(
                    function=baml_def.functions[0],  # Use first extraction function
                    images=images,
                    context=context,
                    timeout_seconds=self.config.timeout_seconds,
                    max_retries=self.config.retry_attempts
                )

                # Execute extraction
                result = await self.baml_interface.execute_extraction(request)

                if result.canonical and result.canonical.extraction_data:
                    extraction_data = result.canonical.extraction_data
                    logger.info(f"BAML extraction successful: {len(extraction_data)} fields")
                else:
                    logger.warning("BAML extraction returned no data")

            else:
                # Fallback: Direct vision extraction without BAML
                logger.info("Using direct vision extraction (BAML not available)")
                extraction_data = await self._direct_vision_extraction(images, schema_json)

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Return empty dict rather than failing the pipeline
            extraction_data = {"extraction_error": str(e)}

        self.lineage_steps.append({
            "stage": "extraction",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "extraction_fields": len(extraction_data)
        })

        return extraction_data

    async def _direct_vision_extraction(self, images: List[Image.Image], schema_json: Dict) -> Dict:
        """
        Perform direct vision extraction without BAML.

        This is a fallback method that uses a simpler approach when BAML
        is not available or not configured.
        """
        logger.info("Performing direct vision extraction")

        # For now, return a placeholder indicating direct extraction is needed
        # In a full implementation, this would call a vision API directly
        return {
            "extraction_method": "direct_vision",
            "image_count": len(images),
            "schema_fields": list(schema_json.get("properties", {}).keys()),
            "status": "requires_vision_api_configuration",
            "message": "Configure BAML client with vision API (e.g., Claude, GPT-4V) for full extraction"
        }

    async def _stage_canonical_output(self, extraction_data: Dict) -> CanonicalData:
        """Stage 6a: Generate canonical truth layer output."""
        step_start = datetime.now()

        logger.info("Stage 6a: Canonical output generation")

        # Create canonical data with extraction results
        canonical_data = CanonicalData(
            processing_id=self.processing_id,
            extraction_data=extraction_data,
            timestamp=datetime.now(),
            source_metadata={
                "pipeline_version": "1.0",
                "extraction_method": extraction_data.get("extraction_method", "baml"),
                "image_count": extraction_data.get("image_count", len(getattr(self, '_current_image_frames', [])))
            }
        )

        self.lineage_steps.append({
            "stage": "canonical_output",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "output_fields": len(extraction_data)
        })

        return canonical_data

    async def _stage_shaped_output(self, canonical_data: CanonicalData) -> ShapedData:
        """Stage 6b: Generate shaped output with 1NF enforcement."""
        step_start = datetime.now()

        logger.info("Stage 6b: Shaped output generation with 1NF enforcement")

        # Apply 1NF transformation to canonical data
        transformed_data = canonical_data.extraction_data.copy() if canonical_data.extraction_data else {}
        transformations = []

        # Basic 1NF: flatten nested structures if needed
        if self.config.enforce_1nf:
            transformed_data, applied_transforms = self._apply_1nf_transforms(transformed_data)
            transformations.extend(applied_transforms)

        shaped_data = ShapedData(
            processing_id=self.processing_id,
            canonical_id=canonical_data.processing_id,
            transformed_data=transformed_data,
            transformations_applied=transformations,
            timestamp=datetime.now()
        )

        self.lineage_steps.append({
            "stage": "shaped_output",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "transformations_applied": len(shaped_data.transformations_applied)
        })

        return shaped_data

    def _apply_1nf_transforms(self, data: Dict) -> Tuple[Dict, List[str]]:
        """Apply 1NF (First Normal Form) transformations to data."""
        transformations = []
        result = {}

        for key, value in data.items():
            if isinstance(value, list):
                # Convert arrays to numbered fields for 1NF
                for i, item in enumerate(value):
                    result[f"{key}_{i}"] = item
                transformations.append(f"flattened_array:{key}")
            elif isinstance(value, dict):
                # Flatten nested objects
                for sub_key, sub_value in value.items():
                    result[f"{key}_{sub_key}"] = sub_value
                transformations.append(f"flattened_object:{key}")
            else:
                result[key] = value

        return result, transformations

    async def _cleanup_resources(self) -> None:
        """Clean up any resources in case of processing failure."""
        logger.info(f"Cleaning up resources for processing: {self.processing_id}")

        # TODO: Implement resource cleanup
        # - Clear any cached data
        # - Release model resources
        # - Clean up temporary files
        pass


# =============================================================================
# CocoIndex Integration - COLPALI-204
# =============================================================================
# Architecture:
#   FLOW 1 (Indexing): PDF → pdf2image → ColPali Embeddings → Qdrant
#   FLOW 2 (Extraction): Query → Qdrant Search → Retrieved Pages → BAML → Structured Output
#
# Based on:
#   - https://github.com/cocoindex-io/cocoindex/tree/main/examples/multi_format_indexing
#   - https://github.com/cocoindex-io/cocoindex/tree/main/examples/patient_intake_extraction_baml
# =============================================================================

import os
import mimetypes
from dataclasses import dataclass as cocoindex_dataclass
from io import BytesIO

try:
    import cocoindex
    from qdrant_client import QdrantClient
    COCOINDEX_AVAILABLE = True
except ImportError:
    COCOINDEX_AVAILABLE = False
    logger.warning("CocoIndex not installed. Run: pip install cocoindex")

# Configuration from environment
QDRANT_GRPC_URL = os.getenv("QDRANT_GRPC_URL", "http://localhost:6334")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "DocumentEmbeddings")
COLPALI_MODEL = os.getenv("COLPALI_MODEL", "vidore/colqwen2-v0.1")
PDF_PATH = os.getenv("PDF_PATH", "pdfs")


@cocoindex_dataclass
class Page:
    """Represents a single page from a document for ColPali embedding."""
    page_number: int | None
    image: bytes


def _file_to_pages_impl(filename: str, content: bytes) -> list:
    """
    Convert document files to a list of page images.

    - PDFs are converted to PNG images at 300 DPI
    - Image files are passed through directly
    - Other file types return empty list
    """
    mime_type, _ = mimetypes.guess_type(filename)

    if mime_type == "application/pdf":
        images = convert_from_bytes(content, dpi=300)
        pages = []
        for i, image in enumerate(images):
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                pages.append(Page(page_number=i + 1, image=buffer.getvalue()))
        return pages
    elif mime_type and mime_type.startswith("image/"):
        return [Page(page_number=None, image=content)]
    else:
        return []


def register_cocoindex_flows():
    """
    Register ColPali-BAML flows with CocoIndex orchestration.

    This creates two flows:
    1. document_indexing_flow: Indexes documents with ColPali embeddings in Qdrant
    2. Extraction functions: BAML extraction with caching for structured output

    Returns the flow functions for external use.
    """
    if not COCOINDEX_AVAILABLE:
        logger.error("Cannot register CocoIndex flows: cocoindex not installed")
        return None

    # Register Qdrant connection
    qdrant_connection = cocoindex.add_auth_entry(
        "qdrant_connection",
        cocoindex.targets.QdrantConnection(grpc_url=QDRANT_GRPC_URL),
    )

    # Wrap file_to_pages with CocoIndex decorator
    @cocoindex.op.function()
    def file_to_pages(filename: str, content: bytes) -> list:
        """CocoIndex wrapper for file to pages conversion."""
        return _file_to_pages_impl(filename, content)

    # FLOW 1: Document Indexing with ColPali
    @cocoindex.flow_def(name="DocumentIndexingFlow")
    def document_indexing_flow(
        flow_builder: cocoindex.FlowBuilder,
        data_scope: cocoindex.DataScope
    ) -> None:
        """
        Index documents with ColPali embeddings in Qdrant.

        Flow:
        1. Load PDF/image files from local directory (binary mode)
        2. Convert PDFs to page images at 300 DPI
        3. Generate ColPali embeddings for each page
        4. Store embeddings in Qdrant with metadata
        """
        # Load documents from local file source (binary mode for PDFs)
        data_scope["documents"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(path=PDF_PATH, binary=True)
        )

        # Collector for page embeddings
        output_embeddings = data_scope.add_collector()

        # Process each document
        with data_scope["documents"].row() as doc:
            # Convert file to pages (PDF→images or pass through images)
            doc["pages"] = flow_builder.transform(
                file_to_pages,
                filename=doc["filename"],
                content=doc["content"]
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

    # Query transformation for ColPali embeddings
    @cocoindex.transform_flow()
    def query_to_colpali_embedding(
        text: cocoindex.DataSlice[str],
    ) -> cocoindex.DataSlice[list]:
        """
        Convert text query to ColPali multi-vector embedding.

        ColPali uses multi-vector embeddings (list of vectors) for
        late interaction retrieval, providing spatial awareness.
        """
        return text.transform(
            cocoindex.functions.ColPaliEmbedQuery(model=COLPALI_MODEL)
        )

    # FLOW 2: BAML Extraction (cached, async)
    @cocoindex.op.function(cache=True, behavior_version=1)
    async def extract_with_baml(page_image: bytes, schema_json: dict) -> dict:
        """
        Extract structured data from a page image using BAML.

        Uses tatforge's BAMLFunctionGenerator for schema-driven extraction.
        Caching (cache=True) prevents redundant LLM calls for the same page.

        Args:
            page_image: PNG image bytes
            schema_json: JSON schema for extraction

        Returns:
            Extracted data as dict matching the schema
        """
        import base64
        try:
            import baml_py
            from baml_client import b
        except ImportError:
            logger.error("BAML not available for extraction")
            return {"error": "BAML not installed"}

        from .schema_manager import SchemaManager
        from .baml_function_generator import BAMLFunctionGenerator

        # Generate BAML function from schema
        schema_manager = SchemaManager()
        baml_def = schema_manager.generate_baml_classes(schema_json)

        function_generator = BAMLFunctionGenerator()
        optimized_functions = function_generator.generate_optimized_functions(baml_def)

        if not optimized_functions:
            return {"error": "No extraction function generated"}

        # Convert image to BAML format
        image_b64 = base64.b64encode(page_image).decode("utf-8")
        image = baml_py.Image.from_base64("image/png", image_b64)

        # Execute the generated extraction function
        function_name = optimized_functions[0].name
        if hasattr(b, function_name):
            result = await getattr(b, function_name)(document_images=[image])
            return result.model_dump() if hasattr(result, 'model_dump') else result
        else:
            return {"error": f"BAML function {function_name} not found"}

    logger.info(f"CocoIndex flows registered:")
    logger.info(f"  - DocumentIndexingFlow (ColPali → Qdrant)")
    logger.info(f"  - query_to_colpali_embedding (text → ColPali embedding)")
    logger.info(f"  - extract_with_baml (page → BAML → structured data)")
    logger.info(f"  Qdrant: {QDRANT_GRPC_URL}")
    logger.info(f"  Collection: {QDRANT_COLLECTION}")
    logger.info(f"  ColPali Model: {COLPALI_MODEL}")

    return {
        "document_indexing_flow": document_indexing_flow,
        "query_to_colpali_embedding": query_to_colpali_embedding,
        "extract_with_baml": extract_with_baml,
        "file_to_pages": file_to_pages,
        "qdrant_connection": qdrant_connection,
    }


def search_documents(query: str, limit: int = 5) -> list:
    """
    Search indexed documents using ColPali embeddings.

    Args:
        query: Natural language search query
        limit: Maximum number of results to return

    Returns:
        List of search results with score, filename, and page number
    """
    if not COCOINDEX_AVAILABLE:
        logger.error("CocoIndex not available for search")
        return []

    flows = register_cocoindex_flows()
    if not flows:
        return []

    client = QdrantClient(url=QDRANT_GRPC_URL, prefer_grpc=True)

    # Convert query to ColPali multi-vector embedding
    query_embedding = flows["query_to_colpali_embedding"].eval(query)

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
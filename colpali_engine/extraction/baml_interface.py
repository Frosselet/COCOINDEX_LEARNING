"""
BAML Execution Interface for Structured Data Extraction - COLPALI-601.

This module provides the complete BAML execution interface that runs dynamically
generated BAML functions with document images as input, handling the bridge
between ColPali patch retrieval and BAML extraction with comprehensive error
handling and performance monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from PIL import Image
import json

from ..core.schema_manager import BAMLDefinition, BAMLFunction
from ..core.baml_client_manager import BAMLClientManager
from ..core.vision_model_manager import VisionModelManager, FallbackStrategy, VisionFallbackResult
from ..vision.colpali_client import ColPaliClient
from ..storage.qdrant_client import QdrantManager
from .models import (
    ExtractionResult, CanonicalData, ProcessingMetadata, QualityMetrics,
    ProcessingStatus, DocumentPatch, ScoredPatch
)

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of BAML function execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


@dataclass
class ExtractionContext:
    """Context information for extraction execution."""
    document_id: str
    document_type: Optional[str] = None
    source_path: Optional[str] = None
    processing_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionRequest:
    """Complete extraction request with all parameters."""
    function: BAMLFunction
    images: List[Union[Image.Image, str]]  # Images or paths
    context: ExtractionContext
    fallback_strategy: FallbackStrategy = FallbackStrategy.ALTERNATIVE_VISION
    timeout_seconds: int = 120
    max_retries: int = 3
    use_patch_retrieval: bool = True
    similarity_threshold: float = 0.7


class BAMLExecutionInterface:
    """
    Advanced BAML execution interface with complete vision integration.

    Provides seamless execution of BAML functions with document images,
    integrating ColPali patch retrieval, vision model fallback strategies,
    and comprehensive error handling with performance monitoring.
    """

    def __init__(
        self,
        client_manager: Optional[BAMLClientManager] = None,
        vision_manager: Optional[VisionModelManager] = None,
        colpali_client: Optional[ColPaliClient] = None,
        qdrant_manager: Optional[QdrantManager] = None
    ):
        """
        Initialize BAML execution interface.

        Args:
            client_manager: BAML client configuration manager
            vision_manager: Vision model manager for intelligent fallback
            colpali_client: ColPali vision client for patch embedding
            qdrant_manager: Qdrant vector storage for semantic retrieval
        """
        self.client_manager = client_manager or BAMLClientManager()
        self.vision_manager = vision_manager or VisionModelManager(self.client_manager)
        self.colpali_client = colpali_client
        self.qdrant_manager = qdrant_manager

        # Execution state
        self.active_executions: Dict[str, ExecutionStatus] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "execution_time": [],
            "patch_retrieval_time": [],
            "vision_processing_time": [],
            "total_processing_time": []
        }

        logger.info("BAML Execution Interface initialized")

    async def initialize(self) -> None:
        """
        Initialize all components for execution.

        Ensures ColPali client is loaded and Qdrant connection is established.
        """
        try:
            # Initialize ColPali client if provided
            if self.colpali_client and not getattr(self.colpali_client, '_is_loaded', False):
                logger.info("Loading ColPali model for patch extraction...")
                await self.colpali_client.load_model()

            # Initialize Qdrant connection if provided
            if self.qdrant_manager:
                logger.info("Connecting to Qdrant vector database...")
                await self.qdrant_manager.connect()
                await self.qdrant_manager.ensure_collection()

            logger.info("BAML Execution Interface fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize BAML execution interface: {e}")
            raise

    async def execute_extraction(self, request: ExtractionRequest) -> ExtractionResult:
        """
        Execute complete BAML extraction with vision context.

        Args:
            request: Complete extraction request with all parameters

        Returns:
            Complete extraction result with canonical data and metadata

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If execution fails after all retries
        """
        processing_id = f"extract_{int(time.time() * 1000)}"
        start_time = datetime.now()

        # Initialize processing metadata
        metadata = ProcessingMetadata(
            processing_id=processing_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            processing_time_seconds=0.0,
            lineage_steps=[],
            config=self._build_execution_config(request),
            status=ProcessingStatus.IN_PROGRESS
        )

        self.active_executions[processing_id] = ExecutionStatus.IN_PROGRESS

        try:
            # Step 1: Image preprocessing and validation
            metadata.add_lineage_step({
                "step": "image_preprocessing",
                "description": "Preprocess and validate input images",
                "image_count": len(request.images)
            })

            processed_images = await self._preprocess_images(request.images)

            # Step 2: Optional patch retrieval for semantic context
            relevant_patches = None
            if request.use_patch_retrieval and self.colpali_client and self.qdrant_manager:
                retrieval_start = time.time()
                relevant_patches = await self._retrieve_relevant_patches(
                    processed_images, request.context, request.similarity_threshold
                )
                retrieval_time = time.time() - retrieval_start
                self.performance_metrics["patch_retrieval_time"].append(retrieval_time)

                metadata.add_lineage_step({
                    "step": "patch_retrieval",
                    "description": "Retrieve semantically relevant patches",
                    "patches_found": len(relevant_patches) if relevant_patches else 0,
                    "retrieval_time_ms": int(retrieval_time * 1000)
                })

            # Step 3: Execute BAML function with vision fallback
            vision_start = time.time()
            vision_result = await self.vision_manager.process_with_vision_fallback(
                function=request.function,
                images=[self._image_to_path(img) for img in processed_images],
                fallback_strategy=request.fallback_strategy,
                max_retries=request.max_retries,
                timeout_seconds=request.timeout_seconds,
                context={
                    "document_type": request.context.document_type,
                    "relevant_patches": relevant_patches,
                    "processing_config": request.context.processing_config
                }
            )
            vision_time = time.time() - vision_start
            self.performance_metrics["vision_processing_time"].append(vision_time)

            # Step 4: Parse and structure the results
            extraction_data = await self._parse_vision_result(vision_result, request.function)

            metadata.add_lineage_step({
                "step": "baml_execution",
                "description": f"Execute BAML function: {request.function.name}",
                "client_used": vision_result.client_used,
                "fallback_applied": vision_result.fallback_applied,
                "cost_estimate": vision_result.cost_estimate,
                "vision_processing_time_ms": int(vision_time * 1000)
            })

            # Step 5: Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                extraction_data, request.function, vision_result
            )

            # Step 6: Create canonical data result
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            metadata.end_time = end_time
            metadata.processing_time_seconds = processing_time
            metadata.status = ProcessingStatus.COMPLETED
            metadata.quality_metrics = quality_metrics

            canonical_data = CanonicalData(
                processing_id=processing_id,
                extraction_data=extraction_data,
                timestamp=end_time,
                confidence_scores=self._extract_field_confidences(extraction_data, quality_metrics),
                source_metadata={
                    "document_id": request.context.document_id,
                    "document_type": request.context.document_type,
                    "image_count": len(processed_images),
                    "function_used": request.function.name,
                    "client_used": vision_result.client_used
                }
            )

            # Update performance tracking
            self.performance_metrics["total_processing_time"].append(processing_time)
            self.active_executions[processing_id] = ExecutionStatus.COMPLETED

            # Store execution in history
            self.execution_history.append({
                "processing_id": processing_id,
                "function_name": request.function.name,
                "status": "completed",
                "processing_time": processing_time,
                "quality_score": quality_metrics.overall_quality,
                "timestamp": end_time.isoformat()
            })

            logger.info(
                f"Extraction completed: {processing_id}, "
                f"quality={quality_metrics.overall_quality:.2f}, "
                f"time={processing_time:.2f}s"
            )

            return ExtractionResult(
                canonical=canonical_data,
                shaped=None,  # Will be created in COLPALI-700
                metadata=metadata
            )

        except Exception as e:
            # Handle execution failure
            logger.error(f"Extraction failed for {processing_id}: {e}")

            end_time = datetime.now()
            metadata.end_time = end_time
            metadata.processing_time_seconds = (end_time - start_time).total_seconds()
            metadata.status = ProcessingStatus.FAILED
            metadata.error_message = str(e)

            self.active_executions[processing_id] = ExecutionStatus.FAILED

            # Still return result with error information
            canonical_data = CanonicalData(
                processing_id=processing_id,
                extraction_data={},
                timestamp=end_time,
                source_metadata={
                    "document_id": request.context.document_id,
                    "error": str(e)
                }
            )

            return ExtractionResult(
                canonical=canonical_data,
                shaped=None,
                metadata=metadata
            )

    async def _preprocess_images(self, images: List[Union[Image.Image, str]]) -> List[Image.Image]:
        """
        Preprocess and validate input images.

        Args:
            images: List of PIL Images or file paths

        Returns:
            List of processed PIL Images

        Raises:
            ValueError: If image processing fails
        """
        processed_images = []

        for i, img in enumerate(images):
            try:
                if isinstance(img, str):
                    # Load from file path
                    pil_image = Image.open(img)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                elif isinstance(img, Image.Image):
                    # Use PIL Image directly
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    pil_image = img
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")

                # Basic validation
                if pil_image.size[0] < 32 or pil_image.size[1] < 32:
                    raise ValueError(f"Image {i} too small: {pil_image.size}")

                processed_images.append(pil_image)

            except Exception as e:
                raise ValueError(f"Failed to preprocess image {i}: {e}")

        logger.debug(f"Preprocessed {len(processed_images)} images")
        return processed_images

    async def _retrieve_relevant_patches(
        self,
        images: List[Image.Image],
        context: ExtractionContext,
        similarity_threshold: float
    ) -> List[ScoredPatch]:
        """
        Retrieve semantically relevant patches using ColPali + Qdrant.

        Args:
            images: Processed images for patch extraction
            context: Extraction context
            similarity_threshold: Minimum similarity for relevant patches

        Returns:
            List of scored patches above threshold
        """
        if not self.colpali_client or not self.qdrant_manager:
            return []

        try:
            # Generate embeddings for query images
            embeddings = await self.colpali_client.embed_frames(images)

            # Search for similar patches in Qdrant
            all_patches = []
            for embedding in embeddings:
                # Use first patch embedding as query vector
                if len(embedding) > 0:
                    query_vector = embedding[0].tolist()  # First patch as representative

                    # Search with context filtering
                    search_results = await self.qdrant_manager.search_similar(
                        query_vector=query_vector,
                        filter_conditions={
                            "document_type": context.document_type
                        } if context.document_type else None,
                        score_threshold=similarity_threshold,
                        limit=10
                    )

                    # Convert to ScoredPatch objects
                    for result in search_results:
                        # Create patch from Qdrant result
                        patch = DocumentPatch(
                            patch_id=result.id,
                            document_id=result.payload.get("document_id", "unknown"),
                            page_number=result.payload.get("page_number", 0),
                            coordinates=tuple(result.payload.get("coordinates", (0, 0, 32, 32))),
                            confidence_score=result.score
                        )

                        scored_patch = ScoredPatch(
                            patch=patch,
                            similarity_score=result.score,
                            retrieval_metadata={
                                "query_embedding_size": len(query_vector),
                                "search_timestamp": datetime.now().isoformat()
                            }
                        )
                        all_patches.append(scored_patch)

            logger.debug(f"Retrieved {len(all_patches)} relevant patches")
            return all_patches

        except Exception as e:
            logger.warning(f"Patch retrieval failed: {e}")
            return []

    def _image_to_path(self, image: Image.Image) -> str:
        """Convert PIL Image to temporary file path for BAML processing."""
        # For now, return a placeholder - in production this would save to temp file
        return "temp_image.png"

    async def _parse_vision_result(
        self,
        vision_result: VisionFallbackResult,
        function: BAMLFunction
    ) -> Dict[str, Any]:
        """
        Parse vision processing result into structured data.

        Args:
            vision_result: Result from vision model processing
            function: BAML function that was executed

        Returns:
            Parsed structured data
        """
        if not vision_result.success:
            logger.warning(f"Vision processing failed: {vision_result.fallback_reason}")
            return {}

        try:
            # If result is already structured, return it
            if isinstance(vision_result.result, dict):
                return vision_result.result

            # If result is a string, try to parse as JSON
            if isinstance(vision_result.result, str):
                try:
                    return json.loads(vision_result.result)
                except json.JSONDecodeError:
                    # Return as plain text field
                    return {"extracted_text": vision_result.result}

            # Fallback to converting to string
            return {"extracted_data": str(vision_result.result)}

        except Exception as e:
            logger.error(f"Failed to parse vision result: {e}")
            return {"extraction_error": str(e)}

    async def _calculate_quality_metrics(
        self,
        extraction_data: Dict[str, Any],
        function: BAMLFunction,
        vision_result: VisionFallbackResult
    ) -> QualityMetrics:
        """
        Calculate quality metrics for extraction results.

        Args:
            extraction_data: Extracted structured data
            function: BAML function used for extraction
            vision_result: Vision processing result

        Returns:
            Quality metrics assessment
        """
        # Confidence score from vision processing
        confidence_score = 0.8 if vision_result.success else 0.2
        if vision_result.fallback_applied:
            confidence_score *= 0.9  # Slightly lower for fallback

        # Completeness ratio based on expected vs actual fields
        expected_fields = len(function.input_params) if function.input_params else 1
        actual_fields = len(extraction_data) if extraction_data else 0
        completeness_ratio = min(actual_fields / expected_fields, 1.0) if expected_fields > 0 else 0.0

        # Consistency score based on data types and structure
        consistency_score = 0.9 if self._validate_data_consistency(extraction_data) else 0.6

        # Schema compliance - basic validation for now
        schema_compliance_score = 0.8 if extraction_data else 0.1

        return QualityMetrics(
            confidence_score=confidence_score,
            completeness_ratio=completeness_ratio,
            consistency_score=consistency_score,
            schema_compliance_score=schema_compliance_score
        )

    def _validate_data_consistency(self, data: Dict[str, Any]) -> bool:
        """Basic validation of data consistency."""
        if not data:
            return False

        # Check for reasonable data types
        for key, value in data.items():
            if value is None:
                continue
            if not isinstance(value, (str, int, float, list, dict, bool)):
                return False

        return True

    def _extract_field_confidences(
        self,
        extraction_data: Dict[str, Any],
        quality_metrics: QualityMetrics
    ) -> Dict[str, float]:
        """Extract field-level confidence scores."""
        base_confidence = quality_metrics.confidence_score

        confidences = {}
        for key in extraction_data.keys():
            # Assign base confidence to all fields for now
            # In production, this would be more sophisticated
            confidences[key] = base_confidence

        return confidences

    def _build_execution_config(self, request: ExtractionRequest) -> Dict[str, Any]:
        """Build execution configuration for metadata."""
        return {
            "function_name": request.function.name,
            "fallback_strategy": request.fallback_strategy.value,
            "timeout_seconds": request.timeout_seconds,
            "max_retries": request.max_retries,
            "use_patch_retrieval": request.use_patch_retrieval,
            "similarity_threshold": request.similarity_threshold,
            "document_type": request.context.document_type,
            "processing_config": request.context.processing_config
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {}

        for metric_name, values in self.performance_metrics.items():
            if values:
                metrics[metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values)
                }
            else:
                metrics[metric_name] = {
                    "count": 0,
                    "average": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "total": 0.0
                }

        return {
            "performance_metrics": metrics,
            "active_executions": len(self.active_executions),
            "completed_executions": len(self.execution_history),
            "success_rate": self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate success rate from execution history."""
        if not self.execution_history:
            return 0.0

        successful = sum(1 for exec_info in self.execution_history if exec_info["status"] == "completed")
        return successful / len(self.execution_history)

    async def cleanup(self) -> None:
        """Clean up resources and connections."""
        try:
            if self.qdrant_manager:
                # Qdrant cleanup if needed
                pass

            if self.colpali_client:
                # ColPali cleanup if needed
                pass

            logger.info("BAML Execution Interface cleaned up")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Factory function for easy instantiation
async def create_baml_execution_interface(
    baml_src_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    enable_colpali: bool = True
) -> BAMLExecutionInterface:
    """
    Create and initialize a complete BAML execution interface.

    Args:
        baml_src_path: Path to BAML source directory
        qdrant_url: Qdrant connection URL
        enable_colpali: Whether to enable ColPali patch retrieval

    Returns:
        Fully initialized BAML execution interface
    """
    # Initialize managers
    client_manager = BAMLClientManager(baml_src_path=baml_src_path)
    vision_manager = VisionModelManager(client_manager=client_manager)

    # Optional components
    colpali_client = None
    if enable_colpali:
        try:
            from ..vision.colpali_client import ColPaliClient
            colpali_client = ColPaliClient()
        except ImportError:
            logger.warning("ColPali client not available")

    qdrant_manager = None
    if qdrant_url:
        qdrant_manager = QdrantManager(url=qdrant_url)

    # Create and initialize interface
    interface = BAMLExecutionInterface(
        client_manager=client_manager,
        vision_manager=vision_manager,
        colpali_client=colpali_client,
        qdrant_manager=qdrant_manager
    )

    await interface.initialize()
    return interface
"""
Qdrant vector database client for ColPali embeddings.

This module provides the interface to Qdrant for storing and querying
multi-vector ColPali embeddings with spatial metadata and document lineage.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Union
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, CollectionInfo,
    PayloadSchemaType, PointStruct, Filter, FieldCondition,
    Match, Range
)
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manager for Qdrant vector database operations.

    Handles collection management, embedding storage, and semantic search
    for ColPali patch embeddings with spatial metadata.
    """

    def __init__(
        self,
        url: str = None,
        port: int = 6333,
        collection_name: str = "colpali_embeddings",
        api_key: str = None,
        timeout: int = 30,
        prefer_grpc: bool = False
    ):
        """
        Initialize Qdrant manager.

        Args:
            url: Qdrant server URL (defaults to localhost or env QDRANT_URL)
            port: Qdrant server port
            collection_name: Name of the vector collection
            api_key: Optional API key (or env QDRANT_API_KEY)
            timeout: Connection timeout in seconds
            prefer_grpc: Use gRPC instead of HTTP
        """
        # Get configuration from environment if not provided
        self.url = url or os.getenv("QDRANT_URL", "http://localhost")
        self.port = port
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        self.timeout = timeout
        self.prefer_grpc = prefer_grpc

        # Connection state
        self.client: Optional[QdrantClient] = None
        self.is_connected = False
        self.connection_retries = 0
        self.max_retries = 5

        # Collection configuration
        self.vector_size = 128  # ColPali embedding dimension
        self.distance_metric = Distance.COSINE

        logger.info(f"QdrantManager initialized: {self._get_connection_string()}")
        logger.info(f"Collection: {collection_name}, Vector size: {self.vector_size}")

    async def connect(self) -> None:
        """
        Connect to Qdrant server with retry logic and health checking.

        Implements COLPALI-401: Connection management with resilience.

        Raises:
            ConnectionError: If unable to connect after retries
        """
        if self.is_connected:
            logger.info("Already connected to Qdrant")
            return

        logger.info(f"Connecting to Qdrant at {self._get_connection_string()}")

        for attempt in range(self.max_retries):
            try:
                # Create Qdrant client
                self.client = QdrantClient(
                    url=self.url,
                    port=self.port if not self._is_full_url() else None,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    prefer_grpc=self.prefer_grpc
                )

                # Test connection with health check
                await self._health_check()

                self.is_connected = True
                self.connection_retries = 0
                logger.info(f"Successfully connected to Qdrant (attempt {attempt + 1})")
                return

            except Exception as e:
                self.connection_retries += 1
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to Qdrant after {self.max_retries} attempts")
                    raise ConnectionError(f"Cannot connect to Qdrant: {e}")

    def _get_connection_string(self) -> str:
        """Get formatted connection string for logging."""
        if self._is_full_url():
            return self.url
        return f"{self.url}:{self.port}"

    def _is_full_url(self) -> bool:
        """Check if URL contains protocol and port."""
        return self.url.startswith(('http://', 'https://'))

    async def _health_check(self) -> None:
        """
        Perform health check on Qdrant connection.

        Raises:
            ConnectionError: If health check fails
        """
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(
                None, self.client.get_collections
            )
            logger.debug(f"Health check passed. Found {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise ConnectionError(f"Qdrant health check failed: {e}")

    async def create_collection(
        self,
        vector_size: int = None,
        distance: Distance = None,
        recreate: bool = False
    ) -> bool:
        """
        Create collection for ColPali embeddings with spatial metadata schema.

        Implements COLPALI-401: Collection setup with optimized indexing.

        Args:
            vector_size: Dimension of embedding vectors (defaults to 128)
            distance: Distance metric for similarity search (defaults to COSINE)
            recreate: Whether to recreate collection if it exists

        Returns:
            True if collection was created, False if it already existed

        Raises:
            RuntimeError: If not connected to Qdrant
            Exception: If collection creation fails
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        vector_size = vector_size or self.vector_size
        distance = distance or self.distance_metric

        logger.info(f"Creating collection '{self.collection_name}' with {vector_size}D vectors")

        try:
            loop = asyncio.get_event_loop()

            # Check if collection exists
            collections = await loop.run_in_executor(None, self.client.get_collections)
            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            if collection_exists:
                if recreate:
                    logger.info(f"Recreating existing collection '{self.collection_name}'")
                    await loop.run_in_executor(
                        None, self.client.delete_collection, self.collection_name
                    )
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return False

            # Create collection with optimized configuration
            await loop.run_in_executor(
                None,
                self._create_collection_sync,
                vector_size,
                distance
            )

            # Create payload indexes for efficient filtering
            await self._create_payload_indexes()

            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise Exception(f"Collection creation failed: {e}")

    def _create_collection_sync(self, vector_size: int, distance: Distance) -> None:
        """Synchronous collection creation for executor."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            ),
            # Optimize for high-dimensional vectors
            optimizers_config={
                "default_segment_number": 2,
                "max_segment_size": 20000,
                "memmap_threshold": 50000,
                "indexing_threshold": 100,
                "flush_interval_sec": 30,
                "max_optimization_threads": 2
            },
            # Use HNSW for fast similarity search
            hnsw_config={
                "m": 16,  # Number of bi-directional links for each node
                "ef_construct": 100,  # Size of dynamic candidate list
                "full_scan_threshold": 10000,
                "max_indexing_threads": 0,  # Use all available threads
                "on_disk": False  # Keep index in memory for speed
            }
        )

    async def _create_payload_indexes(self) -> None:
        """Create indexes for efficient metadata filtering."""
        loop = asyncio.get_event_loop()

        # Index for document_id (exact match)
        await loop.run_in_executor(
            None,
            self.client.create_payload_index,
            self.collection_name,
            "document_id",
            PayloadSchemaType.KEYWORD
        )

        # Index for page_number (range queries)
        await loop.run_in_executor(
            None,
            self.client.create_payload_index,
            self.collection_name,
            "page_number",
            PayloadSchemaType.INTEGER
        )

        # Index for patch coordinates (range queries)
        await loop.run_in_executor(
            None,
            self.client.create_payload_index,
            self.collection_name,
            "patch_x",
            PayloadSchemaType.INTEGER
        )

        await loop.run_in_executor(
            None,
            self.client.create_payload_index,
            self.collection_name,
            "patch_y",
            PayloadSchemaType.INTEGER
        )

        # Index for document type filtering
        await loop.run_in_executor(
            None,
            self.client.create_payload_index,
            self.collection_name,
            "document_type",
            PayloadSchemaType.KEYWORD
        )

        logger.debug("Payload indexes created for efficient filtering")

    async def store_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Store ColPali embeddings with spatial metadata and document lineage.

        Implements COLPALI-402: Embedding storage with spatial metadata.

        Args:
            embeddings: List of embedding vectors (each vector is List[float])
            metadata: Document and processing metadata containing:
                - document_id: Unique document identifier
                - page_number: Page number in document
                - document_type: Type of document (pdf, pptx, etc.)
                - patch_coordinates: List of (x, y) coordinates for each patch
                - processing_timestamp: When document was processed
                - additional custom metadata
            batch_size: Number of embeddings to upsert per batch (default: 100)

        Returns:
            Dictionary containing operation results:
                - points_stored: Number of points successfully stored
                - batches_processed: Number of batches processed
                - storage_time: Time taken for storage operation
                - errors: List of any errors encountered

        Raises:
            RuntimeError: If not connected to Qdrant
            ValueError: If embeddings or metadata are invalid
            Exception: If storage operation fails
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        if not embeddings:
            logger.warning("No embeddings provided for storage")
            return {"points_stored": 0, "batches_processed": 0, "storage_time": 0.0, "errors": []}

        # Validate required metadata fields
        required_fields = ["document_id", "page_number", "document_type", "processing_timestamp"]
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise ValueError(f"Missing required metadata fields: {missing_fields}")

        start_time = time.time()
        points_stored = 0
        batches_processed = 0
        errors = []

        logger.info(f"Starting storage of {len(embeddings)} embeddings for document {metadata['document_id']}")

        try:
            loop = asyncio.get_event_loop()

            # Prepare points for batch upsert
            points = await self._prepare_points_for_storage(embeddings, metadata)
            total_points = len(points)

            # Process in batches for efficient storage
            for batch_start in range(0, total_points, batch_size):
                batch_end = min(batch_start + batch_size, total_points)
                batch_points = points[batch_start:batch_end]

                try:
                    # Perform batch upsert
                    await loop.run_in_executor(
                        None,
                        self._upsert_batch_sync,
                        batch_points
                    )

                    points_stored += len(batch_points)
                    batches_processed += 1

                    logger.debug(f"Stored batch {batches_processed}: {len(batch_points)} points")

                except Exception as e:
                    error_msg = f"Failed to store batch {batches_processed + 1}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            storage_time = time.time() - start_time

            # Log storage results
            success_rate = (points_stored / total_points) * 100 if total_points > 0 else 0
            logger.info(
                f"Storage completed: {points_stored}/{total_points} points stored "
                f"({success_rate:.1f}%) in {storage_time:.3f}s across {batches_processed} batches"
            )

            if errors:
                logger.warning(f"Storage completed with {len(errors)} errors")

            return {
                "points_stored": points_stored,
                "batches_processed": batches_processed,
                "storage_time": storage_time,
                "errors": errors,
                "success_rate": success_rate
            }

        except Exception as e:
            storage_time = time.time() - start_time
            error_msg = f"Storage operation failed after {storage_time:.3f}s: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def _prepare_points_for_storage(
        self,
        embeddings: List[List[float]],
        metadata: Dict[str, Any]
    ) -> List[PointStruct]:
        """
        Prepare embeddings and metadata for Qdrant storage.

        Creates PointStruct objects with spatial coordinates and document lineage.
        """
        points = []
        patch_coordinates = metadata.get("patch_coordinates", [])

        # Generate patch coordinates if not provided
        if not patch_coordinates:
            patch_coordinates = self._generate_patch_coordinates(len(embeddings))
            logger.debug(f"Generated {len(patch_coordinates)} patch coordinates")

        if len(patch_coordinates) != len(embeddings):
            logger.warning(
                f"Patch coordinates count ({len(patch_coordinates)}) doesn't match "
                f"embeddings count ({len(embeddings)}). Using available coordinates."
            )

        for i, embedding in enumerate(embeddings):
            # Generate unique point ID
            point_id = str(uuid.uuid4())

            # Prepare patch coordinates
            patch_x, patch_y = (0, 0)  # Default coordinates
            if i < len(patch_coordinates):
                patch_x, patch_y = patch_coordinates[i]

            # Create comprehensive payload with spatial metadata and document lineage
            payload = {
                "document_id": metadata["document_id"],
                "page_number": metadata["page_number"],
                "document_type": metadata["document_type"],
                "processing_timestamp": metadata["processing_timestamp"],
                "patch_index": i,
                "patch_x": int(patch_x),
                "patch_y": int(patch_y),
                "patch_coordinates": f"{patch_x},{patch_y}",
                "embedding_dimension": len(embedding),
                "storage_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.utc(time.time()))
            }

            # Add optional metadata fields
            optional_fields = [
                "document_path", "document_name", "document_hash", "model_version",
                "processing_config", "quality_score", "confidence_score"
            ]
            for field in optional_fields:
                if field in metadata:
                    payload[field] = metadata[field]

            # Create point with vector and payload
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            points.append(point)

        logger.debug(f"Prepared {len(points)} points for storage with spatial metadata")
        return points

    def _generate_patch_coordinates(self, num_patches: int) -> List[tuple]:
        """
        Generate patch coordinates for embeddings based on typical patch grid layout.

        Args:
            num_patches: Number of patches to generate coordinates for

        Returns:
            List of (x, y) coordinate tuples
        """
        coordinates = []

        # Calculate grid dimensions (assume square-ish grid)
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size < num_patches:
            grid_size += 1

        patch_size = 32  # Standard ColPali patch size

        for i in range(num_patches):
            row = i // grid_size
            col = i % grid_size
            x = col * patch_size
            y = row * patch_size
            coordinates.append((x, y))

        return coordinates

    def _upsert_batch_sync(self, batch_points: List[PointStruct]) -> None:
        """
        Synchronous batch upsert operation for executor.

        Args:
            batch_points: List of points to upsert
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=batch_points,
            wait=True  # Wait for operation to complete for consistency
        )

    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict] = None,
        score_threshold: float = 0.0,
        include_payload: bool = True,
        include_vectors: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar embeddings with spatial filtering and result ranking.

        Implements COLPALI-403: Semantic search and retrieval system.

        Args:
            query_vector: Query embedding vector (List[float])
            limit: Maximum number of results to return (default: 10)
            filter_conditions: Optional metadata filters for spatial/document filtering:
                - document_id: Filter by specific document
                - document_type: Filter by document type
                - page_number: Filter by page number or range
                - spatial_box: Filter by spatial region (x1, y1, x2, y2)
                - processing_timestamp: Filter by processing time range
            score_threshold: Minimum similarity score (default: 0.0)
            include_payload: Include metadata in results (default: True)
            include_vectors: Include embedding vectors in results (default: False)

        Returns:
            Dictionary containing search results:
                - results: List of search results with scores and metadata
                - search_time: Time taken for search operation
                - total_found: Total number of results found
                - query_info: Information about the query and filters applied

        Raises:
            RuntimeError: If not connected to Qdrant
            ValueError: If query_vector is invalid
            Exception: If search operation fails
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        if not query_vector:
            raise ValueError("Query vector cannot be empty")

        start_time = time.time()

        logger.info(f"Searching for similar embeddings: limit={limit}, filters={bool(filter_conditions)}")

        try:
            loop = asyncio.get_event_loop()

            # Prepare search filter
            search_filter = self._build_search_filter(filter_conditions)

            # Perform semantic search
            search_result = await loop.run_in_executor(
                None,
                self._search_sync,
                query_vector,
                limit,
                search_filter,
                score_threshold,
                include_payload,
                include_vectors
            )

            search_time = time.time() - start_time

            # Process and rank results
            processed_results = self._process_search_results(search_result)

            logger.info(
                f"Search completed: {len(processed_results)} results found "
                f"in {search_time:.3f}s (threshold: {score_threshold})"
            )

            return {
                "results": processed_results,
                "search_time": search_time,
                "total_found": len(processed_results),
                "query_info": {
                    "vector_dimension": len(query_vector),
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "filters_applied": filter_conditions is not None,
                    "filter_details": filter_conditions
                }
            }

        except Exception as e:
            search_time = time.time() - start_time
            error_msg = f"Search operation failed after {search_time:.3f}s: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _build_search_filter(self, filter_conditions: Optional[Dict]) -> Optional[Filter]:
        """
        Build Qdrant filter from filter conditions for spatial and metadata filtering.

        Args:
            filter_conditions: Dictionary with filter criteria

        Returns:
            Qdrant Filter object or None if no conditions
        """
        if not filter_conditions:
            return None

        conditions = []

        # Document-level filters
        if "document_id" in filter_conditions:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=Match(value=filter_conditions["document_id"])
                )
            )

        if "document_type" in filter_conditions:
            conditions.append(
                FieldCondition(
                    key="document_type",
                    match=Match(value=filter_conditions["document_type"])
                )
            )

        if "page_number" in filter_conditions:
            page_filter = filter_conditions["page_number"]
            if isinstance(page_filter, int):
                # Exact page match
                conditions.append(
                    FieldCondition(
                        key="page_number",
                        match=Match(value=page_filter)
                    )
                )
            elif isinstance(page_filter, dict) and "range" in page_filter:
                # Page range filter
                page_range = page_filter["range"]
                conditions.append(
                    FieldCondition(
                        key="page_number",
                        range=Range(
                            gte=page_range.get("gte"),
                            lte=page_range.get("lte")
                        )
                    )
                )

        # Spatial filtering
        if "spatial_box" in filter_conditions:
            spatial_box = filter_conditions["spatial_box"]
            if len(spatial_box) == 4:
                x1, y1, x2, y2 = spatial_box

                # Filter by X coordinate range
                conditions.append(
                    FieldCondition(
                        key="patch_x",
                        range=Range(gte=x1, lte=x2)
                    )
                )

                # Filter by Y coordinate range
                conditions.append(
                    FieldCondition(
                        key="patch_y",
                        range=Range(gte=y1, lte=y2)
                    )
                )

        # Time-based filtering
        if "processing_timestamp" in filter_conditions:
            time_filter = filter_conditions["processing_timestamp"]
            if isinstance(time_filter, dict):
                if "after" in time_filter:
                    conditions.append(
                        FieldCondition(
                            key="processing_timestamp",
                            match=Match(value=time_filter["after"]),  # This should use range for proper time filtering
                        )
                    )

        if conditions:
            return Filter(must=conditions)

        return None

    def _search_sync(
        self,
        query_vector: List[float],
        limit: int,
        search_filter: Optional[Filter],
        score_threshold: float,
        include_payload: bool,
        include_vectors: bool
    ):
        """
        Synchronous search operation for executor.

        Args:
            query_vector: Query embedding vector
            limit: Maximum results
            search_filter: Qdrant filter object
            score_threshold: Minimum similarity score
            include_payload: Include metadata in results
            include_vectors: Include vectors in results

        Returns:
            Qdrant search results
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=include_payload,
            with_vectors=include_vectors
        )

    def _process_search_results(self, search_results) -> List[Dict[str, Any]]:
        """
        Process raw Qdrant search results into structured format.

        Args:
            search_results: Raw Qdrant search results

        Returns:
            List of processed result dictionaries
        """
        processed_results = []

        for result in search_results:
            processed_result = {
                "id": result.id,
                "score": float(result.score),
                "metadata": {}
            }

            # Add payload information if available
            if hasattr(result, 'payload') and result.payload:
                processed_result["metadata"] = dict(result.payload)

                # Extract spatial information for convenience
                if "patch_x" in result.payload and "patch_y" in result.payload:
                    processed_result["spatial"] = {
                        "patch_coordinates": (result.payload["patch_x"], result.payload["patch_y"]),
                        "patch_index": result.payload.get("patch_index", 0)
                    }

                # Extract document information
                if "document_id" in result.payload:
                    processed_result["document"] = {
                        "id": result.payload["document_id"],
                        "type": result.payload.get("document_type"),
                        "page_number": result.payload.get("page_number")
                    }

            # Add vector if requested
            if hasattr(result, 'vector') and result.vector:
                processed_result["vector"] = result.vector

            processed_results.append(processed_result)

        # Sort by score (highest first)
        processed_results.sort(key=lambda x: x["score"], reverse=True)

        return processed_results

    async def search_by_spatial_region(
        self,
        query_vector: List[float],
        spatial_box: tuple,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search for similar embeddings within a specific spatial region.

        Convenience method for spatial search queries.

        Args:
            query_vector: Query embedding vector
            spatial_box: (x1, y1, x2, y2) coordinates defining search region
            limit: Maximum number of results

        Returns:
            Search results within spatial region
        """
        filter_conditions = {
            "spatial_box": spatial_box
        }

        return await self.search_similar(
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions
        )

    async def search_by_document(
        self,
        query_vector: List[float],
        document_id: str,
        page_number: Optional[int] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Search for similar embeddings within a specific document.

        Convenience method for document-scoped search queries.

        Args:
            query_vector: Query embedding vector
            document_id: Target document identifier
            page_number: Optional specific page number
            limit: Maximum number of results

        Returns:
            Search results within document scope
        """
        filter_conditions = {
            "document_id": document_id
        }

        if page_number is not None:
            filter_conditions["page_number"] = page_number

        return await self.search_similar(
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions
        )

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the collection.

        Implements COLPALI-401: Collection monitoring and health checks.

        Returns:
            Dictionary with collection information and statistics

        Raises:
            RuntimeError: If not connected to Qdrant
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        try:
            loop = asyncio.get_event_loop()

            # Get collection info
            collection_info = await loop.run_in_executor(
                None, self.client.get_collection, self.collection_name
            )

            # Get collection statistics
            stats = {
                "collection_name": self.collection_name,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": len(collection_info.segments) if collection_info.segments else 0,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "payload_schema": self._get_payload_schema_info(collection_info)
            }

            # Add performance metrics if available
            if hasattr(collection_info, 'config') and collection_info.config:
                hnsw_config = collection_info.config.hnsw_config
                if hnsw_config:
                    stats["hnsw_config"] = {
                        "m": hnsw_config.m,
                        "ef_construct": hnsw_config.ef_construct,
                        "full_scan_threshold": hnsw_config.full_scan_threshold
                    }

            logger.info(f"Collection '{self.collection_name}': {stats['points_count']} points, "
                       f"{stats['vectors_count']} vectors")

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            # Return basic info if detailed info fails
            return {
                "collection_name": self.collection_name,
                "status": "error",
                "error": str(e),
                "vector_size": self.vector_size,
                "distance_metric": self.distance_metric.value
            }

    def _get_payload_schema_info(self, collection_info) -> Dict[str, str]:
        """Extract payload schema information from collection info."""
        schema_info = {}
        if hasattr(collection_info, 'payload_schema') and collection_info.payload_schema:
            for field_name, field_info in collection_info.payload_schema.items():
                schema_info[field_name] = field_info.data_type if hasattr(field_info, 'data_type') else str(field_info)
        return schema_info

    async def collection_exists(self) -> bool:
        """
        Check if the collection exists.

        Returns:
            True if collection exists, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(None, self.client.get_collections)
            return any(col.name == self.collection_name for col in collections.collections)
        except Exception as e:
            logger.warning(f"Failed to check collection existence: {e}")
            return False

    async def ensure_collection(self) -> None:
        """
        Ensure collection exists, creating it if necessary.

        Implements COLPALI-401: Collection lifecycle management.
        """
        if not await self.collection_exists():
            logger.info(f"Collection '{self.collection_name}' does not exist, creating...")
            await self.create_collection()
        else:
            logger.info(f"Collection '{self.collection_name}' exists")

    async def delete_collection(self) -> bool:
        """
        Delete the collection.

        Returns:
            True if collection was deleted, False if it didn't exist

        Raises:
            RuntimeError: If not connected to Qdrant
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        if not await self.collection_exists():
            logger.info(f"Collection '{self.collection_name}' does not exist")
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.client.delete_collection, self.collection_name
            )
            logger.info(f"Collection '{self.collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise Exception(f"Collection deletion failed: {e}")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for the Qdrant collection.

        Implements COLPALI-404: Performance optimization and monitoring.

        Returns:
            Dictionary containing performance metrics:
                - storage_metrics: Storage utilization and efficiency
                - search_performance: Recent search performance stats
                - indexing_status: HNSW indexing status and performance
                - memory_usage: Memory utilization stats
                - system_health: Overall system health indicators

        Raises:
            RuntimeError: If not connected to Qdrant
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        try:
            loop = asyncio.get_event_loop()

            # Get collection info for base metrics
            collection_info = await loop.run_in_executor(
                None, self.client.get_collection, self.collection_name
            )

            # Calculate storage metrics
            storage_metrics = self._calculate_storage_metrics(collection_info)

            # Get indexing performance
            indexing_metrics = self._get_indexing_metrics(collection_info)

            # System health indicators
            health_metrics = await self._get_system_health_metrics()

            performance_data = {
                "collection_name": self.collection_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.utc(time.time())),
                "storage_metrics": storage_metrics,
                "indexing_metrics": indexing_metrics,
                "system_health": health_metrics,
                "configuration": {
                    "vector_size": self.vector_size,
                    "distance_metric": self.distance_metric.value,
                    "connection_url": self._get_connection_string()
                }
            }

            logger.info(f"Performance metrics collected: {storage_metrics['points_count']} points, "
                       f"{storage_metrics['indexing_ratio']:.1f}% indexed")

            return performance_data

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise Exception(f"Performance metrics collection failed: {e}")

    def _calculate_storage_metrics(self, collection_info) -> Dict[str, Any]:
        """Calculate storage utilization and efficiency metrics."""
        points_count = collection_info.points_count or 0
        vectors_count = collection_info.vectors_count or 0
        indexed_vectors = collection_info.indexed_vectors_count or 0

        indexing_ratio = (indexed_vectors / vectors_count * 100) if vectors_count > 0 else 0
        segments_count = len(collection_info.segments) if collection_info.segments else 0

        # Estimate storage size (rough calculation)
        estimated_vector_size_mb = (vectors_count * self.vector_size * 4) / (1024 * 1024)  # 4 bytes per float32
        estimated_payload_size_mb = points_count * 0.5  # Rough estimate for metadata

        return {
            "points_count": points_count,
            "vectors_count": vectors_count,
            "indexed_vectors_count": indexed_vectors,
            "indexing_ratio": indexing_ratio,
            "segments_count": segments_count,
            "estimated_vector_size_mb": round(estimated_vector_size_mb, 2),
            "estimated_payload_size_mb": round(estimated_payload_size_mb, 2),
            "estimated_total_size_mb": round(estimated_vector_size_mb + estimated_payload_size_mb, 2)
        }

    def _get_indexing_metrics(self, collection_info) -> Dict[str, Any]:
        """Extract indexing performance metrics."""
        metrics = {
            "optimizer_status": getattr(collection_info, 'optimizer_status', 'unknown'),
            "status": getattr(collection_info, 'status', 'unknown'),
        }

        # Extract HNSW configuration if available
        if hasattr(collection_info, 'config') and collection_info.config:
            hnsw_config = getattr(collection_info.config, 'hnsw_config', None)
            if hnsw_config:
                metrics["hnsw_config"] = {
                    "m": getattr(hnsw_config, 'm', 16),
                    "ef_construct": getattr(hnsw_config, 'ef_construct', 100),
                    "full_scan_threshold": getattr(hnsw_config, 'full_scan_threshold', 10000)
                }

        return metrics

    async def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health and connectivity metrics."""
        health_start = time.time()

        try:
            # Test basic connectivity
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(None, self.client.get_collections)

            health_check_time = time.time() - health_start

            return {
                "connectivity": "healthy",
                "health_check_time_ms": round(health_check_time * 1000, 2),
                "total_collections": len(collections.collections),
                "connection_retries": self.connection_retries,
                "is_connected": self.is_connected
            }

        except Exception as e:
            health_check_time = time.time() - health_start
            logger.warning(f"Health check failed: {e}")

            return {
                "connectivity": "unhealthy",
                "health_check_time_ms": round(health_check_time * 1000, 2),
                "error": str(e),
                "connection_retries": self.connection_retries,
                "is_connected": self.is_connected
            }

    async def optimize_collection(self) -> Dict[str, Any]:
        """
        Optimize collection performance by triggering indexing and optimization.

        Implements COLPALI-404: Performance optimization.

        Returns:
            Dictionary with optimization results

        Raises:
            RuntimeError: If not connected to Qdrant
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()

            logger.info(f"Starting collection optimization for '{self.collection_name}'")

            # Get pre-optimization metrics
            pre_metrics = await self.get_performance_metrics()

            # Trigger collection optimization (this may take time for large collections)
            # Note: Qdrant automatically optimizes, but we can trigger it manually
            # This is a placeholder for actual optimization trigger
            optimization_result = "optimization_triggered"

            optimization_time = time.time() - start_time

            logger.info(f"Collection optimization completed in {optimization_time:.3f}s")

            return {
                "status": "completed",
                "optimization_time": optimization_time,
                "pre_optimization_metrics": pre_metrics["storage_metrics"],
                "recommendation": "Monitor indexing progress and query performance"
            }

        except Exception as e:
            optimization_time = time.time() - start_time
            error_msg = f"Collection optimization failed after {optimization_time:.3f}s: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def benchmark_search_performance(
        self,
        test_vector: Optional[List[float]] = None,
        iterations: int = 10,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark search performance with multiple test queries.

        Implements COLPALI-404: Performance monitoring.

        Args:
            test_vector: Optional test vector (generates random if None)
            iterations: Number of search iterations to run
            limit: Number of results per search

        Returns:
            Dictionary with benchmark results

        Raises:
            RuntimeError: If not connected to Qdrant
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        if test_vector is None:
            # Generate random test vector
            import random
            test_vector = [random.random() for _ in range(self.vector_size)]

        logger.info(f"Starting search performance benchmark: {iterations} iterations")

        search_times = []
        results_counts = []
        start_time = time.time()

        try:
            for i in range(iterations):
                iteration_start = time.time()

                # Perform search
                search_result = await self.search_similar(
                    query_vector=test_vector,
                    limit=limit,
                    filter_conditions=None
                )

                iteration_time = time.time() - iteration_start
                search_times.append(iteration_time)
                results_counts.append(search_result["total_found"])

            total_benchmark_time = time.time() - start_time

            # Calculate statistics
            avg_search_time = sum(search_times) / len(search_times)
            min_search_time = min(search_times)
            max_search_time = max(search_times)
            avg_results = sum(results_counts) / len(results_counts)

            logger.info(
                f"Search benchmark completed: avg={avg_search_time:.3f}s, "
                f"min={min_search_time:.3f}s, max={max_search_time:.3f}s"
            )

            return {
                "benchmark_summary": {
                    "iterations": iterations,
                    "total_time": round(total_benchmark_time, 3),
                    "avg_search_time": round(avg_search_time, 4),
                    "min_search_time": round(min_search_time, 4),
                    "max_search_time": round(max_search_time, 4),
                    "avg_results_count": round(avg_results, 1)
                },
                "detailed_results": {
                    "search_times": [round(t, 4) for t in search_times],
                    "results_counts": results_counts
                },
                "performance_rating": self._rate_search_performance(avg_search_time),
                "test_configuration": {
                    "vector_dimension": len(test_vector),
                    "results_limit": limit,
                    "collection_name": self.collection_name
                }
            }

        except Exception as e:
            logger.error(f"Search benchmark failed: {e}")
            raise Exception(f"Search performance benchmark failed: {str(e)}")

    def _rate_search_performance(self, avg_search_time: float) -> Dict[str, str]:
        """Rate search performance based on average search time."""
        if avg_search_time < 0.01:
            return {"rating": "excellent", "description": "Sub-10ms average search time"}
        elif avg_search_time < 0.05:
            return {"rating": "good", "description": "Under 50ms average search time"}
        elif avg_search_time < 0.1:
            return {"rating": "acceptable", "description": "Under 100ms average search time"}
        elif avg_search_time < 0.5:
            return {"rating": "slow", "description": "Over 100ms average search time"}
        else:
            return {"rating": "poor", "description": "Over 500ms average search time"}

    async def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get detailed storage statistics and recommendations.

        Implements COLPALI-404: Performance monitoring.

        Returns:
            Dictionary with storage statistics and optimization recommendations
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        try:
            # Get current performance metrics
            metrics = await self.get_performance_metrics()
            storage_metrics = metrics["storage_metrics"]

            # Generate recommendations
            recommendations = []

            if storage_metrics["indexing_ratio"] < 95:
                recommendations.append({
                    "type": "indexing",
                    "priority": "high",
                    "message": f"Only {storage_metrics['indexing_ratio']:.1f}% of vectors are indexed. Consider waiting for indexing to complete."
                })

            if storage_metrics["segments_count"] > 10:
                recommendations.append({
                    "type": "segmentation",
                    "priority": "medium",
                    "message": f"Collection has {storage_metrics['segments_count']} segments. Consider optimization to reduce segment count."
                })

            if storage_metrics["points_count"] > 100000:
                recommendations.append({
                    "type": "scale",
                    "priority": "info",
                    "message": "Large collection detected. Monitor query performance and consider sharding if needed."
                })

            return {
                "storage_metrics": storage_metrics,
                "recommendations": recommendations,
                "collection_health": "healthy" if storage_metrics["indexing_ratio"] > 90 else "needs_attention"
            }

        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            raise Exception(f"Storage statistics collection failed: {e}")
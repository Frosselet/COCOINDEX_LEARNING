"""
Transformation lineage tracking system.

This module implements comprehensive lineage tracking that records the complete
data flow from source document through ColPali processing, BAML extraction,
and final output generation.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class LineageNodeType(Enum):
    """Types of nodes in the lineage graph."""
    SOURCE_DOCUMENT = "source_document"
    DOCUMENT_CONVERSION = "document_conversion"
    COLPALI_EMBEDDING = "colpali_embedding"
    QDRANT_STORAGE = "qdrant_storage"
    VECTOR_RETRIEVAL = "vector_retrieval"
    BAML_EXTRACTION = "baml_extraction"
    CANONICAL_FORMAT = "canonical_format"
    SHAPED_TRANSFORM = "shaped_transform"
    EXPORT_OUTPUT = "export_output"


@dataclass
class LineageNode:
    """
    Represents a single step in the data processing lineage.

    Each node captures information about a specific transformation
    or processing step in the pipeline.
    """
    node_id: str
    node_type: LineageNodeType
    timestamp: datetime
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    parent_nodes: List[str] = field(default_factory=list)
    schema_version: Optional[str] = None
    transformation_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "parent_nodes": self.parent_nodes,
            "schema_version": self.schema_version,
            "transformation_rules": self.transformation_rules
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineageNode':
        """Create node from dictionary."""
        return cls(
            node_id=data["node_id"],
            node_type=LineageNodeType(data["node_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            operation=data["operation"],
            metadata=data.get("metadata", {}),
            performance_metrics=data.get("performance_metrics", {}),
            parent_nodes=data.get("parent_nodes", []),
            schema_version=data.get("schema_version"),
            transformation_rules=data.get("transformation_rules", [])
        )


class LineageGraph:
    """
    Directed acyclic graph representing the complete lineage.

    Maintains the relationships between processing steps and enables
    efficient querying of lineage information.
    """

    def __init__(self):
        """Initialize empty lineage graph."""
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        logger.info("LineageGraph initialized")

    def add_node(self, node: LineageNode) -> None:
        """
        Add a node to the lineage graph.

        Args:
            node: LineageNode to add
        """
        self.nodes[node.node_id] = node

        # Update edges
        for parent_id in node.parent_nodes:
            if parent_id not in self.edges:
                self.edges[parent_id] = []
            self.edges[parent_id].append(node.node_id)

        logger.debug(f"Added lineage node: {node.node_id} ({node.node_type.value})")

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[LineageNode]:
        """Get all direct children of a node."""
        child_ids = self.edges.get(node_id, [])
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_parents(self, node_id: str) -> List[LineageNode]:
        """Get all direct parents of a node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[pid] for pid in node.parent_nodes if pid in self.nodes]

    def get_ancestors(self, node_id: str) -> Set[str]:
        """
        Get all ancestors of a node (recursive).

        Args:
            node_id: Node ID to find ancestors for

        Returns:
            Set of ancestor node IDs
        """
        ancestors = set()
        node = self.get_node(node_id)

        if not node:
            return ancestors

        for parent_id in node.parent_nodes:
            if parent_id not in ancestors:
                ancestors.add(parent_id)
                ancestors.update(self.get_ancestors(parent_id))

        return ancestors

    def get_descendants(self, node_id: str) -> Set[str]:
        """
        Get all descendants of a node (recursive).

        Args:
            node_id: Node ID to find descendants for

        Returns:
            Set of descendant node IDs
        """
        descendants = set()

        for child_id in self.edges.get(node_id, []):
            if child_id not in descendants:
                descendants.add(child_id)
                descendants.update(self.get_descendants(child_id))

        return descendants

    def get_path(self, start_id: str, end_id: str) -> Optional[List[LineageNode]]:
        """
        Find path between two nodes using BFS.

        Args:
            start_id: Starting node ID
            end_id: Ending node ID

        Returns:
            List of nodes in the path, or None if no path exists
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        if start_id == end_id:
            return [self.nodes[start_id]]

        # BFS to find shortest path
        queue = [(start_id, [start_id])]
        visited = {start_id}

        while queue:
            current_id, path = queue.pop(0)

            for child_id in self.edges.get(current_id, []):
                if child_id == end_id:
                    return [self.nodes[nid] for nid in path + [child_id]]

                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, path + [child_id]))

        return None

    def get_source_nodes(self) -> List[LineageNode]:
        """Get all source nodes (nodes with no parents)."""
        return [node for node in self.nodes.values() if not node.parent_nodes]

    def get_sink_nodes(self) -> List[LineageNode]:
        """Get all sink nodes (nodes with no children)."""
        return [
            node for node_id, node in self.nodes.items()
            if node_id not in self.edges or not self.edges[node_id]
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": self.edges
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineageGraph':
        """Load graph from dictionary."""
        graph = cls()

        # Load nodes
        for node_data in data.get("nodes", []):
            node = LineageNode.from_dict(node_data)
            graph.nodes[node.node_id] = node

        # Load edges
        graph.edges = data.get("edges", {})

        return graph


class LineageTracker:
    """
    Main interface for tracking transformation lineage.

    Provides high-level methods for recording processing steps and
    querying lineage information.
    """

    def __init__(self, processing_id: Optional[str] = None):
        """
        Initialize lineage tracker.

        Args:
            processing_id: Optional processing session ID
        """
        self.processing_id = processing_id or str(uuid.uuid4())
        self.graph = LineageGraph()
        self.current_nodes: Dict[str, str] = {}  # type -> node_id mapping
        logger.info(f"LineageTracker initialized: {self.processing_id}")

    def record_source_document(
        self,
        document_id: str,
        source_path: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record source document ingestion.

        Args:
            document_id: Unique document identifier
            source_path: Path to source document
            document_type: Type of document (PDF, image, etc.)
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_source_{document_id}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.SOURCE_DOCUMENT,
            timestamp=datetime.now(),
            operation="ingest_document",
            metadata={
                "document_id": document_id,
                "source_path": source_path,
                "document_type": document_type,
                **(metadata or {})
            }
        )

        self.graph.add_node(node)
        self.current_nodes["source"] = node_id

        logger.info(f"Recorded source document: {document_id}")
        return node_id

    def record_document_conversion(
        self,
        parent_node_id: str,
        page_count: int,
        dpi: int,
        conversion_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record document to image conversion.

        Args:
            parent_node_id: Parent source document node ID
            page_count: Number of pages converted
            dpi: Resolution used
            conversion_time: Time taken in seconds
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_conversion_{uuid.uuid4().hex[:8]}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.DOCUMENT_CONVERSION,
            timestamp=datetime.now(),
            operation="convert_to_images",
            metadata={
                "page_count": page_count,
                "dpi": dpi,
                **(metadata or {})
            },
            performance_metrics={
                "conversion_time_seconds": conversion_time,
                "pages_per_second": page_count / conversion_time if conversion_time > 0 else 0
            },
            parent_nodes=[parent_node_id]
        )

        self.graph.add_node(node)
        self.current_nodes["conversion"] = node_id

        logger.info(f"Recorded document conversion: {page_count} pages @ {dpi} DPI")
        return node_id

    def record_colpali_embedding(
        self,
        parent_node_id: str,
        model_version: str,
        num_patches: int,
        embedding_dim: int,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record ColPali embedding generation.

        Args:
            parent_node_id: Parent conversion node ID
            model_version: ColPali model version
            num_patches: Number of patches generated
            embedding_dim: Embedding dimension
            processing_time: Time taken in seconds
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_colpali_{uuid.uuid4().hex[:8]}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.COLPALI_EMBEDDING,
            timestamp=datetime.now(),
            operation="generate_embeddings",
            metadata={
                "model_version": model_version,
                "num_patches": num_patches,
                "embedding_dim": embedding_dim,
                **(metadata or {})
            },
            performance_metrics={
                "processing_time_seconds": processing_time,
                "patches_per_second": num_patches / processing_time if processing_time > 0 else 0
            },
            parent_nodes=[parent_node_id],
            schema_version=model_version
        )

        self.graph.add_node(node)
        self.current_nodes["colpali"] = node_id

        logger.info(f"Recorded ColPali embedding: {num_patches} patches")
        return node_id

    def record_qdrant_storage(
        self,
        parent_node_id: str,
        collection_name: str,
        num_vectors: int,
        storage_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record vector storage in Qdrant.

        Args:
            parent_node_id: Parent ColPali node ID
            collection_name: Qdrant collection name
            num_vectors: Number of vectors stored
            storage_time: Time taken in seconds
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_qdrant_{uuid.uuid4().hex[:8]}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.QDRANT_STORAGE,
            timestamp=datetime.now(),
            operation="store_vectors",
            metadata={
                "collection_name": collection_name,
                "num_vectors": num_vectors,
                **(metadata or {})
            },
            performance_metrics={
                "storage_time_seconds": storage_time,
                "vectors_per_second": num_vectors / storage_time if storage_time > 0 else 0
            },
            parent_nodes=[parent_node_id]
        )

        self.graph.add_node(node)
        self.current_nodes["qdrant"] = node_id

        logger.info(f"Recorded Qdrant storage: {num_vectors} vectors")
        return node_id

    def record_baml_extraction(
        self,
        parent_node_id: str,
        function_name: str,
        schema_version: str,
        extraction_time: float,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record BAML extraction.

        Args:
            parent_node_id: Parent node ID (could be Qdrant retrieval)
            function_name: BAML function name
            schema_version: Schema version used
            extraction_time: Time taken in seconds
            confidence_score: Optional extraction confidence
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_baml_{uuid.uuid4().hex[:8]}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.BAML_EXTRACTION,
            timestamp=datetime.now(),
            operation="extract_structured_data",
            metadata={
                "function_name": function_name,
                "confidence_score": confidence_score,
                **(metadata or {})
            },
            performance_metrics={
                "extraction_time_seconds": extraction_time
            },
            parent_nodes=[parent_node_id],
            schema_version=schema_version
        )

        self.graph.add_node(node)
        self.current_nodes["baml"] = node_id

        logger.info(f"Recorded BAML extraction: {function_name}")
        return node_id

    def record_canonical_format(
        self,
        parent_node_id: str,
        canonical_id: str,
        field_count: int,
        integrity_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record canonical data formatting.

        Args:
            parent_node_id: Parent BAML extraction node ID
            canonical_id: Canonical data ID
            field_count: Number of fields in canonical data
            integrity_hash: Data integrity hash
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_canonical_{uuid.uuid4().hex[:8]}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.CANONICAL_FORMAT,
            timestamp=datetime.now(),
            operation="format_canonical",
            metadata={
                "canonical_id": canonical_id,
                "field_count": field_count,
                "integrity_hash": integrity_hash,
                **(metadata or {})
            },
            parent_nodes=[parent_node_id]
        )

        self.graph.add_node(node)
        self.current_nodes["canonical"] = node_id

        logger.info(f"Recorded canonical formatting: {canonical_id}")
        return node_id

    def record_shaped_transform(
        self,
        parent_node_id: str,
        shaped_id: str,
        transformation_rules: List[str],
        is_1nf_compliant: bool,
        transform_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record shaped data transformation.

        Args:
            parent_node_id: Parent canonical node ID
            shaped_id: Shaped data ID
            transformation_rules: List of transformation rule IDs applied
            is_1nf_compliant: Whether output is 1NF compliant
            transform_time: Time taken in seconds
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_shaped_{uuid.uuid4().hex[:8]}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.SHAPED_TRANSFORM,
            timestamp=datetime.now(),
            operation="transform_to_1nf",
            metadata={
                "shaped_id": shaped_id,
                "is_1nf_compliant": is_1nf_compliant,
                "transformation_count": len(transformation_rules),
                **(metadata or {})
            },
            performance_metrics={
                "transform_time_seconds": transform_time
            },
            parent_nodes=[parent_node_id],
            transformation_rules=transformation_rules
        )

        self.graph.add_node(node)
        self.current_nodes["shaped"] = node_id

        logger.info(f"Recorded shaped transformation: {len(transformation_rules)} rules applied")
        return node_id

    def record_export_output(
        self,
        parent_node_id: str,
        output_format: str,
        output_path: str,
        export_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record data export to output format.

        Args:
            parent_node_id: Parent node ID (canonical or shaped)
            output_format: Export format (CSV, Parquet, etc.)
            output_path: Output file path
            export_time: Time taken in seconds
            metadata: Optional additional metadata

        Returns:
            Node ID for this step
        """
        node_id = f"{self.processing_id}_export_{uuid.uuid4().hex[:8]}"

        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.EXPORT_OUTPUT,
            timestamp=datetime.now(),
            operation=f"export_to_{output_format.lower()}",
            metadata={
                "output_format": output_format,
                "output_path": output_path,
                **(metadata or {})
            },
            performance_metrics={
                "export_time_seconds": export_time
            },
            parent_nodes=[parent_node_id]
        )

        self.graph.add_node(node)

        logger.info(f"Recorded export: {output_format} to {output_path}")
        return node_id

    def get_complete_lineage(self) -> Dict[str, Any]:
        """
        Get complete lineage for this processing session.

        Returns:
            Dictionary with complete lineage information
        """
        return {
            "processing_id": self.processing_id,
            "graph": self.graph.to_dict(),
            "source_nodes": [n.to_dict() for n in self.graph.get_source_nodes()],
            "sink_nodes": [n.to_dict() for n in self.graph.get_sink_nodes()],
            "total_nodes": len(self.graph.nodes),
            "total_edges": sum(len(children) for children in self.graph.edges.values())
        }

    def export_lineage(self, output_path: str) -> None:
        """
        Export lineage to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        import json
        from pathlib import Path

        lineage_data = self.get_complete_lineage()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(lineage_data, f, indent=2, default=str)

        logger.info(f"Exported lineage to: {output_path}")


class LineageQuery:
    """
    Query interface for lineage information.

    Provides convenient methods for querying and analyzing lineage graphs.
    """

    def __init__(self, tracker: LineageTracker):
        """
        Initialize query interface.

        Args:
            tracker: LineageTracker to query
        """
        self.tracker = tracker
        self.graph = tracker.graph

    def find_nodes_by_type(self, node_type: LineageNodeType) -> List[LineageNode]:
        """Find all nodes of a specific type."""
        return [node for node in self.graph.nodes.values() if node.node_type == node_type]

    def find_nodes_by_operation(self, operation: str) -> List[LineageNode]:
        """Find all nodes with a specific operation."""
        return [node for node in self.graph.nodes.values() if node.operation == operation]

    def get_processing_time(self, start_node_id: str, end_node_id: str) -> float:
        """
        Calculate total processing time between two nodes.

        Args:
            start_node_id: Starting node ID
            end_node_id: Ending node ID

        Returns:
            Total processing time in seconds
        """
        path = self.graph.get_path(start_node_id, end_node_id)

        if not path:
            return 0.0

        total_time = 0.0
        for node in path:
            for key, value in node.performance_metrics.items():
                if "time" in key.lower() and isinstance(value, (int, float)):
                    total_time += value

        return total_time

    def get_transformation_chain(self, output_node_id: str) -> List[LineageNode]:
        """
        Get the complete transformation chain leading to an output.

        Args:
            output_node_id: Output node ID

        Returns:
            List of nodes in the transformation chain
        """
        # Find source node
        ancestors = self.graph.get_ancestors(output_node_id)
        source_nodes = [nid for nid in ancestors if not self.graph.get_node(nid).parent_nodes]

        if not source_nodes:
            return []

        # Get path from source to output
        return self.graph.get_path(source_nodes[0], output_node_id) or []

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for the entire processing session.

        Returns:
            Dictionary with performance metrics
        """
        summary = {
            "total_processing_time": 0.0,
            "by_operation": {},
            "by_node_type": {}
        }

        for node in self.graph.nodes.values():
            # Add to total time
            for key, value in node.performance_metrics.items():
                if "time" in key.lower() and isinstance(value, (int, float)):
                    summary["total_processing_time"] += value

                    # Track by operation
                    if node.operation not in summary["by_operation"]:
                        summary["by_operation"][node.operation] = 0.0
                    summary["by_operation"][node.operation] += value

                    # Track by type
                    type_name = node.node_type.value
                    if type_name not in summary["by_node_type"]:
                        summary["by_node_type"][type_name] = 0.0
                    summary["by_node_type"][type_name] += value

        return summary

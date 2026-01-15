"""
Tests for LineageTracker - Transformation lineage tracking system.

COLPALI-801: Lineage tracking implementation tests.
"""

import pytest
from datetime import datetime
from tatforge.governance.lineage import (
    LineageTracker,
    LineageNode,
    LineageGraph,
    LineageQuery,
    LineageNodeType
)


class TestLineageNode:
    """Test suite for LineageNode."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = LineageNode(
            node_id="test_001",
            node_type=LineageNodeType.SOURCE_DOCUMENT,
            timestamp=datetime.now(),
            operation="test_operation"
        )

        assert node.node_id == "test_001"
        assert node.node_type == LineageNodeType.SOURCE_DOCUMENT
        assert node.operation == "test_operation"
        assert len(node.parent_nodes) == 0

    def test_node_with_metadata(self):
        """Test node with metadata."""
        metadata = {"document_id": "doc_001", "source": "test.pdf"}

        node = LineageNode(
            node_id="test_002",
            node_type=LineageNodeType.DOCUMENT_CONVERSION,
            timestamp=datetime.now(),
            operation="convert",
            metadata=metadata
        )

        assert node.metadata["document_id"] == "doc_001"
        assert node.metadata["source"] == "test.pdf"

    def test_node_with_performance_metrics(self):
        """Test node with performance metrics."""
        metrics = {"processing_time": 1.5, "throughput": 100}

        node = LineageNode(
            node_id="test_003",
            node_type=LineageNodeType.COLPALI_EMBEDDING,
            timestamp=datetime.now(),
            operation="embed",
            performance_metrics=metrics
        )

        assert node.performance_metrics["processing_time"] == 1.5
        assert node.performance_metrics["throughput"] == 100

    def test_node_serialization(self):
        """Test node to_dict serialization."""
        node = LineageNode(
            node_id="test_004",
            node_type=LineageNodeType.BAML_EXTRACTION,
            timestamp=datetime.now(),
            operation="extract",
            metadata={"field": "value"},
            parent_nodes=["parent_001"]
        )

        node_dict = node.to_dict()

        assert node_dict["node_id"] == "test_004"
        assert node_dict["node_type"] == "baml_extraction"
        assert node_dict["operation"] == "extract"
        assert "timestamp" in node_dict
        assert node_dict["metadata"]["field"] == "value"
        assert "parent_001" in node_dict["parent_nodes"]

    def test_node_deserialization(self):
        """Test node from_dict deserialization."""
        node_dict = {
            "node_id": "test_005",
            "node_type": "canonical_format",
            "timestamp": "2024-01-15T10:30:00",
            "operation": "format",
            "metadata": {"test": "data"},
            "performance_metrics": {},
            "parent_nodes": [],
            "schema_version": "1.0",
            "transformation_rules": []
        }

        node = LineageNode.from_dict(node_dict)

        assert node.node_id == "test_005"
        assert node.node_type == LineageNodeType.CANONICAL_FORMAT
        assert node.operation == "format"


class TestLineageGraph:
    """Test suite for LineageGraph."""

    @pytest.fixture
    def graph(self):
        """Create empty graph."""
        return LineageGraph()

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes for testing."""
        source = LineageNode(
            node_id="source_001",
            node_type=LineageNodeType.SOURCE_DOCUMENT,
            timestamp=datetime.now(),
            operation="ingest"
        )

        convert = LineageNode(
            node_id="convert_001",
            node_type=LineageNodeType.DOCUMENT_CONVERSION,
            timestamp=datetime.now(),
            operation="convert",
            parent_nodes=["source_001"]
        )

        embed = LineageNode(
            node_id="embed_001",
            node_type=LineageNodeType.COLPALI_EMBEDDING,
            timestamp=datetime.now(),
            operation="embed",
            parent_nodes=["convert_001"]
        )

        return [source, convert, embed]

    def test_add_node(self, graph, sample_nodes):
        """Test adding nodes to graph."""
        for node in sample_nodes:
            graph.add_node(node)

        assert len(graph.nodes) == 3
        assert "source_001" in graph.nodes
        assert "convert_001" in graph.nodes
        assert "embed_001" in graph.nodes

    def test_get_node(self, graph, sample_nodes):
        """Test retrieving nodes."""
        for node in sample_nodes:
            graph.add_node(node)

        retrieved = graph.get_node("convert_001")
        assert retrieved is not None
        assert retrieved.node_id == "convert_001"

    def test_get_children(self, graph, sample_nodes):
        """Test getting child nodes."""
        for node in sample_nodes:
            graph.add_node(node)

        children = graph.get_children("source_001")
        assert len(children) == 1
        assert children[0].node_id == "convert_001"

    def test_get_parents(self, graph, sample_nodes):
        """Test getting parent nodes."""
        for node in sample_nodes:
            graph.add_node(node)

        parents = graph.get_parents("embed_001")
        assert len(parents) == 1
        assert parents[0].node_id == "convert_001"

    def test_get_ancestors(self, graph, sample_nodes):
        """Test getting all ancestors."""
        for node in sample_nodes:
            graph.add_node(node)

        ancestors = graph.get_ancestors("embed_001")
        assert len(ancestors) == 2
        assert "source_001" in ancestors
        assert "convert_001" in ancestors

    def test_get_descendants(self, graph, sample_nodes):
        """Test getting all descendants."""
        for node in sample_nodes:
            graph.add_node(node)

        descendants = graph.get_descendants("source_001")
        assert len(descendants) == 2
        assert "convert_001" in descendants
        assert "embed_001" in descendants

    def test_get_path(self, graph, sample_nodes):
        """Test finding path between nodes."""
        for node in sample_nodes:
            graph.add_node(node)

        path = graph.get_path("source_001", "embed_001")
        assert path is not None
        assert len(path) == 3
        assert path[0].node_id == "source_001"
        assert path[1].node_id == "convert_001"
        assert path[2].node_id == "embed_001"

    def test_get_path_no_connection(self, graph, sample_nodes):
        """Test path finding with no connection."""
        graph.add_node(sample_nodes[0])

        # Add unconnected node
        isolated = LineageNode(
            node_id="isolated_001",
            node_type=LineageNodeType.EXPORT_OUTPUT,
            timestamp=datetime.now(),
            operation="export"
        )
        graph.add_node(isolated)

        path = graph.get_path("source_001", "isolated_001")
        assert path is None

    def test_get_source_nodes(self, graph, sample_nodes):
        """Test getting source nodes."""
        for node in sample_nodes:
            graph.add_node(node)

        sources = graph.get_source_nodes()
        assert len(sources) == 1
        assert sources[0].node_id == "source_001"

    def test_get_sink_nodes(self, graph, sample_nodes):
        """Test getting sink nodes."""
        for node in sample_nodes:
            graph.add_node(node)

        sinks = graph.get_sink_nodes()
        assert len(sinks) == 1
        assert sinks[0].node_id == "embed_001"

    def test_graph_serialization(self, graph, sample_nodes):
        """Test graph to_dict serialization."""
        for node in sample_nodes:
            graph.add_node(node)

        graph_dict = graph.to_dict()

        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        assert len(graph_dict["nodes"]) == 3

    def test_graph_deserialization(self, graph, sample_nodes):
        """Test graph from_dict deserialization."""
        for node in sample_nodes:
            graph.add_node(node)

        graph_dict = graph.to_dict()
        restored_graph = LineageGraph.from_dict(graph_dict)

        assert len(restored_graph.nodes) == len(graph.nodes)
        assert len(restored_graph.edges) == len(graph.edges)


class TestLineageTracker:
    """Test suite for LineageTracker."""

    @pytest.fixture
    def tracker(self):
        """Create lineage tracker."""
        return LineageTracker(processing_id="test_processing_001")

    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.processing_id == "test_processing_001"
        assert tracker.graph is not None
        assert len(tracker.graph.nodes) == 0

    def test_record_source_document(self, tracker):
        """Test recording source document."""
        node_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        assert node_id is not None
        assert "source" in tracker.current_nodes
        assert tracker.graph.get_node(node_id) is not None

    def test_record_document_conversion(self, tracker):
        """Test recording document conversion."""
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        convert_id = tracker.record_document_conversion(
            parent_node_id=source_id,
            page_count=10,
            dpi=300,
            conversion_time=2.5
        )

        assert convert_id is not None
        node = tracker.graph.get_node(convert_id)
        assert node.metadata["page_count"] == 10
        assert node.metadata["dpi"] == 300
        assert node.performance_metrics["conversion_time_seconds"] == 2.5

    def test_record_colpali_embedding(self, tracker):
        """Test recording ColPali embedding."""
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        convert_id = tracker.record_document_conversion(
            parent_node_id=source_id,
            page_count=5,
            dpi=300,
            conversion_time=1.0
        )

        embed_id = tracker.record_colpali_embedding(
            parent_node_id=convert_id,
            model_version="ColQwen2-v0.1",
            num_patches=1024,
            embedding_dim=128,
            processing_time=3.5
        )

        assert embed_id is not None
        node = tracker.graph.get_node(embed_id)
        assert node.metadata["model_version"] == "ColQwen2-v0.1"
        assert node.metadata["num_patches"] == 1024

    def test_record_qdrant_storage(self, tracker):
        """Test recording Qdrant storage."""
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        storage_id = tracker.record_qdrant_storage(
            parent_node_id=source_id,
            collection_name="test_collection",
            num_vectors=1024,
            storage_time=0.5
        )

        assert storage_id is not None
        node = tracker.graph.get_node(storage_id)
        assert node.metadata["collection_name"] == "test_collection"
        assert node.metadata["num_vectors"] == 1024

    def test_record_baml_extraction(self, tracker):
        """Test recording BAML extraction."""
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        baml_id = tracker.record_baml_extraction(
            parent_node_id=source_id,
            function_name="extract_invoice",
            schema_version="1.0",
            extraction_time=2.0,
            confidence_score=0.95
        )

        assert baml_id is not None
        node = tracker.graph.get_node(baml_id)
        assert node.metadata["function_name"] == "extract_invoice"
        assert node.metadata["confidence_score"] == 0.95

    def test_record_canonical_format(self, tracker):
        """Test recording canonical formatting."""
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        canonical_id = tracker.record_canonical_format(
            parent_node_id=source_id,
            canonical_id="canonical_001",
            field_count=10,
            integrity_hash="abc123def456"
        )

        assert canonical_id is not None
        node = tracker.graph.get_node(canonical_id)
        assert node.metadata["canonical_id"] == "canonical_001"
        assert node.metadata["field_count"] == 10

    def test_record_shaped_transform(self, tracker):
        """Test recording shaped transformation."""
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        shaped_id = tracker.record_shaped_transform(
            parent_node_id=source_id,
            shaped_id="shaped_001",
            transformation_rules=["rule1", "rule2"],
            is_1nf_compliant=True,
            transform_time=1.0
        )

        assert shaped_id is not None
        node = tracker.graph.get_node(shaped_id)
        assert node.metadata["shaped_id"] == "shaped_001"
        assert node.metadata["is_1nf_compliant"] is True
        assert len(node.transformation_rules) == 2

    def test_record_export_output(self, tracker):
        """Test recording export output."""
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        export_id = tracker.record_export_output(
            parent_node_id=source_id,
            output_format="Parquet",
            output_path="/output/data.parquet",
            export_time=0.8
        )

        assert export_id is not None
        node = tracker.graph.get_node(export_id)
        assert node.metadata["output_format"] == "Parquet"
        assert node.metadata["output_path"] == "/output/data.parquet"

    def test_complete_pipeline_lineage(self, tracker):
        """Test recording complete pipeline lineage."""
        # Record complete pipeline
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        convert_id = tracker.record_document_conversion(
            parent_node_id=source_id,
            page_count=5,
            dpi=300,
            conversion_time=1.0
        )

        embed_id = tracker.record_colpali_embedding(
            parent_node_id=convert_id,
            model_version="ColQwen2-v0.1",
            num_patches=1024,
            embedding_dim=128,
            processing_time=2.0
        )

        storage_id = tracker.record_qdrant_storage(
            parent_node_id=embed_id,
            collection_name="test_collection",
            num_vectors=1024,
            storage_time=0.5
        )

        baml_id = tracker.record_baml_extraction(
            parent_node_id=storage_id,
            function_name="extract_invoice",
            schema_version="1.0",
            extraction_time=1.5
        )

        canonical_id = tracker.record_canonical_format(
            parent_node_id=baml_id,
            canonical_id="canonical_001",
            field_count=10,
            integrity_hash="abc123"
        )

        shaped_id = tracker.record_shaped_transform(
            parent_node_id=canonical_id,
            shaped_id="shaped_001",
            transformation_rules=["flatten"],
            is_1nf_compliant=True,
            transform_time=0.5
        )

        export_id = tracker.record_export_output(
            parent_node_id=shaped_id,
            output_format="Parquet",
            output_path="/output/data.parquet",
            export_time=0.3
        )

        # Verify complete lineage
        lineage = tracker.get_complete_lineage()
        assert lineage["total_nodes"] == 8
        assert len(lineage["source_nodes"]) == 1
        assert len(lineage["sink_nodes"]) == 1

    def test_get_complete_lineage(self, tracker):
        """Test getting complete lineage."""
        tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        lineage = tracker.get_complete_lineage()

        assert "processing_id" in lineage
        assert "graph" in lineage
        assert "total_nodes" in lineage
        assert lineage["total_nodes"] == 1


class TestLineageQuery:
    """Test suite for LineageQuery."""

    @pytest.fixture
    def populated_tracker(self):
        """Create tracker with populated lineage."""
        tracker = LineageTracker(processing_id="test_query")

        # Build simple pipeline
        source_id = tracker.record_source_document(
            document_id="doc_001",
            source_path="/path/to/doc.pdf",
            document_type="PDF"
        )

        convert_id = tracker.record_document_conversion(
            parent_node_id=source_id,
            page_count=5,
            dpi=300,
            conversion_time=2.0
        )

        embed_id = tracker.record_colpali_embedding(
            parent_node_id=convert_id,
            model_version="ColQwen2-v0.1",
            num_patches=1024,
            embedding_dim=128,
            processing_time=3.0
        )

        return tracker

    def test_query_initialization(self, populated_tracker):
        """Test query interface initialization."""
        query = LineageQuery(populated_tracker)

        assert query.tracker is not None
        assert query.graph is not None

    def test_find_nodes_by_type(self, populated_tracker):
        """Test finding nodes by type."""
        query = LineageQuery(populated_tracker)

        source_nodes = query.find_nodes_by_type(LineageNodeType.SOURCE_DOCUMENT)
        assert len(source_nodes) == 1

        conversion_nodes = query.find_nodes_by_type(LineageNodeType.DOCUMENT_CONVERSION)
        assert len(conversion_nodes) == 1

    def test_find_nodes_by_operation(self, populated_tracker):
        """Test finding nodes by operation."""
        query = LineageQuery(populated_tracker)

        ingest_nodes = query.find_nodes_by_operation("ingest_document")
        assert len(ingest_nodes) == 1

        convert_nodes = query.find_nodes_by_operation("convert_to_images")
        assert len(convert_nodes) == 1

    def test_get_processing_time(self, populated_tracker):
        """Test calculating processing time."""
        query = LineageQuery(populated_tracker)

        sources = query.find_nodes_by_type(LineageNodeType.SOURCE_DOCUMENT)
        embeddings = query.find_nodes_by_type(LineageNodeType.COLPALI_EMBEDDING)

        if sources and embeddings:
            total_time = query.get_processing_time(sources[0].node_id, embeddings[0].node_id)
            assert total_time > 0

    def test_get_transformation_chain(self, populated_tracker):
        """Test getting transformation chain."""
        query = LineageQuery(populated_tracker)

        embeddings = query.find_nodes_by_type(LineageNodeType.COLPALI_EMBEDDING)

        if embeddings:
            chain = query.get_transformation_chain(embeddings[0].node_id)
            assert len(chain) == 3  # source -> convert -> embed

    def test_get_performance_summary(self, populated_tracker):
        """Test getting performance summary."""
        query = LineageQuery(populated_tracker)

        summary = query.get_performance_summary()

        assert "total_processing_time" in summary
        assert summary["total_processing_time"] > 0
        assert "by_operation" in summary
        assert "by_node_type" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

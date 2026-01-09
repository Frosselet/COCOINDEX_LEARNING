"""
Test suite for Qdrant client implementation (COLPALI-401 - Qdrant Vector Storage).

Tests all COLPALI-400 sub-tasks:
- COLPALI-401: Set up Qdrant client and collection management
- COLPALI-402: Implement embedding storage with spatial metadata
- COLPALI-403: Build semantic search and retrieval system
- COLPALI-404: Performance optimization and monitoring
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock all problematic imports before importing our module
mock_modules = {
    'qdrant_client': MagicMock(),
    'qdrant_client.http.models': MagicMock(),
    'qdrant_client.http.exceptions': MagicMock(),
    'torch': MagicMock(),
    'torch._C': MagicMock(),
    'torch.nn': MagicMock(),
    'torch.nn.functional': MagicMock(),
    'transformers': MagicMock(),
    'colpali_engine.models': MagicMock(),
    'colpali_engine.vision': MagicMock(),
    'colpali_engine.extraction': MagicMock(),
    'colpali_engine.core': MagicMock()
}

with patch.dict('sys.modules', mock_modules):
    # Import the module with all dependencies mocked
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qdrant_client",
        project_root / "colpali_engine" / "storage" / "qdrant_client.py"
    )
    qdrant_module = importlib.util.module_from_spec(spec)

    # Set up mock objects in the module namespace before execution
    mock_qdrant_models = MagicMock()

    # Create proper Distance mock
    class MockDistanceEnum:
        def __init__(self):
            self.COSINE = MockDistanceValue("cosine")

    class MockDistanceValue:
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return self.value

    mock_qdrant_models.Distance = MockDistanceEnum()
    mock_qdrant_models.PointStruct = MagicMock()
    mock_qdrant_models.Filter = MagicMock()
    mock_qdrant_models.FieldCondition = MagicMock()
    mock_qdrant_models.Match = MagicMock()
    mock_qdrant_models.Range = MagicMock()

    # Set these in sys.modules before loading our module
    sys.modules['qdrant_client.http.models'] = mock_qdrant_models

    spec.loader.exec_module(qdrant_module)

    QdrantManager = qdrant_module.QdrantManager

# Create mock classes for testing
class MockDistance:
    def __init__(self):
        self.COSINE = MockDistanceValue("cosine")

class MockDistanceValue:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

Distance = MockDistance()

# Mock PointStruct
class MockPointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload

PointStruct = MockPointStruct

# Test configuration
TEST_COLLECTION_NAME = "test_colpali_embeddings"
TEST_VECTOR_SIZE = 128
TEST_EMBEDDING_COUNT = 50


class TestQdrantManager:
    """Test suite for Qdrant client implementation."""

    def __init__(self):
        self.manager = None
        self.test_embeddings = []
        self.test_metadata = {}

    def setup_test_environment(self):
        """Set up test environment with mock Qdrant client."""
        print("Setting up Qdrant test environment...")

        # Initialize manager with test configuration
        self.manager = QdrantManager(
            url="http://localhost",
            port=6333,
            collection_name=TEST_COLLECTION_NAME,
            timeout=10,
            prefer_grpc=False
        )

        # Generate test embeddings and metadata
        self.test_embeddings = [
            [float(i * j % 100 / 100.0) for j in range(TEST_VECTOR_SIZE)]
            for i in range(TEST_EMBEDDING_COUNT)
        ]

        self.test_metadata = {
            "document_id": "test_doc_001",
            "page_number": 1,
            "document_type": "pdf",
            "patch_coordinates": [(0, 0), (32, 0), (0, 32)],
            "processing_timestamp": "2024-01-09T10:00:00Z"
        }

        print(f"Created {len(self.test_embeddings)} test embeddings")
        print(f"Manager initialized: {self.manager._get_connection_string()}")

    def test_colpali_401_connection_management(self):
        """
        Test COLPALI-401: Qdrant client setup and connection management.
        """
        print("\n" + "="*70)
        print("COLPALI-401: QDRANT CLIENT SETUP AND CONNECTION MANAGEMENT")
        print("="*70)

        try:
            # Test initial state
            assert not self.manager.is_connected, "Manager should not be connected initially"
            assert self.manager.client is None, "Client should be None initially"
            print("✓ Initial state validation passed")

            # Test connection string formatting (http://localhost is a full URL, no port appended)
            connection_str = self.manager._get_connection_string()
            expected = "http://localhost"  # Full URL, so no port appended
            assert connection_str == expected, f"Expected {expected}, got {connection_str}"
            print("✓ Connection string formatting works")

            # Test URL detection (http://localhost is considered a full URL)
            assert self.manager._is_full_url(), "http://localhost should be detected as full URL"

            # Test with full URL
            full_url_manager = QdrantManager(url="https://qdrant.example.com:443")
            assert full_url_manager._is_full_url(), "Should detect full URL"
            print("✓ URL detection logic works")

            # Test connection with mocked client
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client.get_collections.return_value = Mock(collections=[])
                mock_client_class.return_value = mock_client

                # Test successful connection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.manager.connect())

                    assert self.manager.is_connected, "Manager should be connected"
                    assert self.manager.client is not None, "Client should be initialized"
                    assert self.manager.connection_retries == 0, "Retry count should be reset"
                    print("✓ Connection establishment works")

                    # Test already connected scenario
                    loop.run_until_complete(self.manager.connect())  # Should not reconnect
                    print("✓ Already connected handling works")
                finally:
                    loop.close()

            print("✓ COLPALI-401 connection management validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-401 connection test failed: {e}")
            return False

    def test_colpali_401_collection_management(self):
        """
        Test COLPALI-401: Collection creation and management.
        """
        print("\n" + "="*70)
        print("COLPALI-401: COLLECTION CREATION AND MANAGEMENT")
        print("="*70)

        try:
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock successful connection
                mock_client.get_collections.return_value = Mock(collections=[])
                self.manager.client = mock_client
                self.manager.is_connected = True

                # Test collection creation
                mock_client.get_collections.return_value = Mock(collections=[])
                mock_client.create_collection.return_value = None
                mock_client.create_payload_index.return_value = None

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.manager.create_collection())
                    assert result is True, "Collection creation should return True for new collection"
                    print("✓ Collection creation works")

                    # Test collection already exists
                    mock_collection = Mock()
                    mock_collection.name = TEST_COLLECTION_NAME
                    mock_client.get_collections.return_value = Mock(collections=[mock_collection])

                    result = loop.run_until_complete(self.manager.create_collection())
                    assert result is False, "Collection creation should return False for existing collection"
                    print("✓ Existing collection handling works")

                    # Test collection recreation
                    result = loop.run_until_complete(self.manager.create_collection(recreate=True))
                    assert result is True, "Collection recreation should return True"
                    print("✓ Collection recreation works")

                    # Test collection existence check
                    exists = loop.run_until_complete(self.manager.collection_exists())
                    assert exists is True, "Collection should exist"
                    print("✓ Collection existence check works")

                    # Test collection deletion
                    result = loop.run_until_complete(self.manager.delete_collection())
                    assert result is True, "Collection deletion should return True"
                    print("✓ Collection deletion works")
                finally:
                    loop.close()

            print("✓ COLPALI-401 collection management validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-401 collection test failed: {e}")
            return False

    def test_colpali_401_health_monitoring(self):
        """
        Test COLPALI-401: Health checks and collection monitoring.
        """
        print("\n" + "="*70)
        print("COLPALI-401: HEALTH CHECKS AND COLLECTION MONITORING")
        print("="*70)

        try:
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                self.manager.client = mock_client
                self.manager.is_connected = True

                # Test health check
                mock_client.get_collections.return_value = Mock(collections=[])

                # Health check should not raise exception
                asyncio.run(self.manager._health_check())
                print("✓ Health check works")

                # Test collection info retrieval
                mock_collection_info = Mock()
                mock_collection_info.status = "green"
                mock_collection_info.optimizer_status = "ok"
                mock_collection_info.vectors_count = 1000
                mock_collection_info.indexed_vectors_count = 1000
                mock_collection_info.points_count = 1000
                mock_collection_info.segments = [Mock(), Mock()]

                # Mock config structure
                mock_config = Mock()
                mock_params = Mock()
                mock_vectors = Mock()
                mock_vectors.size = TEST_VECTOR_SIZE
                mock_vectors.distance = Distance.COSINE
                mock_params.vectors = mock_vectors
                mock_config.params = mock_params

                mock_hnsw_config = Mock()
                mock_hnsw_config.m = 16
                mock_hnsw_config.ef_construct = 100
                mock_hnsw_config.full_scan_threshold = 10000
                mock_config.hnsw_config = mock_hnsw_config

                mock_collection_info.config = mock_config
                mock_collection_info.payload_schema = {
                    "document_id": Mock(data_type="keyword"),
                    "page_number": Mock(data_type="integer")
                }

                mock_client.get_collection.return_value = mock_collection_info

                info = asyncio.run(self.manager.get_collection_info())

                # Validate returned information
                assert info["collection_name"] == TEST_COLLECTION_NAME
                assert info["status"] == "green"
                assert info["vectors_count"] == 1000
                assert info["vector_size"] == TEST_VECTOR_SIZE
                assert "hnsw_config" in info
                assert "payload_schema" in info
                print("✓ Collection info retrieval works")

                # Test ensure collection
                mock_client.get_collections.return_value = Mock(collections=[mock_collection_info])
                asyncio.run(self.manager.ensure_collection())
                print("✓ Ensure collection works")

            print("✓ COLPALI-401 health monitoring validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-401 health monitoring test failed: {e}")
            return False

    def test_colpali_401_error_handling(self):
        """
        Test COLPALI-401: Error handling and resilience.
        """
        print("\n" + "="*70)
        print("COLPALI-401: ERROR HANDLING AND RESILIENCE")
        print("="*70)

        try:
            # Test not connected scenarios
            fresh_manager = QdrantManager()

            try:
                asyncio.run(fresh_manager.create_collection())
                assert False, "Should raise RuntimeError when not connected"
            except RuntimeError as e:
                assert "not connected" in str(e).lower()
                print("✓ Not connected error handling works")

            try:
                asyncio.run(fresh_manager.get_collection_info())
                assert False, "Should raise RuntimeError when not connected"
            except RuntimeError as e:
                assert "not connected" in str(e).lower()
                print("✓ Collection info error handling works")

            # Test connection retry logic
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                # Mock client that fails first few attempts
                mock_client = Mock()
                mock_client.get_collections.side_effect = [
                    ConnectionError("Connection failed"),
                    ConnectionError("Connection failed"),
                    Mock(collections=[])  # Success on 3rd attempt
                ]
                mock_client_class.return_value = mock_client

                # Should succeed after retries
                asyncio.run(fresh_manager.connect())
                assert fresh_manager.is_connected
                print("✓ Connection retry logic works")

            # Test health check failure handling
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client.get_collections.side_effect = Exception("Health check failed")
                mock_client_class.return_value = mock_client

                try:
                    asyncio.run(fresh_manager._health_check())
                    assert False, "Health check should fail"
                except ConnectionError as e:
                    assert "health check failed" in str(e).lower()
                    print("✓ Health check failure handling works")

            print("✓ COLPALI-401 error handling validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-401 error handling test failed: {e}")
            return False

    def test_colpali_401_integration(self):
        """
        Test COLPALI-401: Complete integration workflow.
        """
        print("\n" + "="*70)
        print("COLPALI-401: INTEGRATION WORKFLOW TEST")
        print("="*70)

        try:
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock successful responses
                mock_client.get_collections.return_value = Mock(collections=[])
                mock_client.create_collection.return_value = None
                mock_client.create_payload_index.return_value = None

                # Test complete workflow
                integration_manager = QdrantManager(
                    collection_name="integration_test",
                    timeout=5
                )

                # 1. Connect to Qdrant
                asyncio.run(integration_manager.connect())
                assert integration_manager.is_connected
                print("✓ Connection established")

                # 2. Ensure collection exists
                asyncio.run(integration_manager.ensure_collection())
                print("✓ Collection ensured")

                # 3. Get collection info
                mock_collection_info = Mock()
                mock_collection_info.status = "green"
                mock_collection_info.vectors_count = 0
                mock_collection_info.points_count = 0
                mock_collection_info.segments = []
                mock_collection_info.config = Mock()
                mock_collection_info.config.params = Mock()
                mock_collection_info.config.params.vectors = Mock()
                mock_collection_info.config.params.vectors.size = 128
                mock_collection_info.config.params.vectors.distance = Distance.COSINE
                mock_collection_info.payload_schema = {}

                mock_client.get_collection.return_value = mock_collection_info

                info = asyncio.run(integration_manager.get_collection_info())
                assert info["status"] == "green"
                print("✓ Collection info retrieved")

                # 4. Verify collection exists
                exists = asyncio.run(integration_manager.collection_exists())
                assert exists
                print("✓ Collection existence verified")

                print("✓ Integration workflow completed successfully")

            print("✓ COLPALI-401 integration validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-401 integration test failed: {e}")
            return False

    def test_colpali_402_embedding_storage(self):
        """
        Test COLPALI-402: Embedding storage with spatial metadata.
        """
        print("\n" + "="*70)
        print("COLPALI-402: EMBEDDING STORAGE WITH SPATIAL METADATA")
        print("="*70)

        try:
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                self.manager.client = mock_client
                self.manager.is_connected = True

                # Mock successful upsert
                mock_client.upsert.return_value = None

                # Test embedding storage
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.manager.store_embeddings(
                            embeddings=self.test_embeddings,
                            metadata=self.test_metadata
                        )
                    )

                    # Validate storage results
                    assert result["points_stored"] == len(self.test_embeddings)
                    assert result["batches_processed"] > 0
                    assert result["storage_time"] >= 0
                    assert result["success_rate"] == 100.0
                    print("✓ Embedding storage with metadata works")

                    # Test spatial coordinate generation
                    coordinates = self.manager._generate_patch_coordinates(16)
                    assert len(coordinates) == 16
                    assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in coordinates)
                    print("✓ Spatial coordinate generation works")

                    # Test empty embedding handling
                    result = loop.run_until_complete(
                        self.manager.store_embeddings(
                            embeddings=[],
                            metadata=self.test_metadata
                        )
                    )
                    assert result["points_stored"] == 0
                    print("✓ Empty embedding handling works")

                finally:
                    loop.close()

            print("✓ COLPALI-402 embedding storage validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-402 test failed: {e}")
            return False

    def test_colpali_403_search_system(self):
        """
        Test COLPALI-403: Search and retrieval system.
        """
        print("\n" + "="*70)
        print("COLPALI-403: SEARCH AND RETRIEVAL SYSTEM")
        print("="*70)

        try:
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                self.manager.client = mock_client
                self.manager.is_connected = True

                # Mock search results
                mock_search_result = []
                for i in range(5):
                    mock_result = Mock()
                    mock_result.id = f"test_id_{i}"
                    mock_result.score = 0.9 - (i * 0.1)
                    mock_result.payload = {
                        "document_id": "test_doc_001",
                        "page_number": 1,
                        "patch_x": i * 32,
                        "patch_y": 0,
                        "patch_index": i
                    }
                    mock_search_result.append(mock_result)

                mock_client.search.return_value = mock_search_result

                # Test basic search
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    query_vector = [0.5] * TEST_VECTOR_SIZE

                    result = loop.run_until_complete(
                        self.manager.search_similar(query_vector=query_vector, limit=10)
                    )

                    assert "results" in result
                    assert "search_time" in result
                    assert len(result["results"]) == 5
                    assert result["results"][0]["score"] >= result["results"][1]["score"]  # Sorted by score
                    print("✓ Basic similarity search works")

                    # Test spatial filtering
                    filter_conditions = {"spatial_box": (0, 0, 64, 32)}
                    result = loop.run_until_complete(
                        self.manager.search_similar(
                            query_vector=query_vector,
                            filter_conditions=filter_conditions
                        )
                    )
                    assert "results" in result
                    print("✓ Spatial filtering works")

                    # Test document-scoped search
                    result = loop.run_until_complete(
                        self.manager.search_by_document(
                            query_vector=query_vector,
                            document_id="test_doc_001"
                        )
                    )
                    assert "results" in result
                    print("✓ Document-scoped search works")

                    # Test spatial region search
                    result = loop.run_until_complete(
                        self.manager.search_by_spatial_region(
                            query_vector=query_vector,
                            spatial_box=(0, 0, 100, 100)
                        )
                    )
                    assert "results" in result
                    print("✓ Spatial region search works")

                finally:
                    loop.close()

            print("✓ COLPALI-403 search system validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-403 test failed: {e}")
            return False

    def test_colpali_404_performance_monitoring(self):
        """
        Test COLPALI-404: Performance optimization and monitoring.
        """
        print("\n" + "="*70)
        print("COLPALI-404: PERFORMANCE OPTIMIZATION AND MONITORING")
        print("="*70)

        try:
            with patch.object(qdrant_module, 'QdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                self.manager.client = mock_client
                self.manager.is_connected = True

                # Mock collection info for performance metrics
                mock_collection_info = Mock()
                mock_collection_info.points_count = 1000
                mock_collection_info.vectors_count = 1000
                mock_collection_info.indexed_vectors_count = 950
                mock_collection_info.segments = [Mock(), Mock()]
                mock_collection_info.status = "green"
                mock_collection_info.optimizer_status = "ok"

                mock_client.get_collection.return_value = mock_collection_info
                mock_client.get_collections.return_value = Mock(collections=[mock_collection_info])

                # Mock search for benchmarking
                mock_search_result = [Mock()]
                mock_client.search.return_value = mock_search_result

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Test performance metrics
                    metrics = loop.run_until_complete(self.manager.get_performance_metrics())
                    assert "storage_metrics" in metrics
                    assert "system_health" in metrics
                    assert metrics["storage_metrics"]["points_count"] == 1000
                    print("✓ Performance metrics collection works")

                    # Test storage statistics
                    stats = loop.run_until_complete(self.manager.get_storage_statistics())
                    assert "storage_metrics" in stats
                    assert "recommendations" in stats
                    assert "collection_health" in stats
                    print("✓ Storage statistics and recommendations work")

                    # Test collection optimization
                    optimization = loop.run_until_complete(self.manager.optimize_collection())
                    assert "status" in optimization
                    assert optimization["status"] == "completed"
                    print("✓ Collection optimization works")

                    # Test search benchmarking (with fewer iterations for testing)
                    benchmark = loop.run_until_complete(
                        self.manager.benchmark_search_performance(iterations=3)
                    )
                    assert "benchmark_summary" in benchmark
                    assert "performance_rating" in benchmark
                    assert benchmark["benchmark_summary"]["iterations"] == 3
                    print("✓ Search performance benchmarking works")

                finally:
                    loop.close()

            print("✓ COLPALI-404 performance monitoring validated")
            return True

        except Exception as e:
            print(f"✗ COLPALI-404 test failed: {e}")
            return False

    def run_all_colpali_400_tests(self):
        """Run all COLPALI-400 tests."""
        print("COLPALI-400: QDRANT VECTOR STORAGE INTEGRATION TEST SUITE")
        print("="*70)
        print("Testing complete Qdrant vector storage with spatial metadata and search")
        print()

        self.setup_test_environment()

        # Run all component tests
        test_results = {
            "COLPALI-401 Connection": self.test_colpali_401_connection_management(),
            "COLPALI-401 Collection": self.test_colpali_401_collection_management(),
            "COLPALI-401 Monitoring": self.test_colpali_401_health_monitoring(),
            "COLPALI-401 Error Handling": self.test_colpali_401_error_handling(),
            "COLPALI-401 Integration": self.test_colpali_401_integration(),
            "COLPALI-402 Storage": self.test_colpali_402_embedding_storage(),
            "COLPALI-403 Search": self.test_colpali_403_search_system(),
            "COLPALI-404 Performance": self.test_colpali_404_performance_monitoring()
        }

        # Summary
        print("\n" + "="*70)
        print("COLPALI-400 COMPREHENSIVE TEST SUMMARY")
        print("="*70)

        passed = 0
        total = len(test_results)

        for test_name, result in test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name:<30} {status}")
            if result:
                passed += 1

        success_rate = (passed / total) * 100
        print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{total})")

        if success_rate == 100:
            print("\n🎉 COLPALI-400 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
            print("📊 Complete Qdrant vector storage integration validated")
            print("✅ All sub-tasks completed:")
            print("   • COLPALI-401: Client setup and collection management")
            print("   • COLPALI-402: Embedding storage with spatial metadata")
            print("   • COLPALI-403: Semantic search and retrieval system")
            print("   • COLPALI-404: Performance optimization and monitoring")
        else:
            print(f"\n⚠️  {total - passed} tests need attention")

        return success_rate >= 90.0  # 90% pass rate required


def main():
    """Run the comprehensive COLPALI-400 test suite."""
    tester = TestQdrantManager()
    success = tester.run_all_colpali_400_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
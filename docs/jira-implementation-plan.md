# ColPali-BAML Vision Processing Engine - Jira Implementation Plan

> **Golden Thread Document**: This document serves as the master implementation plan with detailed Jira stories and tasks for building a containerized vision-native document processing engine using ColPali, BAML, Qdrant, and CocoIndex.

## Project Overview

**Epic**: COLPALI-001 - Vision-Based PDF Processing Engine
**Goal**: Implement a containerized Python package for vision-native document processing
**Architecture**: `(document_blob, schema_json)` → `(canonical_data, shaped_data)` in CSV/Parquet
**Key Technologies**: ColPali, BAML, Qdrant, CocoIndex, Docker

### Success Metrics
- ✅ Process all 15 test PDFs with >95% accuracy
- ✅ Deploy in AWS Lambda container (10GB memory limit)
- ✅ JSON schema → BAML class auto-generation
- ✅ Dual output: canonical + shaped data
- ✅ Complete Docker containerization

---

## Epic: COLPALI-001 - Vision-Based PDF Processing Engine

**Description**: Implement a containerized vision-native document processing engine using ColPali, BAML, Qdrant, and CocoIndex for type-safe table extraction from complex documents.

**Epic Acceptance Criteria**:
- Process all 15 test PDFs with >95% accuracy
- Deploy in AWS Lambda container (10GB memory)
- Support JSON schema → BAML class generation
- Output canonical + shaped data in CSV/Parquet
- Complete Docker containerization

**Timeline**: 10-12 sprints (151 story points)
**Team Requirements**: Backend engineers, DevOps engineer, QA engineer
**Dependencies**: ColPali models, Qdrant instance, AWS Lambda environment

---

## Stories and Tasks Breakdown

### Story 1: COLPALI-100 - Core Infrastructure & Docker Foundation
**Points**: 13
**Sprint**: 1-2
**Priority**: High (Critical Path)

**Description**: Establish the foundational Docker infrastructure and package architecture for the vision-based document processing engine. This includes creating containerized development environments, defining the Python package structure, and setting up dependency management systems.

**Acceptance Criteria**:
- ✅ Multi-stage Docker builds working for all environments (dev, lambda, jupyter)
- ✅ Docker Compose setup with Qdrant service integration
- ✅ Python package structure following clean architecture principles
- ✅ Dependency management system with environment-specific requirements
- ✅ Base containers with all vision processing libraries installed

#### Tasks:

#### COLPALI-101: Set up multi-stage Dockerfiles (dev, lambda, jupyter) [3 pts]
**Assignee**: DevOps Engineer
**Sprint**: 1
**Dependencies**: None

**Description**: Create three optimized Dockerfiles for different deployment targets. Dev container includes development tools and hot-reload capabilities. Lambda container is optimized for AWS Lambda constraints (10GB memory limit). Jupyter container provides interactive notebook environment.

**Acceptance Criteria**:
- Dockerfile.dev with Python 3.13, development tools, debugger support
- Dockerfile.lambda optimized for size and cold start performance
- Dockerfile.jupyter with JupyterLab and visualization libraries
- Multi-stage builds to minimize production image size
- All containers share common base layers for efficiency

**Technical Implementation**:
- Use Python 3.13-slim base image
- Install system dependencies (poppler-utils, wkhtmltopdf, libreoffice)
- Optimize PyTorch for CPU-only inference
- Create non-root user for security
- Multi-stage builds for size optimization

**Definition of Done**:
- [x] All 3 Dockerfiles created and tested ✅
- [x] Docker builds successful for all targets ✅
- [x] Documentation updated with build instructions ✅
- [x] Code review completed ✅

---

#### COLPALI-102: Create docker-compose.yml with Qdrant service [2 pts]
**Assignee**: DevOps Engineer
**Sprint**: 1
**Dependencies**: COLPALI-101

**Description**: Orchestrate the development environment using Docker Compose, including the main application container and Qdrant vector database service with proper networking and volume management.

**Acceptance Criteria**:
- Docker Compose configuration with app and qdrant services
- Persistent Qdrant data storage with named volumes
- Environment variable configuration for service discovery
- Health checks for all services
- Development volume mounts for hot-reload

**Technical Implementation**:
- Map Qdrant port 6333 for external access
- Use environment variables for service URLs
- Mount source code for development hot-reload
- Named volumes for data persistence
- Health check endpoints for service monitoring

**Definition of Done**:
- [x] docker-compose.yml created and tested ✅
- [x] Services start and communicate correctly ✅
- [x] Data persistence verified ✅
- [x] Development workflow validated ✅

---

#### COLPALI-103: Define package structure and core interfaces [3 pts]
**Assignee**: Backend Engineer (Lead)
**Sprint**: 1
**Dependencies**: None

**Description**: Establish the Python package architecture following clean architecture principles with clear separation of concerns between document processing, vision modeling, vector storage, and output generation.

**Acceptance Criteria**:
- Package structure with core/, vision/, storage/, extraction/, outputs/ modules
- Abstract base classes for all major components
- Interface definitions for document adapters, model clients, storage backends
- Plugin system architecture for extensibility
- Proper __init__.py files with public API exports

**Technical Implementation**:
- Use Protocol classes for type hints
- Implement dependency injection pattern
- Follow SOLID principles
- Create abstract base classes for extensibility
- Define clear public APIs

**File Structure**:
```
tatforge/
├── __init__.py                         # Public API exports
├── core/
│   ├── __init__.py
│   ├── pipeline.py                     # Main orchestration
│   ├── document_adapter.py             # Document interfaces
│   └── schema_manager.py               # Schema conversion
├── vision/
│   ├── __init__.py
│   ├── colpali_client.py              # ColPali interface
│   └── image_processor.py             # Image utilities
├── storage/
│   ├── __init__.py
│   └── qdrant_client.py               # Vector DB interface
├── extraction/
│   ├── __init__.py
│   └── baml_interface.py              # BAML execution
└── outputs/
    ├── __init__.py
    ├── canonical.py                    # Truth layer
    └── shaped.py                       # 1NF transformation
```

**Definition of Done**:
- [x] Package structure created ✅
- [x] Abstract interfaces defined ✅
- [x] Type hints implemented ✅
- [x] Import structure validated ✅
- [x] Documentation updated ✅

---

#### COLPALI-104: Set up dependency management (requirements/*.txt) [2 pts]
**Assignee**: DevOps Engineer
**Sprint**: 1
**Dependencies**: COLPALI-103

**Description**: Create environment-specific requirement files optimizing for different deployment scenarios with clear separation between development, production, and Lambda-specific dependencies.

**Acceptance Criteria**:
- requirements/base.txt with core production dependencies
- requirements/dev.txt with development and testing tools
- requirements/lambda.txt with Lambda-optimized versions
- Version pinning strategy for reproducible builds
- Documentation explaining dependency choices

**Technical Implementation**:
- Pin major versions for stability
- Use CPU-optimized PyTorch for Lambda
- Minimize Lambda package size
- Separate development tools from production

**Core Dependencies**:
```
# requirements/base.txt
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0
colpali-engine>=0.3.0
Pillow>=10.0.0
pdf2image>=3.1.0
qdrant-client>=1.7.0
cocoindex>=0.3.0
baml-py>=0.215.2
pydantic>=2.5.0
pandas>=2.1.0
pyarrow>=14.0.0
```

**Definition of Done**:
- [x] All requirement files created ✅
- [x] Dependencies validated and tested ✅
- [x] Version conflicts resolved ✅
- [x] Documentation updated ✅

---

#### COLPALI-105: Implement base container with vision libraries [3 pts]
**Assignee**: DevOps Engineer
**Sprint**: 1-2
**Dependencies**: COLPALI-101, COLPALI-104

**Description**: Build the foundational container image with all system-level dependencies required for document processing and vision model inference, optimized for both development and production use.

**Acceptance Criteria**:
- System dependencies installed (poppler-utils, wkhtmltopdf, libreoffice)
- Python environment with vision processing libraries
- Model cache directory structure
- Proper user permissions for Lambda compatibility
- Health check endpoints implemented

**Technical Implementation**:
- Multi-stage build to reduce final image size
- Create non-root user for security
- Optimize layer caching
- Pre-install system dependencies
- Create model cache directories

**Definition of Done**:
- [x] Base container built successfully ✅
- [x] All dependencies installed and tested ✅
- [x] Health checks working ✅
- [x] Security scan passed ✅
- [x] Size optimization validated ✅

---

### Story 2: COLPALI-200 - Document Processing Pipeline
**Points**: 24 (21 completed, 3 deprioritized)
**Sprint**: 2-3
**Priority**: High (Critical Path) - **COMPLETE** ✅

> **📋 Status Update**: All core tasks are **COMPLETE** including **COLPALI-204 (CocoIndex Integration)**.
> CocoIndex now orchestrates the tatforge pipeline with ColPali embeddings and Qdrant storage.
> COLPALI-205 (caching) remains deprioritized for future release.

**Description**: Implement the document-to-image conversion pipeline that transforms various document formats into standardized image frames for vision processing. This story focuses on the critical adapter layer that ensures format-agnostic processing while maintaining visual fidelity.

**Acceptance Criteria**:
- ✅ PDF documents converted to high-resolution image frames (300 DPI)
- ✅ Consistent image standardization across all document types
- ✅ Plugin architecture for extending to new document formats
- ✅ MIME type detection and routing for multi-format support
- ✅ Performance optimization for large multi-page documents

#### Tasks:

#### COLPALI-201: Implement PDF adapter with pdf2image integration [5 pts]
**Assignee**: Backend Engineer
**Sprint**: 2
**Dependencies**: COLPALI-105

**Description**: Create the core PDF processing adapter that converts PDF pages to high-resolution images using pdf2image library. This adapter will handle complex multi-column layouts, preserve spatial relationships, and maintain visual quality for subsequent ColPali processing.

**Acceptance Criteria**:
- PDF to image conversion at 300 DPI resolution
- Support for complex layouts (multi-column, tables, graphics)
- Memory-efficient processing for large PDFs
- Error handling for corrupted or protected PDFs
- Metadata extraction (page numbers, document properties)
- Batch processing capabilities for multiple PDFs

**Test Data**: All 15 PDFs in `/pdfs/` directory including:
- extreme_multi_column.pdf
- mixed_content_layout.pdf
- Shipping-Stem-2025-09-30.pdf
- Loading-Statement-for-Web-Portal-20250923.pdf

**Technical Implementation**:
- Use pdf2image with poppler backend
- Implement memory streaming for large files
- Extract metadata (page count, dimensions)
- Handle password-protected PDFs
- Batch processing with progress tracking

**Definition of Done**:
- [x] PDF adapter implemented and tested ✅
- [x] All 15 test PDFs processed successfully ✅
- [x] Memory usage optimized ✅
- [x] Error handling validated ✅
- [x] Unit tests written ✅

---

#### COLPALI-202: Create image processor for standardization [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 2
**Dependencies**: COLPALI-201

**Description**: Build an image processing utility that standardizes image frames from different sources into a consistent format suitable for ColPali model input, including resolution normalization, color space conversion, and quality optimization.

**Acceptance Criteria**:
- Consistent image dimensions and resolution across all sources
- Color space normalization (RGB, grayscale handling)
- Quality optimization balancing file size and visual clarity
- Image metadata preservation for lineage tracking
- Support for batch processing

**Technical Implementation**:
- Use Pillow for image operations
- Maintain aspect ratios during resize
- Optimize for ColPali input requirements
- Preserve EXIF data where relevant
- Implement quality assessment metrics

**Definition of Done**:
- [x] Image processor implemented ✅
- [x] Standardization validated across formats ✅
- [x] Performance benchmarks met ✅
- [x] Quality metrics implemented ✅

---

#### COLPALI-203: Build document adapter interface for multi-format support [5 pts]
**Assignee**: Backend Engineer (Lead)
**Sprint**: 2-3
**Dependencies**: COLPALI-202

**Description**: Design and implement a plugin architecture that allows extending the system to support additional document formats (Excel, PowerPoint, Word, HTML) while maintaining a consistent interface for the vision pipeline.

**Acceptance Criteria**:
- Abstract base class for document adapters
- Plugin registration system for new formats
- Consistent error handling across all adapters
- MIME type detection and routing
- Format-specific configuration support

**Technical Implementation**:
- Use ABC for interface definition
- Implement factory pattern for adapter selection
- MIME type detection with python-magic
- Configuration system for adapter parameters
- Error standardization across adapters

**Definition of Done**:
- [x] Adapter interface defined ✅
- [x] Plugin system implemented ✅
- [x] MIME type routing working ✅
- [x] Configuration system tested ✅
- [x] Extension examples provided ✅

---

#### COLPALI-204: Integrate CocoIndex orchestration framework [8 pts] 🔴 **HIGH PRIORITY**
**Assignee**: Backend Engineer (Lead)
**Sprint**: 3
**Dependencies**: COLPALI-203

> **📋 Priority Note**: This is a **CRITICAL** feature. CocoIndex is the central orchestration framework
> that properly wires ColPali vision embeddings and Qdrant vector storage. This follows the official
> CocoIndex ColPali pattern from:
> - https://github.com/cocoindex-io/cocoindex/tree/main/examples/multi_format_indexing (ColPali + Qdrant)
> - https://github.com/cocoindex-io/cocoindex/tree/main/examples/patient_intake_extraction_baml (BAML schema)

**Description**: Integrate CocoIndex as the central orchestration framework that wires together:
1. **Document loading** via `cocoindex.sources.LocalFile` with binary mode for PDFs
2. **PDF to images** via `@cocoindex.op.function()` using pdf2image at 300 DPI
3. **ColPali embeddings** via `cocoindex.functions.ColPaliEmbedImage()` for spatial awareness
4. **Qdrant storage** via `cocoindex.targets.Qdrant()` for multi-vector embeddings
5. **Query handler** via `@flow.query_handler()` with `ColPaliEmbedQuery()` for semantic search
6. **BAML extraction** via tatforge's BAMLFunctionGenerator for structured output AFTER retrieval

**Architecture** (combining both CocoIndex patterns):

```
FLOW 1 - INDEXING (multi_format_indexing pattern):
  PDF → pdf2image (300dpi) → ColPali Embeddings → Qdrant

FLOW 2 - EXTRACTION (patient_intake_extraction_baml pattern):
  Query → ColPali Query Embedding → Qdrant Search → Retrieved Page Images
  → BAML Extraction (tatforge BAMLFunctionGenerator) → Structured Output
```

**Key integration points**:
- Indexing: Use `cocoindex.functions.ColPaliEmbedImage()` for spatial-aware embeddings
- Search: Use `cocoindex.functions.ColPaliEmbedQuery()` for text→embedding conversion
- Extraction: Use `@cocoindex.op.function(cache=True)` to wrap BAML calls with caching

**Acceptance Criteria**:
- CocoIndex flow definition using `@cocoindex.flow_def()` decorator
- PDF→Images conversion via `@cocoindex.op.function()` with pdf2image
- ColPali embeddings via `cocoindex.functions.ColPaliEmbedImage(model="vidore/colqwen2-v0.1")`
- Qdrant export using `cocoindex.targets.Qdrant(collection_name=...)`
- Query handler using `cocoindex.functions.ColPaliEmbedQuery()` for text queries
- BAML schema enforcement via tatforge's BAMLFunctionGenerator (post-retrieval)

**Technical Implementation** (based on multi_format_indexing example):
```python
import cocoindex
import mimetypes
from dataclasses import dataclass
from io import BytesIO
from pdf2image import convert_from_bytes

from tatforge.core.baml_function_generator import BAMLFunctionGenerator

COLPALI_MODEL = "vidore/colqwen2-v0.1"

@dataclass
class Page:
    page_number: int | None
    image: bytes

@cocoindex.op.function()
def file_to_pages(filename: str, content: bytes) -> list[Page]:
    """Convert PDF/images to page images for ColPali embedding."""
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
    return []

@cocoindex.flow_def(name="DocumentIndexingFlow")
def document_indexing_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope
) -> None:
    """Index documents with ColPali embeddings in Qdrant."""
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="pdfs", binary=True)
    )
    output_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as doc:
        doc["pages"] = flow_builder.transform(
            file_to_pages, filename=doc["filename"], content=doc["content"]
        )
        with doc["pages"].row() as page:
            # ColPali generates multi-vector embeddings for spatial awareness
            page["embedding"] = page["image"].transform(
                cocoindex.functions.ColPaliEmbedImage(model=COLPALI_MODEL)
            )
            output_embeddings.collect(
                id=cocoindex.GeneratedField.UUID,
                filename=doc["filename"],
                page=page["page_number"],
                embedding=page["embedding"],
            )

    output_embeddings.export(
        "document_embeddings",
        cocoindex.targets.Qdrant(collection_name="DocumentEmbeddings"),
        primary_key_fields=["id"],
    )

@cocoindex.transform_flow()
def query_to_colpali_embedding(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[list[float]]]:
    """Convert text query to ColPali multi-vector embedding for search."""
    return text.transform(cocoindex.functions.ColPaliEmbedQuery(model=COLPALI_MODEL))

# FLOW 2: BAML Extraction (cached, async - like patient_intake_extraction_baml)
@cocoindex.op.function(cache=True, behavior_version=1)
async def extract_structured_data(page_image: bytes, baml_function_name: str) -> dict:
    """
    Extract structured data from a page image using BAML.
    Uses tatforge's BAMLFunctionGenerator for schema-driven extraction.
    Caching prevents redundant LLM calls for the same page.
    """
    from tatforge.core.baml_function_generator import BAMLFunctionGenerator
    from baml_client import b
    import base64
    import baml_py

    # Convert image to base64 for BAML
    image_b64 = base64.b64encode(page_image).decode("utf-8")
    image = baml_py.Image.from_base64("image/png", image_b64)

    # Use dynamically generated BAML function via tatforge
    result = await getattr(b, baml_function_name)(document_images=[image])
    return result.model_dump() if hasattr(result, 'model_dump') else result
```

**Definition of Done**:
- [x] CocoIndex flow integrated into tatforge/core/pipeline.py ✅
- [x] PDF→Images via pdf2image at 300 DPI ✅
- [x] ColPali embeddings via cocoindex.functions.ColPaliEmbedImage ✅
- [x] Qdrant export via cocoindex.targets.Qdrant ✅
- [x] Query handler with ColPaliEmbedQuery for semantic search ✅
- [x] Integration with tatforge's BAMLFunctionGenerator for post-retrieval extraction ✅
- [x] Integration tested with test PDFs ✅

---

#### COLPALI-205: Implement caching and incremental processing [3 pts] ⚠️ **DEPRIORITIZED**
**Assignee**: Backend Engineer
**Sprint**: 3 → **MOVED TO FUTURE RELEASE**
**Dependencies**: COLPALI-204

> **📋 Priority Note**: This task has been deprioritized as an advanced performance optimization.
> Caching and incremental processing will be valuable for production scale, but the basic
> document processing pipeline should be completed first. Current priority: **LOW**

**Description**: Add intelligent caching mechanisms to avoid reprocessing unchanged documents and implement incremental processing for updated documents, optimizing performance and resource utilization.

**Acceptance Criteria**:
- Document fingerprinting for change detection
- Cached image storage with TTL policies
- Incremental processing for modified documents only
- Cache invalidation strategies
- Storage optimization for cached artifacts

**Technical Implementation**:
- Use content hashing for fingerprinting
- Implement LRU cache with size limits
- TTL policies for cache expiration
- Storage backend for cached images
- Cache hit/miss metrics

**Definition of Done**:
- [ ] Caching system implemented
- [ ] Fingerprinting algorithm validated
- [ ] Cache policies configured
- [ ] Performance improvement measured

---

### Story 3: COLPALI-300 - ColPali Vision Integration
**Points**: 21
**Sprint**: 4-5
**Priority**: High (Critical Path)

**Description**: Integrate the ColPali vision model for semantic patch-level embedding generation from document images. This story implements the core vision processing capability that enables semantic retrieval of table regions without relying on OCR or text extraction.

**Acceptance Criteria**:
- ✅ ColPali model (ColQwen2-v0.1) loaded and operational with 3B parameters
- ✅ Patch-level embeddings generated for 32x32 image regions
- ✅ Memory-optimized processing suitable for Lambda constraints
- ✅ Batch processing support for multiple document pages
- ✅ Performance metrics and model optimization

#### Tasks:

#### COLPALI-301: Set up ColPali model client with memory optimization [8 pts]
**Assignee**: ML Engineer
**Sprint**: 4
**Dependencies**: COLPALI-200 (Document Pipeline)

**Description**: Implement the core ColPali model client that loads and manages the ColQwen2-v0.1 vision model (3B parameters) with aggressive memory optimization for AWS Lambda deployment constraints.

**Acceptance Criteria**:
- ColQwen2-v0.1 model loading with memory management
- Model quantization and optimization for CPU inference
- Memory usage monitoring and cleanup mechanisms
- Error handling for model loading failures
- Performance benchmarking for different batch sizes
- Integration with existing PyTorch ecosystem

**Technical Implementation**:
- Use torch.jit.script for optimization
- Implement model sharing across requests
- Consider INT8 quantization for memory reduction
- Memory profiling and monitoring
- Model warmup strategies

**Definition of Done**:
- [x] ColPali model client implemented ✅
- [x] Memory optimization validated ✅
- [x] Performance benchmarks complete ✅
- [x] Error handling tested ✅
- [x] Integration tests passed ✅

---

#### COLPALI-302: Implement batch processing for image frames [5 pts]
**Assignee**: ML Engineer
**Sprint**: 4
**Dependencies**: COLPALI-301

**Description**: Create efficient batch processing capabilities for multiple document pages, optimizing throughput while respecting memory constraints in containerized environments.

**Acceptance Criteria**:
- Batch processing with dynamic sizing based on available memory
- GPU/CPU hybrid processing when available
- Progress tracking for long-running batch operations
- Memory overflow protection and graceful degradation
- Performance metrics and optimization recommendations

**Technical Implementation**:
- Implement adaptive batch sizing
- Use memory profiling to optimize batch size
- Progress callbacks for long operations
- Memory monitoring and alerts
- Fallback strategies for memory constraints

**Definition of Done**:
- [x] Batch processing implemented ✅
- [x] Memory management validated ✅
- [x] Performance optimized ✅
- [x] Progress tracking working ✅

---

#### COLPALI-303: Create patch-level embedding generation [5 pts]
**Assignee**: ML Engineer
**Sprint**: 5
**Dependencies**: COLPALI-302

**Description**: Implement the core functionality for generating patch-level embeddings from document images, preserving spatial relationships and enabling semantic retrieval of table regions.

**Acceptance Criteria**:
- 32x32 patch extraction from input images
- Spatial coordinate preservation for each patch
- Multi-vector embedding generation (128 dimensions per patch)
- Efficient encoding for storage in vector database
- Quality validation and sanity checks

**Technical Implementation**:
- Maintain spatial metadata (x, y coordinates)
- Optimize embedding compression for storage
- Implement quality validation metrics
- Batch processing for multiple patches
- Spatial indexing for retrieval

**Definition of Done**:
- [x] Patch extraction implemented ✅
- [x] Embeddings generated correctly ✅
- [x] Spatial metadata preserved ✅
- [x] Quality validation working ✅
- [x] Storage format optimized ✅

---

#### COLPALI-304: Optimize model loading for Lambda cold starts [3 pts]
**Assignee**: DevOps Engineer + ML Engineer
**Sprint**: 5
**Dependencies**: COLPALI-303

**Description**: Implement optimization strategies to minimize Lambda cold start time when loading the 3B parameter ColPali model, including model caching, lazy loading, and pre-warming techniques.

**Acceptance Criteria**:
- Model pre-warming during container initialization
- Lazy loading of model components when possible
- Cold start time under 10 seconds
- Model sharing across Lambda invocations
- Monitoring and alerting for cold start performance

**Technical Implementation**:
- Use EFS for model caching
- Implement model loading parallelization
- Pre-warm model in container init
- Model sharing strategies
- Cold start monitoring

**Definition of Done**:
- [x] Cold start optimization implemented ✅
- [x] Performance targets met (<10 seconds) ✅
- [x] Monitoring in place ✅
- [x] Lambda deployment tested ✅

---

### Story 4: COLPALI-400 - Qdrant Vector Storage
**Points**: 16 (Originally 13, expanded with COLPALI-404)
**Sprint**: 5-6
**Priority**: Medium - **COMPLETED** ✅

> **📋 Status Update**: Complete Qdrant vector storage integration is **COMPLETED** and production-ready.
> All four sub-tasks successfully implemented with comprehensive testing and validation.

**Description**: Implement vector database operations using Qdrant for storing and querying ColPali embeddings with spatial metadata. This enables efficient semantic search and retrieval of document patches containing tables and structured data. Extended to include comprehensive performance monitoring and optimization capabilities.

**Acceptance Criteria**:
- ✅ Qdrant collections configured for multi-vector storage
- ✅ Efficient storage of patch embeddings with spatial coordinates
- ✅ Semantic search functionality with similarity thresholds
- ✅ Collection management and optimization
- ✅ Performance tuning for large document collections
- ✅ Performance monitoring and benchmarking capabilities

#### Tasks:

#### COLPALI-401: Set up Qdrant client and collection management [5 pts]
**Assignee**: Backend Engineer
**Sprint**: 5
**Dependencies**: Docker Infrastructure

**Description**: Configure Qdrant vector database client with proper collection schemas for storing ColPali embeddings, including index configuration, similarity metrics, and performance optimization.

**Acceptance Criteria**:
- Qdrant client configuration with connection pooling
- Collection schema for multi-vector embeddings
- Index optimization for similarity search
- Collection lifecycle management (create, update, delete)
- Health monitoring and connection resilience

**Technical Implementation**:
- Use cosine similarity metric
- Implement connection retry logic
- Optimize for high-dimensional vectors
- Connection pooling for performance
- Health check endpoints

**Definition of Done**:
- [x] Qdrant client implemented ✅
- [x] Collection schemas created ✅
- [x] Connection management working ✅
- [x] Health monitoring active ✅

---

#### COLPALI-402: Implement multi-vector embedding storage [5 pts]
**Assignee**: Backend Engineer
**Sprint**: 6
**Dependencies**: COLPALI-401, COLPALI-303

**Description**: Design and implement storage strategy for ColPali's multi-vector output, where each document page generates multiple patch embeddings that need to be stored with spatial metadata and document lineage.

**Acceptance Criteria**:
- Storage of patch embeddings with spatial coordinates (x, y, page_num)
- Document lineage tracking (document_id, source, timestamp)
- Efficient batch insertion for large documents
- Metadata indexing for filtering and retrieval
- Storage optimization and compression

**Technical Implementation**:
- Use payload fields for metadata
- Implement batch upsert operations
- Spatial indexing for coordinate queries
- Document ID hierarchical storage
- Compression strategies

**Definition of Done**:
- [x] Multi-vector storage implemented ✅
- [x] Batch operations working ✅
- [x] Metadata indexing active ✅
- [x] Performance validated ✅

---

#### COLPALI-403: Create retrieval engine for semantic search [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 6
**Dependencies**: COLPALI-402

**Description**: Build the semantic search engine that queries Qdrant for relevant document patches based on schema requirements and table extraction needs, implementing scoring and ranking algorithms.

**Acceptance Criteria**:
- Query interface for semantic similarity search
- Result ranking and scoring mechanisms
- Spatial filtering for region-based queries
- Performance optimization for large collections
- Search result aggregation and deduplication

**Technical Implementation**:
- Implement hybrid search (embedding + metadata)
- Use search result caching
- Scoring algorithms for relevance
- Spatial query optimization
- Result deduplication logic

**Definition of Done**:
- [x] Search engine implemented ✅
- [x] Ranking algorithms working ✅
- [x] Performance optimized ✅
- [x] Caching active ✅

---

#### COLPALI-404: Performance optimization and monitoring [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 6
**Dependencies**: COLPALI-401, COLPALI-402, COLPALI-403

**Description**: Implement comprehensive performance monitoring, optimization, and benchmarking capabilities for the Qdrant vector storage system. Includes metrics collection, search performance benchmarking, and intelligent optimization recommendations.

**Acceptance Criteria**:
- Performance metrics collection and reporting
- Search benchmarking with statistical analysis
- Collection optimization and HNSW tuning
- System health monitoring and alerts
- Intelligent optimization recommendations

**Technical Implementation**:
- Performance metrics collection (storage, indexing, search)
- Search benchmarking with configurable iterations
- Collection optimization triggers
- Health monitoring with connectivity checks
- Recommendation engine based on usage patterns

**Definition of Done**:
- [x] Performance metrics implemented ✅
- [x] Search benchmarking working ✅
- [x] Collection optimization active ✅
- [x] Health monitoring complete ✅

---

### Story 5: COLPALI-500 - BAML Schema System
**Points**: 21
**Sprint**: 6-7
**Priority**: High (Critical Path)

**Description**: Develop the dynamic schema system that converts JSON schemas into BAML classes and functions, enabling type-safe extraction with automatic code generation for diverse document structures.

**Acceptance Criteria**:
- ✅ JSON schema to BAML class conversion engine
- ✅ Dynamic BAML function generation with vision prompts
- ✅ Schema validation and compatibility checking
- ✅ Integration with existing BAML setup and client configurations
- ✅ Support for complex nested schemas and array types

#### Tasks:

#### COLPALI-501: Build JSON to BAML class generator [8 pts]
**Assignee**: Backend Engineer (Lead)
**Sprint**: 6-7
**Dependencies**: Existing BAML setup

**Description**: Implement the core schema conversion engine that translates JSON schemas into BAML class definitions, handling complex nested structures, arrays, and optional fields while maintaining type safety.

**Acceptance Criteria**:
- JSON schema parsing and validation
- BAML class generation with proper type mapping
- Support for nested objects and array types
- Optional field handling and default values
- Generated code validation and syntax checking

**Test Cases**:
- Convert shipping manifest schema
- Convert invoice schema
- Convert table schema from ADR

**Technical Implementation**:
- Use Jinja2 for template generation
- Implement recursive schema traversal
- Type mapping between JSON Schema and BAML
- Validation of generated BAML code
- Error handling for unsupported schemas

**Definition of Done**:
- [x] Schema converter implemented ✅
- [x] All test cases passing ✅
- [x] Generated BAML validated ✅
- [x] Error handling complete ✅

---

#### COLPALI-502: Create dynamic BAML function generation [5 pts]
**Assignee**: Backend Engineer
**Sprint**: 7
**Dependencies**: COLPALI-501

**Description**: Generate BAML extraction functions dynamically based on the converted schemas, including appropriate vision prompts and client configurations for optimal extraction performance.

**Acceptance Criteria**:
- Function template generation with vision-specific prompts
- Client selection based on schema complexity
- Prompt optimization for table extraction
- Function validation and testing framework
- Integration with existing BAML clients (Claude Sonnet 4, GPT-5)

**Technical Implementation**:
- Use template-based prompt generation
- Implement prompt optimization strategies
- Client selection algorithms
- Function testing framework
- Integration with existing BAML setup

**Definition of Done**:
- [x] Function generator implemented ✅
- [x] Prompt templates working ✅
- [x] Client selection active ✅
- [x] Testing framework complete ✅

---

#### COLPALI-503: Implement schema validation and compatibility checks [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 7
**Dependencies**: COLPALI-502

**Description**: Build validation system to ensure JSON schemas are compatible with BAML type system and extraction requirements, providing helpful error messages and suggestions for fixes.

**Acceptance Criteria**:
- JSON schema validation against BAML constraints
- Compatibility checking for supported types
- Helpful error messages and fix suggestions
- Schema migration support for version changes
- Integration with CI/CD pipelines

**Technical Implementation**:
- Use jsonschema library for validation
- Implement custom validators for BAML-specific constraints
- Error message templating
- Schema migration utilities
- CI/CD integration hooks

**Definition of Done**:
- [x] Validation system implemented ✅
- [x] Error messages helpful ✅
- [x] Migration support working ✅
- [x] CI/CD integration active ✅

---

#### COLPALI-504: Integrate with existing BAML client configuration [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 7
**Dependencies**: COLPALI-503

**Description**: Ensure seamless integration with the existing BAML setup in `baml_src/clients.baml`, extending the current configuration to support vision-capable models and dynamic function generation.

**Acceptance Criteria**:
- Backward compatibility with existing BAML setup
- Extension of client configurations for vision models
- Integration with current retry policies and fallback strategies
- Proper namespace management for generated functions
- Configuration validation and testing

**Technical Implementation**:
- Extend existing clients.baml without breaking changes
- Use existing retry policies
- Namespace management for generated functions
- Configuration validation
- Backward compatibility testing

**Definition of Done**:
- [x] Integration complete ✅
- [x] Backward compatibility verified ✅
- [x] Configuration extended ✅
- [x] Testing complete ✅

---

#### COLPALI-505: Add vision-capable model configurations to BAML [2 pts]
**Assignee**: Backend Engineer
**Sprint**: 7
**Dependencies**: COLPALI-504

**Description**: Configure BAML clients to use vision-capable language models (Claude Sonnet 4, GPT-4V) for processing image inputs alongside the existing text-based configurations.

**Acceptance Criteria**:
- Vision model client configurations in BAML
- Image input handling for BAML functions
- Model selection strategies for vision vs text tasks
- Cost optimization for vision model usage
- Error handling for vision model failures

**Technical Implementation**:
- Configure multimodal clients
- Implement fallback to text models when vision unavailable
- Cost optimization strategies
- Error handling for vision failures
- Model selection algorithms

**Definition of Done**:
- [x] Vision clients configured ✅
- [x] Image input working ✅
- [x] Fallback strategies active ✅
- [x] Cost optimization implemented ✅

---

### Story 6: COLPALI-600 - Extraction & Validation
**Points**: 13
**Sprint**: 8
**Priority**: High (Critical Path) - **COMPLETED** ✅

> **📋 Status Update**: Complete extraction and validation pipeline is **COMPLETED** and production-ready.
> All four sub-tasks successfully implemented with comprehensive testing, error handling, and quality analytics.

**Description**: Implement the extraction execution engine that runs BAML functions with image context and validates results against schemas, ensuring high-quality structured data output with comprehensive error handling.

**Acceptance Criteria**:
- ✅ BAML function execution with image inputs
- ✅ Structured output validation against schemas
- ✅ Robust error handling and retry mechanisms
- ✅ Quality metrics and extraction confidence scoring
- ✅ Performance monitoring and optimization

#### Tasks:

#### COLPALI-601: Build BAML execution interface with image context [5 pts]
**Assignee**: Backend Engineer
**Sprint**: 8
**Dependencies**: COLPALI-500 (Schema System), COLPALI-400 (Vector Storage)

**Description**: Create the interface layer that executes dynamically generated BAML functions with document images as input, handling the bridge between ColPali patch retrieval and BAML extraction.

**Acceptance Criteria**:
- BAML function execution with image inputs
- Integration with ColPali patch retrieval results
- Image preprocessing for vision models
- Result parsing and structured data extraction
- Performance monitoring and timeout handling

**Technical Implementation**:
- Use BAML Python client
- Implement async execution for performance
- Image preprocessing pipeline
- Result parsing and validation
- Timeout and error handling

**Definition of Done**:
- [x] BAML execution interface complete ✅
- [x] Image processing working ✅
- [x] Performance optimized ✅
- [x] Error handling active ✅

---

#### COLPALI-602: Implement extraction result validation [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 8
**Dependencies**: COLPALI-601

**Description**: Build comprehensive validation system for extraction results, ensuring outputs conform to declared schemas and meet quality standards before downstream processing.

**Acceptance Criteria**:
- Schema conformance validation
- Data quality checks and sanity validation
- Missing field detection and handling
- Type validation and coercion
- Validation report generation

**Technical Implementation**:
- Use Pydantic for validation
- Implement custom validators for business rules
- Quality scoring algorithms
- Validation reporting
- Error classification

**Definition of Done**:
- [x] Validation system implemented ✅
- [x] Quality checks active ✅
- [x] Reporting working ✅
- [x] Custom validators complete ✅

---

#### COLPALI-603: Create error handling and retry logic [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 8
**Dependencies**: COLPALI-602

**Description**: Implement robust error handling system with intelligent retry mechanisms for model failures, network issues, and extraction quality problems.

**Acceptance Criteria**:
- Retry logic for transient failures
- Error classification and handling strategies
- Graceful degradation for partial failures
- Error reporting and alerting
- Recovery mechanisms for critical failures

**Technical Implementation**:
- Use exponential backoff
- Implement circuit breaker pattern
- Error classification system
- Alerting and monitoring
- Graceful degradation strategies

**Definition of Done**:
- [x] Error handling implemented ✅
- [x] Retry logic working ✅
- [x] Monitoring active ✅
- [x] Recovery tested ✅

---

#### COLPALI-604: Add extraction quality metrics [2 pts]
**Assignee**: Backend Engineer
**Sprint**: 8
**Dependencies**: COLPALI-603

**Description**: Implement quality scoring system that evaluates extraction confidence, completeness, and accuracy to enable monitoring and continuous improvement.

**Acceptance Criteria**:
- Confidence scoring for extraction results
- Completeness metrics (missing fields, partial data)
- Quality benchmarking against known good extractions
- Metrics collection and reporting
- Alerting for quality degradation

**Technical Implementation**:
- Use statistical methods for confidence scoring
- Implement comparison algorithms
- Quality benchmarking framework
- Metrics collection system
- Alerting thresholds

**Definition of Done**:
- [x] Quality metrics implemented ✅
- [x] Confidence scoring working ✅
- [x] Benchmarking active ✅
- [x] Alerting configured ✅

---

### Story 7: COLPALI-700 - Output Management ✅ **COMPLETED**
**Points**: 13
**Sprint**: 9
**Priority**: Medium
**Status**: Production-ready dual-output architecture ✅

**Description**: Implement the dual-output architecture with canonical truth layer and shaped output layer, ensuring data integrity while supporting business transformation requirements with full lineage tracking.

**Acceptance Criteria**:
- ✅ Canonical data formatter preserving extraction truth
- ✅ Shaped data formatter with mandatory 1NF enforcement
- ✅ CSV and Parquet export capabilities with proper encoding
- ✅ Output format validation and quality assurance
- ✅ Metadata and lineage preservation in all outputs

#### Tasks:

#### COLPALI-701: Implement canonical data formatter (truth layer) [5 pts] ✅
**Assignee**: Backend Engineer
**Sprint**: 9
**Dependencies**: COLPALI-600 (Extraction)

**Description**: Build the canonical output formatter that preserves the exact extraction results without any business transformations, serving as the authoritative truth layer for all downstream processing and audit requirements.

**Acceptance Criteria**:
- Faithful preservation of extracted data structure
- No semantic modifications or business logic injection
- Metadata preservation (document source, extraction timestamp, confidence scores)
- Schema validation against original extraction schema
- Immutable output format for audit and compliance

**Technical Implementation**:
- Use Pydantic models for structure validation
- Implement read-only output classes
- Metadata preservation system
- Audit trail generation
- Immutability enforcement

**Definition of Done**:
- [x] Canonical formatter implemented ✅
- [x] Metadata preservation working ✅
- [x] Validation complete ✅
- [x] Audit trail active ✅
- [x] Integrity hash generation (SHA-256) ✅
- [x] Deep copy immutability enforcement ✅
- [x] 20+ comprehensive test cases ✅

---

#### COLPALI-702: Build shaped data formatter with 1NF enforcement [5 pts] ✅
**Assignee**: Backend Engineer
**Sprint**: 9
**Dependencies**: COLPALI-701

**Description**: Implement the business transformation layer that converts canonical data into shaped outputs meeting specific business requirements while enforcing First Normal Form (1NF) compliance for relational data structures.

**Acceptance Criteria**:
- Automatic 1NF normalization (flatten nested structures, eliminate repeating groups)
- Business transformation rules application with versioning
- Data quality validation post-transformation
- Transformation audit trail and lineage tracking
- Configurable transformation rules via JSON/YAML configuration

**Technical Implementation**:
- Implement transformation rule engine
- Use pandas for data manipulation
- 1NF compliance validation
- Transformation versioning
- Audit trail generation

**Definition of Done**:
- [x] Shaped formatter implemented ✅
- [x] 1NF enforcement working ✅
- [x] Transformation rules active ✅
- [x] Lineage tracking complete ✅
- [x] Automatic nested structure flattening ✅
- [x] Multi-type transformation rules (normalize, aggregate, filter, rename, custom) ✅
- [x] Comprehensive 1NF validation (atomic values, unique records, type consistency) ✅
- [x] 25+ comprehensive test cases ✅

---

#### COLPALI-703: Create CSV/Parquet export functionality [3 pts] ✅
**Assignee**: Backend Engineer
**Sprint**: 9
**Dependencies**: COLPALI-702

**Description**: Build export utilities that convert both canonical and shaped data into standard output formats (CSV, Parquet) with proper encoding, compression, and metadata preservation.

**Acceptance Criteria**:
- CSV export with proper encoding (UTF-8) and delimiter handling
- Parquet export with column typing and compression
- Metadata embedding in file headers/schemas
- Large dataset streaming support for memory efficiency
- Export validation and integrity checking

**Technical Implementation**:
- Use pandas for CSV, pyarrow for Parquet
- Implement streaming for large datasets
- Metadata embedding strategies
- Compression optimization
- Validation and integrity checks

**Definition of Done**:
- [x] Export utilities implemented ✅
- [x] Both formats working ✅
- [x] Streaming support active ✅
- [x] Validation complete ✅
- [x] DataExporter with batch and in-memory modes ✅
- [x] StreamingExporter for large datasets ✅
- [x] Metadata embedding in Parquet schema ✅
- [x] Multiple compression options (snappy, gzip, brotli) ✅
- [x] 25+ comprehensive test cases ✅

---

### Story 8: COLPALI-800 - Governance & Lineage
**Points**: 8
**Sprint**: 9-10
**Priority**: Medium

**Description**: Implement comprehensive governance framework ensuring transformation transparency, data lineage tracking, and validation rules that maintain data integrity across the entire processing pipeline.

**Acceptance Criteria**:
- ✅ Complete transformation lineage from source document to final output
- ✅ Governance validation rules preventing data distortion
- ✅ Audit trail for all data transformations
- ✅ Compliance reporting and documentation
- ✅ Data quality metrics and monitoring

#### Tasks:

#### COLPALI-801: Implement transformation lineage tracking [5 pts]
**Assignee**: Backend Engineer
**Sprint**: 9-10
**Dependencies**: COLPALI-700 (Output Management)

**Description**: Build comprehensive lineage tracking system that records the complete data flow from source document through ColPali processing, BAML extraction, and final output generation.

**Acceptance Criteria**:
- Document-to-output lineage tracking with UUIDs
- Transformation step recording (ColPali patches, BAML functions, formatters)
- Version tracking for schemas and transformation rules
- Performance metrics collection at each stage
- Lineage query interface for audit and debugging

**Technical Implementation**:
- Use UUID4 for tracking
- Implement lineage graph data structure
- Step-by-step recording
- Query interface for lineage
- Performance metrics integration

**Definition of Done**:
- [x] Lineage tracking implemented
- [x] UUID tracking working
- [x] Query interface active
- [x] Metrics collection complete

---

#### COLPALI-802: Build governance validation rules [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 10
**Dependencies**: COLPALI-801

**Description**: Implement validation rules that enforce the architectural principles of canonical-first processing and prevent unauthorized data transformations that could compromise data integrity.

**Acceptance Criteria**:
- Validation rules for canonical data integrity
- Business transformation approval workflow
- Data distortion detection and alerting
- Compliance rule checking and reporting
- Automated governance testing framework

**Technical Implementation**:
- Use rule engine pattern
- Implement validation decorators
- Approval workflow system
- Distortion detection algorithms
- Automated testing framework

**Definition of Done**:
- [x] Governance rules implemented
- [x] Validation working
- [x] Approval workflow active
- [x] Testing framework complete

---

### Story 9: COLPALI-900 - Lambda Deployment ✅ **COMPLETED**
**Points**: 21
**Sprint**: 10-11
**Priority**: High
**Status**: Production-ready Lambda deployment infrastructure ✅

**Description**: Optimize the entire system for AWS Lambda deployment, handling the challenges of 3B parameter model deployment, memory management, and cold start optimization in a serverless environment.

**Acceptance Criteria**:
- ✅ Lambda container deployment supporting 3B ColPali model
- ✅ Memory usage under 8GB for 10GB Lambda limit
- ✅ Cold start time under 10 seconds
- ✅ API interface for document processing requests
- ✅ Comprehensive monitoring and alerting

#### Tasks:

#### COLPALI-901: Optimize Lambda container for 3B model deployment [8 pts] ✅
**Assignee**: DevOps Engineer + ML Engineer
**Sprint**: 10-11
**Dependencies**: COLPALI-300 (ColPali Integration)

**Description**: Implement advanced optimization techniques to deploy the 3B parameter ColPali model in AWS Lambda, including model quantization, compression, and efficient memory management.

**Acceptance Criteria**:
- Model quantization (INT8) reducing memory footprint by 50%
- Container image under 10GB total size
- Model loading time under 5 seconds
- Memory usage optimization and monitoring
- GPU-less inference optimization for Lambda CPU

**Technical Implementation**:
- Use torch.quantization
- Implement model pruning
- Optimize container layers
- Memory profiling and optimization
- CPU inference optimization

**Definition of Done**:
- [x] Model optimization complete ✅
- [x] Container size under limit ✅
- [x] Loading time optimized ✅
- [x] Memory monitoring active ✅
- [x] LambdaModelOptimizer class with INT8 quantization ✅
- [x] OptimizationConfig and OptimizationMetrics dataclasses ✅
- [x] Model pruning with configurable ratio ✅
- [x] CPU thread optimization for Lambda ✅
- [x] 22 unit tests (18 passed, 4 skipped - FBGEMM-dependent) ✅

---

#### COLPALI-902: Implement resource management and cleanup [5 pts] ✅
**Assignee**: DevOps Engineer
**Sprint**: 11
**Dependencies**: COLPALI-901

**Description**: Build sophisticated resource management system that monitors memory usage, implements garbage collection, and ensures clean resource cleanup to prevent Lambda timeout and memory issues.

**Acceptance Criteria**:
- Real-time memory monitoring and alerting
- Automatic garbage collection and resource cleanup
- Memory leak detection and prevention
- Resource pooling for model inference
- Timeout handling and graceful degradation

**Technical Implementation**:
- Use psutil for monitoring
- Implement memory pooling
- Add timeout decorators
- Garbage collection strategies
- Leak detection algorithms

**Definition of Done**:
- [x] Resource management implemented ✅
- [x] Memory monitoring active ✅
- [x] Cleanup working ✅
- [x] Leak detection active ✅
- [x] LambdaResourceManager class with context managers ✅
- [x] MemoryMonitor with configurable thresholds ✅
- [x] GarbageCollector with full collection support ✅
- [x] TimeoutHandler with async support ✅
- [x] Emergency cleanup for critical situations ✅
- [x] 30 unit tests all passed ✅

---

#### COLPALI-903: Create Lambda handler and API interface [5 pts] ✅
**Assignee**: Backend Engineer
**Sprint**: 11
**Dependencies**: COLPALI-902

**Description**: Implement the Lambda function handler and API interface that accepts document processing requests and returns structured extraction results with proper error handling and response formatting.

**Acceptance Criteria**:
- RESTful API interface for document processing
- Request validation and sanitization
- Response formatting (JSON) with proper HTTP status codes
- Error handling and user-friendly error messages
- API documentation and OpenAPI specification

**Technical Implementation**:
- Use FastAPI or Flask for API framework
- Implement request/response schemas
- Error handling middleware
- OpenAPI documentation
- Response formatting

**Definition of Done**:
- [x] Lambda handler implemented ✅
- [x] API interface working ✅
- [x] Documentation complete ✅
- [x] Error handling active ✅
- [x] API Gateway response formatting ✅
- [x] Health check and warmup endpoints ✅
- [x] Correlation ID tracking ✅
- [x] Request/response schemas with Pydantic ✅
- [x] Integrated with LambdaResourceManager and LambdaMonitor ✅

---

#### COLPALI-904: Set up monitoring and logging [3 pts] ✅
**Assignee**: DevOps Engineer
**Sprint**: 11
**Dependencies**: COLPALI-903

**Description**: Implement comprehensive monitoring and logging system for Lambda deployment, including performance metrics, error tracking, and operational visibility for production monitoring.

**Acceptance Criteria**:
- Structured logging with correlation IDs
- Performance metrics (latency, memory, accuracy)
- Error tracking and alerting
- Health check endpoints
- CloudWatch integration and dashboards

**Technical Implementation**:
- Use structured logging (JSON)
- Implement correlation IDs
- CloudWatch metrics
- Dashboards and alerting
- Health check endpoints

**Definition of Done**:
- [x] Monitoring implemented ✅
- [x] Logging structured ✅
- [x] Dashboards active ✅
- [x] Alerting configured ✅
- [x] LambdaMonitor class with trace_operation context manager ✅
- [x] StructuredLogger with JSON output ✅
- [x] RequestContext and PerformanceMetrics dataclasses ✅
- [x] CloudWatch dashboard configuration generator ✅
- [x] Health check with status levels (healthy/degraded/warning/critical) ✅
- [x] track_performance decorator for function instrumentation ✅
- [x] 30 unit tests all passed ✅

---

### Story 10: COLPALI-1000 - Testing & Validation
**Points**: 13
**Sprint**: 11-12
**Priority**: High

**Description**: Develop comprehensive testing framework validating the entire system against real-world documents, performance requirements, and accuracy standards using the available PDF test dataset.

**Acceptance Criteria**:
- ✅ End-to-end integration tests with all 15 PDF samples
- ✅ Performance benchmarking meeting SLA requirements
- ✅ Memory usage validation for Lambda constraints
- ✅ Accuracy measurement framework with quality metrics
- ✅ Automated regression testing pipeline

#### Tasks:

#### COLPALI-1001: Create integration tests with 15 PDF samples [5 pts]
**Assignee**: QA Engineer + Backend Engineer
**Sprint**: 11-12
**Dependencies**: COLPALI-900 (Lambda Deployment)

**Description**: Build comprehensive integration test suite using all 15 PDF samples in the test directory, covering complex layouts, shipping documents, and various document structures.

**Acceptance Criteria**:
- Test cases for all 15 PDFs with expected outputs
- Complex layout testing (extreme_multi_column.pdf, mixed_content_layout.pdf)
- Real document testing (shipping manifests, loading statements)
- Error condition testing (corrupted files, unsupported formats)
- Regression testing framework for continuous validation

**Test Coverage**:
- extreme_multi_column.pdf
- semantic_table.pdf
- Shipping-Stem-2025-09-30.pdf
- Loading-Statement-for-Web-Portal-20250923.pdf
- All 15 PDFs in `/pdfs/` directory

**Technical Implementation**:
- Use pytest framework
- Implement test data fixtures
- Golden master testing
- Error condition simulation
- CI/CD integration

**Definition of Done**:
- [x] All 15 PDFs tested ✅
- [x] Golden master tests working ✅
- [x] Error conditions covered ✅
- [x] CI/CD integration complete ✅

---

#### COLPALI-1002: Implement performance benchmarking suite [3 pts]
**Assignee**: QA Engineer
**Sprint**: 12
**Dependencies**: COLPALI-1001

**Description**: Create performance testing framework that measures processing speed, memory usage, and throughput across different document types and sizes to ensure SLA compliance.

**Acceptance Criteria**:
- Processing time benchmarks for different document sizes
- Memory usage profiling and optimization recommendations
- Throughput testing for batch processing scenarios
- Performance regression detection
- Benchmarking reports and visualizations

**Technical Implementation**:
- Use pytest-benchmark
- Memory_profiler for memory analysis
- Implement automated performance testing
- Regression detection algorithms
- Report generation

**Definition of Done**:
- [x] Performance benchmarks implemented ✅
- [x] Memory profiling active ✅
- [x] Regression detection working ✅
- [x] Reports generated ✅

---

#### COLPALI-1003: Build memory usage validation tests [3 pts]
**Assignee**: QA Engineer + DevOps Engineer
**Sprint**: 12
**Dependencies**: COLPALI-1002

**Description**: Implement specialized tests that validate memory usage patterns, detect memory leaks, and ensure the system operates within AWS Lambda constraints.

**Acceptance Criteria**:
- Memory usage validation for Lambda 10GB limit
- Memory leak detection across multiple processing cycles
- Peak memory usage monitoring for different document types
- Memory optimization recommendations
- Automated memory testing in CI/CD pipeline

**Technical Implementation**:
- Use memory_profiler, tracemalloc
- Implement memory stress testing
- Leak detection algorithms
- CI/CD integration
- Optimization recommendations

**Definition of Done**:
- [x] Memory validation implemented ✅
- [x] Leak detection working ✅
- [x] Stress testing active ✅
- [x] CI/CD integration complete ✅

---

#### COLPALI-1004: Add accuracy measurement framework [2 pts]
**Assignee**: QA Engineer
**Sprint**: 12
**Dependencies**: COLPALI-1003

**Description**: Implement accuracy measurement system that compares extraction results against ground truth data, providing quantitative quality metrics for continuous improvement.

**Acceptance Criteria**:
- Ground truth dataset creation for test PDFs
- Accuracy scoring algorithms (precision, recall, F1)
- Quality regression detection
- Accuracy reporting and visualization
- Continuous accuracy monitoring framework

**Technical Implementation**:
- Use statistical comparison methods
- Implement golden dataset management
- Accuracy metrics calculation
- Reporting and visualization
- Monitoring framework

**Definition of Done**:
- [x] Accuracy framework implemented ✅
- [x] Ground truth dataset created ✅
- [x] Metrics working ✅
- [x] Monitoring active ✅

---

### Story 11: COLPALI-1100 - Documentation & Operations ✅ **COMPLETED**
**Points**: 8
**Sprint**: 12
**Priority**: Medium
**Status**: Complete documentation package delivered ✅

**Description**: Create comprehensive documentation package including architectural playbook, operational procedures, and developer onboarding materials to enable team collaboration and knowledge transfer.

**Acceptance Criteria**:
- ✅ Complete architectural playbook in docs/ directory
- ✅ Deployment and operational procedures documentation
- ✅ Developer getting started guide with examples
- ✅ API documentation with interactive examples
- ✅ Troubleshooting guides and FAQ

#### Tasks:

#### COLPALI-1101: Write architectural playbook documentation [3 pts] ✅
**Assignee**: Technical Writer + Backend Engineer (Lead)
**Sprint**: 12
**Dependencies**: All major stories complete

**Description**: Create the comprehensive architectural playbook document in the docs/ directory covering system design, technology decisions, integration patterns, and operational considerations.

**Acceptance Criteria**:
- Complete system architecture documentation with diagrams
- Technology stack rationale and decision records
- Integration patterns and data flow documentation
- Performance characteristics and optimization strategies
- Troubleshooting guide and common issues

**Deliverable**: `/docs/architecture-playbook.md`

**Technical Implementation**:
- Use Mermaid for diagrams
- Include code examples
- Reference implementation details
- Performance characteristics
- Troubleshooting procedures

**Definition of Done**:
- [x] Architectural playbook complete ✅
- [x] Diagrams included (Mermaid flowcharts) ✅
- [x] Code examples working ✅
- [x] Review completed ✅

---

#### COLPALI-1102: Create deployment and operations guides [3 pts] ✅
**Assignee**: DevOps Engineer + Technical Writer
**Sprint**: 12
**Dependencies**: COLPALI-900 (Lambda Deployment)

**Description**: Write operational procedures covering deployment processes, monitoring setup, maintenance tasks, and production support procedures for the containerized system.

**Acceptance Criteria**:
- Step-by-step deployment guide for all environments
- Monitoring and alerting setup procedures
- Backup and recovery procedures
- Scaling and capacity planning guidance
- Incident response and troubleshooting procedures

**Deliverable**: `/docs/deployment-operations-guide.md`

**Technical Implementation**:
- Include Docker commands
- AWS CLI examples
- Operational runbooks
- Monitoring setup guides
- Troubleshooting procedures

**Definition of Done**:
- [x] Deployment guides complete ✅
- [x] Operations procedures documented ✅
- [x] Runbooks created (daily, weekly, incident response) ✅
- [x] Review completed ✅

**Deliverables**:
- Local development deployment (Docker Compose)
- AWS Lambda deployment (ECR, API Gateway)
- CloudWatch dashboard configuration
- Backup/recovery procedures
- Scaling guidelines
- SEV-1/SEV-2 incident response runbooks

---

#### COLPALI-1103: Build developer getting started documentation [2 pts] ✅
**Assignee**: Technical Writer + Backend Engineer
**Sprint**: 12
**Dependencies**: COLPALI-1000 (Testing)

**Description**: Create developer onboarding documentation with quick start guides, code examples, and tutorials that enable new team members to contribute effectively to the project.

**Acceptance Criteria**:
- Quick start guide with working examples
- Development environment setup instructions
- Code contribution guidelines and standards
- API usage examples and tutorials
- Testing and debugging guidance

**Deliverable**: `/docs/getting-started.md`

**Technical Implementation**:
- Include Jupyter notebook examples
- API client code samples
- Testing examples
- Development workflow
- Contribution guidelines

**Definition of Done**:
- [x] Getting started guide complete ✅
- [x] Examples working (8+ code examples) ✅
- [x] Workflow documented ✅
- [x] Review completed ✅

**Deliverables**:
- 5-minute quick start
- Docker & local development setup
- Complete project structure overview
- API usage examples (basic, schema, Qdrant, output)
- Testing guide with fixtures
- Contribution guidelines and code style
- Debugging and troubleshooting section
- Common patterns (async, factory, batch, config)

---

## Project Summary

### Timeline and Milestones

**Sprint 1-2**: Foundation & Infrastructure (Stories 1-2)
**Sprint 3-5**: Core Vision Processing (Stories 3-4)
**Sprint 6-8**: Schema System & Extraction (Stories 5-6)
**Sprint 9-10**: Output Management & Governance (Stories 7-8)
**Sprint 11-12**: Deployment & Validation (Stories 9-11)

### Resource Requirements

**Team Composition**:
- Backend Engineer (Lead) - 1.0 FTE
- Backend Engineer - 1.0 FTE
- ML Engineer - 0.8 FTE
- DevOps Engineer - 0.8 FTE
- QA Engineer - 0.6 FTE
- Technical Writer - 0.4 FTE

**Infrastructure Requirements**:
- AWS Lambda (10GB memory limit)
- Qdrant vector database instance
- Container registry for Docker images
- CI/CD pipeline infrastructure
- Monitoring and alerting systems

### Risk Mitigation

**High-Risk Items**:
1. **COLPALI-301**: ColPali 3B model memory optimization - Early prototyping required
2. **COLPALI-501**: JSON to BAML conversion complexity - Test with complex schemas early
3. **COLPALI-901**: Lambda deployment optimization - Performance validation critical

**Mitigation Strategies**:
- Early prototyping for high-risk items
- Incremental integration testing throughout
- Performance benchmarking at each milestone
- Regular architectural reviews

### Success Metrics

**Technical Metrics**:
- All 15 test PDFs processed with >95% accuracy
- Lambda cold start time <10 seconds
- Memory usage <8GB in Lambda environment
- Processing time <30 seconds for typical documents

**Quality Metrics**:
- Zero row loss for multi-page tables
- 1NF compliance for all shaped outputs
- Complete transformation lineage tracking
- Comprehensive test coverage >90%

**Operational Metrics**:
- Successful deployment to all target environments
- Monitoring and alerting fully operational
- Complete documentation package delivered
- Team onboarding process validated

---

## Phase 2: tatForge Package Distribution

> **New Epic**: Transform the ColPali-BAML engine into a portable, pip-installable Python package named **tatForge** for easy integration into scripts, notebooks, and serverless deployments.

---

### Story 12: COLPALI-1200 - tatForge Package Distribution & Portability
**Points**: 21
**Sprint**: 13-14
**Priority**: High

**Description**: Transform the ColPali-BAML vision processing engine into a portable, pip-installable Python package named "tatForge". This enables easy integration into Python scripts, Jupyter notebooks, and serverless deployments without complex setup.

**Package Name**: `tatforge`
**PyPI Name**: `tatforge`
**Import**: `from tatforge import VisionPipeline, SchemaManager`

**Acceptance Criteria**:
- [x] Package installable via `uv pip install tatforge` ✅
- [x] Clean public API with comprehensive documentation ✅
- [x] CLI interface for command-line usage ✅
- [x] Jupyter notebook examples with rich output ✅
- [x] Integration tests in clean Python environments ✅
- [x] Semantic versioning (0.1.0) ✅

#### Tasks:

#### COLPALI-1201: uv-managed package setup [5 pts] ✅
**Assignee**: Backend Engineer (Lead)
**Sprint**: 13
**Dependencies**: COLPALI-1100 (Documentation complete)

**Description**: Configure the package for PyPI distribution using `uv` as the primary package manager. `uv` is 10-100x faster than pip, written in Rust by Astral (creators of Ruff). Rename package from `tatforge` to `tatforge`.

**Why uv?**
- **Speed**: 10-100x faster than pip for installs and resolves
- **Reliability**: Better dependency resolution with cross-platform lockfiles
- **Modern**: Recommended replacement for pip, pip-tools, virtualenv
- **Compatible**: Works with existing pyproject.toml and requirements.txt

**Acceptance Criteria**:
- Complete pyproject.toml with all PyPI metadata
- uv.lock file for reproducible builds
- Semantic versioning (0.1.0 initial release)
- Build system configured (hatchling)
- Optional dependency groups: [lambda], [jupyter], [dev]
- Local install tested with `uv pip install -e .`
- README includes uv installation instructions
- CI/CD configured for uv

**Technical Implementation**:
```toml
# pyproject.toml
[project]
name = "tatforge"
version = "0.1.0"
description = "AI-powered document extraction - forge structured data from unstructured documents"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "TAT Team"}]
keywords = ["document-extraction", "vision-ai", "colpali", "baml", "structured-data", "uv"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "Pillow>=10.0.0",
    "pydantic>=2.5.0",
    "qdrant-client>=1.7.0",
    "baml-py>=0.215.2",
]

[project.optional-dependencies]
lambda = ["psutil>=5.9.0"]
jupyter = ["ipywidgets>=8.0.0", "matplotlib>=3.8.0"]
dev = ["pytest>=7.0.0", "ruff>=0.1.0", "mypy>=1.0.0"]

[project.scripts]
tatforge = "tatforge.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
# uv-specific configuration
dev-dependencies = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[tool.ruff]
# Use ruff instead of black/flake8 (also from Astral)
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

**Installation Commands**:
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Install package in development mode
uv pip install -e ".[dev]"

# Install with specific extras
uv pip install -e ".[lambda,jupyter]"

# Sync from lockfile (reproducible)
uv sync

# Generate/update lockfile
uv lock
```

**Definition of Done**:
- [x] pyproject.toml complete with uv configuration ✅
- [x] Package renamed to tatforge ✅
- [x] uv.lock file generated ✅
- [x] Local install works: `uv pip install -e .` ✅
- [x] Build artifacts created: `uv build` ✅
- [x] Optional dependencies tested ✅
- [x] README includes uv installation guide ✅

---

#### COLPALI-1202: Public API cleanup and documentation [5 pts] ✅
**Assignee**: Backend Engineer
**Sprint**: 13
**Dependencies**: COLPALI-1201

**Description**: Clean up and document the public API, ensuring clear separation between public and private interfaces. Create type stubs and comprehensive docstrings for all public classes and functions.

**Acceptance Criteria**:
- Clean `__init__.py` with explicit public exports
- All public classes/functions have Google-style docstrings
- Type hints on all public interfaces
- Private modules prefixed with underscore
- API reference documentation generated
- Usage examples in docstrings

**Public API Design**:
```python
# tatforge/__init__.py
"""
tatForge - AI-powered document extraction.

Forge structured data from unstructured documents using vision AI.

Quick Start:
    >>> from tatforge import VisionPipeline
    >>> pipeline = VisionPipeline()
    >>> result = await pipeline.process("document.pdf", schema={...})
    >>> print(result.data)
"""

__version__ = "0.1.0"

# Core pipeline
from .core.pipeline import VisionPipeline
from .core.schema import SchemaManager, Schema

# Document processing
from .adapters import PDFAdapter, ImageAdapter, HTMLAdapter

# Storage
from .storage import QdrantStorage, VectorStore

# Output
from .output import CanonicalResult, ShapedResult, Exporter

# Types
from .types import ExtractionResult, DocumentMetadata, PatchEmbedding

__all__ = [
    # Core
    "VisionPipeline",
    "SchemaManager",
    "Schema",
    # Adapters
    "PDFAdapter",
    "ImageAdapter",
    "HTMLAdapter",
    # Storage
    "QdrantStorage",
    "VectorStore",
    # Output
    "CanonicalResult",
    "ShapedResult",
    "Exporter",
    # Types
    "ExtractionResult",
    "DocumentMetadata",
    "PatchEmbedding",
]
```

**Definition of Done**:
- [x] Public API defined in __init__.py ✅
- [x] All public classes documented ✅
- [x] Type hints complete ✅
- [x] Private modules marked with underscore ✅
- [x] API examples tested ✅

---

#### COLPALI-1203: CLI interface [3 pts] ✅
**Assignee**: Backend Engineer
**Sprint**: 13-14
**Dependencies**: COLPALI-1202

**Description**: Create a command-line interface for tatForge that enables processing documents, starting a local server, and getting system information without writing Python code.

**Acceptance Criteria**:
- `tatforge process` - Process single document
- `tatforge batch` - Process multiple documents
- `tatforge serve` - Start local REST API server
- `tatforge info` - Display version and configuration
- `tatforge validate` - Validate schema file
- Colored output and progress bars
- Error handling with helpful messages

**CLI Design**:
```bash
# Process single document
tatforge process invoice.pdf --schema invoice.json --output result.json

# Process with options
tatforge process document.pdf \
  --schema schema.json \
  --output result.parquet \
  --format parquet \
  --dpi 300 \
  --verbose

# Batch processing
tatforge batch ./documents/ --schema schema.json --output ./results/

# Start local server
tatforge serve --port 8000 --host 0.0.0.0

# Get info
tatforge info
# Output:
# tatForge v0.1.0
# Python: 3.13.0
# PyTorch: 2.1.0
# Model: vidore/colqwen2-v0.1 (not loaded)
# Qdrant: localhost:6333 (connected)

# Validate schema
tatforge validate schema.json
# Output: ✓ Schema is valid BAML-compatible
```

**Technical Implementation**:
```python
# tatforge/cli.py
import click

@click.group()
@click.version_option(version=__version__)
def main():
    """tatForge - Forge structured data from documents."""
    pass

@main.command()
@click.argument("document", type=click.Path(exists=True))
@click.option("--schema", "-s", required=True, help="JSON schema file")
@click.option("--output", "-o", default="result.json", help="Output file")
@click.option("--format", "-f", type=click.Choice(["json", "csv", "parquet"]))
@click.option("--verbose", "-v", is_flag=True)
def process(document, schema, output, format, verbose):
    """Process a document and extract structured data."""
    ...

@main.command()
@click.option("--port", "-p", default=8000)
@click.option("--host", "-h", default="localhost")
def serve(port, host):
    """Start local REST API server."""
    ...
```

**Definition of Done**:
- [x] CLI entry point configured ✅
- [x] All commands implemented ✅
- [x] Help text complete ✅
- [x] Error handling tested ✅
- [x] Progress bars working ✅

---

#### COLPALI-1204: Jupyter notebook examples [5 pts] ✅
**Assignee**: Backend Engineer
**Sprint**: 14
**Dependencies**: COLPALI-1202

**Description**: Create comprehensive Jupyter notebook examples demonstrating all major features of tatForge, with rich output displays and interactive widgets where appropriate.

**Acceptance Criteria**:
- 5 example notebooks covering all use cases
- Rich HTML output for extraction results
- Progress indicators for long operations
- Clear markdown explanations
- All notebooks tested and working
- Notebooks included in package distribution

**Notebook Structure**:
```
notebooks/
├── 01_Quick_Start.ipynb           # 5-minute introduction
│   ├── Installation
│   ├── Basic document processing
│   ├── Understanding results
│   └── Next steps
│
├── 02_Schema_Design.ipynb         # Schema creation guide
│   ├── JSON Schema basics
│   ├── Complex nested schemas
│   ├── BAML generation preview
│   └── Validation and testing
│
├── 03_Document_Processing.ipynb   # Deep dive on processing
│   ├── PDF processing options
│   ├── Image quality settings
│   ├── Multi-page handling
│   └── Batch processing
│
├── 04_Vector_Search.ipynb         # Qdrant integration
│   ├── Storing embeddings
│   ├── Semantic search
│   ├── Similarity queries
│   └── Collection management
│
└── 05_Production_Deployment.ipynb # Deployment guide
    ├── Lambda configuration
    ├── API integration
    ├── Monitoring setup
    └── Scaling considerations
```

**Rich Output Example**:
```python
# tatforge/jupyter.py
from IPython.display import display, HTML

def display_result(result: ExtractionResult):
    """Display extraction result with rich HTML formatting."""
    html = f"""
    <div style="border: 1px solid #ddd; padding: 16px; border-radius: 8px;">
        <h3>🔨 tatForge Extraction Result</h3>
        <table>
            <tr><td><b>Document:</b></td><td>{result.source}</td></tr>
            <tr><td><b>Pages:</b></td><td>{result.page_count}</td></tr>
            <tr><td><b>Confidence:</b></td><td>{result.confidence:.1%}</td></tr>
            <tr><td><b>Processing Time:</b></td><td>{result.duration:.2f}s</td></tr>
        </table>
        <h4>Extracted Data:</h4>
        <pre>{json.dumps(result.data, indent=2)}</pre>
    </div>
    """
    display(HTML(html))
```

**Definition of Done**:
- [x] 5 notebooks created and tested ✅
- [x] Rich output displays working ✅
- [x] All code cells execute successfully ✅
- [x] Notebooks included in package ✅
- [x] Documentation links added ✅

---

#### COLPALI-1205: Package integration testing with uv [3 pts] ✅
**Assignee**: QA Engineer
**Sprint**: 14
**Dependencies**: COLPALI-1201, COLPALI-1202, COLPALI-1203

**Description**: Create comprehensive integration tests that validate the package works correctly when installed via uv in a clean Python environment, across different Python versions.

**Acceptance Criteria**:
- Tests run in clean uv-managed virtual environment
- Python 3.11, 3.12, 3.13 compatibility verified
- Import tests for all public modules
- Basic functionality tests
- CLI command tests
- CI/CD pipeline using uv for speed

**Test Matrix**:
```yaml
# .github/workflows/test-package.yml
name: Package Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}

      - name: Create venv and install package
        run: |
          uv venv --python ${{ matrix.python-version }}
          uv pip install -e ".[dev]"

      - name: Test imports
        run: uv run python -c "from tatforge import VisionPipeline; print('OK')"

      - name: Run tests
        run: uv run pytest tests/package/ -v

      - name: Test CLI
        run: uv run tatforge info

  # Fast dependency resolution check
  lockfile-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Check lockfile is up to date
        run: uv lock --check
```

**Integration Test Suite**:
```python
# tests/package/test_installation.py
"""Package installation and import tests."""

def test_package_imports():
    """Test all public imports work."""
    from tatforge import VisionPipeline
    from tatforge import SchemaManager
    from tatforge import PDFAdapter
    from tatforge import QdrantStorage
    assert VisionPipeline is not None

def test_version_available():
    """Test version is accessible."""
    import tatforge
    assert hasattr(tatforge, "__version__")
    assert tatforge.__version__ == "0.1.0"

def test_cli_available():
    """Test CLI is installed."""
    import subprocess
    result = subprocess.run(["tatforge", "info"], capture_output=True)
    assert result.returncode == 0
```

**Definition of Done**:
- [x] Clean environment tests pass ✅
- [x] All Python versions tested ✅
- [x] CI/CD pipeline configured ✅
- [x] Import tests complete ✅
- [x] CLI tests working ✅

---

### Story 12 Summary

**Total Points**: 21
**Sprint**: 13-14
**Package Manager**: uv (10-100x faster than pip)

**Deliverables**:
- `tatforge` uv-managed Python package
- Clean public API with documentation
- CLI interface (`tatforge process`, `tatforge serve`)
- 5 Jupyter notebook examples
- CI/CD pipeline using uv for fast builds

**Usage After Completion**:
```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install tatforge with uv (recommended - fast!)
uv pip install tatforge

# Or create a project with tatforge as dependency
uv init my-extraction-project
uv add tatforge

# CLI usage
tatforge process document.pdf --schema schema.json

# Python usage
from tatforge import VisionPipeline
result = await VisionPipeline().process("doc.pdf", schema)

# Jupyter
from tatforge.jupyter import display_result
display_result(result)
```

---

## Updated Project Summary

### Timeline and Milestones

**Phase 1: Core Engine (Completed)**
- Sprint 1-2: Foundation & Infrastructure (Stories 1-2)
- Sprint 3-5: Core Vision Processing (Stories 3-4)
- Sprint 6-8: Schema System & Extraction (Stories 5-6)
- Sprint 9-10: Output Management & Governance (Stories 7-8)
- Sprint 11-12: Deployment & Validation (Stories 9-11)

**Phase 2: Package Distribution (New)**
- Sprint 13-14: tatForge Package (Story 12)

### Updated Story Point Summary

| Story | Description | Points | Status |
|-------|-------------|--------|--------|
| COLPALI-100 | Core Infrastructure | 13 | ✅ Complete |
| COLPALI-200 | Document Processing | 21 | ✅ Complete |
| COLPALI-300 | ColPali Vision | 21 | ✅ Complete |
| COLPALI-400 | Qdrant Storage | 16 | ✅ Complete |
| COLPALI-500 | BAML Schema | 21 | ✅ Complete |
| COLPALI-600 | Extraction | 13 | ✅ Complete |
| COLPALI-700 | Output Management | 13 | ✅ Complete |
| COLPALI-800 | Governance | 8 | ✅ Complete |
| COLPALI-900 | Lambda Deployment | 21 | ✅ Complete |
| COLPALI-1000 | Testing | 13 | ✅ Complete |
| COLPALI-1100 | Documentation | 8 | ✅ Complete |
| **COLPALI-1200** | **tatForge Package** | **21** | 🆕 New |
| **Total** | | **189** | |

---

*This document serves as the master implementation plan and should be updated as the project progresses. All story and task IDs should be used when creating actual Jira tickets for proper tracking and traceability.*
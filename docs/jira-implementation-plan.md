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
colpali_engine/
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
**Points**: 21
**Sprint**: 2-3
**Priority**: High (Critical Path)

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

#### COLPALI-204: Integrate CocoIndex orchestration framework [5 pts]
**Assignee**: Backend Engineer (Lead)
**Sprint**: 3
**Dependencies**: COLPALI-203

**Description**: Integrate CocoIndex framework to orchestrate the document processing pipeline, providing workflow management, dependency tracking, and coordination between different processing stages.

**Acceptance Criteria**:
- CocoIndex flow definitions for document processing workflow
- Integration with existing BAML setup
- Dependency management between processing stages
- Error propagation and recovery mechanisms
- Performance monitoring and metrics collection

**Technical Implementation**:
- Define CocoIndex transforms for each processing stage
- Implement async processing where applicable
- Set up workflow dependencies
- Error handling and retry logic
- Metrics collection at each stage

**Definition of Done**:
- [ ] CocoIndex integration complete
- [ ] Workflow definitions created
- [ ] Error handling tested
- [ ] Performance metrics validated
- [ ] BAML integration verified

---

#### COLPALI-205: Implement caching and incremental processing [3 pts]
**Assignee**: Backend Engineer
**Sprint**: 3
**Dependencies**: COLPALI-204

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
- [ ] ColPali model client implemented
- [ ] Memory optimization validated
- [ ] Performance benchmarks complete
- [ ] Error handling tested
- [ ] Integration tests passed

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
- [ ] Batch processing implemented
- [ ] Memory management validated
- [ ] Performance optimized
- [ ] Progress tracking working

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
- [ ] Patch extraction implemented
- [ ] Embeddings generated correctly
- [ ] Spatial metadata preserved
- [ ] Quality validation working
- [ ] Storage format optimized

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
- [ ] Cold start optimization implemented
- [ ] Performance targets met (<10 seconds)
- [ ] Monitoring in place
- [ ] Lambda deployment tested

---

### Story 4: COLPALI-400 - Qdrant Vector Storage
**Points**: 13
**Sprint**: 5-6
**Priority**: Medium

**Description**: Implement vector database operations using Qdrant for storing and querying ColPali embeddings with spatial metadata. This enables efficient semantic search and retrieval of document patches containing tables and structured data.

**Acceptance Criteria**:
- ✅ Qdrant collections configured for multi-vector storage
- ✅ Efficient storage of patch embeddings with spatial coordinates
- ✅ Semantic search functionality with similarity thresholds
- ✅ Collection management and optimization
- ✅ Performance tuning for large document collections

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
- [ ] Qdrant client implemented
- [ ] Collection schemas created
- [ ] Connection management working
- [ ] Health monitoring active

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
- [ ] Multi-vector storage implemented
- [ ] Batch operations working
- [ ] Metadata indexing active
- [ ] Performance validated

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
- [ ] Search engine implemented
- [ ] Ranking algorithms working
- [ ] Performance optimized
- [ ] Caching active

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
- [ ] Schema converter implemented
- [ ] All test cases passing
- [ ] Generated BAML validated
- [ ] Error handling complete

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
- [ ] Function generator implemented
- [ ] Prompt templates working
- [ ] Client selection active
- [ ] Testing framework complete

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
- [ ] Validation system implemented
- [ ] Error messages helpful
- [ ] Migration support working
- [ ] CI/CD integration active

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
- [ ] Integration complete
- [ ] Backward compatibility verified
- [ ] Configuration extended
- [ ] Testing complete

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
- [ ] Vision clients configured
- [ ] Image input working
- [ ] Fallback strategies active
- [ ] Cost optimization implemented

---

### Story 6: COLPALI-600 - Extraction & Validation
**Points**: 13
**Sprint**: 8
**Priority**: High (Critical Path)

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
- [ ] BAML execution interface complete
- [ ] Image processing working
- [ ] Performance optimized
- [ ] Error handling active

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
- [ ] Validation system implemented
- [ ] Quality checks active
- [ ] Reporting working
- [ ] Custom validators complete

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
- [ ] Error handling implemented
- [ ] Retry logic working
- [ ] Monitoring active
- [ ] Recovery tested

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
- [ ] Quality metrics implemented
- [ ] Confidence scoring working
- [ ] Benchmarking active
- [ ] Alerting configured

---

### Story 7: COLPALI-700 - Output Management
**Points**: 13
**Sprint**: 9
**Priority**: Medium

**Description**: Implement the dual-output architecture with canonical truth layer and shaped output layer, ensuring data integrity while supporting business transformation requirements with full lineage tracking.

**Acceptance Criteria**:
- ✅ Canonical data formatter preserving extraction truth
- ✅ Shaped data formatter with mandatory 1NF enforcement
- ✅ CSV and Parquet export capabilities with proper encoding
- ✅ Output format validation and quality assurance
- ✅ Metadata and lineage preservation in all outputs

#### Tasks:

#### COLPALI-701: Implement canonical data formatter (truth layer) [5 pts]
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
- [ ] Canonical formatter implemented
- [ ] Metadata preservation working
- [ ] Validation complete
- [ ] Audit trail active

---

#### COLPALI-702: Build shaped data formatter with 1NF enforcement [5 pts]
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
- [ ] Shaped formatter implemented
- [ ] 1NF enforcement working
- [ ] Transformation rules active
- [ ] Lineage tracking complete

---

#### COLPALI-703: Create CSV/Parquet export functionality [3 pts]
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
- [ ] Export utilities implemented
- [ ] Both formats working
- [ ] Streaming support active
- [ ] Validation complete

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
- [ ] Lineage tracking implemented
- [ ] UUID tracking working
- [ ] Query interface active
- [ ] Metrics collection complete

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
- [ ] Governance rules implemented
- [ ] Validation working
- [ ] Approval workflow active
- [ ] Testing framework complete

---

### Story 9: COLPALI-900 - Lambda Deployment
**Points**: 21
**Sprint**: 10-11
**Priority**: High

**Description**: Optimize the entire system for AWS Lambda deployment, handling the challenges of 3B parameter model deployment, memory management, and cold start optimization in a serverless environment.

**Acceptance Criteria**:
- ✅ Lambda container deployment supporting 3B ColPali model
- ✅ Memory usage under 8GB for 10GB Lambda limit
- ✅ Cold start time under 10 seconds
- ✅ API interface for document processing requests
- ✅ Comprehensive monitoring and alerting

#### Tasks:

#### COLPALI-901: Optimize Lambda container for 3B model deployment [8 pts]
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
- [ ] Model optimization complete
- [ ] Container size under limit
- [ ] Loading time optimized
- [ ] Memory monitoring active

---

#### COLPALI-902: Implement resource management and cleanup [5 pts]
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
- [ ] Resource management implemented
- [ ] Memory monitoring active
- [ ] Cleanup working
- [ ] Leak detection active

---

#### COLPALI-903: Create Lambda handler and API interface [5 pts]
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
- [ ] Lambda handler implemented
- [ ] API interface working
- [ ] Documentation complete
- [ ] Error handling active

---

#### COLPALI-904: Set up monitoring and logging [3 pts]
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
- [ ] Monitoring implemented
- [ ] Logging structured
- [ ] Dashboards active
- [ ] Alerting configured

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
- [ ] All 15 PDFs tested
- [ ] Golden master tests working
- [ ] Error conditions covered
- [ ] CI/CD integration complete

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
- [ ] Performance benchmarks implemented
- [ ] Memory profiling active
- [ ] Regression detection working
- [ ] Reports generated

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
- [ ] Memory validation implemented
- [ ] Leak detection working
- [ ] Stress testing active
- [ ] CI/CD integration complete

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
- [ ] Accuracy framework implemented
- [ ] Ground truth dataset created
- [ ] Metrics working
- [ ] Monitoring active

---

### Story 11: COLPALI-1100 - Documentation & Operations
**Points**: 8
**Sprint**: 12
**Priority**: Medium

**Description**: Create comprehensive documentation package including architectural playbook, operational procedures, and developer onboarding materials to enable team collaboration and knowledge transfer.

**Acceptance Criteria**:
- ✅ Complete architectural playbook in docs/ directory
- ✅ Deployment and operational procedures documentation
- ✅ Developer getting started guide with examples
- ✅ API documentation with interactive examples
- ✅ Troubleshooting guides and FAQ

#### Tasks:

#### COLPALI-1101: Write architectural playbook documentation [3 pts]
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
- [ ] Architectural playbook complete
- [ ] Diagrams included
- [ ] Code examples working
- [ ] Review completed

---

#### COLPALI-1102: Create deployment and operations guides [3 pts]
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

**Technical Implementation**:
- Include Docker commands
- AWS CLI examples
- Operational runbooks
- Monitoring setup guides
- Troubleshooting procedures

**Definition of Done**:
- [ ] Deployment guides complete
- [ ] Operations procedures documented
- [ ] Runbooks created
- [ ] Review completed

---

#### COLPALI-1103: Build developer getting started documentation [2 pts]
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

**Technical Implementation**:
- Include Jupyter notebook examples
- API client code samples
- Testing examples
- Development workflow
- Contribution guidelines

**Definition of Done**:
- [ ] Getting started guide complete
- [ ] Examples working
- [ ] Workflow documented
- [ ] Review completed

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

*This document serves as the master implementation plan and should be updated as the project progresses. All story and task IDs should be used when creating actual Jira tickets for proper tracking and traceability.*
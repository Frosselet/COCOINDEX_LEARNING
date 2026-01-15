# ColPali-BAML Developer Getting Started Guide

> **Developer Onboarding Document**: Quick start guide, setup instructions, code examples, and contribution guidelines for the ColPali-BAML Vision Processing Engine.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [API Usage Examples](#api-usage-examples)
5. [Testing Guide](#testing-guide)
6. [Code Contribution Guidelines](#code-contribution-guidelines)
7. [Debugging & Troubleshooting](#debugging--troubleshooting)
8. [Common Patterns](#common-patterns)

---

## Quick Start

Get up and running in 5 minutes with the development environment.

### Prerequisites

- **Docker Desktop** 4.0+ with 16GB+ memory allocation
- **Python** 3.13+ (for local development)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/colpali-baml.git
cd colpali-baml
```

### 2. Start the Development Environment

```bash
# Build and start all services
docker-compose up -d

# Wait for services to be healthy
docker-compose ps
```

### 3. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check Qdrant health
curl http://localhost:6333/health
```

### 4. Run Your First Test

```bash
# Enter the development container
docker-compose exec colpali-engine bash

# Run tests
PYTHONPATH=. pytest tests/ -v --tb=short
```

### 5. Process Your First Document

```python
# Quick example (run in Python shell or Jupyter)
from colpali_engine import VisionExtractionPipeline, SchemaManager

# Initialize pipeline
pipeline = VisionExtractionPipeline()

# Define extraction schema
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"type": "number"}
                }
            }
        }
    }
}

# Process document
with open("document.pdf", "rb") as f:
    result = await pipeline.process_document(
        document_blob=f.read(),
        schema_json=schema
    )

print(f"Extracted: {result.canonical.extraction_data}")
```

---

## Development Environment Setup

### Option A: Docker Development (Recommended)

The Docker setup provides a consistent development environment with all dependencies pre-installed.

```bash
# Build all containers
docker-compose build

# Start services
docker-compose up -d

# Services available:
# - colpali-engine: Development container (port 8000, 5000)
# - qdrant: Vector database (port 6333, 6334)
# - jupyter: Notebook environment (port 8888)
```

#### Accessing Development Container

```bash
# Interactive bash session
docker-compose exec colpali-engine bash

# Run single command
docker-compose exec colpali-engine pytest tests/unit/ -v
```

#### Accessing Jupyter Notebooks

```bash
# Get Jupyter token
docker-compose logs jupyter | grep token

# Open browser
open http://localhost:8888
```

### Option B: Local Development

For native development without Docker:

#### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate on Windows
```

#### 2. Install Dependencies

```bash
# Install base dependencies
pip install -r requirements/base.txt

# Install development dependencies
pip install -r requirements/dev.txt

# Install system dependencies (macOS)
brew install poppler

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install poppler-utils
```

#### 3. Start Qdrant Locally

```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.7.3

# Or download binary from https://qdrant.tech/
```

#### 4. Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
QDRANT_URL=http://localhost:6333
BAML_ENV=development
LOG_LEVEL=DEBUG
PYTHONPATH=.
EOF

# Load environment
source .env
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.extraPaths": ["${workspaceFolder}"],
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "editor.formatOnSave": true,
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true
}
```

#### PyCharm

1. Set Python interpreter: `.venv/bin/python`
2. Mark `colpali_engine` as Sources Root
3. Configure pytest as test runner
4. Enable Black formatter

---

## Project Structure

```
colpali-qdrant-baml/
├── colpali_engine/              # Main package
│   ├── __init__.py             # Public API exports
│   ├── core/                   # Business logic & orchestration
│   │   ├── pipeline.py         # VisionExtractionPipeline
│   │   ├── document_adapter.py # Abstract document adapter
│   │   ├── schema_manager.py   # JSON → BAML conversion
│   │   └── baml_function_generator.py
│   ├── adapters/               # Format-specific adapters
│   │   ├── pdf_adapter.py      # PDF processing
│   │   ├── image_adapter.py    # Image processing
│   │   └── html_adapter.py     # HTML rendering
│   ├── vision/                 # ColPali integration
│   │   ├── colpali_client.py   # Model loading & embeddings
│   │   └── image_processor.py  # Image standardization
│   ├── storage/                # Vector database
│   │   └── qdrant_client.py    # Qdrant operations
│   ├── extraction/             # BAML extraction
│   │   ├── baml_interface.py   # BAML execution
│   │   ├── validation.py       # Result validation
│   │   ├── error_handling.py   # Retry logic
│   │   └── quality_metrics.py  # Quality scoring
│   ├── outputs/                # Output formatting
│   │   ├── canonical.py        # Truth layer
│   │   ├── shaped.py           # 1NF transformation
│   │   └── exporters.py        # CSV/Parquet export
│   ├── governance/             # Data governance
│   │   ├── lineage.py          # Transformation tracking
│   │   └── validation.py       # Governance rules
│   └── lambda_utils/           # AWS Lambda utilities
│       ├── model_optimizer.py  # Model quantization
│       ├── resource_manager.py # Memory management
│       └── monitoring.py       # CloudWatch integration
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── baml_src/                    # BAML configuration
│   ├── clients.baml            # LLM clients
│   └── generators.baml         # Code generation
├── docs/                        # Documentation
│   ├── architecture-playbook.md
│   ├── deployment-operations-guide.md
│   └── getting-started.md
├── pdfs/                        # Test documents (15 samples)
├── requirements/                # Dependencies
│   ├── base.txt                # Production
│   ├── dev.txt                 # Development
│   └── lambda.txt              # Lambda-optimized
├── docker-compose.yml           # Local development
├── Dockerfile.dev              # Development container
├── Dockerfile.lambda           # Production Lambda
└── lambda_handler.py           # Lambda entry point
```

### Key Modules Explained

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `core/pipeline.py` | Main orchestration | `VisionExtractionPipeline` |
| `core/schema_manager.py` | JSON→BAML conversion | `SchemaManager`, `BAMLClassGenerator` |
| `vision/colpali_client.py` | Vision model | `ColPaliClient` |
| `storage/qdrant_client.py` | Vector storage | `QdrantManager` |
| `extraction/baml_interface.py` | LLM extraction | `BAMLExecutionInterface` |
| `outputs/canonical.py` | Truth layer | `CanonicalFormatter` |
| `outputs/shaped.py` | 1NF normalization | `ShapedFormatter` |

---

## API Usage Examples

### Basic Document Processing

```python
import asyncio
from colpali_engine import VisionExtractionPipeline, SchemaManager

async def process_invoice():
    """Process an invoice document."""

    # Define schema for invoice extraction
    invoice_schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "vendor": {"type": "string"},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"}
                    }
                }
            },
            "subtotal": {"type": "number"},
            "tax": {"type": "number"},
            "total": {"type": "number"}
        },
        "required": ["invoice_number", "total"]
    }

    # Initialize pipeline
    pipeline = VisionExtractionPipeline()

    # Read document
    with open("invoice.pdf", "rb") as f:
        document_blob = f.read()

    # Process document
    result = await pipeline.process_document(
        document_blob=document_blob,
        schema_json=invoice_schema
    )

    # Access results
    print(f"Invoice Number: {result.canonical.extraction_data['invoice_number']}")
    print(f"Total: ${result.canonical.extraction_data['total']}")

    # Get shaped (1NF normalized) output
    if result.shaped:
        for item in result.shaped.transformed_data:
            print(f"Line Item: {item}")

    return result

# Run
result = asyncio.run(process_invoice())
```

### Schema Conversion

```python
from colpali_engine.core.schema_manager import SchemaManager

# Initialize schema manager
schema_manager = SchemaManager()

# Convert JSON Schema to BAML
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
    }
}

baml_code = schema_manager.generate_baml_classes(json_schema)
print(baml_code)
# Output:
# class AutoGeneratedSchema {
#     name string?
#     age int?
#     email string?
# }
```

### Working with Qdrant

```python
from colpali_engine.storage.qdrant_client import QdrantManager

# Initialize client
qdrant = QdrantManager(host="localhost", port=6333)

# Create collection for embeddings
await qdrant.ensure_collection_exists(
    collection_name="document_embeddings",
    vector_size=128,
    distance="Cosine"
)

# Store embeddings
await qdrant.store_embeddings(
    collection_name="document_embeddings",
    embeddings=embeddings,
    metadata={
        "document_id": "doc123",
        "page_number": 1,
        "source": "invoice.pdf"
    }
)

# Search similar patches
results = await qdrant.search(
    collection_name="document_embeddings",
    query_vector=query_embedding,
    limit=10,
    score_threshold=0.7
)
```

### Output Formatting

```python
from colpali_engine.outputs.canonical import CanonicalFormatter
from colpali_engine.outputs.shaped import ShapedFormatter
from colpali_engine.outputs.exporters import DataExporter

# Create canonical (truth) output
canonical_formatter = CanonicalFormatter()
canonical_data = canonical_formatter.format(
    extraction_data=extracted_data,
    source_document="invoice.pdf",
    schema_version="1.0"
)

# Verify integrity
print(f"Integrity Hash: {canonical_data.integrity_hash}")

# Apply 1NF transformation
shaped_formatter = ShapedFormatter()
shaped_data = shaped_formatter.transform_to_1nf(
    canonical_data=canonical_data,
    transformation_rules=[
        {"type": "flatten", "path": "line_items"},
        {"type": "normalize", "field": "date", "format": "ISO8601"}
    ]
)

# Export to files
exporter = DataExporter()

# Export canonical as Parquet
exporter.export_parquet(
    data=canonical_data,
    file_path="output/canonical.parquet",
    compression="snappy"
)

# Export shaped as CSV
exporter.export_csv(
    data=shaped_data,
    file_path="output/shaped.csv"
)
```

### Error Handling

```python
from colpali_engine.extraction.error_handling import (
    ErrorHandler,
    ExtractionError,
    RetryableError
)

# Configure error handling
error_handler = ErrorHandler(
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True
)

async def safe_extraction():
    """Extraction with error handling."""
    try:
        result = await error_handler.execute_with_retry(
            extraction_function,
            document_blob=doc,
            schema=schema
        )
        return result

    except RetryableError as e:
        print(f"Retryable error after {e.attempts} attempts: {e}")
        # Could implement fallback logic

    except ExtractionError as e:
        print(f"Non-retryable error: {e}")
        # Log and alert
```

---

## Testing Guide

### Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   └── test_*.py
├── integration/             # Component interaction tests
│   └── test_*_integration.py
├── e2e/                     # Full pipeline tests
│   ├── test_pdf_integration.py
│   ├── test_performance_benchmarks.py
│   └── test_accuracy_metrics.py
└── conftest.py              # Shared fixtures
```

### Running Tests

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run specific test file
PYTHONPATH=. pytest tests/unit/test_schema_manager.py -v

# Run tests matching pattern
PYTHONPATH=. pytest tests/ -k "pdf" -v

# Run with coverage
PYTHONPATH=. pytest tests/ --cov=colpali_engine --cov-report=html

# Run only fast tests (no PDF processing)
PYTHONPATH=. pytest tests/unit/ -v --ignore=tests/e2e/
```

### Writing Tests

#### Unit Test Example

```python
"""tests/unit/test_schema_converter.py"""

import pytest
from colpali_engine.core.schema_manager import SchemaManager


class TestSchemaManager:
    """Tests for SchemaManager."""

    @pytest.fixture
    def schema_manager(self):
        """Create schema manager instance."""
        return SchemaManager()

    def test_simple_schema_conversion(self, schema_manager):
        """Test converting simple JSON schema to BAML."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        baml_code = schema_manager.generate_baml_classes(schema)

        assert "class" in baml_code
        assert "name string" in baml_code

    def test_nested_schema_conversion(self, schema_manager):
        """Test converting nested JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }

        baml_code = schema_manager.generate_baml_classes(schema)

        assert "Address" in baml_code  # Nested class generated
        assert "street string" in baml_code

    @pytest.mark.parametrize("invalid_schema", [
        None,
        {},
        {"type": "invalid"},
        {"properties": "not_an_object"},
    ])
    def test_invalid_schema_raises_error(self, schema_manager, invalid_schema):
        """Test that invalid schemas raise appropriate errors."""
        with pytest.raises((ValueError, TypeError)):
            schema_manager.generate_baml_classes(invalid_schema)
```

#### Integration Test Example

```python
"""tests/integration/test_extraction_pipeline.py"""

import pytest
from pathlib import Path


class TestExtractionPipeline:
    """Integration tests for the extraction pipeline."""

    @pytest.fixture
    def sample_pdf(self):
        """Get sample PDF for testing."""
        pdf_path = Path(__file__).parent.parent.parent / "pdfs" / "sample.pdf"
        if not pdf_path.exists():
            pytest.skip("Sample PDF not available")
        return pdf_path

    @pytest.mark.asyncio
    async def test_pdf_extraction_end_to_end(self, sample_pdf):
        """Test complete PDF extraction flow."""
        from colpali_engine import VisionExtractionPipeline

        pipeline = VisionExtractionPipeline()

        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"}
            }
        }

        with open(sample_pdf, "rb") as f:
            result = await pipeline.process_document(
                document_blob=f.read(),
                schema_json=schema
            )

        assert result is not None
        assert result.canonical is not None
```

### Test Fixtures

Common fixtures in `conftest.py`:

```python
"""tests/conftest.py"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def pdf_directory(project_root):
    """Get PDF test directory."""
    return project_root / "pdfs"


@pytest.fixture(scope="session")
def all_pdf_files(pdf_directory):
    """Get all PDF test files."""
    return list(pdf_directory.glob("*.pdf"))


@pytest.fixture
def sample_schema():
    """Standard test schema."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "items": {"type": "array", "items": {"type": "object"}}
        }
    }
```

---

## Code Contribution Guidelines

### Branch Naming Convention

```bash
# Feature branches
git checkout -b feature/COLPALI-XXX-description

# Bug fixes
git checkout -b fix/COLPALI-XXX-description

# Documentation
git checkout -b docs/description
```

### Commit Message Format

```bash
git commit -m "$(cat <<'EOF'
feat: COLPALI-XXX - Add schema validation

- Implement JSON schema validation against BAML constraints
- Add error messages with fix suggestions
- Include migration support for version changes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

### Code Style

1. **Formatting**: Use Black with default settings
   ```bash
   black colpali_engine/ tests/
   ```

2. **Linting**: Use flake8
   ```bash
   flake8 colpali_engine/ tests/ --max-line-length=100
   ```

3. **Type Hints**: Use type annotations for all public methods
   ```python
   def process_document(
       self,
       document_blob: bytes,
       schema_json: Dict[str, Any],
       options: Optional[ProcessingOptions] = None
   ) -> ExtractionResult:
       """Process document and return structured result."""
       ...
   ```

4. **Docstrings**: Use Google style
   ```python
   def extract_tables(self, image: Image) -> List[TableData]:
       """Extract tables from document image.

       Args:
           image: PIL Image of document page.

       Returns:
           List of TableData objects containing extracted table content.

       Raises:
           ExtractionError: If table extraction fails.
       """
       ...
   ```

### Pull Request Process

1. **Create feature branch** from `main`
2. **Implement changes** with tests
3. **Run full test suite**: `pytest tests/ -v`
4. **Format code**: `black . && flake8 .`
5. **Update documentation** if needed
6. **Create PR** with description and test results
7. **Address review comments**
8. **Merge** after approval

### Code Review Checklist

- [ ] Tests pass (100% required)
- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Documentation updated

---

## Debugging & Troubleshooting

### Common Issues

#### 1. Docker Build Fails

```bash
# Check Docker logs
docker-compose logs colpali-engine

# Rebuild without cache
docker-compose build --no-cache

# Check disk space
docker system df
```

#### 2. Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=.

# Verify installation
pip list | grep colpali

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 3. Qdrant Connection Issues

```bash
# Check Qdrant is running
curl http://localhost:6333/health

# Verify Docker networking
docker-compose exec colpali-engine curl http://qdrant:6333/health
```

#### 4. Memory Issues

```python
# Monitor memory usage
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")

# Force garbage collection
import gc
gc.collect()
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logging.getLogger("colpali_engine.vision").setLevel(logging.DEBUG)
```

### Using Debugger

```bash
# VS Code launch.json
{
  "configurations": [
    {
      "name": "Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v", "-s", "tests/unit/"],
      "env": {"PYTHONPATH": "${workspaceFolder}"}
    }
  ]
}

# Or with pdb
python -m pdb -c continue script.py
```

---

## Common Patterns

### Async Context Manager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_pipeline():
    """Create pipeline with automatic cleanup."""
    pipeline = VisionExtractionPipeline()
    try:
        await pipeline.initialize()
        yield pipeline
    finally:
        await pipeline.cleanup()

# Usage
async with managed_pipeline() as pipeline:
    result = await pipeline.process_document(doc, schema)
```

### Factory Pattern

```python
from colpali_engine.adapters import PDFAdapter, ImageAdapter, HTMLAdapter

def create_adapter(mime_type: str):
    """Create appropriate adapter for MIME type."""
    adapters = {
        "application/pdf": PDFAdapter,
        "image/jpeg": ImageAdapter,
        "image/png": ImageAdapter,
        "text/html": HTMLAdapter,
    }

    adapter_class = adapters.get(mime_type)
    if adapter_class is None:
        raise ValueError(f"Unsupported MIME type: {mime_type}")

    return adapter_class()
```

### Batch Processing

```python
import asyncio
from typing import List

async def process_batch(
    pipeline: VisionExtractionPipeline,
    documents: List[bytes],
    schema: dict,
    batch_size: int = 5
) -> List[ExtractionResult]:
    """Process documents in batches for memory efficiency."""
    results = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Process batch concurrently
        tasks = [
            pipeline.process_document(doc, schema)
            for doc in batch
        ]

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # Clean up between batches
        import gc
        gc.collect()

    return results
```

### Configuration Management

```python
from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Pipeline configuration from environment."""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    colpali_model: str = "vidore/colqwen2-v0.1"
    batch_size: int = 4
    memory_limit_gb: float = 8.0
    log_level: str = "INFO"

    class Config:
        env_prefix = "COLPALI_"


# Usage
config = PipelineConfig()
pipeline = VisionExtractionPipeline(config=config)
```

---

## Resources

### Documentation
- [Architecture Playbook](architecture-playbook.md) - System design and technology deep dive
- [Deployment Guide](deployment-operations-guide.md) - Deployment and operations procedures
- [JIRA Implementation Plan](jira-implementation-plan.md) - Feature tracking and stories

### External Resources
- [ColPali Paper](https://arxiv.org/abs/2407.01449) - Vision-based document retrieval
- [Qdrant Documentation](https://qdrant.tech/documentation/) - Vector database
- [BAML Documentation](https://docs.boundaryml.com/) - Type-safe LLM extraction
- [PyTorch Documentation](https://pytorch.org/docs/) - ML framework

### Getting Help
- GitHub Issues: Report bugs and feature requests
- Pull Requests: Submit code contributions
- Discussions: Ask questions and share ideas

---

*This guide is maintained by the ColPali-BAML team. For questions or feedback, please open a GitHub issue.*

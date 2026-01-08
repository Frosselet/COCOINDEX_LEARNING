# Test Suite Organization

This directory contains the comprehensive test suite for the ColPali-BAML Vision Processing Engine, organized by type and scope.

## Directory Structure

```
tests/
├── unit/                           # Unit tests for individual components
│   └── test_pdf_adapter.py         # PDF adapter functionality tests
├── integration/                    # Integration tests for multi-component workflows
│   └── test_multi_format_adapter.py # COLPALI-203: Multi-format adapter interface
├── adapters/                       # Adapter-specific test files
│   ├── test_pdf_adapter.py         # Comprehensive PDF adapter validation
│   └── test_image_processor.py     # Image processor validation
└── processors/                     # Processor component tests
    └── test_image_processor.py     # Image standardization processor tests

```

## Test Categories

### Unit Tests
- **Individual component testing**
- **Fast execution, isolated dependencies**
- **Focused on specific functionality**

### Integration Tests
- **Multi-component workflow testing**
- **End-to-end scenario validation**
- **Plugin architecture verification**

### Adapter Tests
- **Format-specific processing validation**
- **Configuration parameter testing**
- **Error handling verification**

### Processor Tests
- **Image standardization pipeline testing**
- **Quality metrics validation**
- **Batch processing verification**

## Running Tests

### Individual Test Files
```bash
# Run specific adapter test
cd tests/unit && python3 test_pdf_adapter.py

# Run integration tests
cd tests/integration && python3 test_multi_format_adapter.py

# Run image processor tests
cd tests/processors && python3 test_image_processor.py
```

### Full Test Suite
```bash
# From project root
python3 -m pytest tests/ -v
```

## Test Requirements

### Dependencies
- All base project dependencies (see requirements/base.txt)
- PIL/Pillow for image processing
- pytest (optional, for pytest runner)

### Test Data
- PDF files in `pdfs/` directory (15 sample documents)
- Generated test images (created dynamically by tests)
- HTML test content (embedded in test files)

## Test Coverage

### COLPALI-201: PDF Adapter
- ✅ Format validation across 15 sample PDFs
- ✅ Metadata extraction (pages, dimensions, title, author)
- ✅ Image conversion with configurable DPI/quality
- ✅ Memory optimization for large documents
- ✅ Batch processing capabilities

### COLPALI-202: Image Standardization Processor
- ✅ Consistent dimension standardization (1024x1024, 2048x2048)
- ✅ Color space normalization (RGB, grayscale handling)
- ✅ Quality optimization with file size limits
- ✅ Aspect ratio preservation with padding
- ✅ Batch processing with concurrent execution
- ✅ Quality metrics generation

### COLPALI-203: Multi-Format Adapter Interface
- ✅ Plugin registration system for new formats
- ✅ MIME type detection and routing (with python-magic integration)
- ✅ Consistent error handling across adapters
- ✅ Format-specific configuration support
- ✅ Extension examples and custom adapter creation

## Success Criteria

All tests must achieve **≥85% success rate** for the following:
- Format detection accuracy
- Processing pipeline completion
- Error handling consistency
- Configuration parameter application
- Multi-format workflow integration

## Continuous Integration

These tests are designed to be run in CI/CD pipelines with:
- Automatic dependency resolution
- Graceful fallbacks for missing system libraries
- Comprehensive error reporting
- Performance metrics collection

---

*Last Updated: 2026-01-08*
*Test Suite Version: 1.0.0*
*Coverage: COLPALI-201, COLPALI-202, COLPALI-203*
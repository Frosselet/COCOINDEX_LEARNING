"""
Shared fixtures for end-to-end testing.

Provides common fixtures for PDF integration tests, performance benchmarks,
memory validation, and accuracy measurement.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# PDF Test Dataset Configuration
# =============================================================================

@dataclass
class PDFTestCase:
    """Test case configuration for a PDF file."""
    filename: str
    filepath: Path
    file_size_bytes: int
    category: str  # 'layout', 'shipping', 'financial', 'synthetic'
    expected_pages: Optional[int] = None
    complexity: str = "medium"  # 'simple', 'medium', 'complex'
    known_challenges: List[str] = field(default_factory=list)
    ground_truth: Optional[Dict[str, Any]] = None


@pytest.fixture(scope="session")
def pdf_directory() -> Path:
    """Get the PDF test directory."""
    pdf_dir = project_root / "pdfs"
    if not pdf_dir.exists():
        pytest.skip(f"PDF directory not found: {pdf_dir}")
    return pdf_dir


@pytest.fixture(scope="session")
def all_pdf_files(pdf_directory: Path) -> List[Path]:
    """Get all PDF files in the test directory."""
    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in test directory")
    return sorted(pdf_files)


@pytest.fixture(scope="session")
def pdf_test_catalog(pdf_directory: Path) -> Dict[str, PDFTestCase]:
    """
    Comprehensive catalog of all 15 PDF test cases with metadata.

    Categories:
    - layout: Tests for document layout handling
    - shipping: Real shipping/logistics documents
    - financial: Financial statements and loading documents
    - synthetic: Synthetically created test documents
    """
    catalog = {}

    # Layout test documents (synthetic)
    layout_docs = {
        "extreme_multi_column.pdf": {
            "category": "layout",
            "complexity": "complex",
            "known_challenges": ["multiple columns", "narrow gutters", "text overflow"],
        },
        "mixed_content_layout.pdf": {
            "category": "layout",
            "complexity": "complex",
            "known_challenges": ["mixed content types", "tables with text", "images"],
        },
        "proper_multi_column.pdf": {
            "category": "layout",
            "complexity": "medium",
            "known_challenges": ["multi-column layout", "standard formatting"],
        },
        "overlapping_text.pdf": {
            "category": "layout",
            "complexity": "complex",
            "known_challenges": ["overlapping text regions", "z-order issues"],
        },
        "single_column_typography.pdf": {
            "category": "layout",
            "complexity": "simple",
            "known_challenges": ["typography variations", "font sizes"],
        },
        "tiny_fonts_spacing.pdf": {
            "category": "layout",
            "complexity": "complex",
            "known_challenges": ["small fonts", "tight spacing", "readability"],
        },
        "semantic_table.pdf": {
            "category": "layout",
            "complexity": "medium",
            "known_challenges": ["table structure", "cell boundaries", "headers"],
        },
    }

    # Real shipping documents
    shipping_docs = {
        "Shipping-Stem-2025-09-30.pdf": {
            "category": "shipping",
            "complexity": "complex",
            "known_challenges": ["shipping manifest", "multiple tables", "dates"],
        },
        "shipping-stem-2025-11-13.pdf": {
            "category": "shipping",
            "complexity": "complex",
            "known_challenges": ["shipping manifest", "vessel information"],
        },
        "shipping_stem-accc-30092025-1.pdf": {
            "category": "shipping",
            "complexity": "medium",
            "known_challenges": ["shipping data", "cargo information"],
        },
        "CBH Shipping Stem 26092025.pdf": {
            "category": "shipping",
            "complexity": "complex",
            "known_challenges": ["CBH format", "grain shipments", "multiple vessels"],
        },
    }

    # Financial/Loading documents
    financial_docs = {
        "Loading-Statement-for-Web-Portal-20250923.pdf": {
            "category": "financial",
            "complexity": "medium",
            "known_challenges": ["loading statement", "quantities", "dates"],
        },
        "Bunge_loadingstatement_2025-09-25.pdf": {
            "category": "financial",
            "complexity": "medium",
            "known_challenges": ["Bunge format", "loading data", "port information"],
        },
        "2857439.pdf": {
            "category": "financial",
            "complexity": "medium",
            "known_challenges": ["financial document", "reference numbers"],
        },
        "document (1).pdf": {
            "category": "financial",
            "complexity": "simple",
            "known_challenges": ["generic document", "standard format"],
        },
    }

    # Combine all document metadata
    all_docs = {**layout_docs, **shipping_docs, **financial_docs}

    # Build catalog with file information
    for filename, metadata in all_docs.items():
        filepath = pdf_directory / filename
        if filepath.exists():
            catalog[filename] = PDFTestCase(
                filename=filename,
                filepath=filepath,
                file_size_bytes=filepath.stat().st_size,
                category=metadata["category"],
                complexity=metadata["complexity"],
                known_challenges=metadata["known_challenges"],
            )

    return catalog


@pytest.fixture(scope="session")
def pdf_test_cases(pdf_test_catalog: Dict[str, PDFTestCase]) -> List[PDFTestCase]:
    """Get list of all PDF test cases."""
    return list(pdf_test_catalog.values())


# =============================================================================
# Category-based fixtures
# =============================================================================

@pytest.fixture(scope="session")
def layout_pdfs(pdf_test_catalog: Dict[str, PDFTestCase]) -> List[PDFTestCase]:
    """Get layout test PDFs."""
    return [tc for tc in pdf_test_catalog.values() if tc.category == "layout"]


@pytest.fixture(scope="session")
def shipping_pdfs(pdf_test_catalog: Dict[str, PDFTestCase]) -> List[PDFTestCase]:
    """Get shipping document PDFs."""
    return [tc for tc in pdf_test_catalog.values() if tc.category == "shipping"]


@pytest.fixture(scope="session")
def financial_pdfs(pdf_test_catalog: Dict[str, PDFTestCase]) -> List[PDFTestCase]:
    """Get financial document PDFs."""
    return [tc for tc in pdf_test_catalog.values() if tc.category == "financial"]


@pytest.fixture(scope="session")
def complex_pdfs(pdf_test_catalog: Dict[str, PDFTestCase]) -> List[PDFTestCase]:
    """Get complex PDFs for stress testing."""
    return [tc for tc in pdf_test_catalog.values() if tc.complexity == "complex"]


@pytest.fixture(scope="session")
def simple_pdfs(pdf_test_catalog: Dict[str, PDFTestCase]) -> List[PDFTestCase]:
    """Get simple PDFs for baseline testing."""
    return [tc for tc in pdf_test_catalog.values() if tc.complexity == "simple"]


# =============================================================================
# Performance and Memory Tracking
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    duration_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    throughput: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def performance_tracker():
    """Fixture for tracking performance metrics."""
    import time
    import psutil
    import os

    class PerformanceTracker:
        def __init__(self):
            self.metrics: List[PerformanceMetrics] = []
            self._start_time: Optional[float] = None
            self._start_memory: Optional[float] = None
            self._operation: Optional[str] = None

        def _get_memory_mb(self) -> float:
            """Get current process memory in MB."""
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)

        def start(self, operation: str) -> None:
            """Start tracking an operation."""
            self._operation = operation
            self._start_memory = self._get_memory_mb()
            self._start_time = time.perf_counter()

        def stop(self, **metadata) -> PerformanceMetrics:
            """Stop tracking and record metrics."""
            if self._start_time is None:
                raise RuntimeError("Tracker not started")

            end_time = time.perf_counter()
            end_memory = self._get_memory_mb()

            metrics = PerformanceMetrics(
                operation=self._operation,
                duration_seconds=end_time - self._start_time,
                memory_before_mb=self._start_memory,
                memory_after_mb=end_memory,
                memory_delta_mb=end_memory - self._start_memory,
                metadata=metadata
            )
            self.metrics.append(metrics)

            self._start_time = None
            self._start_memory = None
            self._operation = None

            return metrics

        def get_summary(self) -> Dict[str, Any]:
            """Get summary of all tracked metrics."""
            if not self.metrics:
                return {"total_operations": 0}

            durations = [m.duration_seconds for m in self.metrics]
            memory_deltas = [m.memory_delta_mb for m in self.metrics]

            return {
                "total_operations": len(self.metrics),
                "total_duration_seconds": sum(durations),
                "avg_duration_seconds": sum(durations) / len(durations),
                "min_duration_seconds": min(durations),
                "max_duration_seconds": max(durations),
                "total_memory_delta_mb": sum(memory_deltas),
                "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
                "max_memory_delta_mb": max(memory_deltas),
            }

    return PerformanceTracker()


# =============================================================================
# Ground Truth and Accuracy
# =============================================================================

@pytest.fixture(scope="session")
def ground_truth_directory() -> Path:
    """Get or create ground truth directory."""
    gt_dir = project_root / "tests" / "e2e" / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    return gt_dir


@pytest.fixture(scope="session")
def ground_truth_data(ground_truth_directory: Path) -> Dict[str, Dict[str, Any]]:
    """Load ground truth data for accuracy testing."""
    gt_data = {}

    for gt_file in ground_truth_directory.glob("*.json"):
        pdf_name = gt_file.stem
        with open(gt_file, 'r') as f:
            gt_data[pdf_name] = json.load(f)

    return gt_data


# =============================================================================
# Module Loading Utilities
# =============================================================================

@pytest.fixture(scope="session")
def load_module():
    """Utility to load modules directly from file path."""
    import importlib.util

    def _load_module(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return _load_module


# =============================================================================
# Test Result Storage
# =============================================================================

@pytest.fixture(scope="session")
def test_results_directory() -> Path:
    """Get or create test results directory."""
    results_dir = project_root / "tests" / "e2e" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@pytest.fixture
def save_test_result(test_results_directory: Path):
    """Fixture to save test results to file."""
    from datetime import datetime

    def _save_result(test_name: str, result: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        filepath = test_results_directory / filename

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return filepath

    return _save_result

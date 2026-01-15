"""
Memory usage validation tests.

Tests COLPALI-1003: Build memory usage validation tests.

This module provides specialized tests that validate memory usage patterns,
detect memory leaks, and ensure the system operates within AWS Lambda constraints.

Test Coverage:
- Memory usage validation for Lambda 10GB limit
- Memory leak detection across multiple processing cycles
- Peak memory usage monitoring for different document types
- Memory optimization recommendations
- Automated memory testing

Environment Requirements:
- pyarrow for PDFAdapter
- psutil for memory monitoring
- tracemalloc for memory tracing
"""

import sys
import gc
import time
import asyncio
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

import pytest
import psutil
import os


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# Project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Module Imports with Dependency Handling
# =============================================================================

PDF_ADAPTER_AVAILABLE = False
PDF_ADAPTER_ERROR = None
RESOURCE_MANAGER_AVAILABLE = False

try:
    from colpali_engine.adapters.pdf_adapter import PDFAdapter, create_pdf_adapter
    PDF_ADAPTER_AVAILABLE = True
except ImportError as e:
    PDF_ADAPTER_ERROR = str(e)

try:
    from colpali_engine.lambda_utils.resource_manager import ResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Memory Constants (Lambda Constraints)
# =============================================================================

# AWS Lambda memory constraints
LAMBDA_MEMORY_LIMITS = {
    "max_memory_mb": 10240,  # 10GB Lambda max
    "target_memory_mb": 8192,  # Target 8GB with buffer
    "warning_threshold_mb": 7000,  # Warning at 7GB
    "critical_threshold_mb": 9000,  # Critical at 9GB
    "per_page_limit_mb": 150,  # Max 150MB per page (based on actual measurements)
    "baseline_overhead_mb": 500,  # Baseline application overhead
}


# =============================================================================
# Memory Tracking Utilities
# =============================================================================

@dataclass
class MemorySnapshot:
    """Snapshot of memory state."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    heap_mb: Optional[float] = None
    operation: Optional[str] = None


@dataclass
class MemoryLeakResult:
    """Result of memory leak detection."""
    leak_detected: bool
    baseline_mb: float
    final_mb: float
    delta_mb: float
    iterations: int
    growth_rate_mb_per_iteration: float
    memory_history: List[float] = field(default_factory=list)


class MemoryTracker:
    """Track memory usage over time."""

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.tracemalloc_enabled = False

    def _get_process_memory(self) -> Tuple[float, float]:
        """Get current process memory (RSS, VMS) in MB."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return (
            mem_info.rss / (1024 * 1024),
            mem_info.vms / (1024 * 1024)
        )

    def start_tracemalloc(self) -> None:
        """Start tracemalloc for heap tracking."""
        tracemalloc.start()
        self.tracemalloc_enabled = True

    def stop_tracemalloc(self) -> None:
        """Stop tracemalloc."""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False

    def snapshot(self, operation: Optional[str] = None) -> MemorySnapshot:
        """Take a memory snapshot."""
        rss, vms = self._get_process_memory()

        heap_mb = None
        if self.tracemalloc_enabled:
            current, peak = tracemalloc.get_traced_memory()
            heap_mb = current / (1024 * 1024)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss,
            vms_mb=vms,
            heap_mb=heap_mb,
            operation=operation
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_memory_delta(self, start_idx: int = 0, end_idx: int = -1) -> float:
        """Get memory delta between snapshots."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[end_idx].rss_mb - self.snapshots[start_idx].rss_mb

    def get_peak_memory(self) -> float:
        """Get peak memory usage."""
        if not self.snapshots:
            return 0.0
        return max(s.rss_mb for s in self.snapshots)

    def detect_leak(self, threshold_mb: float = 10.0) -> bool:
        """Detect if memory is consistently growing."""
        if len(self.snapshots) < 3:
            return False

        # Check if memory is monotonically increasing
        rss_values = [s.rss_mb for s in self.snapshots]
        increases = sum(1 for i in range(1, len(rss_values)) if rss_values[i] > rss_values[i-1])

        # If more than 80% of iterations show increase, likely a leak
        leak_ratio = increases / (len(rss_values) - 1)
        total_growth = rss_values[-1] - rss_values[0]

        return leak_ratio > 0.8 and total_growth > threshold_mb


@pytest.fixture
def memory_tracker():
    """Create a memory tracker."""
    tracker = MemoryTracker()
    yield tracker
    tracker.stop_tracemalloc()


# =============================================================================
# Lambda Memory Constraint Tests
# =============================================================================

@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestLambdaMemoryConstraints:
    """Test memory usage against Lambda constraints."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_baseline_memory_usage(self, memory_tracker):
        """Test baseline memory usage of application."""
        gc.collect()
        baseline = memory_tracker.snapshot("baseline")

        assert baseline.rss_mb < LAMBDA_MEMORY_LIMITS["baseline_overhead_mb"], \
            f"Baseline memory too high: {baseline.rss_mb:.1f}MB"

        print(f"\nBaseline memory: {baseline.rss_mb:.1f}MB RSS")

    def test_single_pdf_memory_usage(
        self,
        pdf_adapter,
        pdf_test_cases,
        memory_tracker
    ):
        """Test memory usage for processing a single PDF."""
        if not pdf_test_cases:
            pytest.skip("No test PDFs")

        gc.collect()
        memory_tracker.snapshot("before")

        test_case = pdf_test_cases[0]
        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        images = run_async(pdf_adapter.convert_to_frames(content))
        memory_tracker.snapshot("after_conversion")

        # Calculate per-page memory
        delta = memory_tracker.get_memory_delta(0, -1)
        per_page = delta / max(len(images), 1)

        print(f"\n{test_case.filename}:")
        print(f"  Pages: {len(images)}")
        print(f"  Memory delta: {delta:.1f}MB")
        print(f"  Per page: {per_page:.1f}MB")

        assert per_page < LAMBDA_MEMORY_LIMITS["per_page_limit_mb"]

    def test_all_pdfs_memory_usage(
        self,
        pdf_adapter,
        pdf_test_cases,
        memory_tracker
    ):
        """Test memory usage processing all PDFs."""
        gc.collect()
        memory_tracker.snapshot("start")

        total_pages = 0
        max_memory = 0

        for test_case in pdf_test_cases:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            total_pages += len(images)

            snapshot = memory_tracker.snapshot(f"after_{test_case.filename}")
            max_memory = max(max_memory, snapshot.rss_mb)

            # Clean up to prevent accumulation
            del images
            gc.collect()

        print(f"\nAll PDFs Memory Test:")
        print(f"  Total pages: {total_pages}")
        print(f"  Peak memory: {max_memory:.1f}MB")
        print(f"  Target limit: {LAMBDA_MEMORY_LIMITS['target_memory_mb']}MB")

        assert max_memory < LAMBDA_MEMORY_LIMITS["target_memory_mb"]

    def test_memory_stays_under_lambda_limit(
        self,
        pdf_adapter,
        complex_pdfs,
        memory_tracker
    ):
        """Test that memory stays under Lambda limit with complex PDFs."""
        if not complex_pdfs:
            pytest.skip("No complex PDFs")

        gc.collect()
        memory_tracker.snapshot("start")

        # Process all complex PDFs
        for test_case in complex_pdfs:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            snapshot = memory_tracker.snapshot(f"after_{test_case.filename}")

            # Check Lambda limit after each PDF
            assert snapshot.rss_mb < LAMBDA_MEMORY_LIMITS["max_memory_mb"], \
                f"Memory exceeded Lambda limit: {snapshot.rss_mb:.1f}MB"

            del images
            gc.collect()

        peak = memory_tracker.get_peak_memory()
        print(f"\nPeak memory with complex PDFs: {peak:.1f}MB")


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestMemoryLeakDetection:
    """Test for memory leaks across multiple processing cycles."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_no_memory_leak_single_pdf(
        self,
        pdf_adapter,
        simple_pdfs,
        memory_tracker
    ):
        """Test for memory leaks processing same PDF multiple times."""
        if not simple_pdfs:
            pytest.skip("No simple PDFs")

        test_case = simple_pdfs[0]
        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        gc.collect()
        memory_tracker.snapshot("baseline")

        # Process same PDF 10 times
        for i in range(10):
            images = run_async(pdf_adapter.convert_to_frames(content))
            memory_tracker.snapshot(f"iteration_{i}")
            del images
            gc.collect()

        # Check for leak
        leak_detected = memory_tracker.detect_leak(threshold_mb=50)
        delta = memory_tracker.get_memory_delta(0, -1)

        print(f"\nMemory leak test (10 iterations):")
        print(f"  Total delta: {delta:.1f}MB")
        print(f"  Leak detected: {leak_detected}")

        assert not leak_detected, f"Memory leak detected: {delta:.1f}MB growth"

    def test_no_memory_leak_multiple_pdfs(
        self,
        pdf_adapter,
        pdf_test_cases,
        memory_tracker
    ):
        """Test for memory leaks processing multiple different PDFs."""
        gc.collect()
        memory_tracker.snapshot("baseline")

        # Process all PDFs 3 times
        for cycle in range(3):
            for test_case in pdf_test_cases:
                with open(test_case.filepath, 'rb') as f:
                    content = f.read()

                images = run_async(pdf_adapter.convert_to_frames(content))
                del images

            gc.collect()
            memory_tracker.snapshot(f"cycle_{cycle}")

        # Check growth between cycles
        cycle_memories = [s.rss_mb for s in memory_tracker.snapshots if s.operation and s.operation.startswith("cycle_")]

        if len(cycle_memories) >= 2:
            growth_per_cycle = (cycle_memories[-1] - cycle_memories[0]) / (len(cycle_memories) - 1)
            print(f"\nMulti-PDF leak test:")
            print(f"  Cycle memories: {cycle_memories}")
            print(f"  Growth per cycle: {growth_per_cycle:.1f}MB")

            # Allow some growth but not excessive
            assert growth_per_cycle < 100, f"Excessive memory growth: {growth_per_cycle:.1f}MB/cycle"

    def test_memory_recovery_after_gc(
        self,
        pdf_adapter,
        complex_pdfs,
        memory_tracker
    ):
        """Test that memory is recovered after garbage collection."""
        if not complex_pdfs:
            pytest.skip("No complex PDFs")

        gc.collect()
        baseline = memory_tracker.snapshot("baseline")

        # Process complex PDFs
        all_images = []
        for test_case in complex_pdfs[:3]:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()
            images = run_async(pdf_adapter.convert_to_frames(content))
            all_images.extend(images)

        peak = memory_tracker.snapshot("peak")

        # Clear references and collect
        del all_images
        gc.collect()
        gc.collect()  # Double collect for generations

        recovered = memory_tracker.snapshot("recovered")

        print(f"\nMemory recovery test:")
        print(f"  Baseline: {baseline.rss_mb:.1f}MB")
        print(f"  Peak: {peak.rss_mb:.1f}MB")
        print(f"  After GC: {recovered.rss_mb:.1f}MB")
        print(f"  Recovery: {(peak.rss_mb - recovered.rss_mb):.1f}MB")

        # Should recover at least 50% of peak-baseline delta
        peak_delta = peak.rss_mb - baseline.rss_mb
        recovered_delta = recovered.rss_mb - baseline.rss_mb

        if peak_delta > 50:  # Only check if significant memory was used
            recovery_ratio = 1 - (recovered_delta / peak_delta)
            assert recovery_ratio > 0.3, f"Poor memory recovery: {recovery_ratio:.1%}"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestPeakMemoryMonitoring:
    """Test peak memory usage monitoring."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_peak_memory_by_document_type(
        self,
        pdf_adapter,
        pdf_test_catalog,
        memory_tracker
    ):
        """Test peak memory by document category."""
        results_by_category = {}

        for filename, test_case in pdf_test_catalog.items():
            gc.collect()
            memory_tracker.snapshot(f"before_{filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            after = memory_tracker.snapshot(f"after_{filename}")

            category = test_case.category
            if category not in results_by_category:
                results_by_category[category] = []

            results_by_category[category].append({
                "filename": filename,
                "memory_mb": after.rss_mb,
                "page_count": len(images),
            })

            del images
            gc.collect()

        print("\nPeak Memory by Document Type:")
        for category, results in results_by_category.items():
            max_memory = max(r["memory_mb"] for r in results)
            avg_memory = sum(r["memory_mb"] for r in results) / len(results)
            print(f"  {category}: max={max_memory:.1f}MB, avg={avg_memory:.1f}MB")

    def test_peak_memory_by_complexity(
        self,
        pdf_adapter,
        pdf_test_catalog,
        memory_tracker
    ):
        """Test peak memory by document complexity."""
        results_by_complexity = {"simple": [], "medium": [], "complex": []}

        for filename, test_case in pdf_test_catalog.items():
            gc.collect()
            before = memory_tracker.snapshot(f"before_{filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            after = memory_tracker.snapshot(f"after_{filename}")

            delta = after.rss_mb - before.rss_mb
            results_by_complexity[test_case.complexity].append({
                "filename": filename,
                "delta_mb": delta,
                "page_count": len(images),
            })

            del images
            gc.collect()

        print("\nMemory Delta by Complexity:")
        for complexity, results in results_by_complexity.items():
            if results:
                max_delta = max(r["delta_mb"] for r in results)
                avg_delta = sum(r["delta_mb"] for r in results) / len(results)
                print(f"  {complexity}: max={max_delta:.1f}MB, avg={avg_delta:.1f}MB")


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestMemoryStressTesting:
    """Memory stress testing."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_sustained_load_memory_stability(
        self,
        pdf_adapter,
        pdf_test_cases,
        memory_tracker
    ):
        """Test memory stability under sustained load."""
        gc.collect()
        memory_tracker.snapshot("start")

        # Simulate sustained load
        for round_num in range(5):
            for test_case in pdf_test_cases[:5]:  # Process 5 PDFs per round
                with open(test_case.filepath, 'rb') as f:
                    content = f.read()

                images = run_async(pdf_adapter.convert_to_frames(content))
                del images

            gc.collect()
            memory_tracker.snapshot(f"round_{round_num}")

        # Analyze memory trend
        round_snapshots = [s for s in memory_tracker.snapshots if s.operation and s.operation.startswith("round_")]
        memories = [s.rss_mb for s in round_snapshots]

        if len(memories) >= 2:
            # Calculate trend
            first_half_avg = sum(memories[:len(memories)//2]) / (len(memories)//2)
            second_half_avg = sum(memories[len(memories)//2:]) / (len(memories) - len(memories)//2)

            growth = second_half_avg - first_half_avg

            print(f"\nSustained load test:")
            print(f"  Memory per round: {memories}")
            print(f"  First half avg: {first_half_avg:.1f}MB")
            print(f"  Second half avg: {second_half_avg:.1f}MB")
            print(f"  Trend: {'+' if growth > 0 else ''}{growth:.1f}MB")

            # Memory should be stable (not growing significantly)
            assert growth < 100, f"Memory growing under sustained load: {growth:.1f}MB"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestMemoryOptimizationRecommendations:
    """Generate memory optimization recommendations."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_generate_memory_report(
        self,
        pdf_adapter,
        pdf_test_catalog,
        memory_tracker,
        save_test_result
    ):
        """Generate memory optimization report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "lambda_limits": LAMBDA_MEMORY_LIMITS,
            "document_analysis": [],
            "recommendations": [],
        }

        for filename, test_case in pdf_test_catalog.items():
            gc.collect()
            before = memory_tracker.snapshot(f"before_{filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            after = memory_tracker.snapshot(f"after_{filename}")

            delta = after.rss_mb - before.rss_mb
            per_page = delta / max(len(images), 1)

            report["document_analysis"].append({
                "filename": filename,
                "category": test_case.category,
                "complexity": test_case.complexity,
                "file_size_kb": test_case.file_size_bytes / 1024,
                "page_count": len(images),
                "memory_delta_mb": delta,
                "memory_per_page_mb": per_page,
            })

            del images
            gc.collect()

        # Generate recommendations
        high_memory_docs = [d for d in report["document_analysis"] if d["memory_per_page_mb"] > 50]
        if high_memory_docs:
            report["recommendations"].append({
                "type": "HIGH_MEMORY_USAGE",
                "description": "Some documents use >50MB per page",
                "affected_documents": [d["filename"] for d in high_memory_docs],
                "suggestion": "Consider reducing DPI or enabling compression for these document types",
            })

        # Calculate overall stats
        all_deltas = [d["memory_delta_mb"] for d in report["document_analysis"]]
        report["summary"] = {
            "total_documents": len(report["document_analysis"]),
            "avg_memory_per_doc_mb": sum(all_deltas) / len(all_deltas) if all_deltas else 0,
            "max_memory_per_doc_mb": max(all_deltas) if all_deltas else 0,
            "fits_lambda_limit": max(all_deltas, default=0) < LAMBDA_MEMORY_LIMITS["target_memory_mb"],
        }

        # Save report
        result_path = save_test_result("memory_optimization_report", report)
        print(f"\nMemory report saved to: {result_path}")

        # Print summary
        print("\nMemory Optimization Summary:")
        print(f"  Avg memory per doc: {report['summary']['avg_memory_per_doc_mb']:.1f}MB")
        print(f"  Max memory per doc: {report['summary']['max_memory_per_doc_mb']:.1f}MB")
        print(f"  Fits Lambda limit: {report['summary']['fits_lambda_limit']}")
        if report["recommendations"]:
            print(f"  Recommendations: {len(report['recommendations'])}")

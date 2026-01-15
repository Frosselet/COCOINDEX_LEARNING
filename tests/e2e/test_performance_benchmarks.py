"""
Performance benchmarking suite.

Tests COLPALI-1002: Implement performance benchmarking suite.

This module provides comprehensive performance testing for:
- Processing time benchmarks for different document sizes
- Memory usage profiling and optimization recommendations
- Throughput testing for batch processing scenarios
- Performance regression detection
- Benchmarking reports and visualizations

Environment Requirements:
- pyarrow for PDFAdapter
- PIL/Pillow for image handling
- psutil for memory profiling
"""

import sys
import time
import asyncio
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
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
IMAGE_PROCESSOR_AVAILABLE = False
PDF_ADAPTER_ERROR = None
IMAGE_PROCESSOR_ERROR = None

try:
    from colpali_engine.adapters.pdf_adapter import PDFAdapter, create_pdf_adapter
    PDF_ADAPTER_AVAILABLE = True
except ImportError as e:
    PDF_ADAPTER_ERROR = str(e)

try:
    from colpali_engine.adapters.image_processor import (
        ImageProcessor,
        create_image_processor,
        ProcessingConfig
    )
    IMAGE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    IMAGE_PROCESSOR_ERROR = str(e)
    ImageProcessor = None
    ProcessingConfig = None


# =============================================================================
# Performance Data Structures
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    operation: str
    iterations: int
    min_time: float
    max_time: float
    mean_time: float
    median_time: float
    std_dev: float
    throughput: Optional[float] = None
    memory_peak_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "iterations": self.iterations,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "mean_time": self.mean_time,
            "median_time": self.median_time,
            "std_dev": self.std_dev,
            "throughput": self.throughput,
            "memory_peak_mb": self.memory_peak_mb,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    operation: str
    baseline_mean: float
    baseline_std: float
    threshold_multiplier: float = 1.5  # Fail if > 1.5x baseline

    def is_regression(self, current_mean: float) -> bool:
        """Check if current performance is a regression."""
        threshold = self.baseline_mean * self.threshold_multiplier
        return current_mean > threshold


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def benchmark_results_dir() -> Path:
    """Directory for storing benchmark results."""
    results_dir = project_root / "tests" / "e2e" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@pytest.fixture
def benchmark_runner():
    """Create a benchmark runner."""

    class BenchmarkRunner:
        def __init__(self, warmup_iterations: int = 1, benchmark_iterations: int = 3):
            self.warmup_iterations = warmup_iterations
            self.benchmark_iterations = benchmark_iterations
            self.results: List[BenchmarkResult] = []

        def run(
            self,
            operation_name: str,
            func,
            *args,
            iterations: Optional[int] = None,
            **kwargs
        ) -> BenchmarkResult:
            """Run benchmark for a function."""
            iterations = iterations or self.benchmark_iterations

            # Warmup
            for _ in range(self.warmup_iterations):
                func(*args, **kwargs)

            # Benchmark
            times = []
            memory_peaks = []

            for _ in range(iterations):
                # Get memory before
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024 * 1024)

                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()

                # Get memory after
                mem_after = process.memory_info().rss / (1024 * 1024)

                times.append(end - start)
                memory_peaks.append(mem_after)

            # Calculate statistics
            benchmark_result = BenchmarkResult(
                operation=operation_name,
                iterations=iterations,
                min_time=min(times),
                max_time=max(times),
                mean_time=statistics.mean(times),
                median_time=statistics.median(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
                memory_peak_mb=max(memory_peaks),
            )

            self.results.append(benchmark_result)
            return benchmark_result

        def get_summary(self) -> Dict[str, Any]:
            """Get summary of all benchmark results."""
            return {
                "total_benchmarks": len(self.results),
                "results": [r.to_dict() for r in self.results],
            }

    return BenchmarkRunner()


# =============================================================================
# SLA Definitions
# =============================================================================

# Performance SLA requirements
PERFORMANCE_SLA = {
    "pdf_conversion_per_page_seconds": 5.0,  # Max 5s per page
    "image_processing_seconds": 1.0,  # Max 1s per image
    "batch_throughput_pages_per_minute": 10,  # Min 10 pages/minute
    "memory_per_page_mb": 100,  # Max 100MB per page
    "total_memory_limit_mb": 8000,  # Max 8GB total (Lambda limit with buffer)
}


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestPDFConversionBenchmarks:
    """Benchmark PDF conversion performance."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_benchmark_simple_pdf_conversion(
        self,
        pdf_adapter,
        simple_pdfs,
        benchmark_runner
    ):
        """Benchmark simple PDF conversion."""
        if not simple_pdfs:
            pytest.skip("No simple PDFs")

        test_case = simple_pdfs[0]

        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        result = benchmark_runner.run(
            f"convert_simple_{test_case.filename}",
            pdf_adapter.convert_to_frames,
            content
        )

        # Check SLA
        assert result.mean_time < PERFORMANCE_SLA["pdf_conversion_per_page_seconds"] * 5
        print(f"\nSimple PDF: {result.mean_time:.3f}s mean, {result.memory_peak_mb:.1f}MB peak")

    def test_benchmark_complex_pdf_conversion(
        self,
        pdf_adapter,
        complex_pdfs,
        benchmark_runner
    ):
        """Benchmark complex PDF conversion."""
        if not complex_pdfs:
            pytest.skip("No complex PDFs")

        for test_case in complex_pdfs[:3]:  # Test first 3 complex PDFs
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            result = benchmark_runner.run(
                f"convert_complex_{test_case.filename}",
                pdf_adapter.convert_to_frames,
                content
            )

            print(f"\n{test_case.filename}: {result.mean_time:.3f}s mean")

    def test_benchmark_all_pdfs_by_size(
        self,
        pdf_adapter,
        pdf_test_cases,
        benchmark_runner,
        benchmark_results_dir
    ):
        """Benchmark all PDFs grouped by file size."""
        # Sort by file size
        sorted_cases = sorted(pdf_test_cases, key=lambda x: x.file_size_bytes)

        results_by_size = {
            "small": [],  # < 10KB
            "medium": [],  # 10KB - 100KB
            "large": [],  # > 100KB
        }

        for test_case in sorted_cases:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            result = benchmark_runner.run(
                f"convert_{test_case.filename}",
                pdf_adapter.convert_to_frames,
                content,
                iterations=2  # Fewer iterations for full suite
            )

            if test_case.file_size_bytes < 10000:
                results_by_size["small"].append(result)
            elif test_case.file_size_bytes < 100000:
                results_by_size["medium"].append(result)
            else:
                results_by_size["large"].append(result)

        # Report by size category
        print("\nBenchmark by File Size:")
        for size_cat, results in results_by_size.items():
            if results:
                mean_times = [r.mean_time for r in results]
                print(f"  {size_cat}: {len(results)} PDFs, avg {statistics.mean(mean_times):.3f}s")


@pytest.mark.skipif(
    not (PDF_ADAPTER_AVAILABLE and IMAGE_PROCESSOR_AVAILABLE),
    reason=f"Adapters not available: PDF={PDF_ADAPTER_ERROR}, ImageProcessor={IMAGE_PROCESSOR_ERROR}"
)
class TestImageProcessingBenchmarks:
    """Benchmark image processing performance."""

    @pytest.fixture
    def image_processor(self):
        """Create image processor."""
        return create_image_processor()

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_benchmark_image_standardization(
        self,
        pdf_adapter,
        image_processor,
        simple_pdfs,
        benchmark_runner
    ):
        """Benchmark image standardization."""
        if not simple_pdfs:
            pytest.skip("No simple PDFs")

        # Get sample images
        test_case = simple_pdfs[0]
        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        images = run_async(pdf_adapter.convert_to_frames(content))
        if not images:
            pytest.skip("No images from PDF")

        config = ProcessingConfig(
            target_width=1024,
            target_height=1024,
            maintain_aspect_ratio=True
        )
        processor = ImageProcessor(config)

        def process_single_image():
            return run_async(processor.process_image(images[0]))

        result = benchmark_runner.run(
            "image_standardization",
            process_single_image,
            iterations=5
        )

        assert result.mean_time < PERFORMANCE_SLA["image_processing_seconds"]
        print(f"\nImage standardization: {result.mean_time:.3f}s mean")

    def test_benchmark_batch_image_processing(
        self,
        pdf_adapter,
        image_processor,
        pdf_test_cases,
        benchmark_runner
    ):
        """Benchmark batch image processing."""
        # Collect images from multiple PDFs
        all_images = []

        for test_case in pdf_test_cases[:5]:  # First 5 PDFs
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            all_images.extend(images[:2])  # Max 2 images per PDF

        if not all_images:
            pytest.skip("No images collected")

        config = ProcessingConfig(
            target_width=1024,
            target_height=1024,
        )
        processor = ImageProcessor(config)

        def process_batch():
            results = []
            for img in all_images:
                processed, _ = run_async(processor.process_image(img))
                results.append(processed)
            return results

        result = benchmark_runner.run(
            f"batch_process_{len(all_images)}_images",
            process_batch,
            iterations=2
        )

        throughput = len(all_images) / result.mean_time
        result.throughput = throughput

        print(f"\nBatch processing: {len(all_images)} images in {result.mean_time:.3f}s")
        print(f"Throughput: {throughput:.2f} images/second")


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestThroughputBenchmarks:
    """Test throughput under various conditions."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_throughput_pages_per_minute(
        self,
        pdf_adapter,
        pdf_test_cases,
        benchmark_runner
    ):
        """Test page processing throughput."""
        total_pages = 0
        start_time = time.perf_counter()

        for test_case in pdf_test_cases:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            total_pages += len(images)

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        pages_per_minute = (total_pages / elapsed_seconds) * 60

        print(f"\nThroughput: {total_pages} pages in {elapsed_seconds:.2f}s")
        print(f"Rate: {pages_per_minute:.1f} pages/minute")

        assert pages_per_minute >= PERFORMANCE_SLA["batch_throughput_pages_per_minute"]

    def test_sustained_throughput(
        self,
        pdf_adapter,
        pdf_test_cases,
        performance_tracker
    ):
        """Test sustained throughput over multiple iterations."""
        iteration_times = []

        # Run 3 iterations over all PDFs
        for iteration in range(3):
            performance_tracker.start(f"iteration_{iteration}")

            for test_case in pdf_test_cases:
                with open(test_case.filepath, 'rb') as f:
                    content = f.read()
                run_async(pdf_adapter.convert_to_frames(content))

            metrics = performance_tracker.stop()
            iteration_times.append(metrics.duration_seconds)

        # Check for performance degradation
        print(f"\nSustained throughput iteration times: {iteration_times}")

        # Last iteration shouldn't be significantly slower than first
        if len(iteration_times) >= 2:
            degradation = iteration_times[-1] / iteration_times[0]
            assert degradation < 1.5, f"Performance degradation detected: {degradation:.2f}x"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestMemoryProfileBenchmarks:
    """Memory profiling benchmarks."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_memory_usage_per_pdf(
        self,
        pdf_adapter,
        pdf_test_cases,
        performance_tracker
    ):
        """Track memory usage for each PDF."""
        memory_usage = []

        for test_case in pdf_test_cases:
            performance_tracker.start(f"memory_{test_case.filename}")

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))

            metrics = performance_tracker.stop(
                page_count=len(images),
                file_size_bytes=test_case.file_size_bytes
            )

            memory_per_page = metrics.memory_delta_mb / max(len(images), 1)
            memory_usage.append({
                "filename": test_case.filename,
                "memory_delta_mb": metrics.memory_delta_mb,
                "page_count": len(images),
                "memory_per_page_mb": memory_per_page,
            })

        # Report
        print("\nMemory Usage by PDF:")
        for usage in sorted(memory_usage, key=lambda x: x["memory_delta_mb"], reverse=True)[:5]:
            print(f"  {usage['filename']}: {usage['memory_delta_mb']:.1f}MB "
                  f"({usage['memory_per_page_mb']:.1f}MB/page)")

        # Check SLA
        max_memory_per_page = max(u["memory_per_page_mb"] for u in memory_usage)
        assert max_memory_per_page < PERFORMANCE_SLA["memory_per_page_mb"]

    def test_peak_memory_usage(
        self,
        pdf_adapter,
        complex_pdfs,
        performance_tracker
    ):
        """Test peak memory usage with complex documents."""
        if not complex_pdfs:
            pytest.skip("No complex PDFs")

        import gc

        # Force garbage collection before test
        gc.collect()

        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / (1024 * 1024)

        peak_memory = baseline_memory

        for test_case in complex_pdfs:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))

            current_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)

            # Clean up
            del images
            gc.collect()

        memory_used = peak_memory - baseline_memory
        print(f"\nPeak memory: {peak_memory:.1f}MB (delta: {memory_used:.1f}MB)")

        assert peak_memory < PERFORMANCE_SLA["total_memory_limit_mb"]


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestRegressionDetection:
    """Performance regression detection tests."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    @pytest.fixture
    def baseline_file(self, benchmark_results_dir) -> Path:
        """Path to baseline file."""
        return benchmark_results_dir / "performance_baseline.json"

    def test_save_baseline(
        self,
        pdf_adapter,
        pdf_test_cases,
        benchmark_runner,
        baseline_file
    ):
        """Save performance baseline for future regression detection."""
        # Skip if baseline already exists (don't overwrite)
        if baseline_file.exists():
            pytest.skip("Baseline already exists")

        baselines = {}

        for test_case in pdf_test_cases:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            result = benchmark_runner.run(
                f"baseline_{test_case.filename}",
                pdf_adapter.convert_to_frames,
                content,
                iterations=3
            )

            baselines[test_case.filename] = {
                "mean_time": result.mean_time,
                "std_dev": result.std_dev,
                "memory_peak_mb": result.memory_peak_mb,
            }

        # Save baseline
        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "baselines": baselines,
        }

        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"\nBaseline saved to: {baseline_file}")

    def test_check_for_regressions(
        self,
        pdf_adapter,
        pdf_test_cases,
        benchmark_runner,
        baseline_file
    ):
        """Check current performance against baseline."""
        if not baseline_file.exists():
            pytest.skip("No baseline file - run test_save_baseline first")

        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)

        baselines = baseline_data["baselines"]
        regressions = []

        for test_case in pdf_test_cases:
            if test_case.filename not in baselines:
                continue

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            result = benchmark_runner.run(
                f"regression_{test_case.filename}",
                pdf_adapter.convert_to_frames,
                content,
                iterations=2
            )

            baseline = baselines[test_case.filename]
            threshold = baseline["mean_time"] * 1.5  # 50% regression threshold

            if result.mean_time > threshold:
                regressions.append({
                    "filename": test_case.filename,
                    "baseline_mean": baseline["mean_time"],
                    "current_mean": result.mean_time,
                    "regression_factor": result.mean_time / baseline["mean_time"],
                })

        if regressions:
            print("\nPerformance Regressions Detected:")
            for reg in regressions:
                print(f"  {reg['filename']}: {reg['regression_factor']:.2f}x slower")

        assert len(regressions) == 0, f"Found {len(regressions)} performance regressions"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestBenchmarkReporting:
    """Generate benchmark reports."""

    def test_generate_benchmark_report(
        self,
        benchmark_runner,
        benchmark_results_dir,
        pdf_test_catalog,
        save_test_result
    ):
        """Generate comprehensive benchmark report."""
        # Load PDF adapter
        pdf_adapter = create_pdf_adapter()

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "sla_requirements": PERFORMANCE_SLA,
            "test_summary": {
                "total_pdfs": len(pdf_test_catalog),
                "categories": {},
            },
            "benchmark_results": [],
        }

        # Count by category
        for tc in pdf_test_catalog.values():
            cat = tc.category
            if cat not in report_data["test_summary"]["categories"]:
                report_data["test_summary"]["categories"][cat] = 0
            report_data["test_summary"]["categories"][cat] += 1

        # Run benchmarks
        for filename, test_case in pdf_test_catalog.items():
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            result = benchmark_runner.run(
                f"report_{filename}",
                pdf_adapter.convert_to_frames,
                content,
                iterations=2
            )

            report_data["benchmark_results"].append({
                "filename": filename,
                "category": test_case.category,
                "complexity": test_case.complexity,
                "file_size_bytes": test_case.file_size_bytes,
                "benchmark": result.to_dict(),
            })

        # Calculate summary statistics
        all_times = [r["benchmark"]["mean_time"] for r in report_data["benchmark_results"]]
        report_data["summary_statistics"] = {
            "total_benchmarks": len(all_times),
            "mean_time": statistics.mean(all_times),
            "median_time": statistics.median(all_times),
            "min_time": min(all_times),
            "max_time": max(all_times),
            "std_dev": statistics.stdev(all_times) if len(all_times) > 1 else 0,
        }

        # Save report
        result_path = save_test_result("benchmark_report", report_data)
        print(f"\nBenchmark report saved to: {result_path}")

        # Print summary
        print("\nBenchmark Summary:")
        print(f"  Total PDFs: {len(pdf_test_catalog)}")
        print(f"  Mean time: {report_data['summary_statistics']['mean_time']:.3f}s")
        print(f"  Median time: {report_data['summary_statistics']['median_time']:.3f}s")

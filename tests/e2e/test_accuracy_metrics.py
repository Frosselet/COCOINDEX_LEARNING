"""
Accuracy measurement framework.

Tests COLPALI-1004: Add accuracy measurement framework.

This module provides accuracy measurement system that compares extraction
results against ground truth data, providing quantitative quality metrics
for continuous improvement.

Test Coverage:
- Ground truth dataset creation for test PDFs
- Accuracy scoring algorithms (precision, recall, F1)
- Quality regression detection
- Accuracy reporting and visualization
- Continuous accuracy monitoring framework

Environment Requirements:
- pyarrow for PDFAdapter
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pytest


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

try:
    from colpali_engine.adapters.pdf_adapter import PDFAdapter, create_pdf_adapter
    PDF_ADAPTER_AVAILABLE = True
except ImportError as e:
    PDF_ADAPTER_ERROR = str(e)


# =============================================================================
# Accuracy Metrics Data Structures
# =============================================================================

class MetricType(Enum):
    """Types of accuracy metrics."""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    PAGE_COUNT_ACCURACY = "page_count_accuracy"
    CONVERSION_SUCCESS = "conversion_success"


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    def calculate_precision(self) -> float:
        """Calculate precision from confusion matrix."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        return self.precision

    def calculate_recall(self) -> float:
        """Calculate recall from confusion matrix."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        return self.recall

    def calculate_f1(self) -> float:
        """Calculate F1 score from precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return self.f1_score

    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        self.accuracy = (self.true_positives + self.true_negatives) / total
        return self.accuracy

    def calculate_all(self) -> "AccuracyMetrics":
        """Calculate all metrics."""
        self.calculate_precision()
        self.calculate_recall()
        self.calculate_f1()
        self.calculate_accuracy()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
        }


@dataclass
class DocumentAccuracyResult:
    """Accuracy result for a single document."""
    filename: str
    category: str
    conversion_success: bool
    expected_pages: Optional[int]
    actual_pages: int
    page_count_match: bool
    image_quality_scores: List[float] = field(default_factory=list)
    metadata_accuracy: float = 0.0
    overall_score: float = 0.0
    errors: List[str] = field(default_factory=list)

    def calculate_overall_score(self) -> float:
        """Calculate overall accuracy score for document."""
        scores = []

        # Conversion success (weight: 40%)
        if self.conversion_success:
            scores.append(1.0 * 0.4)
        else:
            self.overall_score = 0.0
            return 0.0

        # Page count accuracy (weight: 30%)
        if self.expected_pages and self.expected_pages > 0:
            page_accuracy = min(self.actual_pages, self.expected_pages) / max(self.actual_pages, self.expected_pages)
            scores.append(page_accuracy * 0.3)
        else:
            scores.append(0.3)  # No expectation, assume correct

        # Image quality (weight: 20%)
        if self.image_quality_scores:
            avg_quality = sum(self.image_quality_scores) / len(self.image_quality_scores)
            scores.append(avg_quality * 0.2)
        else:
            scores.append(0.2)

        # Metadata accuracy (weight: 10%)
        scores.append(self.metadata_accuracy * 0.1)

        self.overall_score = sum(scores)
        return self.overall_score


@dataclass
class AccuracyReport:
    """Comprehensive accuracy report."""
    timestamp: str
    total_documents: int
    successful_conversions: int
    failed_conversions: int
    overall_accuracy: float
    metrics_by_category: Dict[str, AccuracyMetrics] = field(default_factory=dict)
    document_results: List[DocumentAccuracyResult] = field(default_factory=list)
    regressions_detected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# Ground Truth Management
# =============================================================================

@dataclass
class GroundTruth:
    """Ground truth data for a document."""
    filename: str
    expected_pages: int
    expected_format: str = "RGB"
    min_width: int = 100
    min_height: int = 100
    expected_metadata: Dict[str, Any] = field(default_factory=dict)
    quality_threshold: float = 0.7


class GroundTruthManager:
    """Manage ground truth datasets."""

    def __init__(self, ground_truth_dir: Path):
        self.ground_truth_dir = ground_truth_dir
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, GroundTruth] = {}

    def save_ground_truth(self, filename: str, ground_truth: GroundTruth) -> Path:
        """Save ground truth for a document."""
        gt_file = self.ground_truth_dir / f"{Path(filename).stem}.json"
        data = {
            "filename": ground_truth.filename,
            "expected_pages": ground_truth.expected_pages,
            "expected_format": ground_truth.expected_format,
            "min_width": ground_truth.min_width,
            "min_height": ground_truth.min_height,
            "expected_metadata": ground_truth.expected_metadata,
            "quality_threshold": ground_truth.quality_threshold,
        }
        with open(gt_file, 'w') as f:
            json.dump(data, f, indent=2)
        self._cache[filename] = ground_truth
        return gt_file

    def load_ground_truth(self, filename: str) -> Optional[GroundTruth]:
        """Load ground truth for a document."""
        if filename in self._cache:
            return self._cache[filename]

        gt_file = self.ground_truth_dir / f"{Path(filename).stem}.json"
        if not gt_file.exists():
            return None

        with open(gt_file, 'r') as f:
            data = json.load(f)

        gt = GroundTruth(
            filename=data["filename"],
            expected_pages=data["expected_pages"],
            expected_format=data.get("expected_format", "RGB"),
            min_width=data.get("min_width", 100),
            min_height=data.get("min_height", 100),
            expected_metadata=data.get("expected_metadata", {}),
            quality_threshold=data.get("quality_threshold", 0.7),
        )
        self._cache[filename] = gt
        return gt

    def has_ground_truth(self, filename: str) -> bool:
        """Check if ground truth exists for a document."""
        return self.load_ground_truth(filename) is not None


@pytest.fixture(scope="session")
def ground_truth_manager(ground_truth_directory) -> GroundTruthManager:
    """Create ground truth manager."""
    return GroundTruthManager(ground_truth_directory)


# =============================================================================
# Accuracy Calculation Engine
# =============================================================================

class AccuracyCalculator:
    """Calculate accuracy metrics for document processing."""

    def __init__(self, ground_truth_manager: GroundTruthManager):
        self.gt_manager = ground_truth_manager

    def calculate_image_quality(self, image) -> float:
        """Calculate quality score for a processed image."""
        score = 1.0

        # Check dimensions
        width, height = image.size
        if width < 100 or height < 100:
            score *= 0.5

        # Check mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            score *= 0.8

        # Check for mostly empty images (potential conversion issues)
        if hasattr(image, 'getextrema'):
            try:
                extrema = image.getextrema()
                if isinstance(extrema, tuple) and len(extrema) >= 3:
                    # RGB image
                    r_range = extrema[0][1] - extrema[0][0] if extrema[0] else 0
                    g_range = extrema[1][1] - extrema[1][0] if extrema[1] else 0
                    b_range = extrema[2][1] - extrema[2][0] if extrema[2] else 0
                    avg_range = (r_range + g_range + b_range) / 3
                    if avg_range < 10:  # Very low contrast
                        score *= 0.7
            except Exception:
                pass

        return min(max(score, 0.0), 1.0)

    def evaluate_document(
        self,
        filename: str,
        category: str,
        images: List[Any],
        metadata: Optional[Any] = None
    ) -> DocumentAccuracyResult:
        """Evaluate accuracy for a single document."""
        gt = self.gt_manager.load_ground_truth(filename)

        result = DocumentAccuracyResult(
            filename=filename,
            category=category,
            conversion_success=len(images) > 0,
            expected_pages=gt.expected_pages if gt else None,
            actual_pages=len(images),
            page_count_match=False,
        )

        if not result.conversion_success:
            result.errors.append("Conversion failed - no images produced")
            return result

        # Check page count
        if gt and gt.expected_pages:
            result.page_count_match = result.actual_pages == gt.expected_pages
            if not result.page_count_match:
                result.errors.append(
                    f"Page count mismatch: expected {gt.expected_pages}, got {result.actual_pages}"
                )

        # Calculate image quality scores
        for img in images:
            quality = self.calculate_image_quality(img)
            result.image_quality_scores.append(quality)

        # Evaluate metadata if available
        if metadata and gt and gt.expected_metadata:
            result.metadata_accuracy = self._evaluate_metadata(metadata, gt.expected_metadata)

        result.calculate_overall_score()
        return result

    def _evaluate_metadata(self, actual: Any, expected: Dict[str, Any]) -> float:
        """Evaluate metadata accuracy."""
        if not expected:
            return 1.0

        matches = 0
        total = len(expected)

        for key, expected_value in expected.items():
            actual_value = getattr(actual, key, None) if hasattr(actual, key) else None
            if actual_value == expected_value:
                matches += 1

        return matches / total if total > 0 else 1.0

    def generate_report(
        self,
        document_results: List[DocumentAccuracyResult]
    ) -> AccuracyReport:
        """Generate comprehensive accuracy report."""
        report = AccuracyReport(
            timestamp=datetime.now().isoformat(),
            total_documents=len(document_results),
            successful_conversions=sum(1 for r in document_results if r.conversion_success),
            failed_conversions=sum(1 for r in document_results if not r.conversion_success),
            overall_accuracy=0.0,
            document_results=document_results,
        )

        # Calculate overall accuracy
        if document_results:
            report.overall_accuracy = sum(r.overall_score for r in document_results) / len(document_results)

        # Group by category
        by_category: Dict[str, List[DocumentAccuracyResult]] = {}
        for result in document_results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)

        # Calculate metrics by category
        for category, results in by_category.items():
            metrics = AccuracyMetrics()
            metrics.true_positives = sum(1 for r in results if r.conversion_success and r.page_count_match)
            metrics.false_positives = sum(1 for r in results if r.conversion_success and not r.page_count_match)
            metrics.false_negatives = sum(1 for r in results if not r.conversion_success)
            metrics.calculate_all()
            report.metrics_by_category[category] = metrics

        # Generate recommendations
        if report.failed_conversions > 0:
            report.recommendations.append(
                f"{report.failed_conversions} documents failed conversion - investigate error handling"
            )

        low_quality = [r for r in document_results if r.overall_score < 0.7]
        if low_quality:
            report.recommendations.append(
                f"{len(low_quality)} documents have low quality scores - review processing parameters"
            )

        return report


@pytest.fixture
def accuracy_calculator(ground_truth_manager) -> AccuracyCalculator:
    """Create accuracy calculator."""
    return AccuracyCalculator(ground_truth_manager)


# =============================================================================
# Accuracy Tests
# =============================================================================

class TestGroundTruthManagement:
    """Test ground truth dataset management."""

    def test_create_ground_truth_directory(self, ground_truth_directory):
        """Verify ground truth directory exists."""
        assert ground_truth_directory.exists()

    def test_save_and_load_ground_truth(self, ground_truth_manager, pdf_test_cases):
        """Test saving and loading ground truth."""
        if not pdf_test_cases:
            pytest.skip("No test PDFs")

        test_case = pdf_test_cases[0]

        # Create ground truth
        gt = GroundTruth(
            filename=test_case.filename,
            expected_pages=1,  # Will be updated after actual processing
            expected_format="RGB",
        )

        # Save
        gt_path = ground_truth_manager.save_ground_truth(test_case.filename, gt)
        assert gt_path.exists()

        # Load
        loaded = ground_truth_manager.load_ground_truth(test_case.filename)
        assert loaded is not None
        assert loaded.filename == test_case.filename

    @pytest.mark.skipif(
        not PDF_ADAPTER_AVAILABLE,
        reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
    )
    def test_create_ground_truth_for_all_pdfs(
        self,
        ground_truth_manager,
        pdf_test_catalog
    ):
        """Create ground truth entries for all test PDFs."""
        # Load PDF adapter to get actual page counts
        pdf_adapter = create_pdf_adapter()

        created_count = 0
        for filename, test_case in pdf_test_catalog.items():
            # Skip if ground truth already exists
            if ground_truth_manager.has_ground_truth(filename):
                continue

            # Process PDF to get actual values
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            try:
                images = run_async(pdf_adapter.convert_to_frames(content))

                gt = GroundTruth(
                    filename=filename,
                    expected_pages=len(images),
                    expected_format="RGB",
                    min_width=100,
                    min_height=100,
                )

                ground_truth_manager.save_ground_truth(filename, gt)
                created_count += 1

            except Exception as e:
                print(f"Failed to create ground truth for {filename}: {e}")

        print(f"\nCreated {created_count} new ground truth entries")


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestAccuracyCalculation:
    """Test accuracy calculation functionality."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_image_quality_calculation(self, pdf_adapter, simple_pdfs, accuracy_calculator):
        """Test image quality score calculation."""
        if not simple_pdfs:
            pytest.skip("No simple PDFs")

        test_case = simple_pdfs[0]
        with open(test_case.filepath, 'rb') as f:
            content = f.read()

        images = run_async(pdf_adapter.convert_to_frames(content))
        assert len(images) > 0

        for img in images:
            quality = accuracy_calculator.calculate_image_quality(img)
            assert 0.0 <= quality <= 1.0
            print(f"Image quality: {quality:.3f}")

    def test_document_evaluation(
        self,
        pdf_adapter,
        pdf_test_catalog,
        accuracy_calculator
    ):
        """Test document accuracy evaluation."""
        # Process first few documents
        for filename, test_case in list(pdf_test_catalog.items())[:3]:
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            images = run_async(pdf_adapter.convert_to_frames(content))
            metadata = pdf_adapter.extract_metadata(content)

            result = accuracy_calculator.evaluate_document(
                filename=filename,
                category=test_case.category,
                images=images,
                metadata=metadata
            )

            print(f"\n{filename}:")
            print(f"  Conversion: {'Success' if result.conversion_success else 'Failed'}")
            print(f"  Pages: {result.actual_pages}")
            print(f"  Overall score: {result.overall_score:.3f}")

            assert result.conversion_success
            assert result.overall_score > 0


class TestPrecisionRecallF1:
    """Test precision, recall, and F1 score calculations."""

    def test_metrics_calculation(self):
        """Test basic metrics calculation."""
        metrics = AccuracyMetrics(
            true_positives=80,
            false_positives=10,
            false_negatives=5,
            true_negatives=5
        )

        metrics.calculate_all()

        assert 0.85 < metrics.precision < 0.95  # 80/(80+10) = 0.888
        assert 0.90 < metrics.recall < 1.0  # 80/(80+5) = 0.941
        assert 0.85 < metrics.f1_score < 0.95
        assert 0.80 < metrics.accuracy < 0.90

    def test_edge_cases(self):
        """Test edge cases in metrics calculation."""
        # All true positives
        metrics_perfect = AccuracyMetrics(
            true_positives=100,
            false_positives=0,
            false_negatives=0,
            true_negatives=0
        )
        metrics_perfect.calculate_all()
        assert metrics_perfect.precision == 1.0
        assert metrics_perfect.recall == 1.0

        # No true positives
        metrics_zero = AccuracyMetrics(
            true_positives=0,
            false_positives=10,
            false_negatives=10,
            true_negatives=0
        )
        metrics_zero.calculate_all()
        assert metrics_zero.precision == 0.0
        assert metrics_zero.recall == 0.0
        assert metrics_zero.f1_score == 0.0


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestAccuracyReporting:
    """Test accuracy reporting functionality."""

    @pytest.fixture
    def pdf_adapter(self):
        """Create PDF adapter."""
        return create_pdf_adapter()

    def test_generate_accuracy_report(
        self,
        pdf_adapter,
        pdf_test_catalog,
        accuracy_calculator,
        save_test_result
    ):
        """Generate comprehensive accuracy report."""
        document_results = []

        for filename, test_case in pdf_test_catalog.items():
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            try:
                images = run_async(pdf_adapter.convert_to_frames(content))
                metadata = pdf_adapter.extract_metadata(content)

                result = accuracy_calculator.evaluate_document(
                    filename=filename,
                    category=test_case.category,
                    images=images,
                    metadata=metadata
                )
                document_results.append(result)

            except Exception as e:
                # Record failed conversion
                result = DocumentAccuracyResult(
                    filename=filename,
                    category=test_case.category,
                    conversion_success=False,
                    expected_pages=None,
                    actual_pages=0,
                    page_count_match=False,
                    errors=[str(e)]
                )
                document_results.append(result)

        # Generate report
        report = accuracy_calculator.generate_report(document_results)

        # Save report
        report_data = {
            "timestamp": report.timestamp,
            "total_documents": report.total_documents,
            "successful_conversions": report.successful_conversions,
            "failed_conversions": report.failed_conversions,
            "overall_accuracy": report.overall_accuracy,
            "metrics_by_category": {
                cat: metrics.to_dict()
                for cat, metrics in report.metrics_by_category.items()
            },
            "recommendations": report.recommendations,
        }

        result_path = save_test_result("accuracy_report", report_data)

        # Print summary
        print(f"\nAccuracy Report:")
        print(f"  Total documents: {report.total_documents}")
        print(f"  Successful: {report.successful_conversions}")
        print(f"  Failed: {report.failed_conversions}")
        print(f"  Overall accuracy: {report.overall_accuracy:.3f}")

        print("\nMetrics by Category:")
        for category, metrics in report.metrics_by_category.items():
            print(f"  {category}: P={metrics.precision:.3f} R={metrics.recall:.3f} F1={metrics.f1_score:.3f}")

        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")

        # Assertions
        assert report.successful_conversions == report.total_documents, "All conversions should succeed"
        assert report.overall_accuracy > 0.5, "Overall accuracy should be above 50%"


@pytest.mark.skipif(
    not PDF_ADAPTER_AVAILABLE,
    reason=f"PDFAdapter not available: {PDF_ADAPTER_ERROR}"
)
class TestRegressionDetection:
    """Test quality regression detection."""

    @pytest.fixture
    def accuracy_baseline_file(self, benchmark_results_dir) -> Path:
        """Path to accuracy baseline file."""
        return benchmark_results_dir / "accuracy_baseline.json"

    def test_save_accuracy_baseline(
        self,
        pdf_test_catalog,
        accuracy_calculator,
        accuracy_baseline_file
    ):
        """Save accuracy baseline for regression detection."""
        if accuracy_baseline_file.exists():
            pytest.skip("Baseline already exists")

        pdf_adapter = create_pdf_adapter()
        baselines = {}

        for filename, test_case in pdf_test_catalog.items():
            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            try:
                images = run_async(pdf_adapter.convert_to_frames(content))

                result = accuracy_calculator.evaluate_document(
                    filename=filename,
                    category=test_case.category,
                    images=images,
                )

                baselines[filename] = {
                    "overall_score": result.overall_score,
                    "page_count": result.actual_pages,
                    "image_quality_avg": sum(result.image_quality_scores) / len(result.image_quality_scores) if result.image_quality_scores else 0,
                }
            except Exception:
                pass

        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "baselines": baselines,
        }

        with open(accuracy_baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"\nAccuracy baseline saved to: {accuracy_baseline_file}")

    def test_check_for_regressions(
        self,
        pdf_test_catalog,
        accuracy_calculator,
        accuracy_baseline_file
    ):
        """Check current accuracy against baseline."""
        if not accuracy_baseline_file.exists():
            pytest.skip("No baseline file - run test_save_accuracy_baseline first")

        with open(accuracy_baseline_file, 'r') as f:
            baseline_data = json.load(f)

        baselines = baseline_data["baselines"]
        regressions = []

        pdf_adapter = create_pdf_adapter()

        for filename, test_case in pdf_test_catalog.items():
            if filename not in baselines:
                continue

            with open(test_case.filepath, 'rb') as f:
                content = f.read()

            try:
                images = run_async(pdf_adapter.convert_to_frames(content))

                result = accuracy_calculator.evaluate_document(
                    filename=filename,
                    category=test_case.category,
                    images=images,
                )

                baseline = baselines[filename]

                # Check for significant regression (>10% drop)
                if result.overall_score < baseline["overall_score"] * 0.9:
                    regressions.append({
                        "filename": filename,
                        "baseline_score": baseline["overall_score"],
                        "current_score": result.overall_score,
                        "regression": baseline["overall_score"] - result.overall_score,
                    })

            except Exception as e:
                regressions.append({
                    "filename": filename,
                    "error": str(e),
                })

        if regressions:
            print("\nAccuracy Regressions Detected:")
            for reg in regressions:
                if "error" in reg:
                    print(f"  {reg['filename']}: ERROR - {reg['error']}")
                else:
                    print(f"  {reg['filename']}: {reg['baseline_score']:.3f} -> {reg['current_score']:.3f} (-{reg['regression']:.3f})")

        assert len(regressions) == 0, f"Found {len(regressions)} accuracy regressions"


class TestContinuousMonitoring:
    """Test continuous accuracy monitoring framework."""

    def test_monitoring_infrastructure(self, ground_truth_directory, test_results_directory):
        """Verify monitoring infrastructure is in place."""
        assert ground_truth_directory.exists()
        assert test_results_directory.exists()

    def test_trend_analysis_framework(self, test_results_directory):
        """Test that trend analysis can be performed on results."""
        # Check for existing results
        result_files = list(test_results_directory.glob("accuracy_report_*.json"))

        if len(result_files) < 2:
            pytest.skip("Need at least 2 accuracy reports for trend analysis")

        # Load results chronologically
        results = []
        for rf in sorted(result_files):
            with open(rf, 'r') as f:
                results.append(json.load(f))

        # Analyze trend
        accuracies = [r.get("overall_accuracy", 0) for r in results]

        print(f"\nAccuracy Trend: {accuracies}")

        # Check for declining trend
        if len(accuracies) >= 2:
            trend = accuracies[-1] - accuracies[0]
            print(f"Overall trend: {'+' if trend > 0 else ''}{trend:.3f}")

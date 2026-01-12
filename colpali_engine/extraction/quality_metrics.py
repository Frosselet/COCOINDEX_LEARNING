"""
Extraction Quality Metrics and Analytics System - COLPALI-604.

This module provides comprehensive quality measurement, tracking, and analytics
for vision extraction results. Includes advanced quality algorithms, threshold
validation, performance benchmarking, quality trends analysis, and optimization
recommendations for continuous improvement.
"""

import logging
import statistics
import time
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import json
import math

from .models import QualityMetrics, ExtractionResult, CanonicalData, ProcessingMetadata
from .validation import ValidationReport, ValidationSeverity

logger = logging.getLogger(__name__)


class QualityThreshold(Enum):
    """Quality thresholds for different use cases."""
    MINIMUM_VIABLE = 0.6  # Minimum acceptable quality
    PRODUCTION_READY = 0.8  # Production-ready quality
    HIGH_CONFIDENCE = 0.9  # High confidence quality
    EXCEPTIONAL = 0.95  # Exceptional quality


class QualityDimension(Enum):
    """Different dimensions of extraction quality."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    SCHEMA_COMPLIANCE = "schema_compliance"
    TEXT_QUALITY = "text_quality"
    SPATIAL_ACCURACY = "spatial_accuracy"


class QualityTrendDirection(Enum):
    """Direction of quality trends."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


@dataclass
class QualityBenchmark:
    """Performance benchmark for quality assessment."""
    dimension: QualityDimension
    target_score: float
    current_score: float
    historical_average: float
    percentile_rank: float  # 0-100
    improvement_needed: float = field(init=False)
    meets_target: bool = field(init=False)

    def __post_init__(self):
        """Calculate derived metrics."""
        self.improvement_needed = max(0, self.target_score - self.current_score)
        self.meets_target = self.current_score >= self.target_score


@dataclass
class QualityMetricsReport:
    """Comprehensive quality metrics report."""
    processing_id: str
    timestamp: datetime
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    benchmarks: List[QualityBenchmark]
    quality_grade: str  # A, B, C, D, F
    meets_production_threshold: bool
    improvement_suggestions: List[str]
    risk_factors: List[str]
    performance_metrics: Dict[str, Any]


@dataclass
class QualityTrend:
    """Quality trend analysis over time."""
    dimension: QualityDimension
    period_start: datetime
    period_end: datetime
    direction: QualityTrendDirection
    change_rate: float  # Percentage change per time period
    volatility: float  # Standard deviation of scores
    trend_confidence: float  # 0-1 confidence in trend direction
    sample_count: int


@dataclass
class QualityAlert:
    """Quality degradation alert."""
    alert_id: str
    severity: str  # "warning", "critical"
    dimension: QualityDimension
    current_score: float
    threshold_score: float
    decline_rate: float
    detected_at: datetime
    suggested_actions: List[str]


class BaseQualityAnalyzer(ABC):
    """Abstract base class for quality analyzers."""

    @abstractmethod
    def analyze(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Analyze quality for a specific dimension.

        Args:
            data: Extracted data to analyze
            context: Analysis context with metadata

        Returns:
            Quality score between 0 and 1
        """
        pass


class AccuracyAnalyzer(BaseQualityAnalyzer):
    """Analyzer for extraction accuracy."""

    def analyze(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze extraction accuracy based on confidence scores and validation."""
        if not data:
            return 0.0

        # Get confidence scores from context
        confidence_scores = context.get('confidence_scores', {})
        if not confidence_scores:
            # Fallback to basic accuracy assessment
            return self._assess_basic_accuracy(data)

        # Calculate weighted accuracy based on field importance
        total_weight = 0
        weighted_accuracy = 0

        for field, value in data.items():
            confidence = confidence_scores.get(field, 0.5)
            importance = self._get_field_importance(field, context)

            weight = importance
            total_weight += weight
            weighted_accuracy += confidence * weight

        return weighted_accuracy / total_weight if total_weight > 0 else 0.0

    def _assess_basic_accuracy(self, data: Dict[str, Any]) -> float:
        """Basic accuracy assessment when confidence scores unavailable."""
        if not data:
            return 0.0

        # Simple heuristics for accuracy
        score = 0.7  # Base score

        # Penalize for obvious errors
        for key, value in data.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                score -= 0.1
            elif isinstance(value, str):
                # Check for extraction artifacts
                if any(error_term in value.lower() for error_term in ['error', 'failed', 'null']):
                    score -= 0.2

        return max(0.0, min(1.0, score))

    def _get_field_importance(self, field: str, context: Dict[str, Any]) -> float:
        """Get importance weight for a field."""
        # Important fields get higher weights
        important_fields = ['total', 'amount', 'id', 'date', 'name']

        field_lower = field.lower()
        for important in important_fields:
            if important in field_lower:
                return 1.0

        return 0.5  # Default weight


class CompletenessAnalyzer(BaseQualityAnalyzer):
    """Analyzer for extraction completeness."""

    def analyze(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze extraction completeness."""
        expected_fields = context.get('expected_fields', [])
        if not expected_fields:
            # Fallback to basic completeness check
            return self._assess_basic_completeness(data)

        if not data:
            return 0.0

        # Calculate completeness ratio
        found_fields = 0
        total_importance = 0
        found_importance = 0

        for field in expected_fields:
            importance = self._get_field_importance(field, context)
            total_importance += importance

            if field in data and data[field] is not None:
                if isinstance(data[field], str) and data[field].strip():
                    found_fields += 1
                    found_importance += importance
                elif not isinstance(data[field], str):
                    found_fields += 1
                    found_importance += importance

        # Weighted completeness score
        if total_importance > 0:
            return found_importance / total_importance
        else:
            return found_fields / len(expected_fields) if expected_fields else 0.0

    def _assess_basic_completeness(self, data: Dict[str, Any]) -> float:
        """Basic completeness assessment."""
        if not data:
            return 0.0

        total_fields = len(data)
        if total_fields == 0:
            return 0.0

        filled_fields = sum(
            1 for value in data.values()
            if value is not None and (not isinstance(value, str) or value.strip())
        )

        return filled_fields / total_fields

    def _get_field_importance(self, field: str, context: Dict[str, Any]) -> float:
        """Get field importance for completeness calculation."""
        required_fields = context.get('required_fields', [])

        if field in required_fields:
            return 2.0  # Required fields are twice as important

        return 1.0  # Default importance


class ConsistencyAnalyzer(BaseQualityAnalyzer):
    """Analyzer for extraction consistency."""

    def analyze(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze extraction consistency."""
        if not data:
            return 0.0

        consistency_score = 1.0

        # Check type consistency
        consistency_score *= self._check_type_consistency(data, context)

        # Check format consistency
        consistency_score *= self._check_format_consistency(data, context)

        # Check value consistency
        consistency_score *= self._check_value_consistency(data, context)

        return consistency_score

    def _check_type_consistency(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Check if extracted data types are consistent with expectations."""
        expected_types = context.get('expected_types', {})
        if not expected_types:
            return 1.0

        consistent_fields = 0
        total_fields = 0

        for field, expected_type in expected_types.items():
            if field in data and data[field] is not None:
                total_fields += 1
                actual_value = data[field]

                if self._type_matches(actual_value, expected_type):
                    consistent_fields += 1

        return consistent_fields / total_fields if total_fields > 0 else 1.0

    def _check_format_consistency(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Check format consistency (dates, numbers, etc.)."""
        score = 1.0

        # Check date formats
        date_fields = [k for k in data.keys() if 'date' in k.lower()]
        if date_fields:
            score *= self._check_date_format_consistency(data, date_fields)

        # Check number formats
        score *= self._check_number_format_consistency(data)

        return score

    def _check_value_consistency(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Check logical value consistency."""
        # Check for duplicate values where they shouldn't exist
        string_values = []
        for value in data.values():
            if isinstance(value, str) and len(value.strip()) > 3:
                string_values.append(value.strip().lower())

        if len(string_values) > 0:
            unique_values = len(set(string_values))
            duplicate_penalty = 1.0 - (len(string_values) - unique_values) / len(string_values)
            return duplicate_penalty * 0.9 + 0.1  # Allow some duplication

        return 1.0

    def _type_matches(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            'string': str,
            'int': int,
            'integer': int,
            'float': float,
            'number': (int, float),
            'bool': bool,
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_python_type = type_mapping.get(expected_type.lower())
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True  # Unknown type, assume consistent

    def _check_date_format_consistency(self, data: Dict[str, Any], date_fields: List[str]) -> float:
        """Check consistency of date formats."""
        if len(date_fields) <= 1:
            return 1.0

        date_patterns = []
        for field in date_fields:
            if field in data and isinstance(data[field], str):
                pattern = self._detect_date_pattern(data[field])
                if pattern:
                    date_patterns.append(pattern)

        if not date_patterns:
            return 1.0

        # Check if all dates use the same pattern
        unique_patterns = set(date_patterns)
        return 1.0 if len(unique_patterns) <= 1 else 0.7

    def _check_number_format_consistency(self, data: Dict[str, Any]) -> float:
        """Check consistency of number formats."""
        # This is a simplified check - could be enhanced
        return 1.0

    def _detect_date_pattern(self, date_str: str) -> Optional[str]:
        """Detect the pattern of a date string."""
        import re

        patterns = {
            'YYYY-MM-DD': r'^\d{4}-\d{2}-\d{2}$',
            'MM/DD/YYYY': r'^\d{1,2}/\d{1,2}/\d{4}$',
            'DD.MM.YYYY': r'^\d{1,2}\.\d{1,2}\.\d{4}$'
        }

        for pattern_name, regex in patterns.items():
            if re.match(regex, date_str.strip()):
                return pattern_name

        return None


class ReliabilityAnalyzer(BaseQualityAnalyzer):
    """Analyzer for extraction reliability."""

    def analyze(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze extraction reliability based on error rates and stability."""
        # Get processing metadata
        processing_metadata = context.get('processing_metadata', {})
        error_count = processing_metadata.get('error_count', 0)
        retry_count = processing_metadata.get('retry_count', 0)
        fallback_applied = processing_metadata.get('fallback_applied', False)

        # Base reliability score
        reliability = 1.0

        # Penalize for errors and retries
        if error_count > 0:
            reliability *= (1.0 - min(0.5, error_count * 0.1))

        if retry_count > 0:
            reliability *= (1.0 - min(0.3, retry_count * 0.1))

        if fallback_applied:
            reliability *= 0.8

        # Check for processing anomalies
        processing_time = processing_metadata.get('processing_time', 0)
        if processing_time > 30:  # Unusually long processing
            reliability *= 0.9

        return max(0.0, reliability)


class PerformanceAnalyzer(BaseQualityAnalyzer):
    """Analyzer for extraction performance metrics."""

    def analyze(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze extraction performance."""
        processing_metadata = context.get('processing_metadata', {})

        # Processing time score (faster = better)
        processing_time = processing_metadata.get('processing_time', 30)
        time_score = self._calculate_time_score(processing_time)

        # Throughput score
        field_count = len(data) if data else 0
        throughput_score = self._calculate_throughput_score(field_count, processing_time)

        # Resource efficiency score
        resource_score = processing_metadata.get('resource_efficiency', 0.8)

        return (time_score + throughput_score + resource_score) / 3

    def _calculate_time_score(self, processing_time: float) -> float:
        """Calculate score based on processing time."""
        # Excellent: < 5s, Good: < 15s, Acceptable: < 30s, Poor: > 30s
        if processing_time < 5:
            return 1.0
        elif processing_time < 15:
            return 0.9
        elif processing_time < 30:
            return 0.7
        else:
            return max(0.2, 1.0 - (processing_time - 30) / 60)

    def _calculate_throughput_score(self, field_count: int, processing_time: float) -> float:
        """Calculate throughput score (fields per second)."""
        if processing_time <= 0 or field_count <= 0:
            return 0.5

        throughput = field_count / processing_time

        # Good throughput: > 1 field/sec
        if throughput >= 1.0:
            return min(1.0, throughput / 2.0)
        else:
            return throughput


class SchemaComplianceAnalyzer(BaseQualityAnalyzer):
    """Analyzer for schema compliance quality."""

    def analyze(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze schema compliance quality."""
        validation_report = context.get('validation_report')
        if not validation_report:
            return 0.8  # Default score when validation unavailable

        if isinstance(validation_report, dict):
            # Extract key metrics from validation report dict
            is_valid = validation_report.get('is_valid', False)
            total_issues = validation_report.get('total_issues', 0)
            critical_issues = validation_report.get('critical_issues', 0)
            error_issues = validation_report.get('error_issues', 0)
        else:
            # Handle ValidationReport object
            is_valid = validation_report.is_valid
            total_issues = validation_report.total_issues
            critical_issues = validation_report.issues_by_severity.get(ValidationSeverity.CRITICAL, 0)
            error_issues = validation_report.issues_by_severity.get(ValidationSeverity.ERROR, 0)

        if is_valid and total_issues == 0:
            return 1.0

        # Calculate compliance score based on issue severity
        score = 1.0

        # Heavy penalty for critical issues
        if critical_issues > 0:
            score *= max(0.2, 1.0 - critical_issues * 0.3)

        # Moderate penalty for errors
        if error_issues > 0:
            score *= max(0.4, 1.0 - error_issues * 0.2)

        # Light penalty for other issues
        other_issues = total_issues - critical_issues - error_issues
        if other_issues > 0:
            score *= max(0.7, 1.0 - other_issues * 0.05)

        return score


class ExtractionQualityManager:
    """
    Comprehensive extraction quality management system.

    Provides advanced quality measurement, tracking, trend analysis,
    and optimization recommendations for vision extraction pipelines.
    """

    def __init__(self, enable_trending: bool = True, enable_alerting: bool = True):
        """
        Initialize quality manager with analytics capabilities.

        Args:
            enable_trending: Enable quality trend analysis
            enable_alerting: Enable quality degradation alerts
        """
        self.enable_trending = enable_trending
        self.enable_alerting = enable_alerting

        # Initialize analyzers
        self.analyzers = {
            QualityDimension.ACCURACY: AccuracyAnalyzer(),
            QualityDimension.COMPLETENESS: CompletenessAnalyzer(),
            QualityDimension.CONSISTENCY: ConsistencyAnalyzer(),
            QualityDimension.RELIABILITY: ReliabilityAnalyzer(),
            QualityDimension.PERFORMANCE: PerformanceAnalyzer(),
            QualityDimension.SCHEMA_COMPLIANCE: SchemaComplianceAnalyzer()
        }

        # Quality history for trending
        self.quality_history: List[Dict[str, Any]] = []

        # Quality benchmarks
        self.benchmarks: Dict[QualityDimension, float] = {
            QualityDimension.ACCURACY: 0.85,
            QualityDimension.COMPLETENESS: 0.90,
            QualityDimension.CONSISTENCY: 0.80,
            QualityDimension.RELIABILITY: 0.95,
            QualityDimension.PERFORMANCE: 0.75,
            QualityDimension.SCHEMA_COMPLIANCE: 0.90
        }

        # Active alerts
        self.active_alerts: List[QualityAlert] = []

        logger.info("Extraction Quality Manager initialized")

    async def assess_extraction_quality(
        self,
        extraction_result: ExtractionResult,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityMetricsReport:
        """
        Perform comprehensive quality assessment of extraction results.

        Args:
            extraction_result: Complete extraction result to assess
            context: Additional context for quality assessment

        Returns:
            Comprehensive quality metrics report
        """
        assessment_context = context or {}

        # Prepare analysis context
        analysis_context = {
            **assessment_context,
            'confidence_scores': extraction_result.canonical.confidence_scores or {},
            'processing_metadata': {
                'processing_time': extraction_result.metadata.processing_time_seconds,
                'error_count': 0,  # Would be populated from error handling system
                'retry_count': 0,  # Would be populated from error handling system
                'fallback_applied': False,  # Would be populated from processing
                'resource_efficiency': 0.8  # Would be calculated from actual resource usage
            }
        }

        # Run quality analysis for each dimension
        dimension_scores = {}
        benchmarks = []

        for dimension, analyzer in self.analyzers.items():
            try:
                score = analyzer.analyze(
                    extraction_result.canonical.extraction_data,
                    analysis_context
                )
                dimension_scores[dimension] = score

                # Create benchmark
                historical_avg = self._get_historical_average(dimension)
                percentile = self._calculate_percentile_rank(dimension, score)

                benchmark = QualityBenchmark(
                    dimension=dimension,
                    target_score=self.benchmarks[dimension],
                    current_score=score,
                    historical_average=historical_avg,
                    percentile_rank=percentile
                )
                benchmarks.append(benchmark)

                logger.debug(f"{dimension.value} quality score: {score:.3f}")

            except Exception as e:
                logger.error(f"Failed to analyze {dimension.value}: {e}")
                dimension_scores[dimension] = 0.5  # Fallback score

        # Calculate overall quality score
        overall_score = self._calculate_overall_score(dimension_scores)

        # Determine quality grade
        quality_grade = self._assign_quality_grade(overall_score)

        # Check production readiness
        meets_production = overall_score >= QualityThreshold.PRODUCTION_READY.value

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            dimension_scores, benchmarks
        )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(dimension_scores, extraction_result)

        # Create quality report
        report = QualityMetricsReport(
            processing_id=extraction_result.canonical.processing_id,
            timestamp=datetime.now(),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            benchmarks=benchmarks,
            quality_grade=quality_grade,
            meets_production_threshold=meets_production,
            improvement_suggestions=improvement_suggestions,
            risk_factors=risk_factors,
            performance_metrics=self._extract_performance_metrics(extraction_result)
        )

        # Update quality history
        if self.enable_trending:
            self._update_quality_history(report)

        # Check for quality alerts
        if self.enable_alerting:
            await self._check_quality_alerts(report)

        logger.info(
            f"Quality assessment completed: {overall_score:.3f} "
            f"({quality_grade}), production ready: {meets_production}"
        )

        return report

    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, float]) -> float:
        """Calculate weighted overall quality score."""
        # Weights for different dimensions
        weights = {
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.CONSISTENCY: 0.15,
            QualityDimension.RELIABILITY: 0.20,
            QualityDimension.PERFORMANCE: 0.10,
            QualityDimension.SCHEMA_COMPLIANCE: 0.10
        }

        weighted_sum = 0
        total_weight = 0

        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _assign_quality_grade(self, overall_score: float) -> str:
        """Assign quality grade based on overall score."""
        if overall_score >= 0.95:
            return "A+"
        elif overall_score >= 0.90:
            return "A"
        elif overall_score >= 0.85:
            return "B+"
        elif overall_score >= 0.80:
            return "B"
        elif overall_score >= 0.70:
            return "C+"
        elif overall_score >= 0.60:
            return "C"
        else:
            return "F"

    def _generate_improvement_suggestions(
        self,
        dimension_scores: Dict[QualityDimension, float],
        benchmarks: List[QualityBenchmark]
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []

        # Find dimensions that need improvement
        poor_dimensions = [
            dim for dim, score in dimension_scores.items()
            if score < self.benchmarks[dim]
        ]

        for dimension in poor_dimensions:
            score = dimension_scores[dimension]
            target = self.benchmarks[dimension]
            improvement_needed = target - score

            if dimension == QualityDimension.ACCURACY:
                if improvement_needed > 0.2:
                    suggestions.append("Consider using higher-capability vision models for better accuracy")
                else:
                    suggestions.append("Fine-tune confidence score thresholds or add validation rules")

            elif dimension == QualityDimension.COMPLETENESS:
                if improvement_needed > 0.15:
                    suggestions.append("Review extraction prompts to ensure all required fields are requested")
                else:
                    suggestions.append("Add field-specific extraction hints or examples")

            elif dimension == QualityDimension.CONSISTENCY:
                suggestions.append("Implement stricter data type validation and format normalization")

            elif dimension == QualityDimension.RELIABILITY:
                suggestions.append("Increase error handling robustness and reduce retry failures")

            elif dimension == QualityDimension.PERFORMANCE:
                suggestions.append("Optimize processing pipeline or consider faster vision models")

            elif dimension == QualityDimension.SCHEMA_COMPLIANCE:
                suggestions.append("Strengthen schema validation rules and improve prompt engineering")

        if not suggestions:
            suggestions.append("Quality metrics are meeting targets - consider optimizing for cost efficiency")

        return suggestions

    def _identify_risk_factors(
        self,
        dimension_scores: Dict[QualityDimension, float],
        extraction_result: ExtractionResult
    ) -> List[str]:
        """Identify potential risk factors in extraction quality."""
        risk_factors = []

        # Check for critical quality issues
        if dimension_scores.get(QualityDimension.ACCURACY, 1.0) < 0.6:
            risk_factors.append("Low accuracy may lead to incorrect business decisions")

        if dimension_scores.get(QualityDimension.RELIABILITY, 1.0) < 0.7:
            risk_factors.append("Poor reliability may cause processing failures in production")

        if dimension_scores.get(QualityDimension.SCHEMA_COMPLIANCE, 1.0) < 0.7:
            risk_factors.append("Schema violations may break downstream integrations")

        # Check processing metadata for risks
        processing_time = extraction_result.metadata.processing_time_seconds
        if processing_time > 60:
            risk_factors.append("Long processing times may impact user experience")

        # Check data completeness
        field_count = extraction_result.canonical.field_count
        if field_count < 3:
            risk_factors.append("Very few extracted fields may indicate extraction failure")

        return risk_factors

    def _extract_performance_metrics(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """Extract performance metrics from extraction result."""
        return {
            "processing_time_seconds": extraction_result.metadata.processing_time_seconds,
            "processing_time_ms": extraction_result.metadata.duration_ms,
            "field_count": extraction_result.canonical.field_count,
            "throughput_fields_per_second": (
                extraction_result.canonical.field_count /
                max(extraction_result.metadata.processing_time_seconds, 0.1)
            ),
            "lineage_steps": len(extraction_result.metadata.lineage_steps),
            "has_shaped_output": extraction_result.has_shaped_output
        }

    def _get_historical_average(self, dimension: QualityDimension) -> float:
        """Get historical average for a quality dimension."""
        if not self.quality_history:
            return 0.8  # Default when no history

        dimension_scores = [
            entry.get('dimension_scores', {}).get(dimension, 0.8)
            for entry in self.quality_history[-20:]  # Last 20 entries
        ]

        return statistics.mean(dimension_scores) if dimension_scores else 0.8

    def _calculate_percentile_rank(self, dimension: QualityDimension, score: float) -> float:
        """Calculate percentile rank for a score in historical context."""
        if not self.quality_history:
            return 50.0  # Default percentile

        historical_scores = [
            entry.get('dimension_scores', {}).get(dimension, 0.8)
            for entry in self.quality_history
        ]

        if not historical_scores:
            return 50.0

        # Count scores below current score
        below_count = sum(1 for s in historical_scores if s < score)
        total_count = len(historical_scores)

        return (below_count / total_count) * 100 if total_count > 0 else 50.0

    def _update_quality_history(self, report: QualityMetricsReport) -> None:
        """Update quality history for trend analysis."""
        history_entry = {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "dimension_scores": {
                dim.value: score for dim, score in report.dimension_scores.items()
            },
            "quality_grade": report.quality_grade,
            "meets_production": report.meets_production_threshold
        }

        self.quality_history.append(history_entry)

        # Keep only last 100 entries
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]

    async def _check_quality_alerts(self, report: QualityMetricsReport) -> None:
        """Check for quality degradation and generate alerts."""
        if len(self.quality_history) < 5:
            return  # Need minimum history for trend analysis

        # Check for significant quality degradation
        recent_scores = [
            entry["overall_score"] for entry in self.quality_history[-5:]
        ]

        current_score = report.overall_score
        recent_avg = statistics.mean(recent_scores)

        # Alert if current score is significantly below recent average
        if current_score < recent_avg - 0.15:
            alert = QualityAlert(
                alert_id=f"quality_alert_{int(time.time())}",
                severity="critical",
                dimension=QualityDimension.ACCURACY,  # Primary concern
                current_score=current_score,
                threshold_score=recent_avg - 0.15,
                decline_rate=(recent_avg - current_score) / recent_avg,
                detected_at=datetime.now(),
                suggested_actions=[
                    "Investigate recent processing changes",
                    "Review input data quality",
                    "Check system resource availability"
                ]
            )

            self.active_alerts.append(alert)
            logger.warning(f"Quality degradation alert: {alert.alert_id}")

    def analyze_quality_trends(
        self,
        lookback_days: int = 7,
        minimum_samples: int = 10
    ) -> Dict[QualityDimension, QualityTrend]:
        """
        Analyze quality trends over specified time period.

        Args:
            lookback_days: Days to look back for trend analysis
            minimum_samples: Minimum samples required for reliable trends

        Returns:
            Quality trends for each dimension
        """
        if not self.enable_trending:
            logger.warning("Quality trending is disabled")
            return {}

        # Filter history to lookback period
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_history = [
            entry for entry in self.quality_history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]

        if len(recent_history) < minimum_samples:
            logger.info(f"Insufficient data for trend analysis: {len(recent_history)} < {minimum_samples}")
            return {}

        trends = {}
        period_start = datetime.fromisoformat(recent_history[0]["timestamp"])
        period_end = datetime.fromisoformat(recent_history[-1]["timestamp"])

        for dimension in QualityDimension:
            scores = [
                entry["dimension_scores"].get(dimension.value, 0.8)
                for entry in recent_history
            ]

            if len(scores) < minimum_samples:
                continue

            # Calculate trend direction and statistics
            direction = self._calculate_trend_direction(scores)
            change_rate = self._calculate_change_rate(scores, lookback_days)
            volatility = statistics.stdev(scores) if len(scores) > 1 else 0.0
            confidence = self._calculate_trend_confidence(scores)

            trend = QualityTrend(
                dimension=dimension,
                period_start=period_start,
                period_end=period_end,
                direction=direction,
                change_rate=change_rate,
                volatility=volatility,
                trend_confidence=confidence,
                sample_count=len(scores)
            )

            trends[dimension] = trend

        logger.info(f"Analyzed trends for {len(trends)} quality dimensions")
        return trends

    def _calculate_trend_direction(self, scores: List[float]) -> QualityTrendDirection:
        """Calculate trend direction from score sequence."""
        if len(scores) < 3:
            return QualityTrendDirection.STABLE

        # Simple linear trend calculation
        n = len(scores)
        x = list(range(n))

        # Calculate slope using least squares
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(scores)

        numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return QualityTrendDirection.STABLE

        slope = numerator / denominator

        # Determine trend direction
        if abs(slope) < 0.001:
            return QualityTrendDirection.STABLE
        elif slope > 0.005:
            return QualityTrendDirection.IMPROVING
        elif slope < -0.005:
            return QualityTrendDirection.DECLINING
        else:
            # Check volatility for stable vs volatile
            volatility = statistics.stdev(scores)
            return QualityTrendDirection.VOLATILE if volatility > 0.1 else QualityTrendDirection.STABLE

    def _calculate_change_rate(self, scores: List[float], days: int) -> float:
        """Calculate percentage change rate per day."""
        if len(scores) < 2:
            return 0.0

        first_score = scores[0]
        last_score = scores[-1]

        if first_score == 0:
            return 0.0

        total_change = (last_score - first_score) / first_score
        return (total_change / days) * 100  # Percentage change per day

    def _calculate_trend_confidence(self, scores: List[float]) -> float:
        """Calculate confidence in trend direction."""
        if len(scores) < 3:
            return 0.0

        # Calculate R-squared to measure how well the trend fits the data
        n = len(scores)
        x = list(range(n))

        x_mean = statistics.mean(x)
        y_mean = statistics.mean(scores)

        # Linear regression
        numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        predicted = [slope * x[i] + intercept for i in range(n)]
        ss_res = sum((scores[i] - predicted[i]) ** 2 for i in range(n))
        ss_tot = sum((scores[i] - y_mean) ** 2 for i in range(n))

        if ss_tot == 0:
            return 1.0

        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))

    def get_quality_recommendations(
        self,
        target_score: float = 0.9,
        budget_constraint: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get quality improvement recommendations with cost considerations.

        Args:
            target_score: Target overall quality score
            budget_constraint: Optional budget limit for improvements

        Returns:
            Recommendations for quality improvements
        """
        if not self.quality_history:
            return {"recommendations": [], "message": "Insufficient quality history for recommendations"}

        current_scores = self.quality_history[-1]["dimension_scores"] if self.quality_history else {}
        current_overall = self.quality_history[-1]["overall_score"] if self.quality_history else 0.8

        recommendations = []

        if current_overall >= target_score:
            recommendations.append({
                "type": "optimization",
                "priority": "low",
                "description": "Quality targets met - consider cost optimization",
                "estimated_impact": 0.0,
                "estimated_cost": 0.0
            })
        else:
            # Find dimensions that need the most improvement
            improvement_opportunities = []

            for dim_str, score in current_scores.items():
                try:
                    dimension = QualityDimension(dim_str)
                    target_dim_score = self.benchmarks.get(dimension, 0.85)

                    if score < target_dim_score:
                        improvement_needed = target_dim_score - score
                        improvement_opportunities.append({
                            "dimension": dimension,
                            "current_score": score,
                            "improvement_needed": improvement_needed,
                            "estimated_impact": improvement_needed * 0.8  # Simplified impact calculation
                        })
                except ValueError:
                    continue  # Skip unknown dimensions

            # Sort by improvement impact
            improvement_opportunities.sort(key=lambda x: x["estimated_impact"], reverse=True)

            # Generate specific recommendations
            for opportunity in improvement_opportunities[:3]:  # Top 3 opportunities
                dim = opportunity["dimension"]

                recommendation = {
                    "type": "improvement",
                    "dimension": dim.value,
                    "priority": "high" if opportunity["improvement_needed"] > 0.2 else "medium",
                    "current_score": opportunity["current_score"],
                    "target_score": self.benchmarks.get(dim, 0.85),
                    "estimated_impact": opportunity["estimated_impact"],
                    "estimated_cost": self._estimate_improvement_cost(dim, opportunity["improvement_needed"]),
                    "description": self._get_improvement_description(dim, opportunity["improvement_needed"])
                }

                if budget_constraint is None or recommendation["estimated_cost"] <= budget_constraint:
                    recommendations.append(recommendation)

        return {
            "recommendations": recommendations,
            "current_overall_score": current_overall,
            "target_overall_score": target_score,
            "estimated_total_cost": sum(rec.get("estimated_cost", 0) for rec in recommendations),
            "estimated_total_impact": sum(rec.get("estimated_impact", 0) for rec in recommendations)
        }

    def _estimate_improvement_cost(self, dimension: QualityDimension, improvement_needed: float) -> float:
        """Estimate cost of improving a quality dimension."""
        # Simplified cost estimation - would be more sophisticated in production
        base_costs = {
            QualityDimension.ACCURACY: 50.0,  # Higher model costs
            QualityDimension.COMPLETENESS: 20.0,  # Prompt engineering
            QualityDimension.CONSISTENCY: 30.0,  # Validation rules
            QualityDimension.RELIABILITY: 40.0,  # Infrastructure improvements
            QualityDimension.PERFORMANCE: 60.0,  # Hardware/optimization
            QualityDimension.SCHEMA_COMPLIANCE: 25.0  # Schema improvements
        }

        base_cost = base_costs.get(dimension, 30.0)
        return base_cost * improvement_needed

    def _get_improvement_description(self, dimension: QualityDimension, improvement_needed: float) -> str:
        """Get description for improvement recommendation."""
        descriptions = {
            QualityDimension.ACCURACY: f"Improve accuracy by {improvement_needed:.2f} through better vision models or validation",
            QualityDimension.COMPLETENESS: f"Increase completeness by {improvement_needed:.2f} with enhanced extraction prompts",
            QualityDimension.CONSISTENCY: f"Improve consistency by {improvement_needed:.2f} with stricter format validation",
            QualityDimension.RELIABILITY: f"Enhance reliability by {improvement_needed:.2f} through better error handling",
            QualityDimension.PERFORMANCE: f"Boost performance by {improvement_needed:.2f} via optimization or faster models",
            QualityDimension.SCHEMA_COMPLIANCE: f"Strengthen schema compliance by {improvement_needed:.2f} with improved validation rules"
        }

        return descriptions.get(dimension, f"Improve {dimension.value} by {improvement_needed:.2f}")

    def export_quality_analytics(self) -> Dict[str, Any]:
        """Export comprehensive quality analytics for external analysis."""
        analytics = {
            "quality_history": self.quality_history,
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "dimension": alert.dimension.value,
                    "current_score": alert.current_score,
                    "threshold_score": alert.threshold_score,
                    "decline_rate": alert.decline_rate,
                    "detected_at": alert.detected_at.isoformat(),
                    "suggested_actions": alert.suggested_actions
                }
                for alert in self.active_alerts
            ],
            "benchmarks": {
                dim.value: score for dim, score in self.benchmarks.items()
            },
            "analytics_metadata": {
                "total_assessments": len(self.quality_history),
                "trending_enabled": self.enable_trending,
                "alerting_enabled": self.enable_alerting,
                "export_timestamp": datetime.now().isoformat()
            }
        }

        logger.info(f"Exported quality analytics with {len(self.quality_history)} assessments")
        return analytics


# Factory function for easy instantiation
def create_quality_manager(
    enable_trending: bool = True,
    enable_alerting: bool = True,
    custom_benchmarks: Optional[Dict[QualityDimension, float]] = None
) -> ExtractionQualityManager:
    """
    Create extraction quality manager with optional custom configuration.

    Args:
        enable_trending: Enable quality trend analysis
        enable_alerting: Enable quality degradation alerts
        custom_benchmarks: Custom quality benchmarks

    Returns:
        Configured extraction quality manager
    """
    manager = ExtractionQualityManager(
        enable_trending=enable_trending,
        enable_alerting=enable_alerting
    )

    if custom_benchmarks:
        manager.benchmarks.update(custom_benchmarks)

    return manager
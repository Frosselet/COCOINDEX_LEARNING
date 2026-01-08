"""
Data models for extraction results and processing metadata.

This module defines the core data structures used throughout the vision
processing pipeline, ensuring type safety and proper lineage tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class ProcessingStatus(Enum):
    """Status of processing operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QualityMetrics:
    """Quality metrics for extraction results."""
    confidence_score: float
    completeness_ratio: float
    consistency_score: float
    schema_compliance_score: float
    overall_quality: float = field(init=False)

    def __post_init__(self):
        """Calculate overall quality score."""
        self.overall_quality = (
            self.confidence_score +
            self.completeness_ratio +
            self.consistency_score +
            self.schema_compliance_score
        ) / 4


@dataclass
class ProcessingMetadata:
    """Metadata about the processing pipeline execution."""
    processing_id: str
    start_time: datetime
    end_time: datetime
    processing_time_seconds: float
    lineage_steps: List[Dict[str, Any]]
    config: Dict[str, Any]
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_message: Optional[str] = None
    quality_metrics: Optional[QualityMetrics] = None

    @property
    def duration_ms(self) -> int:
        """Get processing duration in milliseconds."""
        return int(self.processing_time_seconds * 1000)

    def add_lineage_step(self, step_data: Dict[str, Any]) -> None:
        """Add a step to the processing lineage."""
        self.lineage_steps.append({
            **step_data,
            "timestamp": datetime.now(),
            "processing_id": self.processing_id
        })


@dataclass
class CanonicalData:
    """
    Canonical truth layer data.

    This represents the faithful extraction from the document without any
    business transformations - serving as the authoritative source of truth.
    """
    processing_id: str
    extraction_data: Dict[str, Any]
    timestamp: datetime
    schema_version: str = "1.0"
    confidence_scores: Optional[Dict[str, float]] = None
    source_metadata: Optional[Dict[str, Any]] = None

    @property
    def field_count(self) -> int:
        """Get the number of extracted fields."""
        return len(self.extraction_data) if self.extraction_data else 0

    def get_field_confidence(self, field_name: str) -> Optional[float]:
        """Get confidence score for a specific field."""
        if not self.confidence_scores:
            return None
        return self.confidence_scores.get(field_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processing_id": self.processing_id,
            "extraction_data": self.extraction_data,
            "timestamp": self.timestamp.isoformat(),
            "schema_version": self.schema_version,
            "confidence_scores": self.confidence_scores,
            "source_metadata": self.source_metadata,
            "field_count": self.field_count
        }


@dataclass
class TransformationRule:
    """Represents a data transformation rule."""
    rule_id: str
    rule_type: str  # "normalize", "aggregate", "filter", etc.
    description: str
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ShapedData:
    """
    Shaped output layer data.

    This represents business-transformed data with 1NF enforcement and
    complete transformation lineage tracking.
    """
    processing_id: str
    canonical_id: str
    transformed_data: Dict[str, Any]
    transformations_applied: List[TransformationRule]
    timestamp: datetime
    is_1nf_compliant: bool = True
    transformation_metadata: Optional[Dict[str, Any]] = None

    @property
    def transformation_count(self) -> int:
        """Get the number of transformations applied."""
        return len(self.transformations_applied)

    def add_transformation(self, rule: TransformationRule) -> None:
        """Add a transformation rule to the applied list."""
        self.transformations_applied.append(rule)

    def validate_1nf_compliance(self) -> bool:
        """
        Validate that the transformed data is in 1NF.

        First Normal Form (1NF) requirements:
        1. Each table cell contains a single value
        2. Each record is unique
        3. Each column contains values of a single type
        """
        # TODO: Implement comprehensive 1NF validation
        # This will be implemented in COLPALI-702
        return self.is_1nf_compliant

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processing_id": self.processing_id,
            "canonical_id": self.canonical_id,
            "transformed_data": self.transformed_data,
            "transformations_applied": [
                {
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    "description": rule.description,
                    "version": rule.version,
                    "created_at": rule.created_at.isoformat(),
                    "parameters": rule.parameters
                }
                for rule in self.transformations_applied
            ],
            "timestamp": self.timestamp.isoformat(),
            "is_1nf_compliant": self.is_1nf_compliant,
            "transformation_count": self.transformation_count
        }


@dataclass
class ExtractionResult:
    """
    Complete result from the vision extraction pipeline.

    Contains both canonical and shaped data along with processing metadata
    and quality metrics.
    """
    canonical: CanonicalData
    shaped: Optional[ShapedData]
    metadata: ProcessingMetadata

    @property
    def has_shaped_output(self) -> bool:
        """Check if shaped output is available."""
        return self.shaped is not None

    @property
    def extraction_quality(self) -> Optional[float]:
        """Get overall extraction quality score."""
        return (
            self.metadata.quality_metrics.overall_quality
            if self.metadata.quality_metrics
            else None
        )

    def export_canonical(self) -> Dict[str, Any]:
        """Export canonical data for serialization."""
        return self.canonical.to_dict()

    def export_shaped(self) -> Optional[Dict[str, Any]]:
        """Export shaped data for serialization."""
        return self.shaped.to_dict() if self.shaped else None

    def export_metadata(self) -> Dict[str, Any]:
        """Export processing metadata for serialization."""
        return {
            "processing_id": self.metadata.processing_id,
            "start_time": self.metadata.start_time.isoformat(),
            "end_time": self.metadata.end_time.isoformat(),
            "processing_time_seconds": self.metadata.processing_time_seconds,
            "duration_ms": self.metadata.duration_ms,
            "status": self.metadata.status.value,
            "lineage_steps": self.metadata.lineage_steps,
            "config": self.metadata.config,
            "error_message": self.metadata.error_message,
            "quality_metrics": (
                {
                    "confidence_score": self.metadata.quality_metrics.confidence_score,
                    "completeness_ratio": self.metadata.quality_metrics.completeness_ratio,
                    "consistency_score": self.metadata.quality_metrics.consistency_score,
                    "schema_compliance_score": self.metadata.quality_metrics.schema_compliance_score,
                    "overall_quality": self.metadata.quality_metrics.overall_quality
                }
                if self.metadata.quality_metrics
                else None
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete result to dictionary for serialization."""
        return {
            "canonical": self.export_canonical(),
            "shaped": self.export_shaped(),
            "metadata": self.export_metadata(),
            "has_shaped_output": self.has_shaped_output,
            "extraction_quality": self.extraction_quality
        }


@dataclass
class DocumentPatch:
    """Represents a patch/region from a document image."""
    patch_id: str
    document_id: str
    page_number: int
    coordinates: tuple  # (x, y, width, height)
    embedding: Optional[List[float]] = None
    confidence_score: float = 0.0
    extracted_text: Optional[str] = None
    patch_metadata: Optional[Dict[str, Any]] = None

    @property
    def area(self) -> int:
        """Calculate patch area."""
        return self.coordinates[2] * self.coordinates[3]

    def overlaps_with(self, other: 'DocumentPatch') -> bool:
        """Check if this patch overlaps with another patch."""
        x1, y1, w1, h1 = self.coordinates
        x2, y2, w2, h2 = other.coordinates

        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


@dataclass
class ScoredPatch:
    """Document patch with similarity score for retrieval."""
    patch: DocumentPatch
    similarity_score: float
    retrieval_metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_qdrant_result(cls, qdrant_result: Any) -> 'ScoredPatch':
        """Create ScoredPatch from Qdrant search result."""
        # TODO: Implement Qdrant result parsing
        # This will be implemented in COLPALI-403
        pass
"""
Canonical data formatter for truth layer preservation.

This module implements the canonical output formatter that preserves exact
extraction results without any business transformations, serving as the
authoritative source of truth.
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, ValidationError
from ..extraction.models import CanonicalData

logger = logging.getLogger(__name__)


class CanonicalFormatter:
    """
    Formatter for canonical truth layer data.

    Preserves extraction results faithfully without any semantic modifications
    or business logic injection, maintaining data integrity for audit and
    compliance requirements.
    """

    def __init__(self, schema_version: str = "1.0"):
        """
        Initialize canonical formatter.

        Args:
            schema_version: Version of the canonical schema
        """
        self.schema_version = schema_version
        self._immutable_fields = {"processing_id", "timestamp", "schema_version"}
        logger.info(f"CanonicalFormatter initialized with schema version {schema_version}")

    async def format_extraction_result(
        self,
        processing_id: str,
        extraction_data: Dict[str, Any],
        confidence_scores: Optional[Dict[str, float]] = None,
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> CanonicalData:
        """
        Format extraction result as canonical data.

        Implements faithful data preservation without any semantic modifications,
        ensuring the canonical layer represents the exact truth from extraction.

        Args:
            processing_id: Unique processing identifier
            extraction_data: Raw extraction results
            confidence_scores: Optional field-level confidence scores
            source_metadata: Optional source document metadata

        Returns:
            CanonicalData object with preserved truth

        Raises:
            ValueError: If extraction_data is None or empty
        """
        logger.info(f"Formatting canonical data for processing_id: {processing_id}")

        # Validate input data
        if extraction_data is None:
            raise ValueError("extraction_data cannot be None")

        if not isinstance(extraction_data, dict):
            raise ValueError(f"extraction_data must be a dict, got {type(extraction_data)}")

        # Create deep copy to ensure immutability
        preserved_data = self._deep_copy_data(extraction_data)

        # Preserve metadata without modification
        preserved_metadata = None
        if source_metadata is not None:
            preserved_metadata = self._deep_copy_data(source_metadata)
            # Add preservation timestamp
            preserved_metadata["preserved_at"] = datetime.now().isoformat()

        # Preserve confidence scores
        preserved_confidence = None
        if confidence_scores is not None:
            preserved_confidence = dict(confidence_scores)

        # Create canonical data object
        canonical_data = CanonicalData(
            processing_id=processing_id,
            extraction_data=preserved_data,
            timestamp=datetime.now(),
            schema_version=self.schema_version,
            confidence_scores=preserved_confidence,
            source_metadata=preserved_metadata
        )

        # Calculate and add integrity hash
        integrity_hash = self._calculate_integrity_hash(canonical_data)
        if canonical_data.source_metadata is None:
            canonical_data.source_metadata = {}
        canonical_data.source_metadata["integrity_hash"] = integrity_hash

        logger.info(
            f"Canonical data formatted successfully: {canonical_data.field_count} fields, "
            f"integrity_hash: {integrity_hash[:8]}..."
        )

        return canonical_data

    def validate_data_integrity(
        self,
        canonical_data: CanonicalData
    ) -> bool:
        """
        Validate canonical data integrity.

        Checks that the canonical data hasn't been tampered with by verifying
        the integrity hash and ensuring immutable fields haven't changed.

        Args:
            canonical_data: CanonicalData to validate

        Returns:
            True if data integrity is maintained, False otherwise
        """
        logger.info(f"Validating canonical data integrity for: {canonical_data.processing_id}")

        try:
            # Check for required fields
            if not canonical_data.processing_id:
                logger.error("Missing processing_id")
                return False

            if canonical_data.extraction_data is None:
                logger.error("Missing extraction_data")
                return False

            # Verify integrity hash if present
            if (canonical_data.source_metadata and
                "integrity_hash" in canonical_data.source_metadata):

                stored_hash = canonical_data.source_metadata["integrity_hash"]
                calculated_hash = self._calculate_integrity_hash(canonical_data)

                if stored_hash != calculated_hash:
                    logger.error(
                        f"Integrity hash mismatch! Stored: {stored_hash[:8]}..., "
                        f"Calculated: {calculated_hash[:8]}..."
                    )
                    return False

            # Validate metadata consistency
            if canonical_data.source_metadata:
                if "preserved_at" not in canonical_data.source_metadata:
                    logger.warning("Missing preserved_at timestamp in metadata")

            # Check schema version compatibility
            if canonical_data.schema_version != self.schema_version:
                logger.warning(
                    f"Schema version mismatch: data={canonical_data.schema_version}, "
                    f"formatter={self.schema_version}"
                )

            logger.info(f"Canonical data integrity validated successfully")
            return True

        except Exception as e:
            logger.error(f"Error validating canonical data integrity: {e}")
            return False

    def export_for_audit(
        self,
        canonical_data: CanonicalData
    ) -> Dict[str, Any]:
        """
        Export canonical data for audit purposes.

        Args:
            canonical_data: CanonicalData to export

        Returns:
            Audit-ready data dictionary
        """
        return {
            **canonical_data.to_dict(),
            "audit_metadata": {
                "exported_at": datetime.now().isoformat(),
                "format_version": f"canonical-v{self.schema_version}",
                "integrity_validated": self.validate_data_integrity(canonical_data),
                "field_count": canonical_data.field_count,
                "has_confidence_scores": canonical_data.confidence_scores is not None,
                "has_source_metadata": canonical_data.source_metadata is not None
            }
        }

    def _deep_copy_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a deep copy of data to ensure immutability.

        Args:
            data: Data to copy

        Returns:
            Deep copy of the data
        """
        import copy
        return copy.deepcopy(data)

    def _calculate_integrity_hash(self, canonical_data: CanonicalData) -> str:
        """
        Calculate integrity hash for canonical data.

        Uses SHA-256 hashing of the extraction data to detect any modifications.
        Note: The hash excludes the integrity_hash itself to avoid circular dependency.

        Args:
            canonical_data: CanonicalData to hash

        Returns:
            SHA-256 hash of the data
        """
        # Create hashable representation
        hashable_dict = {
            "processing_id": canonical_data.processing_id,
            "extraction_data": canonical_data.extraction_data,
            "timestamp": canonical_data.timestamp.isoformat(),
            "schema_version": canonical_data.schema_version,
            "confidence_scores": canonical_data.confidence_scores
        }

        # Convert to deterministic JSON string
        json_str = json.dumps(hashable_dict, sort_keys=True, default=str)

        # Calculate SHA-256 hash
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()

    def check_for_modifications(
        self,
        original: CanonicalData,
        current: CanonicalData
    ) -> List[str]:
        """
        Check for unauthorized modifications between two canonical data objects.

        Args:
            original: Original canonical data
            current: Current canonical data to compare

        Returns:
            List of modified fields (empty if no modifications)
        """
        modifications = []

        # Check immutable fields
        if original.processing_id != current.processing_id:
            modifications.append("processing_id")

        if original.timestamp != current.timestamp:
            modifications.append("timestamp")

        if original.schema_version != current.schema_version:
            modifications.append("schema_version")

        # Check data integrity
        original_hash = self._calculate_integrity_hash(original)
        current_hash = self._calculate_integrity_hash(current)

        if original_hash != current_hash:
            modifications.append("extraction_data (integrity hash mismatch)")

        return modifications
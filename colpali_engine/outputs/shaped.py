"""
Shaped data formatter with 1NF enforcement.

This module implements the business transformation layer that converts canonical
data into shaped outputs meeting specific business requirements while enforcing
First Normal Form (1NF) compliance.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from ..extraction.models import CanonicalData, ShapedData, TransformationRule

logger = logging.getLogger(__name__)


class ShapedFormatter:
    """
    Formatter for shaped output with 1NF enforcement.

    Applies business transformations to canonical data while maintaining
    complete transformation lineage and enforcing relational database
    normalization requirements.
    """

    def __init__(self):
        """Initialize shaped formatter."""
        self.transformation_rules: Dict[str, TransformationRule] = {}
        logger.info("ShapedFormatter initialized")

    async def transform_to_1nf(
        self,
        canonical_data: CanonicalData,
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> ShapedData:
        """
        Transform canonical data to 1NF-compliant shaped output.

        Implements automatic normalization to First Normal Form including:
        - Flattening nested structures
        - Eliminating repeating groups
        - Ensuring atomic values in all fields
        - Applying business transformation rules

        Args:
            canonical_data: Source canonical data
            transformation_config: Optional transformation configuration with keys:
                - flatten_nested: bool (default True) - Flatten nested dicts/lists
                - preserve_arrays: bool (default False) - Keep arrays as-is
                - transformation_rules: List[str] - Rule IDs to apply

        Returns:
            ShapedData with 1NF-compliant transformations

        Raises:
            ValueError: If canonical_data is invalid
        """
        logger.info(f"Transforming to 1NF shaped output for: {canonical_data.processing_id}")

        # Validate input
        if canonical_data.extraction_data is None:
            raise ValueError("canonical_data.extraction_data cannot be None")

        # Default configuration
        config = transformation_config or {}
        flatten_nested = config.get("flatten_nested", True)
        preserve_arrays = config.get("preserve_arrays", False)
        rule_ids = config.get("transformation_rules", [])

        # Track transformations applied
        transformations_applied = []

        # Step 1: Flatten nested structures for 1NF
        transformed_data = canonical_data.extraction_data.copy()

        if flatten_nested:
            transformed_data, flatten_rule = self._flatten_to_1nf(
                transformed_data,
                preserve_arrays=preserve_arrays
            )
            if flatten_rule:
                transformations_applied.append(flatten_rule)

        # Step 2: Apply registered transformation rules
        for rule_id in rule_ids:
            if rule_id in self.transformation_rules:
                rule = self.transformation_rules[rule_id]
                transformed_data = self.apply_transformation_rule(transformed_data, rule_id)
                transformations_applied.append(rule)
                logger.info(f"Applied transformation rule: {rule_id}")

        # Step 3: Validate 1NF compliance
        is_1nf_compliant = self.validate_1nf_compliance(transformed_data)

        if not is_1nf_compliant:
            logger.warning("Transformed data may not be fully 1NF compliant")

        # Create shaped data object
        shaped_data = ShapedData(
            processing_id=f"{canonical_data.processing_id}_shaped",
            canonical_id=canonical_data.processing_id,
            transformed_data=transformed_data,
            transformations_applied=transformations_applied,
            timestamp=datetime.now(),
            is_1nf_compliant=is_1nf_compliant,
            transformation_metadata={
                "flatten_nested": flatten_nested,
                "preserve_arrays": preserve_arrays,
                "rule_count": len(transformations_applied),
                "source_field_count": canonical_data.field_count
            }
        )

        logger.info(
            f"1NF transformation complete: {len(transformations_applied)} transformations applied, "
            f"1NF compliant: {is_1nf_compliant}"
        )

        return shaped_data

    def register_transformation_rule(
        self,
        rule: TransformationRule
    ) -> None:
        """
        Register a transformation rule.

        Args:
            rule: TransformationRule to register
        """
        self.transformation_rules[rule.rule_id] = rule
        logger.info(f"Registered transformation rule: {rule.rule_id}")

    def apply_transformation_rule(
        self,
        data: Dict[str, Any],
        rule_id: str
    ) -> Dict[str, Any]:
        """
        Apply a specific transformation rule to data.

        Args:
            data: Data to transform
            rule_id: ID of transformation rule to apply

        Returns:
            Transformed data

        Raises:
            KeyError: If rule_id not found in registered rules
        """
        if rule_id not in self.transformation_rules:
            raise KeyError(f"Transformation rule '{rule_id}' not found")

        rule = self.transformation_rules[rule_id]
        logger.info(f"Applying transformation rule: {rule_id} ({rule.rule_type})")

        # Apply transformation based on rule type
        try:
            if rule.rule_type == "normalize":
                return self._apply_normalization(data, rule.parameters)
            elif rule.rule_type == "aggregate":
                return self._apply_aggregation(data, rule.parameters)
            elif rule.rule_type == "filter":
                return self._apply_filter(data, rule.parameters)
            elif rule.rule_type == "rename":
                return self._apply_rename(data, rule.parameters)
            elif rule.rule_type == "custom":
                # Custom transformations via callable
                if rule.parameters and "transform_fn" in rule.parameters:
                    transform_fn = rule.parameters["transform_fn"]
                    return transform_fn(data)
                return data
            else:
                logger.warning(f"Unknown rule type: {rule.rule_type}, skipping")
                return data

        except Exception as e:
            logger.error(f"Error applying transformation rule {rule_id}: {e}")
            raise

    def validate_1nf_compliance(
        self,
        data: Dict[str, Any]
    ) -> bool:
        """
        Validate that data meets 1NF requirements.

        First Normal Form (1NF) requirements:
        1. Each table cell contains a single value (atomic values)
        2. Each record is unique
        3. Each column contains values of a single type

        Args:
            data: Data to validate

        Returns:
            True if data is 1NF compliant
        """
        logger.info("Validating 1NF compliance")

        try:
            # Check 1: All values must be atomic (no nested dicts or lists)
            if not self._check_atomic_values(data):
                logger.warning("1NF violation: Non-atomic values found")
                return False

            # Check 2: If data represents a table (list of dicts), check uniqueness
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                if not self._check_unique_records(data):
                    logger.warning("1NF violation: Duplicate records found")
                    return False

            # Check 3: Type consistency within columns (if data is tabular)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                if not self._check_type_consistency(data):
                    logger.warning("1NF violation: Inconsistent column types")
                    return False

            logger.info("1NF compliance validated successfully")
            return True

        except Exception as e:
            logger.error(f"Error validating 1NF compliance: {e}")
            return False

    def get_available_transformations(self) -> List[str]:
        """
        Get list of available transformation rules.

        Returns:
            List of transformation rule IDs
        """
        return list(self.transformation_rules.keys())

    def export_transformation_lineage(
        self,
        shaped_data: ShapedData
    ) -> Dict[str, Any]:
        """
        Export complete transformation lineage for governance.

        Args:
            shaped_data: ShapedData with transformation history

        Returns:
            Complete lineage information
        """
        return {
            "processing_id": shaped_data.processing_id,
            "canonical_id": shaped_data.canonical_id,
            "transformation_count": shaped_data.transformation_count,
            "transformations": [
                {
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    "description": rule.description,
                    "version": rule.version,
                    "applied_at": rule.created_at.isoformat(),
                    "parameters": rule.parameters
                }
                for rule in shaped_data.transformations_applied
            ],
            "1nf_compliant": shaped_data.is_1nf_compliant,
            "export_timestamp": datetime.now().isoformat()
        }

    # Helper methods for 1NF transformation and validation

    def _flatten_to_1nf(
        self,
        data: Dict[str, Any],
        preserve_arrays: bool = False,
        parent_key: str = "",
        sep: str = "_"
    ) -> tuple[Dict[str, Any], Optional[TransformationRule]]:
        """
        Flatten nested dictionary to 1NF-compliant structure.

        Args:
            data: Data to flatten
            preserve_arrays: If True, keep arrays as JSON strings
            parent_key: Parent key for recursion
            sep: Separator for flattened keys

        Returns:
            Tuple of (flattened_data, transformation_rule)
        """
        items = []

        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                # Recursively flatten nested dicts
                items.extend(
                    self._flatten_dict(v, new_key, sep).items()
                )
            elif isinstance(v, list) and not preserve_arrays:
                # Flatten lists by creating indexed keys
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            self._flatten_dict(item, f"{new_key}{sep}{i}", sep).items()
                        )
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))

        flattened_data = dict(items)

        # Create transformation rule
        rule = TransformationRule(
            rule_id="auto_flatten_1nf",
            rule_type="normalize",
            description="Automatic flattening to First Normal Form",
            version="1.0",
            parameters={"preserve_arrays": preserve_arrays, "separator": sep}
        )

        return flattened_data, rule

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "_"
    ) -> Dict[str, Any]:
        """
        Recursively flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for flattened keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _check_atomic_values(self, data: Union[Dict, List]) -> bool:
        """
        Check if all values in data are atomic (not nested structures).

        Args:
            data: Data to check

        Returns:
            True if all values are atomic
        """
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, (dict, list)):
                    # Allow empty lists/dicts as atomic
                    if value:
                        return False
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if not self._check_atomic_values(item):
                        return False
                elif isinstance(item, list):
                    return False

        return True

    def _check_unique_records(self, data: List[Dict[str, Any]]) -> bool:
        """
        Check if all records in a list are unique.

        Args:
            data: List of dictionaries representing records

        Returns:
            True if all records are unique
        """
        import json

        # Convert records to JSON strings for comparison
        record_strings = []
        for record in data:
            record_str = json.dumps(record, sort_keys=True)
            if record_str in record_strings:
                return False
            record_strings.append(record_str)

        return True

    def _check_type_consistency(self, data: List[Dict[str, Any]]) -> bool:
        """
        Check if column types are consistent across all records.

        Args:
            data: List of dictionaries representing records

        Returns:
            True if types are consistent within each column
        """
        if not data:
            return True

        # Collect types for each column
        column_types: Dict[str, set] = {}

        for record in data:
            for key, value in record.items():
                value_type = type(value).__name__
                if key not in column_types:
                    column_types[key] = set()
                column_types[key].add(value_type)

        # Check if any column has mixed types (excluding None/null)
        for column, types in column_types.items():
            # Allow None alongside other types
            types_without_none = types - {"NoneType"}
            if len(types_without_none) > 1:
                logger.warning(f"Column '{column}' has mixed types: {types}")
                return False

        return True

    # Transformation rule implementations

    def _apply_normalization(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply normalization transformation."""
        # Example: Convert all string values to lowercase
        if parameters and parameters.get("lowercase", False):
            result = {}
            for k, v in data.items():
                if isinstance(v, str):
                    result[k] = v.lower()
                else:
                    result[k] = v
            return result
        return data

    def _apply_aggregation(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply aggregation transformation."""
        # Example: Sum numeric fields
        if parameters and "sum_fields" in parameters:
            result = data.copy()
            sum_fields = parameters["sum_fields"]
            total = sum(data.get(field, 0) for field in sum_fields if isinstance(data.get(field), (int, float)))
            result["_aggregated_sum"] = total
            return result
        return data

    def _apply_filter(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply filter transformation."""
        # Example: Keep only specified fields
        if parameters and "keep_fields" in parameters:
            keep_fields = parameters["keep_fields"]
            return {k: v for k, v in data.items() if k in keep_fields}
        return data

    def _apply_rename(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply rename transformation."""
        # Example: Rename fields according to mapping
        if parameters and "field_mapping" in parameters:
            result = {}
            field_mapping = parameters["field_mapping"]
            for k, v in data.items():
                new_key = field_mapping.get(k, k)
                result[new_key] = v
            return result
        return data
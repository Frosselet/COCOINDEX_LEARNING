"""
Output management and formatting components.

This module handles the dual-output architecture with canonical truth layer
and shaped output layer, ensuring data integrity and transformation governance.
"""

from .canonical import CanonicalFormatter
from .shaped import ShapedFormatter
from .exporters import DataExporter, StreamingExporter

__all__ = [
    "CanonicalFormatter",
    "ShapedFormatter",
    "DataExporter",
    "StreamingExporter",
]
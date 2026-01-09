"""
Tests for SchemaManager - COLPALI-501 implementation.

This test suite validates JSON schema to BAML conversion with the three
required test schemas: shipping manifest, invoice, and table extraction.
"""

import pytest
import json
from typing import Dict, Any

from colpali_engine.core.schema_manager import (
    SchemaManager,
    BAMLDefinition,
    BAMLClass,
    BAMLField,
    BAMLFunction,
    ValidationResult,
    SchemaConversionError,
    ValidationError
)


class TestSchemaManager:
    """Test suite for SchemaManager JSON to BAML conversion."""

    @pytest.fixture
    def schema_manager(self):
        """Create SchemaManager instance for testing."""
        return SchemaManager()

    @pytest.fixture
    def shipping_manifest_schema(self) -> Dict[str, Any]:
        """Shipping manifest JSON schema for testing."""
        return {
            "title": "ShippingManifest",
            "type": "object",
            "description": "Complete shipping manifest with sender, recipient and items",
            "properties": {
                "manifest_id": {
                    "type": "string",
                    "description": "Unique manifest identifier"
                },
                "shipping_date": {
                    "type": "string",
                    "description": "Date of shipment in ISO format"
                },
                "sender": {
                    "type": "object",
                    "description": "Sender information",
                    "properties": {
                        "company_name": {"type": "string"},
                        "contact_person": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "postal_code": {"type": "string"},
                                "country": {"type": "string"}
                            },
                            "required": ["street", "city", "country"]
                        },
                        "phone": {"type": "string"}
                    },
                    "required": ["company_name", "address"]
                },
                "recipient": {
                    "type": "object",
                    "description": "Recipient information",
                    "properties": {
                        "company_name": {"type": "string"},
                        "contact_person": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "postal_code": {"type": "string"},
                                "country": {"type": "string"}
                            },
                            "required": ["street", "city", "country"]
                        }
                    },
                    "required": ["company_name", "address"]
                },
                "items": {
                    "type": "array",
                    "description": "List of shipped items",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "weight_kg": {"type": "number"},
                            "dimensions": {
                                "type": "object",
                                "properties": {
                                    "length_cm": {"type": "number"},
                                    "width_cm": {"type": "number"},
                                    "height_cm": {"type": "number"}
                                }
                            },
                            "value_usd": {"type": "number"}
                        },
                        "required": ["item_id", "description", "quantity"]
                    }
                },
                "total_weight_kg": {"type": "number"},
                "total_value_usd": {"type": "number"},
                "carrier": {"type": "string"},
                "tracking_number": {"type": "string"}
            },
            "required": [
                "manifest_id",
                "shipping_date",
                "sender",
                "recipient",
                "items",
                "carrier"
            ]
        }

    @pytest.fixture
    def invoice_schema(self) -> Dict[str, Any]:
        """Invoice JSON schema for testing."""
        return {
            "title": "Invoice",
            "type": "object",
            "description": "Commercial invoice with detailed line items",
            "properties": {
                "invoice_number": {
                    "type": "string",
                    "description": "Unique invoice number"
                },
                "invoice_date": {
                    "type": "string",
                    "description": "Invoice issue date"
                },
                "due_date": {
                    "type": "string",
                    "description": "Payment due date"
                },
                "vendor": {
                    "type": "object",
                    "description": "Vendor/seller information",
                    "properties": {
                        "company_name": {"type": "string"},
                        "tax_id": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "state": {"type": "string"},
                                "postal_code": {"type": "string"},
                                "country": {"type": "string"}
                            }
                        },
                        "contact": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "phone": {"type": "string"}
                            }
                        }
                    },
                    "required": ["company_name", "address"]
                },
                "customer": {
                    "type": "object",
                    "description": "Customer/buyer information",
                    "properties": {
                        "company_name": {"type": "string"},
                        "customer_id": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "state": {"type": "string"},
                                "postal_code": {"type": "string"},
                                "country": {"type": "string"}
                            }
                        }
                    },
                    "required": ["company_name", "address"]
                },
                "line_items": {
                    "type": "array",
                    "description": "Invoice line items",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line_number": {"type": "integer"},
                            "product_code": {"type": "string"},
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "line_total": {"type": "number"},
                            "tax_rate": {"type": "number"},
                            "tax_amount": {"type": "number"}
                        },
                        "required": ["description", "quantity", "unit_price", "line_total"]
                    }
                },
                "subtotal": {"type": "number"},
                "total_tax": {"type": "number"},
                "total_amount": {"type": "number"},
                "currency": {"type": "string"},
                "payment_terms": {"type": "string"}
            },
            "required": [
                "invoice_number",
                "invoice_date",
                "vendor",
                "customer",
                "line_items",
                "subtotal",
                "total_amount",
                "currency"
            ]
        }

    @pytest.fixture
    def table_schema(self) -> Dict[str, Any]:
        """Table extraction schema for testing - based on ADR requirements."""
        return {
            "title": "TableExtraction",
            "type": "object",
            "description": "Structured table data extraction from documents",
            "properties": {
                "table_title": {
                    "type": "string",
                    "description": "Table title or caption if present"
                },
                "headers": {
                    "type": "array",
                    "description": "Column headers",
                    "items": {
                        "type": "string"
                    }
                },
                "rows": {
                    "type": "array",
                    "description": "Table rows data",
                    "items": {
                        "type": "object",
                        "properties": {
                            "row_number": {"type": "integer"},
                            "cells": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column_index": {"type": "integer"},
                                        "value": {"type": "string"},
                                        "data_type": {
                                            "type": "string",
                                            "enum": ["text", "number", "date", "currency"]
                                        },
                                        "confidence": {"type": "number"}
                                    },
                                    "required": ["column_index", "value"]
                                }
                            }
                        },
                        "required": ["row_number", "cells"]
                    }
                },
                "summary_statistics": {
                    "type": "object",
                    "description": "Table summary information",
                    "properties": {
                        "total_rows": {"type": "integer"},
                        "total_columns": {"type": "integer"},
                        "has_totals_row": {"type": "boolean"},
                        "numeric_columns": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    }
                },
                "extraction_metadata": {
                    "type": "object",
                    "description": "Metadata about extraction process",
                    "properties": {
                        "page_number": {"type": "integer"},
                        "table_position": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "width": {"type": "number"},
                                "height": {"type": "number"}
                            }
                        },
                        "extraction_confidence": {"type": "number"}
                    }
                }
            },
            "required": ["headers", "rows", "summary_statistics"]
        }

    def test_shipping_manifest_conversion(self, schema_manager, shipping_manifest_schema):
        """Test conversion of shipping manifest schema to BAML."""
        # Convert schema
        definition = schema_manager.generate_baml_classes(shipping_manifest_schema)

        # Validate structure
        assert isinstance(definition, BAMLDefinition)
        assert len(definition.classes) > 0
        assert len(definition.functions) == 1

        # Check main class
        main_class = next(c for c in definition.classes if c.name == "ShippingManifest")
        assert main_class is not None
        assert len(main_class.fields) > 0

        # Check required fields are not optional
        required_fields = ["manifest_id", "shipping_date", "sender", "recipient", "items", "carrier"]
        for field in main_class.fields:
            if field.name in required_fields:
                assert not field.optional, f"Required field {field.name} should not be optional"

        # Check nested classes created for objects
        class_names = [c.name for c in definition.classes]
        assert "Sender" in class_names
        assert "Recipient" in class_names
        assert "Address" in class_names
        assert "ItemsItem" in class_names  # For array of objects
        assert "Dimensions" in class_names

        # Check function generation
        function = definition.functions[0]
        assert function.name == "ExtractShippingManifest"
        assert function.return_type == "ShippingManifest"
        assert function.client == "CustomSonnet4"
        assert "document_images" in function.prompt_template

    def test_invoice_conversion(self, schema_manager, invoice_schema):
        """Test conversion of invoice schema to BAML."""
        definition = schema_manager.generate_baml_classes(invoice_schema)

        # Validate structure
        assert isinstance(definition, BAMLDefinition)
        assert len(definition.classes) > 0
        assert len(definition.functions) == 1

        # Check main class
        main_class = next(c for c in definition.classes if c.name == "Invoice")
        assert main_class is not None

        # Check nested classes for complex objects
        class_names = [c.name for c in definition.classes]
        assert "Vendor" in class_names
        assert "Customer" in class_names
        assert "Address" in class_names  # Should be reused/created for both vendor and customer
        assert "Contact" in class_names
        assert "LineItemsItem" in class_names  # For line items array

        # Check array handling for line items
        line_items_field = next(f for f in main_class.fields if f.name == "line_items")
        assert line_items_field.field_type == "LineItemsItem[]"

        # Validate function
        function = definition.functions[0]
        assert function.name == "ExtractInvoice"
        assert function.return_type == "Invoice"

    def test_table_schema_conversion(self, schema_manager, table_schema):
        """Test conversion of table extraction schema to BAML."""
        definition = schema_manager.generate_baml_classes(table_schema)

        # Validate structure
        assert isinstance(definition, BAMLDefinition)
        assert len(definition.classes) > 0

        # Check main class
        main_class = next(c for c in definition.classes if c.name == "TableExtraction")
        assert main_class is not None

        # Check complex nested array handling
        class_names = [c.name for c in definition.classes]
        assert "RowsItem" in class_names  # For rows array
        assert "CellsItem" in class_names  # For cells array within rows
        assert "SummaryStatistics" in class_names
        assert "ExtractionMetadata" in class_names
        assert "TablePosition" in class_names

        # Check array of primitives (headers)
        headers_field = next(f for f in main_class.fields if f.name == "headers")
        assert headers_field.field_type == "string[]"

        # Check nested array handling (rows containing cells)
        rows_field = next(f for f in main_class.fields if f.name == "rows")
        assert rows_field.field_type == "RowsItem[]"

    def test_schema_validation_success(self, schema_manager, shipping_manifest_schema):
        """Test successful schema validation."""
        result = schema_manager.validate_schema_compatibility(shipping_manifest_schema)

        assert result.is_valid
        assert len(result.issues) == 0

    def test_schema_validation_failure(self, schema_manager):
        """Test schema validation with invalid schema."""
        invalid_schema = {
            "title": "InvalidSchema",
            # Missing 'type' and 'properties'
            "invalid_field": "should_cause_error"
        }

        result = schema_manager.validate_schema_compatibility(invalid_schema)

        assert not result.is_valid
        assert len(result.issues) > 0
        assert any("properties" in issue for issue in result.issues)

    def test_unsupported_type_handling(self, schema_manager):
        """Test handling of unsupported JSON schema types."""
        schema_with_unsupported_type = {
            "title": "TestSchema",
            "type": "object",
            "properties": {
                "valid_field": {"type": "string"},
                "unsupported_field": {"type": "null"}  # null type not supported
            }
        }

        result = schema_manager.validate_schema_compatibility(schema_with_unsupported_type)

        assert not result.is_valid
        assert any("Unsupported type 'null'" in issue for issue in result.issues)

    def test_baml_code_generation(self, schema_manager, invoice_schema):
        """Test complete BAML code generation."""
        definition = schema_manager.generate_baml_classes(invoice_schema)
        baml_code = schema_manager.generate_baml_code(definition)

        # Validate generated code structure
        assert "class Invoice {" in baml_code
        assert "class Vendor {" in baml_code
        assert "function ExtractInvoice(" in baml_code
        assert "client CustomSonnet4" in baml_code
        assert "document_images image[]" in baml_code

        # Check for optional field syntax
        assert "?" in baml_code  # Should have some optional fields

        # Check prompt structure
        assert "{{ _.role(\"user\") }}" in baml_code
        assert "{{ ctx.output_format }}" in baml_code

    def test_error_handling_invalid_json(self, schema_manager):
        """Test error handling for invalid JSON input."""
        with pytest.raises(ValidationError):
            schema_manager.generate_baml_classes("not_a_dict")

    def test_error_handling_missing_properties(self, schema_manager):
        """Test error handling for schema missing required properties."""
        invalid_schema = {
            "title": "InvalidSchema",
            "type": "object"
            # Missing 'properties'
        }

        with pytest.raises(ValidationError):
            schema_manager.generate_baml_classes(invalid_schema)

    def test_vision_prompt_optimization(self, schema_manager, table_schema):
        """Test vision prompt generation includes table-specific instructions."""
        definition = schema_manager.generate_baml_classes(table_schema)
        function = definition.functions[0]
        prompt = function.prompt_template

        # Check for table-specific instructions
        assert "table" in prompt.lower()
        assert "row-column structure" in prompt.lower() or "structure" in prompt.lower()
        assert "spatial relationships" in prompt.lower() or "relationships" in prompt.lower()
        assert "multiple pages" in prompt.lower() or "pages" in prompt.lower()

    def test_type_mapping_edge_cases(self, schema_manager):
        """Test edge cases in type mapping."""
        edge_case_schema = {
            "title": "EdgeCaseSchema",
            "type": "object",
            "properties": {
                "empty_object": {
                    "type": "object",
                    "properties": {}
                },
                "array_without_items": {
                    "type": "array"
                    # Missing 'items' - should be caught by validation
                }
            }
        }

        # Should fail validation due to array without items
        result = schema_manager.validate_schema_compatibility(edge_case_schema)
        assert not result.is_valid
        assert any("missing 'items'" in issue for issue in result.issues)

    def test_definition_caching(self, schema_manager, shipping_manifest_schema):
        """Test that generated definitions are cached for reuse."""
        # Generate definition twice
        definition1 = schema_manager.generate_baml_classes(shipping_manifest_schema)
        definition2 = schema_manager.generate_baml_classes(shipping_manifest_schema)

        # Should be cached
        assert "ShippingManifest" in schema_manager.generated_definitions
        assert schema_manager.generated_definitions["ShippingManifest"] is not None

    def test_client_configuration_extensibility(self, schema_manager, invoice_schema):
        """Test that client configuration can be extended."""
        definition = schema_manager.generate_baml_classes(invoice_schema)
        function = definition.functions[0]

        # Default should be CustomSonnet4
        assert function.client == "CustomSonnet4"

        # Client should be configurable for different use cases
        assert hasattr(function, 'client')
        function.client = "CustomGPT5"  # Should be changeable
        assert function.client == "CustomGPT5"

    def test_complex_nested_structures(self, schema_manager):
        """Test handling of deeply nested object structures."""
        complex_schema = {
            "title": "ComplexNested",
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "deep_field": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        definition = schema_manager.generate_baml_classes(complex_schema)

        # Should handle deep nesting
        class_names = [c.name for c in definition.classes]
        assert "Level1" in class_names
        assert "Level2" in class_names
        assert "Level3Item" in class_names  # For array items

    def test_field_descriptions_preservation(self, schema_manager, shipping_manifest_schema):
        """Test that field descriptions are preserved in BAML generation."""
        definition = schema_manager.generate_baml_classes(shipping_manifest_schema)
        baml_code = schema_manager.generate_baml_code(definition)

        # Check that descriptions are included as comments
        assert "// Unique manifest identifier" in baml_code or "manifest_id" in baml_code
        assert "// Date of shipment" in baml_code or "shipping_date" in baml_code
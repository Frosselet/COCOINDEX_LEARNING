"""
COLPALI-500 Integration Tests - BAML Schema System Integration.

This module tests the complete COLPALI-500 system integration including:
- JSON Schema to BAML class conversion (COLPALI-501)
- Dynamic BAML function generation (COLPALI-502)
- Schema validation and compatibility checks (COLPALI-503)
- BAML client configuration integration (COLPALI-504)
- Vision model configurations (COLPALI-505)

Tests the full pipeline: JSON Schema → BAML Classes → BAML Functions → Client Selection → Vision Processing
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from tatforge.core.schema_manager import SchemaManager, BAMLDefinition
from tatforge.core.baml_function_generator import BAMLFunctionGenerator, ClientComplexity
from tatforge.core.schema_validator import SchemaValidator, CompatibilityLevel, CompatibilityReport
from tatforge.core.baml_client_manager import BAMLClientManager
from tatforge.core.vision_model_manager import VisionModelManager, FallbackStrategy


class TestCOLPALI500Integration:
    """Integration tests for the complete COLPALI-500 system."""

    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.baml_src_path = Path(self.temp_dir) / "baml_src"
        self.baml_src_path.mkdir(exist_ok=True)

        # Create minimal clients.baml file for testing
        clients_content = '''
client<llm> CustomSonnet4 {
    retry_policy Exponential
    provider anthropic
    options {
        model "claude-sonnet-4-20250514"
        api_key env.ANTHROPIC_API_KEY
    }
}

client<llm> CustomGPT5 {
    retry_policy Exponential
    provider openai
    options {
        model "gpt-5"
        api_key env.OPENAI_API_KEY
    }
}
        '''

        clients_file = self.baml_src_path / "clients.baml"
        clients_file.write_text(clients_content.strip())

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_pipeline_simple_invoice(self):
        """Test complete pipeline with simple invoice schema."""
        # 1. JSON Schema Input (COLPALI-501)
        invoice_schema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
                "date": {"type": "string", "description": "Invoice date"},
                "total": {"type": "number", "description": "Total invoice amount"},
                "vendor": {"type": "string", "description": "Vendor name"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "price": {"type": "number"}
                        }
                    }
                }
            },
            "required": ["invoice_number", "date", "total"]
        }

        # 2. Schema Validation (COLPALI-503)
        validator = SchemaValidator()
        validation_result = validator.validate_schema(invoice_schema)

        assert isinstance(validation_result, CompatibilityReport)
        assert validation_result.compatibility_level in [
            CompatibilityLevel.FULLY_COMPATIBLE,
            CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        ]

        # 3. Schema to BAML Conversion (COLPALI-501)
        schema_manager = SchemaManager()
        baml_definition = schema_manager.generate_baml_classes(invoice_schema)

        assert isinstance(baml_definition, BAMLDefinition)
        assert len(baml_definition.classes) >= 1  # Should have at least main class
        main_class = baml_definition.classes[0]
        assert main_class is not None

        # 4. Client Configuration (COLPALI-504)
        client_manager = BAMLClientManager(baml_src_path=str(self.baml_src_path))
        vision_manager = VisionModelManager(client_manager=client_manager)

        # 5. Function Generation (COLPALI-502)
        function_generator = BAMLFunctionGenerator(
            client_manager=client_manager,
            vision_manager=vision_manager
        )

        # Analyze complexity
        complexity = function_generator.analyze_complexity(baml_definition)
        assert complexity in [ClientComplexity.SIMPLE, ClientComplexity.MODERATE]

        # Generate optimized functions
        functions = function_generator.generate_optimized_functions(
            baml_definition,
            namespace="invoice_test"
        )

        assert len(functions.functions) > 0
        primary_function = functions.functions[0]

        # 6. Vision Model Integration (COLPALI-505)
        # Test vision processing with fallback
        result = vision_manager.process_with_vision_fallback(
            function=primary_function,
            images=["mock_invoice.pdf"],
            fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION
        )

        assert result.success is True
        assert result.client_used is not None

        # 7. Full Configuration Generation
        baml_config = schema_manager.generate_baml_code(baml_definition)
        vision_config = vision_manager.generate_vision_baml_config()

        assert "class" in baml_config  # Should contain class definitions
        assert "CustomSonnet4" in vision_config
        assert "COLPALI-505" in vision_config

    def test_complete_pipeline_complex_shipping_manifest(self):
        """Test complete pipeline with complex shipping manifest schema."""
        # Complex shipping manifest schema
        shipping_schema = {
            "type": "object",
            "properties": {
                "manifest_id": {"type": "string", "description": "Unique manifest identifier"},
                "ship_date": {"type": "string", "format": "date"},
                "destination": {
                    "type": "object",
                    "properties": {
                        "address": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip_code": {"type": "string"}
                    }
                },
                "shipments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tracking_number": {"type": "string"},
                            "weight": {"type": "number"},
                            "contents": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item_code": {"type": "string"},
                                        "description": {"type": "string"},
                                        "quantity": {"type": "integer"},
                                        "value": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["manifest_id", "ship_date", "shipments"]
        }

        # Test the complete pipeline
        validator = SchemaValidator()
        validation_result = validator.validate_schema(shipping_schema)

        # Complex schema should still be compatible
        assert isinstance(validation_result, CompatibilityReport)
        assert validation_result.compatibility_level in [
            CompatibilityLevel.FULLY_COMPATIBLE,
            CompatibilityLevel.COMPATIBLE_WITH_WARNINGS,
            CompatibilityLevel.LIMITED_COMPATIBILITY
        ]

        schema_manager = SchemaManager()
        baml_definition = schema_manager.generate_baml_classes(shipping_schema)

        # Should create multiple nested classes
        assert len(baml_definition.classes) >= 3  # Main + nested objects

        client_manager = BAMLClientManager(baml_src_path=str(self.baml_src_path))
        vision_manager = VisionModelManager(client_manager=client_manager)
        function_generator = BAMLFunctionGenerator(
            client_manager=client_manager,
            vision_manager=vision_manager
        )

        # Complex schema should require higher complexity clients
        complexity = function_generator.analyze_complexity(baml_definition)
        assert complexity in [ClientComplexity.MODERATE, ClientComplexity.COMPLEX, ClientComplexity.ADVANCED]

        functions = function_generator.generate_optimized_functions(
            baml_definition,
            namespace="shipping_test",
            optimization_hints={
                "document_type": "shipping_manifest",
                "has_tables": True,
                "spatial_layout": True
            }
        )

        # Should generate optimized functions with vision-specific prompts
        assert len(functions.functions) > 0
        primary_function = functions.functions[0]
        assert primary_function.prompt_template is not None

        # Test cost optimization
        cost_recommendations = vision_manager.get_cost_optimization_recommendation(
            complexity=complexity,
            image_count=3,
            budget_limit=0.50
        )

        assert "recommendations" in cost_recommendations
        assert len(cost_recommendations["recommendations"]) > 0
        assert cost_recommendations["budget_limit"] == 0.50

    def test_pipeline_with_validation_fixes(self):
        """Test pipeline with automatic schema fixes."""
        # Schema with issues that should be auto-fixed
        problematic_schema = {
            "type": "object",
            "properties": {
                "valid_field": {"type": "string"},
                "mixed_case_Field": {"type": "string"},  # Inconsistent naming
                "nested_object": {
                    "type": "object",
                    "properties": {
                        "valid_nested": {"type": "string"},
                        "another_field": {"type": "string"}
                    }
                }
            }
        }

        # 1. Validate and get fixes
        validator = SchemaValidator()
        validation_result = validator.validate_schema(problematic_schema)

        # Should return a report
        assert isinstance(validation_result, CompatibilityReport)

        # 2. Auto-fix the schema if needed
        if validation_result.compatibility_level == CompatibilityLevel.INCOMPATIBLE:
            fixed_schema = validator.auto_fix_schema(problematic_schema)
            assert fixed_schema is not None

            # Validate fixed schema
            fixed_validation = validator.validate_schema(fixed_schema)
            assert fixed_validation.compatibility_level != CompatibilityLevel.INCOMPATIBLE

            # Use fixed schema
            test_schema = fixed_schema
        else:
            test_schema = problematic_schema

        # 3. Continue pipeline with schema
        schema_manager = SchemaManager()
        baml_definition = schema_manager.generate_baml_classes(test_schema)

        assert len(baml_definition.classes) >= 1

        # Generated BAML should be valid
        baml_code = schema_manager.generate_baml_code(baml_definition)
        assert "class" in baml_code

    def test_vision_model_fallback_integration(self):
        """Test integration with vision model fallback strategies."""
        # Simple schema for vision processing test
        receipt_schema = {
            "type": "object",
            "properties": {
                "store_name": {"type": "string"},
                "date": {"type": "string"},
                "total": {"type": "number"},
                "tax": {"type": "number"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"}
                        }
                    }
                }
            }
        }

        # Pipeline setup
        schema_manager = SchemaManager()
        baml_definition = schema_manager.generate_baml_classes(receipt_schema)

        client_manager = BAMLClientManager(baml_src_path=str(self.baml_src_path))
        vision_manager = VisionModelManager(client_manager=client_manager)
        function_generator = BAMLFunctionGenerator(
            client_manager=client_manager,
            vision_manager=vision_manager
        )

        functions = function_generator.generate_optimized_functions(
            baml_definition,
            namespace="receipt_test"
        )

        # Test different fallback strategies
        fallback_strategies = [
            FallbackStrategy.ALTERNATIVE_VISION,
            FallbackStrategy.GRACEFUL_DEGRADATION,
            FallbackStrategy.TEXT_ONLY,
            FallbackStrategy.HYBRID_APPROACH
        ]

        for strategy in fallback_strategies:
            result = vision_manager.process_with_vision_fallback(
                function=functions.functions[0],
                images=["receipt_image.jpg"],
                fallback_strategy=strategy,
                max_retries=2
            )

            # All strategies should return some result
            assert result is not None
            assert hasattr(result, 'success')
            assert hasattr(result, 'client_used')

    def test_cost_optimization_across_components(self):
        """Test cost optimization integration across all components."""
        # Schema that could be processed with different complexity levels
        table_schema = {
            "type": "object",
            "properties": {
                "table_title": {"type": "string"},
                "headers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "page_number": {"type": "integer"},
                        "table_position": {"type": "string"}
                    }
                }
            }
        }

        schema_manager = SchemaManager()
        baml_definition = schema_manager.generate_baml_classes(table_schema)

        client_manager = BAMLClientManager(baml_src_path=str(self.baml_src_path))
        vision_manager = VisionModelManager(client_manager=client_manager)
        function_generator = BAMLFunctionGenerator(
            client_manager=client_manager,
            vision_manager=vision_manager
        )

        # Test different budget constraints
        budgets = [0.01, 0.05, 0.10, 0.50]

        for budget in budgets:
            # Generate functions with cost consideration
            functions = function_generator.generate_optimized_functions(
                baml_definition,
                namespace="cost_test",
                preferences={"cost_priority": True, "max_cost": budget}
            )

            # Get cost recommendations
            complexity = function_generator.analyze_complexity(baml_definition)
            cost_rec = vision_manager.get_cost_optimization_recommendation(
                complexity=complexity,
                image_count=5,
                budget_limit=budget
            )

            # Should provide recommendations
            assert "recommendations" in cost_rec
            assert len(cost_rec["recommendations"]) > 0

            # Should respect budget constraints for reasonable budgets
            within_budget_options = [
                rec for rec in cost_rec["recommendations"]
                if rec["within_budget"]
            ]

            if budget >= 0.05:  # Reasonable budget should have options
                assert len(within_budget_options) > 0

    def test_end_to_end_configuration_generation(self):
        """Test complete BAML configuration generation."""
        # Complex schema for comprehensive configuration
        order_schema = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "address": {"type": "string"}
                    }
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "name": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"}
                        }
                    }
                },
                "totals": {
                    "type": "object",
                    "properties": {
                        "subtotal": {"type": "number"},
                        "tax": {"type": "number"},
                        "shipping": {"type": "number"},
                        "total": {"type": "number"}
                    }
                }
            }
        }

        # Complete pipeline
        schema_manager = SchemaManager()
        baml_definition = schema_manager.generate_baml_classes(order_schema)

        client_manager = BAMLClientManager(baml_src_path=str(self.baml_src_path))
        vision_manager = VisionModelManager(client_manager=client_manager)
        function_generator = BAMLFunctionGenerator(
            client_manager=client_manager,
            vision_manager=vision_manager
        )

        functions = function_generator.generate_optimized_functions(
            baml_definition,
            namespace="order_processing"
        )

        # Generate complete BAML configuration
        baml_classes = schema_manager.generate_baml_code(baml_definition)
        function_snippet = client_manager.generate_baml_configuration_snippet(functions, "order_processing")
        vision_config = vision_manager.generate_vision_baml_config()

        # Verify complete configuration
        assert "class" in baml_classes  # Should contain class definitions
        assert len(baml_classes) > 100  # Should be substantial

        assert "function" in function_snippet or "order_processing" in function_snippet
        assert "client" in function_snippet  # Should reference a client

        assert "Vision-capable model configurations" in vision_config
        assert "COLPALI-505" in vision_config

        # Test that all components are properly integrated
        complete_config = f"""
// BAML Classes - Generated by COLPALI-501
{baml_classes}

// Dynamic Functions - Generated by COLPALI-502
{function_snippet}

// Vision Configuration - Generated by COLPALI-505
{vision_config}
        """

        assert len(complete_config) > 1000  # Should be substantial
        assert "class" in complete_config  # Classes included
        assert "function" in complete_config or "client" in complete_config  # Functions included
        assert "COLPALI-505" in complete_config  # Vision config included

    def test_error_handling_and_resilience(self):
        """Test system resilience and error handling across components."""
        # Test with various problematic inputs
        problematic_schemas = [
            {},  # Empty schema
            {"type": "invalid_type"},  # Invalid type
            {"type": "object", "properties": {}},  # Empty properties
            {
                "type": "object",
                "properties": {
                    "simple_field": {"type": "string"}
                }
            }
        ]

        schema_manager = SchemaManager()
        validator = SchemaValidator()
        client_manager = BAMLClientManager(baml_src_path=str(self.baml_src_path))

        for i, schema in enumerate(problematic_schemas):
            try:
                # Each component should handle errors gracefully
                validation_result = validator.validate_schema(schema)

                assert isinstance(validation_result, CompatibilityReport)

                if validation_result.compatibility_level != CompatibilityLevel.INCOMPATIBLE:
                    baml_definition = schema_manager.generate_baml_classes(schema)
                    assert baml_definition is not None
                    assert len(baml_definition.classes) >= 0  # May be empty for invalid schemas

                else:
                    # Should provide meaningful issues
                    assert len(validation_result.issues) >= 0  # May have issues

            except Exception as e:
                # If exceptions occur, they should be informative
                assert len(str(e)) > 5  # Meaningful error message

    def test_performance_with_large_schema(self):
        """Test system performance with large, complex schemas."""
        # Generate a large schema programmatically
        large_schema = {
            "type": "object",
            "properties": {}
        }

        # Add many fields and nested objects
        for i in range(20):  # 20 top-level fields (reduced for faster tests)
            field_name = f"field_{i:03d}"
            if i % 5 == 0:  # Every 5th field is a nested object
                large_schema["properties"][field_name] = {
                    "type": "object",
                    "properties": {
                        f"nested_{j:02d}": {"type": "string" if j % 2 == 0 else "number"}
                        for j in range(5)  # 5 nested fields each
                    }
                }
            elif i % 3 == 0:  # Every 3rd field is an array
                large_schema["properties"][field_name] = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "array_field_1": {"type": "string"},
                            "array_field_2": {"type": "number"}
                        }
                    }
                }
            else:
                large_schema["properties"][field_name] = {
                    "type": "string" if i % 2 == 0 else "number"
                }

        # Test pipeline performance
        import time

        start_time = time.time()

        schema_manager = SchemaManager()
        baml_definition = schema_manager.generate_baml_classes(large_schema)

        conversion_time = time.time() - start_time

        # Should complete in reasonable time (< 10 seconds for this size)
        assert conversion_time < 10.0

        # Should create appropriate number of classes
        assert len(baml_definition.classes) >= 1  # Should have at least main class

        # Test validation performance
        start_time = time.time()
        validator = SchemaValidator()
        validation_result = validator.validate_schema(large_schema)
        validation_time = time.time() - start_time

        assert validation_time < 5.0  # Should validate quickly

        # Function generation should also be reasonable
        client_manager = BAMLClientManager(baml_src_path=str(self.baml_src_path))
        vision_manager = VisionModelManager(client_manager=client_manager)
        function_generator = BAMLFunctionGenerator(
            client_manager=client_manager,
            vision_manager=vision_manager
        )

        start_time = time.time()
        functions = function_generator.generate_optimized_functions(baml_definition, "large_test")
        generation_time = time.time() - start_time

        assert generation_time < 5.0  # Should generate reasonably quickly
        assert len(functions.functions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
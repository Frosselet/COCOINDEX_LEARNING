"""
Tests for BAML Function Generator - COLPALI-502 implementation.

This test suite validates dynamic BAML function generation with vision-optimized
prompts and intelligent client selection based on schema complexity.
"""

import pytest
from typing import Dict, Any

from tatforge.core.baml_function_generator import (
    BAMLFunctionGenerator,
    ClientComplexity,
    PromptOptimization,
    ClientConfiguration,
    PromptTemplate,
    FunctionGenerationError
)
from tatforge.core.schema_manager import (
    SchemaManager,
    BAMLDefinition,
    BAMLClass,
    BAMLField,
    BAMLFunction
)


class TestBAMLFunctionGenerator:
    """Test suite for BAMLFunctionGenerator optimization system."""

    @pytest.fixture
    def function_generator(self):
        """Create BAMLFunctionGenerator instance for testing."""
        return BAMLFunctionGenerator()

    @pytest.fixture
    def schema_manager(self):
        """Create SchemaManager instance for generating test data."""
        return SchemaManager()

    @pytest.fixture
    def simple_definition(self, schema_manager) -> BAMLDefinition:
        """Simple schema definition for basic testing."""
        simple_schema = {
            "title": "SimpleDocument",
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "author": {"type": "string", "description": "Document author"},
                "date": {"type": "string", "description": "Publication date"}
            },
            "required": ["title", "author"]
        }
        return schema_manager.generate_baml_classes(simple_schema)

    @pytest.fixture
    def complex_table_definition(self, schema_manager) -> BAMLDefinition:
        """Complex table schema for testing advanced optimization."""
        table_schema = {
            "title": "ComplexTable",
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
                                        "data_type": {"type": "string"},
                                        "position": {
                                            "type": "object",
                                            "properties": {
                                                "x": {"type": "number"},
                                                "y": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return schema_manager.generate_baml_classes(table_schema)

    @pytest.fixture
    def invoice_definition(self, schema_manager) -> BAMLDefinition:
        """Invoice schema for testing financial document optimization."""
        invoice_schema = {
            "title": "Invoice",
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "vendor": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "string"},
                        "total_amount": {"type": "number"}
                    }
                },
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "total": {"type": "number"}
                        }
                    }
                }
            }
        }
        return schema_manager.generate_baml_classes(invoice_schema)

    def test_complexity_analysis_simple(self, function_generator, simple_definition):
        """Test complexity analysis for simple schemas."""
        complexity = function_generator._analyze_schema_complexity(simple_definition)
        assert complexity == ClientComplexity.SIMPLE

    def test_complexity_analysis_complex_table(self, function_generator, complex_table_definition):
        """Test complexity analysis for complex table schemas."""
        complexity = function_generator._analyze_schema_complexity(complex_table_definition)
        assert complexity in [ClientComplexity.COMPLEX, ClientComplexity.ADVANCED]

    def test_complexity_analysis_invoice(self, function_generator, invoice_definition):
        """Test complexity analysis for invoice schemas."""
        complexity = function_generator._analyze_schema_complexity(invoice_definition)
        assert complexity in [ClientComplexity.MODERATE, ClientComplexity.COMPLEX]

    def test_client_selection_simple(self, function_generator):
        """Test client selection for simple complexity."""
        config = function_generator._select_client_configuration(ClientComplexity.SIMPLE)

        assert config.primary_client == "CustomHaiku"
        assert config.fallback_client == "CustomSonnet4"
        assert config.retry_policy == "Constant"
        assert config.max_tokens == 2000
        assert "simple" in config.reasoning.lower()

    def test_client_selection_moderate(self, function_generator):
        """Test client selection for moderate complexity."""
        config = function_generator._select_client_configuration(ClientComplexity.MODERATE)

        assert config.primary_client == "CustomSonnet4"
        assert config.fallback_client == "CustomOpus4"
        assert config.retry_policy == "Exponential"
        assert config.max_tokens == 4000

    def test_client_selection_complex(self, function_generator):
        """Test client selection for complex schemas."""
        config = function_generator._select_client_configuration(ClientComplexity.COMPLEX)

        assert config.primary_client == "CustomOpus4"
        assert config.fallback_client == "CustomGPT5"
        assert config.max_tokens == 8000

    def test_client_selection_advanced(self, function_generator):
        """Test client selection for advanced complexity."""
        config = function_generator._select_client_configuration(ClientComplexity.ADVANCED)

        assert config.primary_client == "CustomGPT5"
        assert config.fallback_client == "CustomOpus4"
        assert config.max_tokens == 12000
        assert config.temperature == 0.05

    def test_client_selection_with_preference(self, function_generator):
        """Test client selection with user preference."""
        hints = {"preferred_client": "CustomOpus4"}
        config = function_generator._select_client_configuration(ClientComplexity.SIMPLE, hints)

        # Should respect user preference despite simple complexity
        assert "CustomOpus4" in config.primary_client
        assert "preference" in config.reasoning.lower()

    def test_optimization_strategy_table_detection(self, function_generator, complex_table_definition):
        """Test automatic detection of table optimization strategy."""
        strategy = function_generator._determine_optimization_strategy(complex_table_definition)
        assert strategy == PromptOptimization.TABLE_FOCUSED

    def test_optimization_strategy_invoice_detection(self, function_generator, invoice_definition):
        """Test automatic detection of invoice optimization strategy."""
        strategy = function_generator._determine_optimization_strategy(invoice_definition)
        assert strategy == PromptOptimization.INVOICE_FOCUSED

    def test_optimization_strategy_explicit_hint(self, function_generator, simple_definition):
        """Test explicit optimization strategy from hints."""
        hints = {"optimization_strategy": "spatial"}
        strategy = function_generator._determine_optimization_strategy(simple_definition, hints)
        assert strategy == PromptOptimization.SPATIAL_FOCUSED

    def test_optimization_strategy_general_fallback(self, function_generator, simple_definition):
        """Test fallback to general optimization strategy."""
        strategy = function_generator._determine_optimization_strategy(simple_definition)
        assert strategy == PromptOptimization.GENERAL

    def test_generate_optimized_functions_simple(self, function_generator, simple_definition):
        """Test function generation for simple schemas."""
        functions = function_generator.generate_optimized_functions(simple_definition)

        assert len(functions) == 1
        function = functions[0]

        # Should use efficient client for simple task
        assert function.client in ["CustomHaiku", "CustomSonnet4"]

        # Should have optimized prompt
        assert "{{ _.role(\"user\") }}" in function.prompt_template
        assert "{{ document_images }}" in function.prompt_template
        assert "{{ ctx.output_format }}" in function.prompt_template
        assert "title" in function.prompt_template.lower()
        assert "author" in function.prompt_template.lower()

    def test_generate_optimized_functions_table(self, function_generator, complex_table_definition):
        """Test function generation for complex table schemas."""
        functions = function_generator.generate_optimized_functions(complex_table_definition)

        assert len(functions) == 1
        function = functions[0]

        # Should use powerful client for complex table
        assert function.client in ["CustomOpus4", "CustomGPT5"]

        # Should have table-specific optimizations
        prompt = function.prompt_template.lower()
        assert "table" in prompt
        assert "row" in prompt or "column" in prompt
        assert "structure" in prompt

    def test_generate_optimized_functions_invoice(self, function_generator, invoice_definition):
        """Test function generation for invoice schemas."""
        functions = function_generator.generate_optimized_functions(invoice_definition)

        assert len(functions) == 1
        function = functions[0]

        # Should have invoice-specific optimizations
        prompt = function.prompt_template.lower()
        assert "vendor" in prompt or "invoice" in prompt
        assert "total" in prompt or "amount" in prompt

    def test_generate_optimized_functions_with_hints(self, function_generator, simple_definition):
        """Test function generation with optimization hints."""
        hints = {
            "preferred_client": "CustomGPT5",
            "optimization_strategy": "table"
        }

        functions = function_generator.generate_optimized_functions(simple_definition, hints)
        function = functions[0]

        # Should respect hints
        assert function.client == "CustomGPT5"
        prompt = function.prompt_template.lower()
        assert "table" in prompt

    def test_function_validation_valid(self, function_generator):
        """Test validation of valid BAML function."""
        valid_function = BAMLFunction(
            name="ExtractTest",
            input_params=[{"document_images": "image[]"}],
            return_type="TestClass",
            client="CustomSonnet4",
            prompt_template='''{{ _.role("user") }}
Extract data with precision.

Target fields:
- test_field: Test description

Instructions:
1. Analyze document images
2. Extract visible information

Document images:
{{ document_images }}

Return the extracted data:
{{ ctx.output_format }}''',
            description="Test extraction function"
        )

        result = function_generator.validate_function(valid_function)

        assert result["is_valid"] == True
        assert len(result["issues"]) == 0
        assert result["optimization_score"] > 80.0

    def test_function_validation_invalid(self, function_generator):
        """Test validation of invalid BAML function."""
        invalid_function = BAMLFunction(
            name="",  # Missing name
            input_params=[],  # Missing input params
            return_type="",  # Missing return type
            client="InvalidClient",  # Invalid client
            prompt_template="Bad prompt",  # Missing required elements
            description=""
        )

        result = function_generator.validate_function(invalid_function)

        assert result["is_valid"] == False
        assert len(result["issues"]) > 0
        assert result["optimization_score"] < 50.0

    def test_function_validation_warnings(self, function_generator):
        """Test validation warnings for suboptimal functions."""
        warning_function = BAMLFunction(
            name="ExtractTest",
            input_params=[{"document_images": "image[]"}],
            return_type="TestClass",
            client="CustomSonnet4",
            prompt_template="Short prompt without required elements",
            description="Test function"
        )

        result = function_generator.validate_function(warning_function)

        assert result["is_valid"] == True  # Structurally valid
        assert len(result["warnings"]) > 0  # But has optimization issues
        assert result["optimization_score"] < 80.0

    def test_field_instructions_generation(self, function_generator):
        """Test generation of field-specific instructions."""
        test_class = BAMLClass(
            name="TestClass",
            fields=[
                BAMLField(name="name", field_type="string", description="Person name"),
                BAMLField(name="items", field_type="Item[]", description="List of items"),
                BAMLField(name="total_amount", field_type="float", description="Total cost"),
                BAMLField(name="birth_date", field_type="string", description="Date of birth")
            ]
        )

        instructions = function_generator._generate_field_instructions(
            test_class, PromptOptimization.GENERAL
        )

        assert len(instructions) == 4
        assert any("Person name" in instr for instr in instructions)
        assert any("list" in instr.lower() for instr in instructions)  # Array hint
        assert any("numeric" in instr.lower() for instr in instructions)  # Number hint
        assert any("date" in instr.lower() for instr in instructions)  # Date hint

    def test_table_specific_instructions(self, function_generator):
        """Test generation of table-specific instruction sections."""
        test_class = BAMLClass(
            name="TableData",
            fields=[
                BAMLField(name="rows", field_type="Row[]"),
                BAMLField(name="cells", field_type="Cell[]")
            ]
        )

        instructions = function_generator._generate_strategy_sections(
            test_class, PromptOptimization.TABLE_FOCUSED
        )

        table_instructions = [instr for instr in instructions if "table" in instr.lower()]
        assert len(table_instructions) > 0

        # Should have specific table handling instructions
        assert any("structure" in instr.lower() for instr in instructions)
        assert any("row" in instr.lower() or "column" in instr.lower() for instr in instructions)

    def test_invoice_specific_instructions(self, function_generator):
        """Test generation of invoice-specific instruction sections."""
        test_class = BAMLClass(
            name="Invoice",
            fields=[
                BAMLField(name="vendor", field_type="Vendor"),
                BAMLField(name="total_amount", field_type="float")
            ]
        )

        instructions = function_generator._generate_strategy_sections(
            test_class, PromptOptimization.INVOICE_FOCUSED
        )

        # Should have invoice-specific instructions
        assert any("vendor" in instr.lower() for instr in instructions)
        assert any("total" in instr.lower() or "calculate" in instr.lower() for instr in instructions)

    def test_spatial_specific_instructions(self, function_generator):
        """Test generation of spatial-specific instruction sections."""
        instructions = function_generator._generate_strategy_sections(
            None, PromptOptimization.SPATIAL_FOCUSED
        )

        # Should have spatial relationship instructions
        assert any("spatial" in instr.lower() for instr in instructions)
        assert any("position" in instr.lower() for instr in instructions)

    def test_error_handling(self, function_generator):
        """Test error handling for invalid inputs."""
        # Test with None input
        with pytest.raises(FunctionGenerationError):
            function_generator.generate_optimized_functions(None)

        # Test with empty definition
        empty_definition = BAMLDefinition(classes=[], functions=[])
        with pytest.raises(FunctionGenerationError):
            function_generator.generate_optimized_functions(empty_definition)

    def test_optimization_score_calculation(self, function_generator):
        """Test optimization score calculation logic."""
        # High-quality function
        good_function = BAMLFunction(
            name="ExtractData",
            input_params=[{"document_images": "image[]"}],
            return_type="Data",
            client="CustomSonnet4",
            prompt_template='''{{ _.role("user") }}
Extract structured data with high precision.

Target fields:
- field1: Description

Instructions:
1. Analyze document structure
2. Maintain table relationships

Document images:
{{ document_images }}

Return accurate data:
{{ ctx.output_format }}''',
            description="Extract data"
        )

        validation = function_generator.validate_function(good_function)
        good_score = validation["optimization_score"]

        # Poor-quality function
        bad_function = BAMLFunction(
            name="Bad",
            input_params=[{"images": "image[]"}],
            return_type="Data",
            client="CustomSonnet4",
            prompt_template="Do something",
            description=""
        )

        validation = function_generator.validate_function(bad_function)
        bad_score = validation["optimization_score"]

        assert good_score > bad_score
        assert good_score >= 85.0
        assert bad_score < 50.0

    def test_client_configuration_initialization(self, function_generator):
        """Test that client configurations are properly initialized."""
        configs = function_generator.client_configurations

        required_clients = ["CustomHaiku", "CustomSonnet4", "CustomOpus4", "CustomGPT5"]
        for client in required_clients:
            assert client in configs
            config = configs[client]
            assert isinstance(config, ClientConfiguration)
            assert config.primary_client == client
            assert config.reasoning is not None

    def test_prompt_template_initialization(self, function_generator):
        """Test that prompt templates are properly initialized."""
        templates = function_generator.prompt_templates

        required_strategies = [
            PromptOptimization.GENERAL,
            PromptOptimization.TABLE_FOCUSED,
            PromptOptimization.INVOICE_FOCUSED,
            PromptOptimization.FORM_FOCUSED,
            PromptOptimization.SPATIAL_FOCUSED
        ]

        for strategy in required_strategies:
            assert strategy in templates
            template = templates[strategy]
            assert isinstance(template, PromptTemplate)
            assert template.base_template is not None
            assert template.optimization_strategy == strategy

    def test_complexity_threshold_edge_cases(self, function_generator):
        """Test complexity analysis at threshold boundaries."""
        # Create definitions at complexity boundaries

        # Exactly at simple/moderate boundary
        moderate_classes = []
        for i in range(3):  # Just above simple threshold (2 classes)
            moderate_classes.append(BAMLClass(
                name=f"Class{i}",
                fields=[
                    BAMLField(name="field1", field_type="string"),
                    BAMLField(name="nested", field_type="NestedClass")  # Adds complexity
                ]
            ))

        moderate_definition = BAMLDefinition(
            classes=moderate_classes,
            functions=[BAMLFunction(
                name="Extract", input_params=[], return_type="Class0",
                client="test", prompt_template="test"
            )]
        )

        complexity = function_generator._analyze_schema_complexity(moderate_definition)
        assert complexity in [ClientComplexity.MODERATE, ClientComplexity.COMPLEX]

    def test_end_to_end_optimization_pipeline(self, function_generator, schema_manager):
        """Test complete end-to-end optimization pipeline."""
        # Create a realistic complex schema
        complex_schema = {
            "title": "FinancialReport",
            "type": "object",
            "properties": {
                "report_header": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "period": {"type": "string"},
                        "prepared_by": {"type": "string"}
                    }
                },
                "financial_tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string"},
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "account": {"type": "string"},
                                        "current_period": {"type": "number"},
                                        "previous_period": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                },
                "summary_totals": {
                    "type": "object",
                    "properties": {
                        "total_assets": {"type": "number"},
                        "total_liabilities": {"type": "number"},
                        "net_worth": {"type": "number"}
                    }
                }
            }
        }

        # Generate schema definition
        definition = schema_manager.generate_baml_classes(complex_schema)

        # Generate optimized functions
        optimized_functions = function_generator.generate_optimized_functions(definition)

        # Validate results
        assert len(optimized_functions) == 1
        function = optimized_functions[0]

        # Should select appropriate client for complex financial document
        assert function.client in ["CustomOpus4", "CustomGPT5"]

        # Should have comprehensive prompt
        prompt = function.prompt_template
        assert "{{ _.role(\"user\") }}" in prompt
        assert "financial" in prompt.lower() or "table" in prompt.lower()
        assert "{{ document_images }}" in prompt
        assert "{{ ctx.output_format }}" in prompt

        # Validate the optimized function
        validation = function_generator.validate_function(function)
        assert validation["is_valid"] == True
        assert validation["optimization_score"] >= 75.0  # Should be well optimized
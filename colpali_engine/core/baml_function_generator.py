"""
BAML Function Generation - COLPALI-502 implementation.

This module generates dynamic BAML extraction functions with vision-optimized
prompts and intelligent client selection based on schema complexity.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .schema_manager import BAMLDefinition, BAMLClass, BAMLFunction

logger = logging.getLogger(__name__)


class ClientComplexity(Enum):
    """Schema complexity levels for client selection."""
    SIMPLE = "simple"      # Basic text extraction
    MODERATE = "moderate"  # Nested objects, few arrays
    COMPLEX = "complex"    # Deep nesting, many arrays, tables
    ADVANCED = "advanced"  # Multi-page, spatial relationships, complex tables


class PromptOptimization(Enum):
    """Prompt optimization strategies."""
    GENERAL = "general"           # General document extraction
    TABLE_FOCUSED = "table"       # Table-specific instructions
    FORM_FOCUSED = "form"         # Form field extraction
    INVOICE_FOCUSED = "invoice"   # Invoice/financial documents
    SPATIAL_FOCUSED = "spatial"   # Spatial relationship preservation


@dataclass
class ClientConfiguration:
    """BAML client configuration for extraction tasks."""
    primary_client: str
    fallback_client: Optional[str] = None
    retry_policy: str = "Exponential"
    max_tokens: Optional[int] = None
    temperature: float = 0.1
    reasoning: str = ""


@dataclass
class PromptTemplate:
    """Vision-optimized prompt template."""
    base_template: str
    field_instructions: List[str]
    optimization_strategy: PromptOptimization
    table_specific_instructions: Optional[str] = None
    multi_page_handling: Optional[str] = None
    quality_requirements: Optional[str] = None


class BAMLFunctionGenerator:
    """
    Generates dynamic BAML extraction functions with optimized prompts and client selection.

    This class takes BAMLDefinition objects from the schema manager and creates
    sophisticated extraction functions tailored to the specific document type
    and complexity level.
    """

    def __init__(self):
        self.client_configurations = self._initialize_client_configs()
        self.prompt_templates = self._initialize_prompt_templates()
        self.complexity_thresholds = self._initialize_complexity_thresholds()

        logger.info("BAMLFunctionGenerator initialized")

    def generate_optimized_functions(
        self,
        definition: BAMLDefinition,
        optimization_hints: Optional[Dict[str, Any]] = None
    ) -> List[BAMLFunction]:
        """
        Generate optimized BAML functions for the given schema definition.

        Args:
            definition: BAMLDefinition from schema manager
            optimization_hints: Optional hints for optimization strategy

        Returns:
            List of optimized BAMLFunction objects
        """
        try:
            # Validate input
            if definition is None:
                raise FunctionGenerationError("Definition cannot be None")

            if not definition.classes:
                raise FunctionGenerationError("Definition must contain at least one class")

            if not definition.functions:
                raise FunctionGenerationError("Definition must contain at least one function")

            logger.info(f"Generating optimized functions for {len(definition.classes)} classes")

            # Analyze schema complexity
            complexity = self._analyze_schema_complexity(definition)
            logger.info(f"Detected complexity level: {complexity.value}")

            # Select optimal client configuration
            client_config = self._select_client_configuration(complexity, optimization_hints)
            logger.info(f"Selected client: {client_config.primary_client} ({client_config.reasoning})")

            # Determine optimization strategy
            optimization = self._determine_optimization_strategy(definition, optimization_hints)
            logger.info(f"Using optimization strategy: {optimization.value}")

            # Generate optimized functions
            functions = []
            for i, function in enumerate(definition.functions):
                optimized_function = self._optimize_function(
                    function,
                    definition.classes,
                    client_config,
                    optimization,
                    optimization_hints
                )
                functions.append(optimized_function)
                logger.debug(f"Optimized function {i+1}: {optimized_function.name}")

            logger.info(f"Generated {len(functions)} optimized functions")
            return functions

        except Exception as e:
            logger.error(f"Function generation failed: {e}")
            raise FunctionGenerationError(f"Failed to generate optimized functions: {e}")

    def validate_function(self, function: BAMLFunction) -> Dict[str, Any]:
        """
        Validate generated BAML function for correctness and optimization.

        Args:
            function: BAMLFunction to validate

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "optimization_score": 0.0,
            "recommendations": []
        }

        try:
            # Validate function structure
            self._validate_function_structure(function, validation_results)

            # Validate prompt quality
            self._validate_prompt_quality(function, validation_results)

            # Validate client selection
            self._validate_client_selection(function, validation_results)

            # Calculate optimization score
            validation_results["optimization_score"] = self._calculate_optimization_score(
                function, validation_results
            )

            logger.debug(f"Function validation complete: {validation_results['optimization_score']:.2f}")

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation error: {e}")

        return validation_results

    def _analyze_schema_complexity(self, definition: BAMLDefinition) -> ClientComplexity:
        """Analyze schema complexity to determine optimal client selection."""
        complexity_score = 0

        # Count classes and nesting depth
        total_classes = len(definition.classes)
        max_nesting_depth = 0
        total_fields = 0
        array_fields = 0
        object_fields = 0

        for baml_class in definition.classes:
            total_fields += len(baml_class.fields)

            for field in baml_class.fields:
                # Count arrays
                if "[]" in field.field_type:
                    array_fields += 1
                    complexity_score += 2  # Arrays add complexity

                # Count nested objects
                if field.field_type in [c.name for c in definition.classes]:
                    object_fields += 1
                    complexity_score += 1  # Nested objects add complexity

                # Detect table-like structures
                if "row" in field.name.lower() or "cell" in field.name.lower():
                    complexity_score += 3  # Table structures are complex

        # Schema complexity analysis - adjusted thresholds
        if total_classes <= 2 and complexity_score < 3:
            return ClientComplexity.SIMPLE
        elif total_classes <= 4 and complexity_score < 10:
            return ClientComplexity.MODERATE
        elif total_classes <= 8 and complexity_score < 20:
            return ClientComplexity.COMPLEX
        else:
            return ClientComplexity.ADVANCED

    def _select_client_configuration(
        self,
        complexity: ClientComplexity,
        hints: Optional[Dict[str, Any]] = None
    ) -> ClientConfiguration:
        """Select optimal BAML client based on complexity and hints."""

        # Check for explicit client preference in hints
        if hints and "preferred_client" in hints:
            preferred = hints["preferred_client"]
            config = ClientConfiguration(
                primary_client=preferred,
                fallback_client="CustomSonnet4",
                retry_policy="Exponential",
                max_tokens=4000,
                temperature=0.1,
                reasoning=f"User specified preference: {preferred}"
            )
            return config

        # Select based on complexity
        if complexity == ClientComplexity.SIMPLE:
            config = ClientConfiguration(
                primary_client="CustomHaiku",
                fallback_client="CustomSonnet4",
                retry_policy="Constant",
                max_tokens=2000,
                temperature=0.0,
                reasoning="Simple schema - fast, cost-effective client"
            )
        elif complexity == ClientComplexity.MODERATE:
            config = ClientConfiguration(
                primary_client="CustomSonnet4",
                fallback_client="CustomOpus4",
                retry_policy="Exponential",
                max_tokens=4000,
                temperature=0.1,
                reasoning="Moderate complexity - balanced performance and accuracy"
            )
        elif complexity == ClientComplexity.COMPLEX:
            config = ClientConfiguration(
                primary_client="CustomOpus4",
                fallback_client="CustomGPT5",
                retry_policy="Exponential",
                max_tokens=8000,
                temperature=0.1,
                reasoning="Complex schema - high-accuracy model required"
            )
        else:  # ADVANCED
            config = ClientConfiguration(
                primary_client="CustomGPT5",
                fallback_client="CustomOpus4",
                retry_policy="Exponential",
                max_tokens=12000,
                temperature=0.05,
                reasoning="Advanced complexity - maximum capability model"
            )

        return config

    def _determine_optimization_strategy(
        self,
        definition: BAMLDefinition,
        hints: Optional[Dict[str, Any]] = None
    ) -> PromptOptimization:
        """Determine the best prompt optimization strategy."""

        # Check explicit strategy in hints
        if hints and "optimization_strategy" in hints:
            strategy = hints["optimization_strategy"]
            strategy_map = {
                "general": PromptOptimization.GENERAL,
                "table": PromptOptimization.TABLE_FOCUSED,
                "invoice": PromptOptimization.INVOICE_FOCUSED,
                "form": PromptOptimization.FORM_FOCUSED,
                "spatial": PromptOptimization.SPATIAL_FOCUSED
            }
            if strategy.lower() in strategy_map:
                return strategy_map[strategy.lower()]

        # Analyze schema patterns to determine strategy
        schema_indicators = {
            "table": 0,
            "invoice": 0,
            "form": 0,
            "spatial": 0
        }

        # Examine class and field names for patterns
        for baml_class in definition.classes:
            class_name_lower = baml_class.name.lower()

            # Table indicators
            if any(indicator in class_name_lower for indicator in ["table", "row", "cell", "column"]):
                schema_indicators["table"] += 3

            # Invoice indicators
            if any(indicator in class_name_lower for indicator in ["invoice", "vendor", "customer", "line", "total"]):
                schema_indicators["invoice"] += 2

            # Form indicators
            if any(indicator in class_name_lower for indicator in ["form", "field", "input", "checkbox"]):
                schema_indicators["form"] += 2

            # Spatial indicators
            if any(indicator in class_name_lower for indicator in ["position", "coordinate", "spatial", "location"]):
                schema_indicators["spatial"] += 2

            # Check field names
            for field in baml_class.fields:
                field_name_lower = field.name.lower()

                if any(indicator in field_name_lower for indicator in ["row", "cell", "header", "column"]):
                    schema_indicators["table"] += 1

                if any(indicator in field_name_lower for indicator in ["amount", "total", "price", "quantity"]):
                    schema_indicators["invoice"] += 1

                if any(indicator in field_name_lower for indicator in ["position", "x", "y", "width", "height"]):
                    schema_indicators["spatial"] += 1

        # Select strategy with highest score
        max_score = max(schema_indicators.values())
        if max_score >= 3:
            for strategy, score in schema_indicators.items():
                if score == max_score:
                    if strategy == "table":
                        return PromptOptimization.TABLE_FOCUSED
                    elif strategy == "invoice":
                        return PromptOptimization.INVOICE_FOCUSED
                    elif strategy == "form":
                        return PromptOptimization.FORM_FOCUSED
                    elif strategy == "spatial":
                        return PromptOptimization.SPATIAL_FOCUSED

        return PromptOptimization.GENERAL

    def _optimize_function(
        self,
        original_function: BAMLFunction,
        classes: List[BAMLClass],
        client_config: ClientConfiguration,
        optimization: PromptOptimization,
        hints: Optional[Dict[str, Any]] = None
    ) -> BAMLFunction:
        """Optimize a single BAML function with advanced prompting and client selection."""

        # Generate optimized prompt
        optimized_prompt = self._generate_optimized_prompt(
            original_function, classes, optimization, hints
        )

        # Create optimized function
        optimized_function = BAMLFunction(
            name=original_function.name,
            input_params=original_function.input_params,
            return_type=original_function.return_type,
            client=client_config.primary_client,
            prompt_template=optimized_prompt,
            description=original_function.description
        )

        return optimized_function

    def _generate_optimized_prompt(
        self,
        function: BAMLFunction,
        classes: List[BAMLClass],
        optimization: PromptOptimization,
        hints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate vision-optimized prompt based on strategy."""

        # Get base template for optimization strategy
        base_template = self.prompt_templates[optimization]

        # Find the return class for detailed field analysis
        return_class = None
        for cls in classes:
            if cls.name == function.return_type:
                return_class = cls
                break

        # Generate field-specific instructions
        field_instructions = self._generate_field_instructions(return_class, optimization)

        # Generate strategy-specific sections
        strategy_sections = self._generate_strategy_sections(return_class, optimization, hints)

        # Combine into complete prompt
        prompt_parts = [
            "{{ _.role(\"user\") }}",
            base_template.base_template,
            "",
            "Target fields:"
        ]

        # Add field instructions
        for instruction in field_instructions:
            prompt_parts.append(f"- {instruction}")

        prompt_parts.extend([
            "",
            "Instructions:"
        ])

        # Add strategy-specific instructions
        for i, section in enumerate(strategy_sections, 1):
            prompt_parts.append(f"{i}. {section}")

        # Add standard closing
        prompt_parts.extend([
            "",
            "Document images:",
            "{{ document_images }}",
            "",
            f"Return the extracted data in the specified format:",
            "{{ ctx.output_format }}"
        ])

        return "\n".join(prompt_parts)

    def _generate_field_instructions(
        self,
        baml_class: Optional[BAMLClass],
        optimization: PromptOptimization
    ) -> List[str]:
        """Generate field-specific extraction instructions."""
        if not baml_class:
            return ["Extract all visible data with high precision"]

        instructions = []
        for field in baml_class.fields:
            instruction = field.name

            if field.description:
                instruction += f": {field.description}"

            # Add type-specific hints
            if "[]" in field.field_type:
                instruction += " (extract all items in list/table format)"
            elif field.field_type.lower() in ["float", "number"]:
                instruction += " (extract numeric value only)"
            elif "date" in field.name.lower():
                instruction += " (preserve date format as shown)"

            # Add optimization-specific hints
            if optimization == PromptOptimization.TABLE_FOCUSED:
                if "row" in field.name.lower() or "cell" in field.name.lower():
                    instruction += " (maintain exact row-column structure)"

            instructions.append(instruction)

        return instructions

    def _generate_strategy_sections(
        self,
        baml_class: Optional[BAMLClass],
        optimization: PromptOptimization,
        hints: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate strategy-specific instruction sections."""
        base_instructions = [
            "Analyze all provided document images carefully",
            "Extract only visible, clearly readable information",
            "Preserve original formatting and structure where possible"
        ]

        if optimization == PromptOptimization.TABLE_FOCUSED:
            base_instructions.extend([
                "Maintain exact table structure with proper row-column alignment",
                "Preserve header relationships and data organization",
                "For merged cells, repeat values appropriately",
                "Extract table metadata (totals, summaries) separately"
            ])

        elif optimization == PromptOptimization.INVOICE_FOCUSED:
            base_instructions.extend([
                "Identify vendor and customer information accurately",
                "Calculate totals and verify arithmetic consistency",
                "Preserve line item details and quantities precisely",
                "Extract payment terms and due dates clearly"
            ])

        elif optimization == PromptOptimization.SPATIAL_FOCUSED:
            base_instructions.extend([
                "Preserve spatial relationships between document elements",
                "Maintain relative positioning information",
                "Note layout structure and visual hierarchies",
                "Extract coordinate information if available"
            ])

        elif optimization == PromptOptimization.FORM_FOCUSED:
            base_instructions.extend([
                "Identify form field labels and corresponding values",
                "Preserve checkbox and selection states",
                "Extract form structure and field relationships",
                "Note required vs optional field indicators"
            ])

        else:  # GENERAL
            base_instructions.extend([
                "If data spans multiple pages, combine appropriately",
                "For complex layouts, maintain logical reading order",
                "Preserve hierarchical relationships between data elements"
            ])

        return base_instructions

    def _initialize_client_configs(self) -> Dict[str, ClientConfiguration]:
        """Initialize default client configurations."""
        return {
            "CustomHaiku": ClientConfiguration(
                primary_client="CustomHaiku",
                fallback_client="CustomSonnet4",
                retry_policy="Constant",
                max_tokens=2000,
                temperature=0.0,
                reasoning="Fast, cost-effective for simple tasks"
            ),
            "CustomSonnet4": ClientConfiguration(
                primary_client="CustomSonnet4",
                fallback_client="CustomOpus4",
                retry_policy="Exponential",
                max_tokens=4000,
                temperature=0.1,
                reasoning="Balanced performance and accuracy"
            ),
            "CustomOpus4": ClientConfiguration(
                primary_client="CustomOpus4",
                fallback_client="CustomGPT5",
                retry_policy="Exponential",
                max_tokens=8000,
                temperature=0.1,
                reasoning="High accuracy for complex documents"
            ),
            "CustomGPT5": ClientConfiguration(
                primary_client="CustomGPT5",
                fallback_client="CustomOpus4",
                retry_policy="Exponential",
                max_tokens=12000,
                temperature=0.05,
                reasoning="Maximum capability for advanced tasks"
            )
        }

    def _initialize_prompt_templates(self) -> Dict[PromptOptimization, PromptTemplate]:
        """Initialize prompt templates for different optimization strategies."""
        return {
            PromptOptimization.GENERAL: PromptTemplate(
                base_template="Extract structured data from these document images with high precision.",
                field_instructions=[],
                optimization_strategy=PromptOptimization.GENERAL
            ),
            PromptOptimization.TABLE_FOCUSED: PromptTemplate(
                base_template="Extract structured table data from these document images with precise row-column alignment.",
                field_instructions=[],
                optimization_strategy=PromptOptimization.TABLE_FOCUSED,
                table_specific_instructions="Maintain exact table structure and relationships"
            ),
            PromptOptimization.INVOICE_FOCUSED: PromptTemplate(
                base_template="Extract financial and invoice data from these documents with accurate calculations.",
                field_instructions=[],
                optimization_strategy=PromptOptimization.INVOICE_FOCUSED
            ),
            PromptOptimization.FORM_FOCUSED: PromptTemplate(
                base_template="Extract form field data and selections with precise label-value mapping.",
                field_instructions=[],
                optimization_strategy=PromptOptimization.FORM_FOCUSED
            ),
            PromptOptimization.SPATIAL_FOCUSED: PromptTemplate(
                base_template="Extract data while preserving spatial layout and positional relationships.",
                field_instructions=[],
                optimization_strategy=PromptOptimization.SPATIAL_FOCUSED
            )
        }

    def _initialize_complexity_thresholds(self) -> Dict[str, int]:
        """Initialize complexity analysis thresholds."""
        return {
            "simple_max_classes": 2,
            "simple_max_score": 5,
            "moderate_max_classes": 5,
            "moderate_max_score": 15,
            "complex_max_classes": 10,
            "complex_max_score": 30
        }

    def _validate_function_structure(self, function: BAMLFunction, results: Dict[str, Any]) -> None:
        """Validate function structure and parameters."""
        if not function.name:
            results["issues"].append("Function name is required")
            results["is_valid"] = False

        if not function.input_params:
            results["issues"].append("Function must have input parameters")
            results["is_valid"] = False

        if not function.return_type:
            results["issues"].append("Function must have return type")
            results["is_valid"] = False

        if not function.client:
            results["issues"].append("Function must specify client")
            results["is_valid"] = False

        if not function.prompt_template:
            results["issues"].append("Function must have prompt template")
            results["is_valid"] = False

    def _validate_prompt_quality(self, function: BAMLFunction, results: Dict[str, Any]) -> None:
        """Validate prompt template quality and completeness."""
        prompt = function.prompt_template

        required_elements = [
            ("{{ _.role(\"user\") }}", "User role specification"),
            ("{{ document_images }}", "Document images placeholder"),
            ("{{ ctx.output_format }}", "Output format placeholder")
        ]

        missing_critical_elements = 0
        for element, description in required_elements:
            if element not in prompt:
                results["warnings"].append(f"Missing {description}: {element}")
                missing_critical_elements += 1

        # Check for good prompt practices
        if len(prompt.split("\n")) < 5:
            results["warnings"].append("Prompt may be too brief for optimal results")

        if "extract" not in prompt.lower():
            results["warnings"].append("Prompt should explicitly mention 'extract' or similar action")

        # Check for minimal prompt content - this is critical
        if len(prompt) < 20:
            results["issues"].append("Prompt is too short to be effective")

        # If missing all critical elements, make it an issue
        if missing_critical_elements >= 3:
            results["issues"].append("Missing all critical template elements")

    def _validate_client_selection(self, function: BAMLFunction, results: Dict[str, Any]) -> None:
        """Validate client selection appropriateness."""
        valid_clients = [
            "CustomHaiku", "CustomSonnet4", "CustomOpus4", "CustomGPT5",
            "CustomGPT5Mini", "CustomGPT5Chat", "CustomFast", "OpenaiFallback"
        ]

        if function.client not in valid_clients:
            results["issues"].append(f"Invalid client '{function.client}'. Valid options: {valid_clients}")
            results["is_valid"] = False

    def _calculate_optimization_score(self, function: BAMLFunction, results: Dict[str, Any]) -> float:
        """Calculate optimization score for the function."""
        score = 100.0

        # Deduct points for issues and warnings
        score -= len(results["issues"]) * 20
        score -= len(results["warnings"]) * 5

        # Bonus points for optimization features
        prompt = function.prompt_template.lower()

        if "target fields" in prompt:
            score += 5
        if "instructions" in prompt:
            score += 5
        if any(word in prompt for word in ["table", "structure", "relationship"]):
            score += 5
        if "precision" in prompt or "accurate" in prompt:
            score += 5

        return max(0.0, min(100.0, score))


# Custom exceptions
class FunctionGenerationError(Exception):
    """Raised when function generation fails."""
    pass
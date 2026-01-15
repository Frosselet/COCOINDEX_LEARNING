"""
Schema management and BAML code generation.

This module handles the conversion of JSON schemas into BAML class definitions
and extraction functions, enabling dynamic type-safe extraction with automatic
code generation for diverse document structures.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from jinja2 import Template

logger = logging.getLogger(__name__)


class BAMLType(Enum):
    """BAML-supported data types."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class BAMLField:
    """Represents a field in a BAML class."""
    name: str
    field_type: str
    optional: bool = False
    description: Optional[str] = None


@dataclass
class BAMLClass:
    """Represents a BAML class definition."""
    name: str
    fields: List[BAMLField]
    description: Optional[str] = None


@dataclass
class BAMLFunction:
    """Represents a BAML extraction function."""
    name: str
    input_params: List[Dict[str, str]]
    return_type: str
    client: str
    prompt_template: str
    description: Optional[str] = None


@dataclass
class BAMLDefinition:
    """Complete BAML definition with classes and functions."""
    classes: List[BAMLClass]
    functions: List[BAMLFunction]
    imports: List[str] = None
    version: str = "0.215.2"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]


class SchemaManager:
    """
    Manages conversion of JSON schemas to BAML definitions.

    This class handles the dynamic generation of BAML classes and functions
    from JSON schemas, enabling type-safe extraction with vision-optimized
    prompting strategies.
    """

    def __init__(self):
        self.baml_templates = self._initialize_templates()
        self.type_mapping = self._initialize_type_mapping()
        self.generated_definitions: Dict[str, BAMLDefinition] = {}
        self.class_registry: Dict[str, BAMLClass] = {}  # Track generated classes to avoid duplicates

        logger.info("SchemaManager initialized")

    def generate_baml_classes(self, schema_json: Dict) -> BAMLDefinition:
        """
        Convert JSON schema to BAML class definitions.

        Args:
            schema_json: JSON schema dictionary

        Returns:
            BAMLDefinition with generated classes and functions

        Raises:
            SchemaConversionError: If conversion fails
            ValidationError: If schema is invalid
        """
        try:
            # Validate input schema
            self._validate_json_schema(schema_json)

            # Clear class registry for new conversion
            self.class_registry = {}

            # Extract schema information
            schema_name = schema_json.get("title", "ExtractedData")
            schema_properties = schema_json.get("properties", {})
            required_fields = schema_json.get("required", [])

            logger.info(f"Converting schema '{schema_name}' with {len(schema_properties)} properties")

            # Generate BAML classes
            classes = self._convert_properties_to_classes(
                schema_name,
                schema_properties,
                required_fields
            )

            # Generate extraction function
            functions = self._generate_extraction_functions(schema_name, classes[0])

            # Create complete definition
            definition = BAMLDefinition(
                classes=classes,
                functions=functions
            )

            # Cache the definition
            self.generated_definitions[schema_name] = definition

            logger.info(f"Successfully generated BAML definition for '{schema_name}'")
            return definition

        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"BAML generation failed: {e}")
            raise SchemaConversionError(f"Failed to convert schema: {e}")

    def generate_baml_code(self, definition: BAMLDefinition) -> str:
        """
        Generate BAML code from definition.

        Args:
            definition: BAMLDefinition object

        Returns:
            Complete BAML code as string
        """
        code_parts = []

        # Generate imports (if any)
        if definition.imports:
            for import_stmt in definition.imports:
                code_parts.append(f"import {import_stmt}")
            code_parts.append("")

        # Generate classes
        for baml_class in definition.classes:
            class_code = self._generate_class_code(baml_class)
            code_parts.append(class_code)
            code_parts.append("")

        # Generate functions
        for baml_function in definition.functions:
            function_code = self._generate_function_code(baml_function)
            code_parts.append(function_code)
            code_parts.append("")

        return "\n".join(code_parts)

    def validate_schema_compatibility(self, schema_json: Dict) -> ValidationResult:
        """
        Validate JSON schema compatibility with BAML.

        Args:
            schema_json: JSON schema to validate

        Returns:
            ValidationResult with compatibility information
        """
        issues = []
        warnings = []

        try:
            # Check schema structure
            if not isinstance(schema_json, dict):
                issues.append("Schema must be a dictionary")

            if "type" not in schema_json:
                warnings.append("Schema missing 'type' field")

            if "properties" not in schema_json:
                issues.append("Schema must have 'properties' field")

            # Check property types
            properties = schema_json.get("properties", {})
            for prop_name, prop_def in properties.items():
                prop_type = prop_def.get("type")

                if prop_type not in self.type_mapping:
                    issues.append(f"Unsupported type '{prop_type}' for property '{prop_name}'")

                # Check nested objects
                if prop_type == "object" and "properties" not in prop_def:
                    warnings.append(f"Object property '{prop_name}' has no nested properties")

                # Check arrays
                if prop_type == "array" and "items" not in prop_def:
                    issues.append(f"Array property '{prop_name}' missing 'items' definition")

        except Exception as e:
            issues.append(f"Schema validation error: {e}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )

    def _initialize_templates(self) -> Dict[str, Template]:
        """Initialize Jinja2 templates for BAML code generation."""
        class_template = Template("""
{%- if description %}
// {{ description }}
{%- endif %}
class {{ name }} {
{%- for field in fields %}
    {{ field.name }} {{ field.field_type }}{{ '?' if field.optional else '' }}{% if field.description %} // {{ field.description }}{% endif %}
{%- endfor %}
}""".strip())

        function_template = Template("""
{%- if description %}
// {{ description }}
{%- endif %}
function {{ name }}({{ input_params | join(', ') }}) -> {{ return_type }} {
    client {{ client }}
    prompt #"
        {{ prompt_template | indent(8) }}
    "#
}""".strip())

        return {
            "class": class_template,
            "function": function_template
        }

    def _initialize_type_mapping(self) -> Dict[str, str]:
        """Initialize mapping from JSON Schema types to BAML types."""
        return {
            "string": "string",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "array",
            "object": "object"
        }

    def _validate_json_schema(self, schema: Any) -> None:
        """Validate JSON schema structure."""
        if not isinstance(schema, dict):
            raise ValidationError("Schema must be a dictionary")

        if "properties" not in schema:
            raise ValidationError("Schema must have 'properties' field")

        # Validate properties structure
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValidationError("Schema 'properties' field must be a dictionary")

        # Validate each property has a valid type
        for prop_name, prop_def in properties.items():
            if not isinstance(prop_def, dict):
                raise ValidationError(f"Property '{prop_name}' definition must be a dictionary")

            prop_type = prop_def.get("type")
            if prop_type and prop_type not in self.type_mapping:
                # Allow this to pass validation but log as warning
                logger.warning(f"Property '{prop_name}' has unsupported type '{prop_type}', will default to string")

        # Additional validation can be added here
        pass

    def _convert_properties_to_classes(
        self,
        class_name: str,
        properties: Dict,
        required_fields: List[str]
    ) -> List[BAMLClass]:
        """Convert JSON schema properties to BAML classes."""
        classes = []
        main_fields = []

        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "string")
            is_required = prop_name in required_fields
            description = prop_def.get("description")

            # Handle nested objects
            if prop_type == "object":
                nested_properties = prop_def.get("properties", {})
                nested_required = prop_def.get("required", [])
                nested_class_name = self._format_class_name(prop_name)

                # Check if this class structure already exists
                class_key = self._generate_class_key(nested_class_name, nested_properties)

                if class_key in self.class_registry:
                    # Reuse existing class
                    field_type = self.class_registry[class_key].name
                    logger.debug(f"Reusing existing class '{field_type}' for property '{prop_name}'")
                else:
                    # Create new nested classes recursively
                    nested_classes = self._convert_properties_to_classes(
                        nested_class_name,
                        nested_properties,
                        nested_required
                    )

                    # Register the main nested class
                    if nested_classes:
                        self.class_registry[class_key] = nested_classes[0]

                    # Add ALL nested classes (main + any deeply nested ones)
                    classes.extend(nested_classes)

                    field_type = nested_class_name

                # Reference the nested class in the current class
                main_fields.append(BAMLField(
                    name=prop_name,
                    field_type=field_type,
                    optional=not is_required,
                    description=description
                ))

            # Handle arrays
            elif prop_type == "array":
                items_def = prop_def.get("items", {})
                items_type = items_def.get("type", "string")

                if items_type == "object":
                    # Array of objects
                    item_class_name = f"{self._format_class_name(prop_name)}Item"
                    item_properties = items_def.get("properties", {})
                    item_required = items_def.get("required", [])

                    # Check if this item class structure already exists
                    item_class_key = self._generate_class_key(item_class_name, item_properties)

                    if item_class_key in self.class_registry:
                        # Reuse existing item class
                        field_type = f"{self.class_registry[item_class_key].name}[]"
                    else:
                        # Create item classes recursively
                        item_classes = self._convert_properties_to_classes(
                            item_class_name,
                            item_properties,
                            item_required
                        )

                        # Register the main item class
                        if item_classes:
                            self.class_registry[item_class_key] = item_classes[0]

                        # Add ALL item classes (main + any nested ones)
                        classes.extend(item_classes)

                        field_type = f"{item_class_name}[]"
                else:
                    # Array of primitives
                    baml_type = self.type_mapping.get(items_type, "string")
                    field_type = f"{baml_type}[]"

                main_fields.append(BAMLField(
                    name=prop_name,
                    field_type=field_type,
                    optional=not is_required,
                    description=description
                ))

            # Handle primitive types
            else:
                baml_type = self.type_mapping.get(prop_type, "string")
                main_fields.append(BAMLField(
                    name=prop_name,
                    field_type=baml_type,
                    optional=not is_required,
                    description=description
                ))

        # Create main class
        main_class = BAMLClass(
            name=class_name,
            fields=main_fields,
            description=f"Generated from JSON schema for {class_name}"
        )

        # Return main class first, then all nested classes
        return [main_class] + classes

    def _generate_class_key(self, class_name: str, properties: Dict) -> str:
        """Generate a unique key for a class based on its structure."""
        # Create a signature based on property names and types
        prop_sig = []
        for prop_name, prop_def in sorted(properties.items()):
            prop_type = prop_def.get("type", "string")
            prop_sig.append(f"{prop_name}:{prop_type}")

        return f"{class_name}|{','.join(prop_sig)}"

    def _format_class_name(self, prop_name: str) -> str:
        """Format property name to valid class name."""
        # Convert snake_case or kebab-case to PascalCase
        # Examples: line_items -> LineItems, table_data -> TableData
        parts = prop_name.replace('-', '_').split('_')
        return ''.join(part.capitalize() for part in parts)

    def _generate_extraction_functions(
        self,
        schema_name: str,
        main_class: BAMLClass
    ) -> List[BAMLFunction]:
        """Generate BAML extraction functions for the schema."""
        function_name = f"Extract{schema_name}"

        # Generate vision-optimized prompt
        prompt_template = self._generate_vision_prompt(main_class)

        extraction_function = BAMLFunction(
            name=function_name,
            input_params=[{"document_images": "image[]"}],
            return_type=main_class.name,
            client="CustomSonnet4",  # Default to Claude Sonnet 4
            prompt_template=prompt_template,
            description=f"Extract {schema_name} data from document images"
        )

        return [extraction_function]

    def _generate_vision_prompt(self, baml_class: BAMLClass) -> str:
        """Generate vision-optimized prompt for document extraction."""
        field_descriptions = []
        for field in baml_class.fields:
            desc = f"- {field.name}"
            if field.description:
                desc += f": {field.description}"
            field_descriptions.append(desc)

        prompt = f"""{{{{ _.role("user") }}}}
Extract structured data from these document images with high precision.

Target fields:
{chr(10).join(field_descriptions)}

Instructions:
1. Analyze all provided document images carefully
2. Preserve spatial relationships and visual context
3. For tables, maintain row-column structure
4. If data spans multiple pages, combine appropriately
5. Extract only visible, clearly readable information
6. Return structured data matching the {baml_class.name} schema

Document images:
{{{{ document_images }}}}

Return the extracted data in the specified format:
{{{{ ctx.output_format }}}}"""

        return prompt

    def _generate_class_code(self, baml_class: BAMLClass) -> str:
        """Generate BAML code for a class definition."""
        return self.baml_templates["class"].render(
            name=baml_class.name,
            fields=baml_class.fields,
            description=baml_class.description
        )

    def _generate_function_code(self, baml_function: BAMLFunction) -> str:
        """Generate BAML code for a function definition."""
        input_params = [f"{name} {type_}" for param in baml_function.input_params
                       for name, type_ in param.items()]

        return self.baml_templates["function"].render(
            name=baml_function.name,
            input_params=input_params,
            return_type=baml_function.return_type,
            client=baml_function.client,
            prompt_template=baml_function.prompt_template,
            description=baml_function.description
        )


# Custom exceptions
class SchemaConversionError(Exception):
    """Raised when schema conversion fails."""
    pass


class ValidationError(Exception):
    """Raised when schema validation fails."""
    pass
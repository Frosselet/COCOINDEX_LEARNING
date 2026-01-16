"""
CocoIndex extraction flow for BAML-based document extraction.

FLOW 2: Query -> Qdrant Search -> Retrieved Pages -> BAML -> Structured Output

Uses @cocoindex.op.function(cache=True) for efficient cached extraction.

This follows the official pattern from:
https://github.com/cocoindex-io/cocoindex/blob/main/examples/patient_intake_extraction_baml/main.py
"""

import os
import base64
import logging

import cocoindex

logger = logging.getLogger(__name__)


@cocoindex.op.function(cache=True, behavior_version=1)
async def extract_with_baml(page_image: bytes, extraction_prompt: str) -> dict:
    """
    Extract structured data from a page image using BAML.

    Uses BAML's native image support for extraction.
    Caching (cache=True) prevents redundant LLM calls for the same page.

    Args:
        page_image: PNG image bytes of the document page
        extraction_prompt: Prompt describing what to extract

    Returns:
        Extracted data as dict
    """
    try:
        import baml_py
        from baml_client import b
    except ImportError:
        logger.error("BAML not available for extraction")
        return {"error": "BAML not installed. Run: pip install baml-py"}

    # Convert image to BAML format
    image_b64 = base64.b64encode(page_image).decode("utf-8")
    image = baml_py.Image.from_base64("image/png", image_b64)

    try:
        # Use the generic extraction function
        result = await b.ExtractDocumentFieldsFromImage(
            document_image=image,
            extraction_prompt=extraction_prompt,
        )
        return {"extracted_text": result, "status": "success"}
    except AttributeError:
        # Fallback if ExtractDocumentFieldsFromImage doesn't exist
        logger.warning("ExtractDocumentFieldsFromImage not found, using fallback")
        return {
            "status": "fallback",
            "message": "BAML function not found. Ensure BAML client is generated.",
        }
    except Exception as e:
        logger.error(f"BAML extraction failed: {e}")
        return {"error": str(e), "status": "failed"}


@cocoindex.op.function(cache=True, behavior_version=1)
async def extract_with_schema(
    page_image: bytes,
    schema_json: dict,
) -> dict:
    """
    Extract structured data from a page image using a JSON schema.

    Uses tatforge's BAMLFunctionGenerator for schema-driven extraction.
    Caching prevents redundant LLM calls for the same page+schema combination.

    Args:
        page_image: PNG image bytes of the document page
        schema_json: JSON schema defining the extraction structure

    Returns:
        Extracted data as dict matching the schema
    """
    try:
        import baml_py
        from baml_client import b
    except ImportError:
        logger.error("BAML not available for extraction")
        return {"error": "BAML not installed"}

    try:
        from tatforge.core.schema_manager import SchemaManager
        from tatforge.core.baml_function_generator import BAMLFunctionGenerator
    except ImportError:
        logger.error("tatforge schema manager not available")
        return {"error": "tatforge schema manager not installed"}

    # Generate BAML function from schema
    schema_manager = SchemaManager()
    baml_def = schema_manager.generate_baml_classes(schema_json)

    function_generator = BAMLFunctionGenerator()
    optimized_functions = function_generator.generate_optimized_functions(baml_def)

    if not optimized_functions:
        return {"error": "No extraction function generated from schema"}

    # Convert image to BAML format
    image_b64 = base64.b64encode(page_image).decode("utf-8")
    image = baml_py.Image.from_base64("image/png", image_b64)

    # Execute the generated extraction function
    function_name = optimized_functions[0].name
    if hasattr(b, function_name):
        try:
            result = await getattr(b, function_name)(document_images=[image])
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        except Exception as e:
            return {"error": f"Extraction failed: {e}"}
    else:
        return {"error": f"BAML function {function_name} not found in client"}

// Generic document extraction function for vision-based processing
// This function extracts structured data from document images based on a schema

// Generic extraction result - flexible key-value pairs
class ExtractionField {
  field_name string
  value string
  confidence float?
}

class DocumentExtractionResult {
  fields ExtractionField[]
  document_type string?
  extraction_notes string?
}

// Function to extract data from a document image
function ExtractFromDocumentImage(
  document_image: image,
  schema_description: string,
  field_names: string[]
) -> DocumentExtractionResult {
  // Use a vision-capable model via chat completions API
  client "openai/gpt-4o"
  prompt #"
    You are a document extraction expert. Analyze this document image and extract the requested information.

    Document Image:
    {{ document_image }}

    Schema Description:
    {{ schema_description }}

    Fields to Extract:
    {{ field_names }}

    Instructions:
    1. Carefully examine the document image
    2. Extract values for each requested field
    3. If a field is not found or unclear, set its value to "NOT_FOUND"
    4. Provide confidence scores (0.0 to 1.0) when possible
    5. Note the document type if identifiable

    {{ ctx.output_format }}
  "#
}

// Simpler extraction function that returns raw JSON string for flexibility
function ExtractDocumentFields(
  document_image: image,
  extraction_prompt: string
) -> string {
  client "openai/gpt-4o"
  prompt #"
    Analyze this document image and extract the requested information.

    Document Image:
    {{ document_image }}

    {{ extraction_prompt }}

    Return the extracted data as valid JSON matching the requested structure.
  "#
}

// Test with a sample extraction
test sample_extraction {
  functions [ExtractDocumentFields]
  args {
    document_image {
      url "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PDF_file_icon.svg/1200px-PDF_file_icon.svg.png"
    }
    extraction_prompt "Extract any text visible in this image. Return as JSON with a 'text' field."
  }
}

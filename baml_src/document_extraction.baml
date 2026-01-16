// Generic document extraction function for vision-based processing
// This function extracts structured data from PDF documents using native PDF support

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

// Function to extract data from a PDF document using native PDF support
function ExtractFromPDF(
  document: pdf,
  schema_description: string,
  field_names: string[]
) -> DocumentExtractionResult {
  // Use GPT-4o which supports PDF/image inputs
  client "openai/gpt-4o"
  prompt #"
    You are a document extraction expert. Analyze this document and extract the requested information.

    Document:
    {{ document }}

    Schema Description:
    {{ schema_description }}

    Fields to Extract:
    {{ field_names }}

    Instructions:
    1. Carefully examine the document
    2. Extract values for each requested field
    3. If a field is not found or unclear, set its value to "NOT_FOUND"
    4. Provide confidence scores (0.0 to 1.0) when possible
    5. Note the document type if identifiable

    {{ ctx.output_format }}
  "#
}

// Simpler extraction function that returns raw JSON string for flexibility
function ExtractDocumentFieldsFromPDF(
  document: pdf,
  extraction_prompt: string
) -> string {
  client "openai/gpt-4o"
  prompt #"
    Analyze this document and extract the requested information.

    Document:
    {{ document }}

    {{ extraction_prompt }}

    Return the extracted data as valid JSON matching the requested structure.
  "#
}

// Image-based extraction function for ColPali pipeline
// ColPali converts PDFs to 300 DPI page images, so we need image input support
function ExtractDocumentFieldsFromImage(
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

// Structured extraction from image with schema
function ExtractFromImage(
  document_image: image,
  schema_description: string,
  field_names: string[]
) -> DocumentExtractionResult {
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

// Test with a sample extraction (note: test requires actual PDF file)
test sample_extraction {
  functions [ExtractDocumentFieldsFromPDF]
  args {
    document {
      // Use a base64-encoded PDF or file path for testing
      file "../../pdfs/sample.pdf"
    }
    extraction_prompt "Extract any text visible in this document. Return as JSON with a 'text' field."
  }
}

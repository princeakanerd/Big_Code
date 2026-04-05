from docling.document_converter import DocumentConverter

def parse_and_chunk_policy(file_path: str):
    print(f"Parser: Analyzing document layout for {file_path}...")
    
    # Initialize the Docling converter
    converter = DocumentConverter()
    result = converter.convert(file_path)
    document = result.document
    
    chunks = []
    
    # Iterate through all text elements recognized by Docling
    for element in document.texts:
        text_content = element.text.strip()
        
        # Only process elements that actually contain text
        if text_content:
            # Safely extract the page number (Docling stores provenance as a list)
            page_no = "Unknown"
            if hasattr(element, "prov") and element.prov:
                page_no = element.prov[0].page_no
            
            # Format the chunk exactly how our Adjudicator Agent expects it
            formatted_chunk = f"[Page {page_no}]: {text_content}"
            chunks.append(formatted_chunk)
            
    print(f"Parser: Successfully extracted {len(chunks)} citable chunks.")
    return chunks

# Test the parser if this script is run directly
if __name__ == "__main__":
    sample_pdf = "data/sample_policy.pdf"
    try:
        extracted_chunks = parse_and_chunk_policy(sample_pdf)
        if extracted_chunks:
            print("Sample Chunk:", extracted_chunks[0])
    except Exception as e:
        print(f"Error reading PDF: {e}")

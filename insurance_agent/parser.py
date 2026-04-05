from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
import json

def parse_and_chunk_policy(file_path: str):
    print(f"Parser: Analyzing document layout for {file_path}...")
    
    converter = DocumentConverter()
    result = converter.convert(file_path)
    document = result.document
    
    chunks = []
    
    for element in document.texts:
        if element.label in ["paragraph", "list_item"]:
            text_content = element.text
            
            page_no = element.prov[0].page_no if element.prov else "Unknown"
            
            formatted_chunk = f"[Page {page_no}]: {text_content}"
            chunks.append(formatted_chunk)
            
    print(f"Parser: Successfully extracted {len(chunks)} citable chunks.")
    return chunks

if __name__ == "__main__":
    sample_pdf = "data/sample_policy.pdf"
    try:
        extracted_chunks = parse_and_chunk_policy(sample_pdf)
        print("Sample Chunk:", extracted_chunks[0] if extracted_chunks else "No content extracted")
    except Exception as e:
        print("Error reading PDF:", e)

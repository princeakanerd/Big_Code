import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from parser import parse_and_chunk_policy

load_dotenv()

# Initialize the Google Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def build_vector_database(pdf_path: str, persist_directory: str = "./chroma_db"):
    print("Database: Initializing vector database build...")
    
    # 1. Get the layout-aware chunks from our parser
    chunks = parse_and_chunk_policy(pdf_path)
    
    # 2. Store them in ChromaDB
    # We are using persistent storage so it saves to your hard drive
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="insurance_policies"
    )
    print(f"Database: Successfully stored {len(chunks)} chunks in ChromaDB at {persist_directory}")
    return vectorstore

if __name__ == "__main__":
    # Make sure you have a sample PDF at data/sample_policy.pdf
    build_vector_database("data/sample_policy.pdf")

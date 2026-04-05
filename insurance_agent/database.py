import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from parser import parse_and_chunk_policy

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def build_vector_database(pdf_path: str, persist_directory: str = "./chroma_db"):
    print("Database: Initializing vector database build...")
    
    chunks = parse_and_chunk_policy(pdf_path)
    
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="insurance_policies"
    )
    print(f"Database: Successfully stored {len(chunks)} chunks in ChromaDB at {persist_directory}")
    return vectorstore

if __name__ == "__main__":
    build_vector_database("data/sample_policy.pdf")

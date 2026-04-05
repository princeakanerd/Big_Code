import networkx as nx
import pickle
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List
from parser import parse_and_chunk_policy
import os
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def safe_invoke(chain, payload):
    """Executes a LangChain invoke with exponential backoff for 429 rate limits."""
    return chain.invoke(payload)


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

class KnowledgeTriplet(BaseModel):
    subject: str = Field(description="The primary entity (e.g., CPT-93015, Cardiovascular stress test)")
    predicate: str = Field(description="The relationship (e.g., HAS_LIMIT, REQUIRES, IS_A)")
    object_: str = Field(description="The target entity (e.g., $300.00, Prior Authorization)")

class TripletExtraction(BaseModel):
    triplets: List[KnowledgeTriplet] = Field(description="Extracted knowledge triplets from the text")

def build_knowledge_graph(pdf_path: str, save_path: str = "knowledge_graph.pkl"):
    print("GraphRAG: Initializing knowledge graph extraction...")
    chunks = parse_and_chunk_policy(pdf_path)
    
    G = nx.DiGraph()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert medical ontology extractor. Extract knowledge triplets (subject-predicate-object) from the given insurance policy chunk. Keep entities concise."),
        ("user", "{chunk}")
    ])
    
    extractor = llm.with_structured_output(TripletExtraction)
    
    for chunk in chunks:
        print(f"GraphRAG: Processing chunk... {chunk[:50]}...")
        try:
            result = safe_invoke(prompt | extractor, {"chunk": chunk})
            
            for t in result.triplets:
                G.add_node(t.subject.lower())
                G.add_node(t.object_.lower())
                G.add_edge(t.subject.lower(), t.object_.lower(), 
                           relation=t.predicate, 
                           source_chunk=chunk)
        except Exception as e:
            print(f"Error extracting triplets from chunk: {e}")
            
    with open(save_path, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"GraphRAG: Successfully constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. Saved to {save_path}.")
    return G

if __name__ == "__main__":
    build_knowledge_graph("data/sample_policy.pdf")

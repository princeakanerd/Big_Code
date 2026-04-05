import networkx as nx
import pickle
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List
from parser import parse_and_chunk_policy
import os
from dotenv import load_dotenv

load_dotenv()

# We will use Flash for faster processing
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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
            result = (prompt | extractor).invoke({"chunk": chunk})
            
            for t in result.triplets:
                # Add nodes and edges, embedding the original chunk directly onto the edge 
                # to preserve the Docling spatial citation [Page X] for our Streamlit UI
                G.add_node(t.subject.lower())
                G.add_node(t.object_.lower())
                G.add_edge(t.subject.lower(), t.object_.lower(), 
                           relation=t.predicate, 
                           source_chunk=chunk)
        except Exception as e:
            print(f"Error extracting triplets from chunk: {e}")
            
    # Serialize the graph
    with open(save_path, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"GraphRAG: Successfully constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. Saved to {save_path}.")
    return G

if __name__ == "__main__":
    build_knowledge_graph("data/sample_policy.pdf")

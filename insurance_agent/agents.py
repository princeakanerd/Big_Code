import os
from state import ClaimState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import networkx as nx
import pickle
import os
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def safe_invoke(chain, payload):
    """Executes a LangChain invoke with exponential backoff for 429 rate limits."""
    return chain.invoke(payload)



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class ExtractedClaim(BaseModel):
    procedure_code: str = Field(description="The CPT or ICD-10 medical procedure code")
    cost: float = Field(description="The total cost of the procedure")
    diagnosis: str = Field(description="The medical diagnosis provided")

class CodeValidationResult(BaseModel):
    is_valid: bool = Field(description="True if the extracted procedure code accurately matches the raw text and is a valid medical code.")
    corrected_code: str = Field(description="If valid, output the normalized code. If absent or invalid, output 'INVALID_CODE'.")

class VerificationResult(BaseModel):
    is_verified: bool = Field(description="True if the citations are completely accurate and supported by the text, False if there is a hallucination.")
    feedback: str = Field(description="If False, explain exactly which citation failed and why.")


def information_extraction_agent(state: ClaimState):
    print("Agent: Extracting hospital bill data...")
    
    raw_bill_text = state.get("claim_details", {}).get("raw_text", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the procedure code, cost, and diagnosis from the text."),
        ("user", "{bill_text}")
    ])
    
    extractor_llm = llm.with_structured_output(ExtractedClaim)
    chain = prompt | extractor_llm
    
    extracted_data = safe_invoke(chain, {"bill_text": raw_bill_text})
    
    print(f"Agent: Auditing extracted candidate code: {extracted_data.procedure_code}...")
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a stringent medical auditing agent. Review the proposed procedure code against the raw hospital bill. If the code is hallucinatory or unsupported by the text, flag it as invalid. Normalizing formats like 93015 to CPT-93015 is acceptable."),
        ("user", "Raw Bill: {bill_text}\n\nProposed Code: {proposed_code}")
    ])
    
    auditor_llm = llm.with_structured_output(CodeValidationResult)
    qa_chain = qa_prompt | auditor_llm
    
    audit_result = safe_invoke(qa_chain, {
        "bill_text": raw_bill_text,
        "proposed_code": extracted_data.procedure_code
    })
    
    if audit_result.is_valid:
        extracted_data.procedure_code = audit_result.corrected_code
        print("Agent: Candidate code passed audit.")
    else:
        print(f"Agent: REJECTED invalid medical code: {extracted_data.procedure_code}")
        extracted_data.procedure_code = "INVALID_CODE"
    
    return {"claim_details": extracted_data.dict()}

def planner_retrieval_agent(state: ClaimState):
    print("Agent: Retrieving relevant policy clauses from Knowledge Graph...")
    
    graph_path = "knowledge_graph.pkl"
    if not os.path.exists(graph_path):
        print("Agent: Graph not found! Defaulting to empty chunks.")
        return {"retrieved_policy_chunks": []}
        
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
        
    procedure_code = state.get("claim_details", {}).get("procedure_code", "").lower()
    diagnosis = state.get("claim_details", {}).get("diagnosis", "").lower()
    
    search_terms = []
    if procedure_code: search_terms.append(procedure_code)
    if diagnosis: search_terms.append(diagnosis)
    
    relevant_chunks = set()
    
    for node in G.nodes():
        for term in search_terms:
            if term in node:
                for u, v, data in G.edges(node, data=True):
                    if "source_chunk" in data:
                        relevant_chunks.add(data["source_chunk"])
                for u, v, data in G.in_edges(node, data=True):
                    if "source_chunk" in data:
                        relevant_chunks.add(data["source_chunk"])
                        
    retrieved_chunks = list(relevant_chunks)
    print(f"Agent: Found {len(retrieved_chunks)} relevant clauses via Graph Traversal.")
    
    return {"retrieved_policy_chunks": retrieved_chunks}

def adjudication_agent(state: ClaimState):
    print("Agent: Drafting adjudication decision...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an adjudication agent drafting decisions on hospital claims. You must base your decision STRICTLY on the Policy Rules provided. Include precise inline spatial citations based on the provided chunks. Do not use any outside knowledge."),
        ("user", "Claim Details: {claim}\n\nPolicy Rules: {chunks}")
    ])
    
    chain = prompt | llm
    response = safe_invoke(chain, {
        "claim": state.get("claim_details"),
        "chunks": "\n".join(state.get("retrieved_policy_chunks"))
    })
    
    return {"draft_decision": response.content}

def citation_verifier_agent(state: ClaimState):
    print("Agent: Verifying citations and logic...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a verification judge. Evaluate whether the drafted decision correctly cites the policy rules. Check for any hallucinations, such as coverage limits or procedures not explicitly listed in the chunks."),
        ("user", "Policy Rules:\n{chunks}\n\nDraft Decision:\n{draft}")
    ])
    
    verifier_llm = llm.with_structured_output(VerificationResult)
    chain = prompt | verifier_llm
    
    result = safe_invoke(chain, {
        "chunks": "\n".join(state.get("retrieved_policy_chunks")),
        "draft": state.get("draft_decision")
    })
    
    if result.is_verified:
        return {"final_status": "Verified", "errors": []}
    else:
        return {"final_status": "Needs Revision", "errors": [result.feedback]}

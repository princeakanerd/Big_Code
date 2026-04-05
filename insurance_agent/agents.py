import os
from state import ClaimState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)


class ExtractedClaim(BaseModel):
    procedure_code: str = Field(description="The CPT or ICD-10 medical procedure code")
    cost: float = Field(description="The total cost of the procedure")
    diagnosis: str = Field(description="The medical diagnosis provided")

class VerificationResult(BaseModel):
    is_verified: bool = Field(description="True if the citations are completely accurate and supported by the text, False if there is a hallucination.")
    feedback: str = Field(description="If False, explain exactly which citation failed and why.")


def information_extraction_agent(state: ClaimState):
    print("Agent: Extracting hospital bill data...")
    
    raw_bill_text = "Patient underwent a cardiovascular stress test (CPT-93015) on Oct 12th. Total cost: $450.00. Diagnosis: suspected arrhythmia."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the requested information from the bill text."),
        ("user", "{bill_text}")
    ])
    
    extractor_llm = llm.with_structured_output(ExtractedClaim)
    chain = prompt | extractor_llm
    
    extracted_data = chain.invoke({"bill_text": raw_bill_text})
    
    return {"claim_details": extracted_data.dict()}

def planner_retrieval_agent(state: ClaimState):
    print("Agent: Retrieving relevant policy clauses...")
    
    mock_retrieved_chunks = [
        "[Page 14, Paragraph 2]: Cardiovascular stress tests (CPT-93015) are covered up to a maximum of $300.00.",
        "[Page 15, Paragraph 1]: Diagnostic tests for arrhythmia are subject to a 20% co-pay."
    ]
    return {"retrieved_policy_chunks": mock_retrieved_chunks}

def adjudication_agent(state: ClaimState):
    print("Agent: Drafting adjudication decision...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an adjudication agent drafting decisions on hospital claims. You must base your decision STRICTLY on the Policy Rules provided. Include precise inline spatial citations based on the provided chunks. Do not use any outside knowledge."),
        ("user", "Claim Details: {claim}\n\nPolicy Rules: {chunks}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
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
    
    result = chain.invoke({
        "chunks": "\n".join(state.get("retrieved_policy_chunks")),
        "draft": state.get("draft_decision")
    })
    
    if result.is_verified:
        return {"final_status": "Verified", "errors": []}
    else:
        return {"final_status": "Needs Revision", "errors": [result.feedback]}

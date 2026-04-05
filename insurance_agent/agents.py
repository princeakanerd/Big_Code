import os
from state import ClaimState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class ExtractedClaim(BaseModel):
    procedure_code: str = Field(description="The CPT or ICD-10 medical procedure code")
    cost: float = Field(description="The total cost of the procedure")
    diagnosis: str = Field(description="The medical diagnosis provided")

class VerificationResult(BaseModel):
    is_verified: bool = Field(description="True if the citations are completely accurate and supported by the text, False if there is a hallucination.")
    feedback: str = Field(description="If False, explain exactly which citation failed and why.")


def information_extraction_agent(state: ClaimState):
    print("Agent: Extracting hospital bill data...")
    
    # Read the dynamic bill text passed in from the evaluation script
    raw_bill_text = state.get("claim_details", {}).get("raw_text", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the procedure code, cost, and diagnosis from the text."),
        ("user", "{bill_text}")
    ])
    
    extractor_llm = llm.with_structured_output(ExtractedClaim)
    chain = prompt | extractor_llm
    
    extracted_data = chain.invoke({"bill_text": raw_bill_text})
    
    return {"claim_details": extracted_data.dict()}

def planner_retrieval_agent(state: ClaimState):
    print("Agent: Retrieving relevant policy clauses from vector database...")
    
    # Load the existing database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="insurance_policies"
    )
    
    # Formulate a search query based on the extracted hospital bill
    procedure_code = state.get("claim_details", {}).get("procedure_code", "")
    diagnosis = state.get("claim_details", {}).get("diagnosis", "")
    query = f"What are the coverage rules, limits, and co-pays for procedure {procedure_code} and diagnosis {diagnosis}?"
    
    # Retrieve the top 5 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    
    retrieved_chunks = [doc.page_content for doc in docs]
    print(f"Agent: Found {len(retrieved_chunks)} relevant clauses.")
    
    return {"retrieved_policy_chunks": retrieved_chunks}

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

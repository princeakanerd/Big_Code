from typing import TypedDict, List

class ClaimState(TypedDict):
    claim_details: dict            
    retrieved_policy_chunks: List[str] 
    draft_decision: str            
    verified_citations: List[str]  
    final_status: str              
    errors: List[str]              

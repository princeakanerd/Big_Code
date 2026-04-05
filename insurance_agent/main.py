from langgraph.graph import StateGraph, END
from state import ClaimState
from agents import (
    information_extraction_agent,
    planner_retrieval_agent,
    adjudication_agent,
    citation_verifier_agent
)

def verification_router(state: ClaimState):
    if state.get("final_status") == "Needs Revision":
        print("Router: Hallucination detected! Routing back to Adjudicator...")
        return "revise"
    print("Router: Decision verified! Ending workflow...")
    return "end"

workflow = StateGraph(ClaimState)

workflow.add_node("Extractor", information_extraction_agent)
workflow.add_node("Planner", planner_retrieval_agent)
workflow.add_node("Adjudicator", adjudication_agent)
workflow.add_node("Verifier", citation_verifier_agent)

workflow.set_entry_point("Extractor")
workflow.add_edge("Extractor", "Planner")
workflow.add_edge("Planner", "Adjudicator")
workflow.add_edge("Adjudicator", "Verifier")
workflow.add_conditional_edges(
    "Verifier",
    verification_router,
    {
        "revise": "Adjudicator",
        "end": END
    }
)

app = workflow.compile()

if __name__ == "__main__":
    print("Starting Insurance Claim Settlement Agent Workflow...\n")
    initial_state = {}
    
    final_state = app.invoke(initial_state)
    print("\nFinal Output:", final_state.get("draft_decision"))

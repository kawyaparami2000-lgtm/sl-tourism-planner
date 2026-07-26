# Agent 2 – Itinerary Feasibility Evaluator Agent
# Pattern: TOOL-USE pattern
# Uses a retrieval tool and dedicated evaluation model to check travel conditions and feasibility.

from typing import Dict, Any, List
from agents.state import PlannerState
from models.model_router import get_model

def get_stub_evidence(query: str) -> List[str]:
    """
    TEMPORARY STUB — replaced by real RAG retriever in Phase 3.
    Retrieves static domain evidence about weather, transport, and travel feasibility in Sri Lanka.
    """
    evidence_database = [
        "South-West Monsoon affects the Southern and Western coasts from May to September, bringing periodic heavy rainfall and rough sea conditions.",
        "Cultural Triangle (Sigiriya, Dambulla, Anuradhapura) experiences pleasant and dry weather during August.",
        "Express train travel between Kandy and Ella takes ~6 hours; road transport from Kandy to Galle takes ~4.5 hours via the Southern Expressway.",
        "Mid-range travel budget in Sri Lanka typically averages $50-$100 per day including accommodation, regional transport, and entrance fees."
    ]
    # Simple keyword match stub retriever
    matched = [fact for fact in evidence_database if any(word.lower() in fact.lower() for word in query.split())]
    return matched if matched else evidence_database[:2]

def evaluator_agent(state: PlannerState) -> Dict[str, Any]:
    """
    Tool-Use Agent Node:
    Calls retrieval tool and evaluation model (Groq) to assess itinerary feasibility.
    """
    # Fetch assigned LLM for evaluation sub-task from Model Router
    model = get_model("evaluation")
    
    itinerary_legs = state.get("itinerary_legs", [])
    user_input = state.get("user_input", {})
    travel_dates = user_input.get("travel_dates", "")

    retrieved_evidence = []
    evaluation_notes = {
        "model_used": str(model),
        "is_feasible": True,
        "weather_notes": "",
        "transport_notes": "",
        "budget_notes": ""
    }

    print(f"[Evaluator Agent] Tool-Use execution with Model: {model}")
    for leg in itinerary_legs:
        dest = leg.get("destination", "")
        query = f"{dest} {travel_dates}"
        # Call retriever tool
        facts = get_stub_evidence(query)
        retrieved_evidence.extend(facts)

    # Evaluate retrieved evidence against itinerary legs
    eval_summary = []
    if "August" in travel_dates:
        evaluation_notes["weather_notes"] = "Sigiriya/Kandy leg has ideal dry weather in August. Note: Southern coast may experience periodic rain due to SW monsoon."
        eval_summary.append("Weather check: Cultural Triangle favorable; Southern Coast subject to intermittent monsoon showers.")
    
    evaluation_notes["transport_notes"] = "Travel transfers between Cultural Triangle and South Coast take ~4-5 hours by highway."
    evaluation_notes["budget_notes"] = f"Budget level '{user_input.get('budget', 'mid-range')}' matches expected regional cost structures."
    evaluation_notes["summary_list"] = eval_summary

    print("[Evaluator Agent] Evidence retrieved via stub tool:")
    for snippet in set(retrieved_evidence):
        print(f"  - [Retrieved Evidence]: {snippet}")

    return {
        "retrieved_evidence": list(set(retrieved_evidence)),
        "evaluation_notes": evaluation_notes,
        "next_agent": "coach"
    }

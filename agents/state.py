# Shared LangGraph State Schema for Sri Lanka Tourism Planner
from typing import TypedDict, List, Dict, Any

class PlannerState(TypedDict):
    """
    Shared state passed between LangGraph nodes across all agent steps.
    """
    user_input: Dict[str, Any]         # travel_dates, budget, group_type, interests
    itinerary_legs: List[Dict[str, Any]] # list of planned travel legs (destination, nights, why_it_fits)
    retrieved_evidence: List[str]      # list of RAG evidence snippets (filled by Evaluator)
    evaluation_notes: Dict[str, Any]   # feasibility flags (weather, travel time, budget)
    coach_feedback: Dict[str, Any]     # strengths, areas_to_improve, alternatives
    next_agent: str                    # next node in workflow (planner, evaluator, coach, END)

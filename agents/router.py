# Orchestrator / Router Node
# Pattern: ROUTER / ORCHESTRATOR pattern
# Evaluates graph state to determine conditional edge transitions between agents.

from typing import Literal
from agents.state import PlannerState

def route_next_agent(state: PlannerState) -> Literal["planner", "evaluator", "coach", "__end__"]:
    """
    Router/Orchestrator function:
    Reads state['next_agent'] and determines node execution sequence:
    planner -> evaluator -> coach -> END
    (Supports conditional feedback loop back to planner if evaluator notes critical unfeasibility).
    """
    next_agent = state.get("next_agent", "planner")
    eval_notes = state.get("evaluation_notes", {})

    # Check for conditional loopback: if evaluator flags major unfeasibility, route back to planner
    if next_agent == "coach" and not eval_notes.get("is_feasible", True):
        print("[Router] Evaluator flagged major feasibility issues. Routing back to Planner...")
        return "planner"

    if next_agent == "evaluator":
        print("[Router] Routing state to Evaluator Agent...")
        return "evaluator"
    elif next_agent == "coach":
        print("[Router] Routing state to Coach Agent...")
        return "coach"
    elif next_agent in ["END", "__end__"]:
        print("[Router] Workflow complete. Routing to END.")
        return "__end__"

    print("[Router] Default routing to Planner Agent...")
    return "planner"

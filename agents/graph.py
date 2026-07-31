# LangGraph Graph Definition wiring Planner, Evaluator, Coach agents and Router together

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from agents.state import PlannerState
from agents.planner_agent import planner_agent
from agents.evaluator_agent import evaluator_agent
from agents.coach_agent import coach_agent
from agents.router import route_next_agent

def create_planner_graph():
    """
    Constructs and compiles the LangGraph multi-agent workflow.
    """
    builder = StateGraph(PlannerState)

    # Register agent nodes
    builder.add_node("planner", planner_agent)
    builder.add_node("evaluator", evaluator_agent)
    builder.add_node("coach", coach_agent)

    # Set entry point
    builder.add_edge(START, "planner")

    # Wire conditional edges driven by router
    builder.add_conditional_edges("planner", route_next_agent, {"evaluator": "evaluator", "planner": "planner"})
    builder.add_conditional_edges("evaluator", route_next_agent, {"coach": "coach", "planner": "planner"})
    builder.add_conditional_edges("coach", route_next_agent, {"__end__": END})

    return builder.compile()

# Single compiled graph instance
planner_graph = create_planner_graph()

def run_trip_planner(user_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exposed main function to execute the full multi-agent tourism planner workflow.
    """
    initial_state: PlannerState = {
        "user_input": user_input,
        "itinerary_legs": [],
        "retrieved_evidence": [],
        "evaluation_notes": {},
        "coach_feedback": {},
        "next_agent": "planner"
    }

    final_state = planner_graph.invoke(initial_state)
    return final_state

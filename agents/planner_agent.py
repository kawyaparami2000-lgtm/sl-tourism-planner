# Agent 1 – Travel Planner Agent
# Pattern: PLANNING / TASK-DECOMPOSITION pattern
# Decomposes high-level travel requirements (dates, budget, group, interests) into structured itinerary legs.

from typing import Dict, Any
from agents.state import PlannerState

def planner_agent(state: PlannerState) -> Dict[str, Any]:
    """
    Planning/Task-Decomposition Agent Node:
    Analyzes user preferences and decomposes the trip into structured itinerary legs.
    """
    user_input = state.get("user_input", {})
    travel_dates = user_input.get("travel_dates", "August, 7 days")
    budget = user_input.get("budget", "mid-range")
    group_type = user_input.get("group_type", "couple")
    interests = user_input.get("interests", ["culture", "beaches"])

    # Task Decomposition: Break 7-day trip into multi-destination itinerary legs
    planned_legs = [
        {
            "leg_number": 1,
            "destination": "Cultural Triangle (Sigiriya & Kandy)",
            "nights": 3,
            "why_it_fits": f"Matches interests in {', '.join(interests)} with historical sites and scenic landscape suitable for a {group_type}."
        },
        {
            "leg_number": 2,
            "destination": "Southern Coast (Galle & Mirissa)",
            "nights": 4,
            "why_it_fits": f"Provides beach relaxation and ocean activities aligned with mid-range budget of {budget}."
        }
    ]

    print("[Planner Agent] Generated decomposed itinerary legs:")
    for leg in planned_legs:
        print(f"  - Leg {leg['leg_number']}: {leg['destination']} ({leg['nights']} nights) -> {leg['why_it_fits']}")

    return {
        "itinerary_legs": planned_legs,
        "next_agent": "evaluator"
    }

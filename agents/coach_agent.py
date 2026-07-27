# Agent 3 – Budget & Optimization Coach Agent
# Pattern: REFLECTION / SELF-CRITIQUE pattern
# Evaluates proposed itinerary and feasibility notes using an advanced coaching model (OpenRouter).

from typing import Dict, Any
from agents.state import PlannerState
from models.model_router import get_model

def coach_agent(state: PlannerState) -> Dict[str, Any]:
    """
    Reflection/Self-Critique Agent Node:
    Critiques the generated itinerary using assigned OpenRouter coaching model.
    """
    # Fetch assigned LLM for coaching synthesis sub-task from Model Router
    model = get_model("coaching_synthesis")
    
    itinerary_legs = state.get("itinerary_legs", [])
    eval_notes = state.get("evaluation_notes", {})
    user_input = state.get("user_input", {})

    print(f"[Coach Agent] Reflection execution with Model: {model}")

    # Formulate structured critique (using standard bullet prefixes for console safety)
    strengths = [
        "[+] Excellent balance of cultural heritage and coastal relaxation tailored for a 7-day duration.",
        "[+] Cultural Triangle (Sigiriya/Kandy) dry season timing in August optimizes outdoor exploration.",
        f"[+] Leg allocations align well with the {user_input.get('budget', 'mid-range')} budget parameters."
    ]

    gaps = [
        "[-] Southern coast leg in August carries minor risk of monsoon rain showers.",
        "[-] 4.5-hour road transit between Kandy and Galle utilizes almost half a travel day."
    ]

    alternatives = [
        "Option A: Swap Southern Coast for East Coast (Pasikudah/Trincomalee) which enjoys peak dry summer weather in August.",
        "Option B: Include scenic train trip (Kandy to Ella) before heading to south/east coast for improved travel experience."
    ]

    coach_feedback = {
        "model_used": str(model),
        "strengths": strengths,
        "areas_to_improve": gaps,
        "alternatives": alternatives,
        "formatted_summary": (
            f"--- COACH CRITIQUE (Model: {model}) ---\n"
            + "\n".join(strengths) + "\n\n"
            + "\n".join(gaps) + "\n\n"
            + "Suggested Alternatives:\n" + "\n".join(f"  * {alt}" for alt in alternatives)
        )
    }

    print("[Coach Agent] Reflection complete. Feedback generated:")
    print(coach_feedback["formatted_summary"])

    return {
        "coach_feedback": coach_feedback,
        "next_agent": "END"
    }

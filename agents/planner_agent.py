# Agent 1 – Travel Planner Agent
# Pattern: PLANNING / TASK-DECOMPOSITION pattern
# Decomposes high-level travel requirements (dates, budget, group, interests) into structured itinerary legs using LLM reasoning.

import json
import re
from typing import Dict, Any
from agents.state import PlannerState
from models.model_router import get_model

def planner_agent(state: PlannerState) -> Dict[str, Any]:
    """
    Planning/Task-Decomposition Agent Node:
    Analyzes user preferences and calls the LLM via Model Router to decompose the trip into structured itinerary legs.
    """
    user_input = state.get("user_input", {})
    
    # Extract values from user_input state. If a field is missing, fallbacks are used with explicit comments as required.
    travel_dates = user_input.get("travel_dates")
    if not travel_dates:
        # Fallback default if travel_dates is missing in user_input
        travel_dates = "Flexible, 7 days"
        
    budget = user_input.get("budget")
    if not budget:
        # Fallback default if budget is missing in user_input
        budget = "mid-range"
        
    group_type = user_input.get("group_type")
    if not group_type:
        # Fallback default if group_type is missing in user_input
        group_type = "couple"
        
    interests = user_input.get("interests")
    if not interests:
        # Fallback default if interests is missing in user_input
        interests = ["general sightseeing"]

    # Fetch assigned LLM for planning sub-task from Model Router
    model = get_model("planning")
    
    if isinstance(model, str) or not hasattr(model, "invoke"):
        raise RuntimeError(
            f"Planner Agent failed: LLM is not properly configured. Router returned: '{model}'. "
            "Please ensure GROQ_API_KEY is set in environment or secrets."
        )

    interests_str = ", ".join(interests) if isinstance(interests, list) else str(interests)

    system_prompt = (
        "You are an expert Sri Lanka Travel Planner agent following a Task-Decomposition pattern.\n"
        "Your role is to decompose high-level user travel requirements into exactly 2 structured itinerary legs in Sri Lanka.\n\n"
        "IMPORTANT RULES:\n"
        "1. Recommend real Sri Lankan destinations matching the user's specific interests:\n"
        "   - Wildlife & National Parks -> Yala, Udawalawe, Wilpattu, Minneriya\n"
        "   - Beaches & Coastal / Surfing -> Galle, Mirissa, Trincomalee, Arugam Bay, Tangalle, Hikkaduwa, Bentota\n"
        "   - Culture & Heritage -> Sigiriya, Kandy, Anuradhapura, Polonnaruwa, Dambulla\n"
        "   - Hiking, Nature & Mountains -> Ella, Nuwara Eliya, Horton Plains, Adams Peak\n"
        "2. Do NOT default to the exact same pair (Sigiriya & Kandy + Galle & Mirissa) unless it directly aligns with the user's interests.\n"
        "3. You MUST respond with ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "itinerary_legs": [\n'
        "    {\n"
        '      "leg_number": 1,\n'
        '      "destination": "Destination Name(s)",\n'
        '      "nights": 3,\n'
        '      "why_it_fits": "Explanation of why this destination fits their interests, budget, and group type."\n'
        "    },\n"
        "    {\n"
        '      "leg_number": 2,\n'
        '      "destination": "Destination Name(s)",\n'
        '      "nights": 4,\n'
        '      "why_it_fits": "Explanation of why this destination fits their interests, budget, and group type."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Output ONLY raw JSON. No conversational filler or markdown formatting outside the JSON block."
    )

    user_prompt = (
        f"Trip Details:\n"
        f"- Travel Dates/Duration: {travel_dates}\n"
        f"- Budget Level: {budget}\n"
        f"- Group Type: {group_type}\n"
        f"- Selected Interests: {interests_str}\n\n"
        "Decompose this trip into 2 tailored itinerary legs in JSON format."
    )

    try:
        response = model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        raw_text = response.content.strip()
    except Exception as e:
        print(f"[Planner Agent] Primary LLM call failed ({e}). Attempting fallback model...")
        fallback_model = get_model("planning", fallback=True)
        if fallback_model and hasattr(fallback_model, "invoke") and fallback_model != model:
            try:
                response = fallback_model.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                raw_text = response.content.strip()
            except Exception as inner_e:
                raise RuntimeError(f"Planner Agent LLM invocation failed: {str(e)}") from e
        else:
            raise RuntimeError(f"Planner Agent LLM invocation failed: {str(e)}") from e

    # Parse JSON from response
    parsed_json = _parse_json_response(raw_text)
    planned_legs = parsed_json.get("itinerary_legs", [])

    if not isinstance(planned_legs, list) or len(planned_legs) < 2:
        raise ValueError(
            f"Planner Agent returned unparseable or incomplete itinerary structure. Raw output:\n{raw_text}"
        )

    # Validate each leg dictionary schema
    for idx, leg in enumerate(planned_legs, start=1):
        if not all(k in leg for k in ("leg_number", "destination", "nights", "why_it_fits")):
            raise ValueError(
                f"Planner Agent leg #{idx} is missing required fields. Leg content: {leg}"
            )

    print("[Planner Agent] Successfully generated itinerary legs via LLM reasoning:")
    for leg in planned_legs:
        print(f"  - Leg {leg['leg_number']}: {leg['destination']} ({leg['nights']} nights) -> {leg['why_it_fits']}")

    return {
        "itinerary_legs": planned_legs,
        "next_agent": "evaluator"
    }

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Helper to extract JSON object from LLM response text."""
    clean_text = text.strip()
    clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\s*```$", "", clean_text).strip()
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Planner Agent failed to parse valid JSON from LLM response text:\n{text}")


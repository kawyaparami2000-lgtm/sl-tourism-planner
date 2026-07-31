# Agent 3 – Budget & Optimization Coach Agent
# Pattern: REFLECTION / SELF-CRITIQUE pattern
# Evaluates proposed itinerary and feasibility notes using an advanced coaching model (OpenRouter).

import json
import re
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
    
    if isinstance(model, str) or not hasattr(model, "invoke"):
        raise RuntimeError(
            f"Coach Agent failed: LLM is not properly configured. Router returned: '{model}'. "
            "Please ensure OPENROUTER_API_KEY is set in environment or secrets."
        )

    itinerary_legs = state.get("itinerary_legs", [])
    eval_notes = state.get("evaluation_notes", {})
    user_input = state.get("user_input", {})

    print(f"[Coach Agent] Reflection execution with Model: {model}")

    system_prompt = (
        "You are an expert Sri Lanka Travel Coach following a Reflection & Self-Critique pattern.\n"
        "Your task is to review the proposed itinerary legs and feasibility notes, and provide constructive critique and alternative optimizations.\n\n"
        "You MUST respond with ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "strengths": [\n'
        '    "[+] Positive highlight 1",\n'
        '    "[+] Positive highlight 2"\n'
        '  ],\n'
        '  "areas_to_improve": [\n'
        '    "[-] Potential risk or inefficiency 1",\n'
        '    "[-] Potential risk or inefficiency 2"\n'
        '  ],\n'
        '  "alternatives": [\n'
        '    "Option A: Alternative route or activity suggestion",\n'
        '    "Option B: Alternative route or activity suggestion"\n'
        '  ]\n'
        "}\n"
        "Output ONLY raw JSON. No conversational filler or markdown formatting outside the JSON block."
    )

    user_prompt = (
        f"User Input:\n{json.dumps(user_input, indent=2)}\n\n"
        f"Proposed Itinerary Legs:\n{json.dumps(itinerary_legs, indent=2)}\n\n"
        f"Feasibility Evaluator Notes:\n{json.dumps(eval_notes, indent=2)}\n\n"
        "Critique this itinerary and propose optimizations in JSON format."
    )

    try:
        response = model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        raw_text = response.content.strip()
    except Exception as e:
        raise RuntimeError(f"Coach Agent LLM invocation failed: {str(e)}") from e

    parsed_json = _parse_json_response(raw_text)

    strengths = parsed_json.get("strengths", [])
    gaps = parsed_json.get("areas_to_improve", [])
    alternatives = parsed_json.get("alternatives", [])

    formatted_summary = (
        f"--- COACH CRITIQUE (Model: {model}) ---\n"
        + "\n".join(strengths) + "\n\n"
        + "\n".join(gaps) + "\n\n"
        + "Suggested Alternatives:\n" + "\n".join(f"  * {alt}" for alt in alternatives)
    )

    coach_feedback = {
        "model_used": str(model),
        "strengths": strengths,
        "areas_to_improve": gaps,
        "alternatives": alternatives,
        "formatted_summary": formatted_summary
    }

    print("[Coach Agent] Reflection complete. Feedback generated:")
    print(coach_feedback["formatted_summary"])

    return {
        "coach_feedback": coach_feedback,
        "next_agent": "END"
    }

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Helper to extract JSON object from LLM response text."""
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text)
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Coach Agent failed to parse valid JSON from LLM response:\n{text}")

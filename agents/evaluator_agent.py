# Agent 2 – Itinerary Feasibility Evaluator Agent
# Pattern: TOOL-USE pattern
# Uses real RAG vector retrieval tool and dedicated evaluation model to check travel conditions and feasibility.

import json
import re
from typing import Dict, Any, List
from agents.state import PlannerState
from models.model_router import get_model
from rag.retriever import retrieve

def evaluator_agent(state: PlannerState) -> Dict[str, Any]:
    """
    Tool-Use Agent Node:
    Calls real RAG vector retriever tool and evaluation model (Groq) to assess itinerary feasibility.
    """
    # Fetch assigned LLM for evaluation sub-task from Model Router
    model = get_model("evaluation")
    
    if isinstance(model, str) or not hasattr(model, "invoke"):
        raise RuntimeError(
            f"Evaluator Agent failed: LLM is not properly configured. Router returned: '{model}'. "
            "Please ensure GROQ_API_KEY is set in environment or secrets."
        )

    itinerary_legs = state.get("itinerary_legs", [])
    user_input = state.get("user_input", {})
    travel_dates = user_input.get("travel_dates", "Flexible")
    budget = user_input.get("budget", "mid-range")

    retrieved_evidence = []
    
    print(f"[Evaluator Agent] Tool-Use execution with Model: {model}")
    for leg in itinerary_legs:
        dest = leg.get("destination", "")
        query = f"{dest} {travel_dates} {budget}"
        
        # Call real RAG retriever tool
        results = retrieve(query, k=3)
        for r in results:
            evidence_snippet = f"[{r['category']}/{r['source_file']}]: {r['text']}"
            retrieved_evidence.append(evidence_snippet)

    unique_evidence = list(set(retrieved_evidence))

    system_prompt = (
        "You are an expert Sri Lanka Travel Feasibility Evaluator following a Tool-Use & Verification pattern.\n"
        "Your task is to analyze the proposed itinerary legs, user travel parameters, and retrieved RAG context snippets to evaluate travel feasibility.\n\n"
        "Assess:\n"
        "1. Weather Feasibility (e.g. monsoon patterns in Sri Lanka during the specified travel dates)\n"
        "2. Transport Feasibility (transit times between destinations)\n"
        "3. Budget Feasibility (alignment with local costs)\n\n"
        "You MUST respond with ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "is_feasible": true,\n'
        '  "weather_notes": "Specific weather findings for these legs and dates.",\n'
        '  "transport_notes": "Specific transit times and route considerations.",\n'
        '  "budget_notes": "Budget alignment notes.",\n'
        '  "summary_list": [\n'
        '    "Brief key bullet point 1",\n'
        '    "Brief key bullet point 2"\n'
        '  ]\n'
        "}\n"
        "Output ONLY raw JSON. No conversational filler or markdown formatting outside the JSON block."
    )

    user_prompt = (
        f"User Travel Inputs:\n"
        f"- Travel Dates: {travel_dates}\n"
        f"- Budget Level: {budget}\n\n"
        f"Proposed Itinerary Legs:\n{json.dumps(itinerary_legs, indent=2)}\n\n"
        f"Retrieved Context Snippets from Knowledge Base:\n" + "\n".join(f"- {e}" for e in unique_evidence[:8]) + "\n\n"
        "Evaluate the feasibility based strictly on the user input, legs, and evidence snippets above."
    )

    try:
        response = model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        raw_text = response.content.strip()
    except Exception as e:
        raise RuntimeError(f"Evaluator Agent LLM invocation failed: {str(e)}") from e

    parsed_json = _parse_json_response(raw_text)

    # Attach model_used
    parsed_json["model_used"] = str(model)

    print("[Evaluator Agent] Evidence retrieved via RAG Vector Store:")
    for snippet in unique_evidence:
        print(f"  - {snippet}")
    print(f"[Evaluator Agent] Feasibility result: {parsed_json.get('is_feasible')}")

    return {
        "retrieved_evidence": unique_evidence,
        "evaluation_notes": parsed_json,
        "next_agent": "coach"
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
        raise ValueError(f"Evaluator Agent failed to parse valid JSON from LLM response:\n{text}")

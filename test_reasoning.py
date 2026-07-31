import json
from agents.graph import run_trip_planner

def main():
    input_wildlife = {
        "travel_dates": "August, 7 days",
        "budget": "mid-range",
        "group_type": "couple",
        "interests": ["Wildlife & National Parks"]
    }

    input_beaches = {
        "travel_dates": "August, 7 days",
        "budget": "mid-range",
        "group_type": "couple",
        "interests": ["Beaches & Coastal"]
    }

    print("==================================================")
    print("=== TEST 1: Wildlife & National Parks ===")
    print("==================================================")
    res1 = run_trip_planner(input_wildlife)
    print("\n--- [TEST 1 RESULT] ITINERARY LEGS ---")
    for leg in res1["itinerary_legs"]:
        print(f"Leg {leg['leg_number']}: {leg['destination']} ({leg['nights']} nights)")
        print(f"  Why it fits: {leg['why_it_fits']}")

    print("\n--- [TEST 1 RESULT] EVALUATOR NOTES ---")
    print(f"Is Feasible: {res1['evaluation_notes'].get('is_feasible')}")
    print(f"Weather: {res1['evaluation_notes'].get('weather_notes')}")
    print(f"Transport: {res1['evaluation_notes'].get('transport_notes')}")

    print("\n--- [TEST 1 RESULT] COACH FEEDBACK ---")
    print(res1["coach_feedback"].get("formatted_summary"))

    print("\n\n==================================================")
    print("=== TEST 2: Beaches & Coastal ===")
    print("==================================================")
    res2 = run_trip_planner(input_beaches)
    print("\n--- [TEST 2 RESULT] ITINERARY LEGS ---")
    for leg in res2["itinerary_legs"]:
        print(f"Leg {leg['leg_number']}: {leg['destination']} ({leg['nights']} nights)")
        print(f"  Why it fits: {leg['why_it_fits']}")

    print("\n--- [TEST 2 RESULT] EVALUATOR NOTES ---")
    print(f"Is Feasible: {res2['evaluation_notes'].get('is_feasible')}")
    print(f"Weather: {res2['evaluation_notes'].get('weather_notes')}")
    print(f"Transport: {res2['evaluation_notes'].get('transport_notes')}")

    print("\n--- [TEST 2 RESULT] COACH FEEDBACK ---")
    print(res2["coach_feedback"].get("formatted_summary"))

if __name__ == "__main__":
    main()

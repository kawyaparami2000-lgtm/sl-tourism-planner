# Sri Lanka Tourism Planner — Streamlit Web Application

import sys
import traceback
import streamlit as st
from agents.graph import run_trip_planner

# -----------------------------------------------------------------------------
# 1. Page Configuration & Header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka Tourism Planner",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇱🇰 Sri Lanka Tourism Planner")
st.caption("An AI-powered multi-agent travel application that creates personalized itineraries, evaluates travel feasibility via RAG, and optimizes your trip budget.")

# -----------------------------------------------------------------------------
# 2. Input Preferences Form
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Trip Preferences")

with st.sidebar.form(key="trip_form"):
    travel_dates_str = st.text_input("Travel Month / Timing", value="August", help="e.g. August, December to January")
    trip_length = st.number_input("Trip Duration (Days)", min_value=1, max_value=30, value=7, step=1)
    
    budget_tier = st.selectbox(
        "Budget Tier",
        options=["Mid-range", "Budget", "Luxury"],
        index=0
    )
    
    group_type = st.selectbox(
        "Group Type",
        options=["Couple", "Solo", "Family", "Friends group"],
        index=0
    )
    
    interests = st.multiselect(
        "Travel Interests",
        options=[
            "Culture & Heritage",
            "Hill Country & Nature",
            "Beaches & Coastal",
            "Wildlife & National Parks",
            "Adventure & Hiking",
            "Wellness & Ayurveda",
            "Food & Culinary"
        ],
        default=["Culture & Heritage", "Beaches & Coastal"]
    )
    
    submit_button = st.form_submit_button(label="Plan My Trip", use_container_width=True)

# -----------------------------------------------------------------------------
# 3. Form Submission & Graph Execution
# -----------------------------------------------------------------------------
if submit_button:
    if not travel_dates_str.strip():
        st.warning("Please specify your intended travel month or timing.")
        st.stop()
        
    if not interests:
        st.warning("Please select at least one travel interest.")
        st.stop()

    # Build schema payload expected by LangGraph workflow
    user_input_payload = {
        "travel_dates": f"{travel_dates_str.strip()}, {trip_length} days",
        "budget": budget_tier.lower(),
        "group_type": group_type.lower(),
        "interests": [item.lower() for item in interests]
    }

    try:
        with st.spinner("🤖 Executing Agentic AI Workflow (Planning -> RAG Retrieval -> Feasibility -> Coaching)..."):
            final_state = run_trip_planner(user_input_payload)
            st.session_state["trip_plan_result"] = final_state
            st.session_state["last_payload"] = user_input_payload
            st.success("Travel plan generated successfully!")
    except Exception as e:
        st.error("Failed to generate travel plan. Please check your system configuration or try again.")
        print(f"[App Execution Error]: {e}", file=sys.stderr)
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 4. Displaying Results
# -----------------------------------------------------------------------------
result_state = st.session_state.get("trip_plan_result")

if result_state:
    st.subheader("📌 Your Personalized Travel Plan")
    
    # Summary Info Pills
    payload = st.session_state.get("last_payload", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration & Timing", payload.get("travel_dates", "N/A"))
    with col2:
        st.metric("Budget Tier", payload.get("budget", "N/A").title())
    with col3:
        st.metric("Group Type", payload.get("group_type", "N/A").title())
    with col4:
        st.metric("Interests", ", ".join(payload.get("interests", [])).title())

    st.markdown("---")

    # Main Result Tabs
    tab_itinerary, tab_evaluator, tab_coach = st.tabs([
        "🗺️ Itinerary Legs",
        "🔍 RAG Feasibility Notes",
        "💡 Coach Optimization"
    ])

    # Tab 1: Itinerary Legs
    with tab_itinerary:
        st.markdown("### Decomposed Travel Itinerary")
        legs = result_state.get("itinerary_legs", [])
        if legs:
            for leg in legs:
                with st.expander(f"📍 Leg {leg.get('leg_number', 1)}: {leg.get('destination', 'Destination')} ({leg.get('nights', 1)} nights)", expanded=True):
                    st.write(f"**Why it fits your profile:** {leg.get('why_it_fits', '')}")
        else:
            st.info("No itinerary legs generated.")

    # Tab 2: Evaluator & RAG Notes
    with tab_evaluator:
        st.markdown("### RAG Evidence & Feasibility Evaluation")
        eval_notes = result_state.get("evaluation_notes", {})
        
        st.markdown(f"**Assigned Evaluation LLM Model:** `{eval_notes.get('model_used', 'N/A')}`")
        
        col_w, col_t, col_b = st.columns(3)
        with col_w:
            st.info(f"**🌤️ Weather Assessment**\n\n{eval_notes.get('weather_notes', 'N/A')}")
        with col_t:
            st.info(f"**🚆 Transport Assessment**\n\n{eval_notes.get('transport_notes', 'N/A')}")
        with col_b:
            st.info(f"**💰 Budget Assessment**\n\n{eval_notes.get('budget_notes', 'N/A')}")

        st.markdown("#### 📚 Knowledge Base Evidence Retrieved via RAG:")
        evidence_list = result_state.get("retrieved_evidence", [])
        if evidence_list:
            for snippet in evidence_list:
                st.caption(f"• {snippet}")
        else:
            st.write("No evidence retrieved.")

    # Tab 3: Coach Feedback
    with tab_coach:
        st.markdown("### Reflective Critique & Optimization")
        coach_feedback = result_state.get("coach_feedback", {})
        
        st.markdown(f"**Assigned Coaching LLM Model:** `{coach_feedback.get('model_used', 'N/A')}`")

        col_s, col_g = st.columns(2)
        with col_s:
            st.success("#### [+] Strengths")
            for item in coach_feedback.get("strengths", []):
                st.write(item)
        with col_g:
            st.warning("#### [-] Areas to Improve")
            for item in coach_feedback.get("areas_to_improve", []):
                st.write(item)

        st.markdown("#### 💡 Suggested Alternatives")
        for alt in coach_feedback.get("alternatives", []):
            st.info(alt)

# -----------------------------------------------------------------------------
# 5. How This Works Section
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ How this works"):
    st.write(
        "The Sri Lanka Tourism Planner uses a 3-agent AI orchestration architecture powered by LangGraph. "
        "The **Planner Agent** breaks your trip into structured legs, the **Evaluator Agent** queries a ground-truth "
        "Retrieval-Augmented Generation (RAG) knowledge base to check weather and transport feasibility, and the "
        "**Coach Agent** provides reflective critique and optimization suggestions."
    )

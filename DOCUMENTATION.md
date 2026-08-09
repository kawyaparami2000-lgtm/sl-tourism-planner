# Sri Lanka Tourism Planner — Technical Documentation & Architecture Guide

Welcome to the comprehensive technical documentation for the **Sri Lanka Tourism Planner**, an AI-powered travel planning application built for an Agentic AI assignment.

This project combines **Multi-Agent Orchestration (LangGraph)**, **Retrieval-Augmented Generation (RAG via ChromaDB & SentenceTransformers)**, and a **Task-Based Model Router (Groq & OpenRouter)** wrapped in an interactive **Streamlit Web Application**.

---

## 📐 1. System Architecture & Agent Patterns

The system operates on a state-graph workflow where 3 distinct AI agents collaborate across 3 recognized Agentic AI design patterns:

```text
               +----------------------------------+
               |     User Inputs (Streamlit UI)   |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    1. Travel Planner Agent       |
               | (Planning & Task-Decomposition)  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | 2. Feasibility Evaluator Agent   |
               |   (Tool-Use Pattern via RAG)     |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    3. Budget & Coach Agent       |
               |  (Reflection & Self-Critique)    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |     Router / Orchestrator        |  ---> (If unfeasible: loop back)
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |        Final UI Output           |
               +----------------------------------+
```

### Agent Design Patterns Summary

| Agent | Design Pattern | Description & Responsibilities | Model Used |
| :--- | :--- | :--- | :--- |
| **Travel Planner Agent** | **Planning / Task-Decomposition** | Breaks high-level user travel requirements (dates, duration, budget, interests) into structured multi-destination itinerary legs. | Rule/Model Decomposer |
| **Itinerary Feasibility Evaluator** | **Tool-Use** | Queries the ground-truth RAG vector store for facts on weather monsoons, transport times, and regional costs to assess plan feasibility. | **Groq** (`llama-3.1-8b-instant`) |
| **Optimization Coach Agent** | **Reflection / Self-Critique** | Critiques the proposed itinerary against feasibility notes, highlighting `[+] Strengths`, `[-] Areas to Improve`, and suggested alternatives. | **OpenRouter** (`openai/gpt-4o-mini`) |
| **Router Node** | **Orchestrator / Routing** | Dynamically inspects graph state `next_agent` to route execution: `planner -> evaluator -> coach -> END`. | Graph Logic |

---

## 📚 2. RAG Knowledge Base & Vector Pipeline

Recommendations are grounded in a domain knowledge base composed of 20 original text files across 9 subcategories under `data/`:

- `data/destinations/`: `kandy.txt`, `galle.txt`, `ella.txt`, `sigiriya.txt`, `nuwara_eliya.txt`, `mirissa.txt`, `trincomalee.txt`, `yala.txt`
- `data/transport/`: `trains.txt`, `buses.txt`, `tuktuks_and_drivers.txt`, `domestic_flights.txt`
- `data/accommodation/`: `budget_stays.txt`, `midrange_stays.txt`, `luxury_stays.txt`
- `data/seasonal/`: `yala_season.txt`, `maha_season.txt`
- `data/food/`: `sri_lankan_cuisine.txt`
- `data/safety_visa/`: `visa_requirements.txt`, `safety_advisories.txt`
- `data/festivals/`: `festival_calendar.txt`
- `data/wildlife/`: `national_parks.txt`
- `data/budget/`: `average_costs.txt`

### Vector Ingestion & Storage Details

1. **Ingestion ([rag/ingest.py](file:///c:/Users/acer/Desktop/AI%20Bot/sl-tourism-planner/rag/ingest.py))**: Recursively loads text files, splits them into ~400 character chunks with 50-character overlap, and attaches `category` and `source_file` metadata.
2. **Embedding ([rag/embed_store.py](file:///c:/Users/acer/Desktop/AI%20Bot/sl-tourism-planner/rag/embed_store.py))**: Embeds chunks using the local, free `all-MiniLM-L6-v2` SentenceTransformers model and stores vectors in a persistent local `./chroma_db/` folder.
3. **Self-Healing Vector Store**: `get_vector_store()` checks if `./chroma_db/` exists and is non-empty. On a fresh container (e.g. Streamlit Cloud), it automatically builds the vector store from `data/` at startup.
4. **Retrieval Tool ([rag/retriever.py](file:///c:/Users/acer/Desktop/AI%20Bot/sl-tourism-planner/rag/retriever.py))**: Exposes `retrieve(query, k=3)` returning top-k matching chunks with source metadata.

---

## 🔀 3. Model Router & Secret Precedence

Model selection is handled by `models/model_router.py` through `get_model(task_name)`:

```python
def _get_api_key(key_name: str) -> str:
    # 1. Check Streamlit Secrets precedence (Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            sec_val = str(st.secrets[key_name]).strip()
            if sec_val:
                return sec_val
    except Exception:
        pass

    # 2. Fall back to os.environ / python-dotenv (Local environment)
    env_val = os.getenv(key_name, "").strip()
    if env_val:
        return env_val
    return ""
```

---

## 📁 4. Project Folder Structure

```text
sl-tourism-planner/
├── app.py                  # Streamlit web application UI
├── .env.example            # Environment template for API keys
├── .gitignore              # Ignores .env, .streamlit/secrets.toml, chroma_db/
├── README.md               # Quickstart guide
├── DOCUMENTATION.md        # Comprehensive technical documentation
├── requirements.txt        # Version-pinned Python dependencies
├── agents/
│   ├── __init__.py
│   ├── state.py             # Shared LangGraph PlannerState TypedDict
│   ├── planner_agent.py      # Agent 1: Planning / Task-Decomposition pattern
│   ├── evaluator_agent.py    # Agent 2: Tool-Use pattern via RAG retriever
│   ├── coach_agent.py        # Agent 3: Reflection / Self-Critique pattern
│   ├── router.py             # Orchestrator node driving state transitions
│   └── graph.py              # Compiled LangGraph StateGraph & run_trip_planner entrypoint
├── rag/
│   ├── __init__.py
│   ├── ingest.py             # Document ingestion & metadata chunking
│   ├── embed_store.py        # SentenceTransformers embeddings & self-healing ChromaDB
│   ├── retriever.py          # Vector similarity search retriever tool
│   └── eval_queries.py       # 5 retrieval quality test queries
├── data/                     # Domain knowledge corpus across 9 subcategories
│   ├── accommodation/
│   ├── budget/
│   ├── destinations/
│   ├── festivals/
│   ├── food/
│   ├── safety_visa/
│   ├── seasonal/
│   ├── transport/
│   └── wildlife/
├── models/
│   └── model_router.py       # Task-based model selector with secrets precedence
├── notebooks/
│   └── architecture_test.ipynb # Multi-agent workflow test notebook
└── tests/
    └── test_retrieval.py     # RAG pipeline test placeholder
```

---

## 🚀 5. How to Run Locally

### 1. Configure `.env`
Create a `.env` file in the project root folder:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Launch Streamlit Web UI
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

### 3. Run RAG Retrieval Evaluation Script
```bash
python -m rag.eval_queries
```

### 4. Run Multi-Agent Graph Test in Python
```bash
python -c "from agents.graph import run_trip_planner; print(run_trip_planner({'travel_dates': 'August, 7 days', 'budget': 'mid-range', 'group_type': 'couple', 'interests': ['culture', 'beaches']}))"
```

---

## ☁️ 6. Streamlit Cloud Deployment Guide

1. Push the `main` branch to your GitHub repository:
   ```bash
   git push origin main
   ```
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Click **New App**, select your repo, set **Branch** to `main`, and **Main file path** to `app.py`.
4. Under **App Settings -> Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   OPENROUTER_API_KEY = "your_openrouter_api_key_here"
   ```
5. Click **Deploy!**. The vector store will automatically build itself on first run.

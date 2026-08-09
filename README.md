# 🇱🇰 Sri Lanka Tourism Planner

Sri Lanka Tourism Planner is an AI-powered travel planning application designed to generate personalized itineraries, evaluate itinerary feasibility via RAG, and optimize travel budgets for trips across Sri Lanka using multi-agent orchestration and retrieval-augmented generation (RAG).

For full architectural details, agent pattern breakdowns, and RAG specifications, see [DOCUMENTATION.md](./DOCUMENTATION.md).

---
## 🔗 Live Demo

Try the deployed app here: [Sri Lanka Tourism Planner](https://sl-tourism-planner-tt4f9kjbtjtvcmmiiq2idq.streamlit.app/)

---

## 🌟 Key Features

- **Multi-Agent Orchestration (LangGraph)**:
  - **Planner Agent**: Planning / Task-Decomposition pattern.
  - **Evaluator Agent**: Tool-Use pattern querying ground-truth RAG vector store.
  - **Coach Agent**: Reflection / Self-Critique pattern providing optimization advice.
  - **Router Node**: Dynamic orchestrator managing workflow execution order.
- **RAG Knowledge Base (ChromaDB + SentenceTransformers)**:
  - Grounded in 20 original domain text files across 9 subcategories (`destinations`, `transport`, `accommodation`, `seasonal`, `food`, `safety_visa`, `festivals`, `wildlife`, `budget`).
  - **Self-healing vector store**: Automatically builds Chroma vector store on first run if database does not exist.
- **Task-Based Model Router**:
  - Evaluation sub-task: **Groq** (`llama-3.1-8b-instant`).
  - Coaching sub-task: **OpenRouter** (`openai/gpt-4o-mini`).
- **Interactive Streamlit Web UI**:
  - Preference form, status loading, error preflight check, and tabbed result visualization.

---

## ⚡ Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

### 4. Run RAG Evaluation Test Script
```bash
python -m rag.eval_queries
```

---

## 📖 Complete Documentation

For detailed technical specifications, architecture diagrams, and Streamlit Cloud deployment steps, refer to [DOCUMENTATION.md](file:///c:/Users/acer/Desktop/AI%20Bot/sl-tourism-planner/DOCUMENTATION.md).
# Central Definition of Model Selection per Sub-Task
import os
from dotenv import load_dotenv

load_dotenv()

def get_model(task_name: str):
    """
    Central Model Router:
    Selects and initializes the appropriate LLM for a given sub-task:
    - 'evaluation': Fast Groq model (llama-3.1-8b-instant)
    - 'coaching_synthesis': Stronger OpenRouter model (openai/gpt-4o-mini via OpenRouter)
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if task_name == "evaluation":
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                print("[Model Router] Selected model for 'evaluation': Groq (llama-3.1-8b-instant)")
                return ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    groq_api_key=groq_api_key,
                    temperature=0.2
                )
            except Exception as e:
                print(f"[Model Router] Groq initialization note: {e}")
        print("[Model Router] Configured model for 'evaluation': Groq (llama-3.1-8b-instant)")
        return "Groq: llama-3.1-8b-instant"

    elif task_name == "coaching_synthesis":
        if openrouter_api_key:
            try:
                from langchain_openai import ChatOpenAI
                print("[Model Router] Selected model for 'coaching_synthesis': OpenRouter (openai/gpt-4o-mini)")
                return ChatOpenAI(
                    model_name="openai/gpt-4o-mini",
                    openai_api_key=openrouter_api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.4
                )
            except Exception as e:
                print(f"[Model Router] OpenRouter initialization note: {e}")
        print("[Model Router] Configured model for 'coaching_synthesis': OpenRouter (openai/gpt-4o-mini)")
        return "OpenRouter: openai/gpt-4o-mini"

    else:
        raise ValueError(f"Unknown task_name: '{task_name}'")

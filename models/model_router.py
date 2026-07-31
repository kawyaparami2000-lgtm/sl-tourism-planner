# Central Definition of Model Selection per Sub-Task with Streamlit Secrets Fallback
import os
from dotenv import load_dotenv

load_dotenv()

def _get_api_key(key_name: str) -> str:
    """
    Retrieves API key following strict precedence order:
    1. st.secrets first (if running on Streamlit Cloud)
    2. os.environ / python-dotenv locally as fallback
    """
    # 1. Check Streamlit Secrets precedence
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            sec_val = str(st.secrets[key_name]).strip()
            if sec_val:
                return sec_val
    except Exception:
        pass

    # 2. Fall back to os.environ / python-dotenv
    env_val = os.getenv(key_name, "").strip()
    if env_val:
        return env_val

    return ""

def get_model(task_name: str):
    """
    Central Model Router:
    Selects and initializes the appropriate LLM for a given sub-task:
    - 'evaluation': Fast Groq model (llama-3.1-8b-instant)
    - 'coaching_synthesis': Stronger OpenRouter model (openai/gpt-4o-mini via OpenRouter)
    """
    groq_api_key = _get_api_key("GROQ_API_KEY")
    openrouter_api_key = _get_api_key("OPENROUTER_API_KEY")

    if task_name == "evaluation":
        if groq_api_key and groq_api_key != "your_groq_api_key_here":
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
        if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
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

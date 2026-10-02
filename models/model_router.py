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
        # pyrefly: ignore [missing-import]
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

def get_model(task_name: str, fallback: bool = False):
    """
    Central Model Router with automatic Fallback:
    - Primary for 'planning' & 'evaluation': Groq (llama-3.1-8b-instant). Fallback: OpenRouter.
    - Primary for 'coaching_synthesis': OpenRouter (openai/gpt-4o-mini). Fallback: Groq.
    If fallback=True, swaps primary and secondary model attempts.
    """
    groq_api_key = _get_api_key("GROQ_API_KEY")
    openrouter_api_key = _get_api_key("OPENROUTER_API_KEY")

    groq_valid = bool(groq_api_key and groq_api_key != "your_groq_api_key_here")
    openrouter_valid = bool(openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here")

    def _init_groq():
        if groq_valid:
            try:
                # pyrefly: ignore [missing-import]
                from langchain_groq import ChatGroq
                print(f"[Model Router] Initializing Groq (openai/gpt-oss-20b) for '{task_name}'")
                return ChatGroq(
                    model_name="openai/gpt-oss-20b",
                    groq_api_key=groq_api_key,
                    temperature=0.2
                )
            except Exception as e:
                print(f"[Model Router] Groq initialization note: {e}")
        return None

    def _init_openrouter():
        if openrouter_valid:
            try:
                # pyrefly: ignore [missing-import]
                from langchain_openai import ChatOpenAI
                print(f"[Model Router] Initializing OpenRouter (openai/gpt-4o-mini) for '{task_name}'")
                return ChatOpenAI(
                    model_name="openai/gpt-4o-mini",
                    openai_api_key=openrouter_api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.4
                )
            except Exception as e:
                print(f"[Model Router] OpenRouter initialization note: {e}")
        return None

    if task_name in ["planning", "evaluation"]:
        primary_fn = _init_openrouter if fallback else _init_groq
        secondary_fn = _init_groq if fallback else _init_openrouter
    elif task_name == "coaching_synthesis":
        primary_fn = _init_groq if fallback else _init_openrouter
        secondary_fn = _init_openrouter if fallback else _init_groq
    else:
        raise ValueError(f"Unknown task_name: '{task_name}'")

    model = primary_fn()
    if model is not None:
        return model

    model = secondary_fn()
    if model is not None:
        print(f"[Model Router] Secondary fallback model selected for '{task_name}'")
        return model

    return f"No valid API key available for '{task_name}'. Please configure GROQ_API_KEY or OPENROUTER_API_KEY."


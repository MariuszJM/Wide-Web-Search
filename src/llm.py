# src/llm.py

from langchain_ollama import ChatOllama
from config import LOCAL_LLM_MODEL, LLM_TEMPERATURE

def initialize_llm() -> tuple[ChatOllama, ChatOllama]:
    """Initialize LLM models for generating text and JSON output."""
    llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=LLM_TEMPERATURE)
    llm_json = ChatOllama(model=LOCAL_LLM_MODEL, temperature=LLM_TEMPERATURE, format="json")
    return llm, llm_json
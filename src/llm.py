# src/llm.py

from langchain_ollama import ChatOllama
from langchain_ollama.chat_ollama import ChatOllamaConfig
from config import LOCAL_LLM_MODEL, LLM_TEMPERATURE

def initialize_llm() -> tuple[ChatOllama, ChatOllama]:
    """Initialize LLM models for generating text and JSON output."""
    config = ChatOllamaConfig(model=LOCAL_LLM_MODEL, temperature=LLM_TEMPERATURE)
    llm = ChatOllama(config)
    llm_json = ChatOllama(config, format="json")
    return llm, llm_json
from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_TEMPERATURE = int(os.getenv("LLM_TEMPERATURE", 0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 7500))
OUTPUT_FOLDER = os.getenv(
    "OUTPUT_FOLDER", "./runs"
)

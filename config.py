from dotenv import load_dotenv
import os


SEARCH_QUERIES = [
    "High-level overview of intelligent user interfaces and their impact on modern UI/UX design",
    "Tools and frameworks for building intelligent user interfaces: A 2024 guide",
]

CONTENT_QUESTIONS = [
    "What are the latest trends in intelligent user interfaces, and how are they shaping user experience?",
    "What are the best practices for ensuring accessibility and inclusivity in AI-powered user interfaces?",
]

TIME_HORIZON_DAYS = 185
MAX_TOP_SOURCES = 1
PLATFORM = "youtube"
MAX_SOURCES_PER_SEARCH_QUERY = 10

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL")
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_TEMPERATURE = int(os.getenv("LLM_TEMPERATURE", 0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 7500))
OUTPUT_FOLDER = os.getenv(
    "OUTPUT_FOLDER", "./runs"
)

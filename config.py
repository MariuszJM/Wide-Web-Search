# config.py

SEARCH_QUERIES = [
    "High-level overview of intelligent user interfaces and their impact on modern UI/UX design",
    "Tools and frameworks for building intelligent user interfaces: A 2024 guide"
]

SPECIFIC_QUESTIONS = [
    "What are the latest trends in intelligent user interfaces, and how are they shaping user experience?",
    "What are the best practices for ensuring accessibility and inclusivity in AI-powered user interfaces?",
]

TIME_HORIZON_DAYS = 185

MAX_RESULTS = 1

PLATFORM = 'google'

SOURCES_PER_QUERY = 3

LOCAL_LLM_MODEL = "llama3.2:latest"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 7500

OUTPUT_FOLDER = './runs'
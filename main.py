import logging
import os
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from src.llm import get_llm, get_llm_json_mode
from src.nodes import (
    google_retrieve_urls,
    youtube_retrieve_urls,
    google_process_content,
    youtube_process_content
)
from typing import List
from typing_extensions import TypedDict
from pprint import pprint


# Load environment variables
load_dotenv()

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Search Queries
SEARCH_QUERIES = [
    "High-level overview of intelligent user interfaces and their impact on modern UI/UX design",
    "Tools and frameworks for building intelligent user interfaces: A 2024 guide"
]

# Specific Questions
SPECIFIC_QUESTIONS = [
    "What are the latest trends in intelligent user interfaces, and how are they shaping user experience?",
    "What are the best practices for ensuring accessibility and inclusivity in AI-powered user interfaces?",
]

# Configuration constants
TIME_HORIZON = 185  # In days
MAX_OUTPUTS = 5
PLATFORM = 'google'
SOURCES_PER_QUERY = 10
LLM_NAME = "ollama"  # Options: "ollama", "groq"

# Initialize LLM
llm = get_llm(LLM_NAME)
llm_json_mode = get_llm_json_mode(LLM_NAME)

# Save configuration data
config_data = {
    "search_queries": SEARCH_QUERIES,
    "specific_questions": SPECIFIC_QUESTIONS,
    "platform": PLATFORM,
    "time_horizon": TIME_HORIZON,
    "max_outputs": MAX_OUTPUTS,
    "llm_name": LLM_NAME,
}

# Optional tracing setup using LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

# Graph State definition
class GraphState(TypedDict):
    search_queries: List[str]
    specific_questions: List[str]
    platform: str
    llm: object
    llm_json_mode: object
    time_horizon: int
    max_outputs: int
    logger: object
    unique_urls: List[str]
    sources_per_query: int

# Routing function to determine the next node based on platform
def route_platform(state):
    if state["platform"].lower() == 'google':
        return "google_retrieve_urls"
    elif state["platform"].lower() == 'youtube':
        return "youtube_retrieve_urls"
    else:
        raise ValueError("Invalid platform specified in state.")

# Define the workflow graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("google_retrieve_urls", google_retrieve_urls)
workflow.add_node("youtube_retrieve_urls", youtube_retrieve_urls)
workflow.add_node("google_process_content", google_process_content)
workflow.add_node("youtube_process_content", youtube_process_content)

# Define edges
workflow.set_conditional_entry_point(
    route_platform,
    {
        "google_retrieve_urls": "google_retrieve_urls",
        "youtube_retrieve_urls": "youtube_retrieve_urls",
    },
)

# Platform-specific flows
workflow.add_edge("google_retrieve_urls", "google_process_content")
workflow.add_edge("youtube_retrieve_urls", "youtube_process_content")

# Merge flows after platform-specific processing
workflow.add_edge("google_process_content", END)
workflow.add_edge("youtube_process_content", END)

# Compile the graph
graph = workflow.compile()

# Run the graph
initial_state = GraphState({
    "search_queries": SEARCH_QUERIES,
    "specific_questions": SPECIFIC_QUESTIONS,
    "platform": PLATFORM,
    "llm": llm,
    "llm_json_mode": llm_json_mode,
    "time_horizon": TIME_HORIZON,
    "max_outputs": MAX_OUTPUTS,
    "logger": logger,
    "sources_per_query": SOURCES_PER_QUERY,
})

# Stream the graph execution
for event in graph.stream(initial_state, stream_mode="values"):
    pprint(event)

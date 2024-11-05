import pytest
from unittest.mock import MagicMock
from src.processing import ContentProcessor

CONTENT_QUESTIONS = ["What is the main topic?", "What are the key points?"]
MAX_TOP_SOURCES = 2
LLM_MAX_TOKENS = 7500

@pytest.fixture
def mock_llm_handler():
    llm_handler = MagicMock()
    llm_handler.invoke_text.return_value.content = "Mocked response"
    llm_handler.invoke_json.return_value.content = '{"binary_score": "yes"}'
    llm_handler.llm.get_num_tokens.return_value = 10
    return llm_handler

class Document:
    def __init__(self, page_content):
        self.page_content = page_content

CONTENT_QUESTIONS = ["What is the main topic?", "What are the key points?"]
MAX_TOP_SOURCES = 2
LLM_MAX_TOKENS = 7500

@pytest.fixture
def mock_llm_handler():
    llm_handler = MagicMock()
    llm_handler.invoke_text.return_value.content = "Mocked response"
    llm_handler.invoke_json.return_value.content = '{"binary_score": "yes"}'
    llm_handler.llm.get_num_tokens.return_value = 10
    return llm_handler

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

@pytest.fixture
def sample_documents():
    return [Document("Sample document content")]

def test_is_relevant_chunk(mock_llm_handler):
    processor = ContentProcessor(mock_llm_handler, LLM_MAX_TOKENS)
    result = processor.is_relevant_chunk(
        "Sample document content", CONTENT_QUESTIONS[0]
    )
    assert result is True

def test_generate_answer(mock_llm_handler, sample_documents):
    processor = ContentProcessor(mock_llm_handler, LLM_MAX_TOKENS)
    answer = processor.generate_answer(CONTENT_QUESTIONS[0], sample_documents)
    assert answer == "Mocked response"

def test_check_hallucination(mock_llm_handler, sample_documents):
    processor = ContentProcessor(mock_llm_handler, LLM_MAX_TOKENS)
    result = processor.check_hallucination("Mocked response", sample_documents)
    assert result == "yes"
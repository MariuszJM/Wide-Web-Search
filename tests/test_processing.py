import pytest
from unittest.mock import MagicMock, patch
from src.processing import ContentProcessor

SPECIFIC_QUESTIONS = ["What is the main topic?", "What are the key points?"]
MAX_RESULTS = 2

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value.content = "Mocked response"
    llm.get_num_tokens.return_value = 10
    return llm

@pytest.fixture
def mock_llm_json():
    llm_json = MagicMock()
    llm_json.invoke.return_value.content = '{"binary_score": "yes"}'
    return llm_json

class Document:
    def __init__(self, page_content):
        self.page_content = page_content

@pytest.fixture
def sample_documents():
    return [Document("Sample document content")]

@pytest.fixture
def sample_source_items(sample_documents):
    return {
        "Sample Title": {
            "url": "http://example.com",
            "documents": sample_documents
        }
    }

@patch("src.processing.SPECIFIC_QUESTIONS", SPECIFIC_QUESTIONS)
@patch("src.processing.MAX_RESULTS", MAX_RESULTS)
def test_is_relevant_chunk(mock_llm, mock_llm_json):
    processor = ContentProcessor(mock_llm, mock_llm_json)
    result = processor.is_relevant_chunk("Sample document content", "What is the main topic?")
    assert result is True

@patch("src.processing.SPECIFIC_QUESTIONS", SPECIFIC_QUESTIONS)
@patch("src.processing.MAX_RESULTS", MAX_RESULTS)
def test_generate_answer(mock_llm, mock_llm_json, sample_documents):
    processor = ContentProcessor(mock_llm, mock_llm_json)
    answer = processor.generate_answer("What is the main topic?", sample_documents)
    assert answer == "Mocked response"

@patch("src.processing.SPECIFIC_QUESTIONS", SPECIFIC_QUESTIONS)
@patch("src.processing.MAX_RESULTS", MAX_RESULTS)
def test_check_hallucination(mock_llm, mock_llm_json, sample_documents):
    processor = ContentProcessor(mock_llm, mock_llm_json)
    result = processor.check_hallucination("Mocked response", sample_documents)
    assert result == "yes"

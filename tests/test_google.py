import pytest
from unittest.mock import patch, MagicMock
from src.nodes import google_retrieve_urls, google_process_content


@pytest.fixture
def mock_getenv():
    with patch('src.nodes.os.getenv') as mock:
        yield mock

def setup_state():
    return {
        'logger': MagicMock(),
        'search_queries': ['test query'],
        'time_horizon': '1',
        'sources_per_query': 10
    }

def test_missing_api_keys(mock_getenv):
    # Test case: API keys are missing, raise EnvironmentError
    mock_getenv.side_effect = lambda key: None
    state = setup_state()
    
    with pytest.raises(EnvironmentError):
        google_retrieve_urls(state)
import pytest
from unittest.mock import patch, MagicMock
from src.nodes import google_retrieve_urls, google_process_content


@pytest.fixture
def mock_getenv():
    with patch('src.nodes.os.getenv') as mock:
        yield mock

@pytest.fixture
def mock_state():
    return {
        'logger': MagicMock(),
        'unique_urls': ['url1', 'url2']
    }

def setup_state():
    return {
        'logger': MagicMock(),
        'search_queries': ['test query'],
        'time_horizon': '1d',
        'sources_per_query': 10
    }

def test_missing_api_keys(mock_getenv):
    # Test case: API keys are missing, raise EnvironmentError
    mock_getenv.side_effect = lambda key: None
    state = setup_state()
    
    with pytest.raises(EnvironmentError):
        google_retrieve_urls(state)

@patch('src.nodes.WebBaseLoader')
def test_exception_handling(mock_loader, mock_state):
    # Mock loader to raise an exception during content loading
    mock_loader.return_value.load.side_effect = Exception('Test exception')

    # Call the function and validate exception handling
    result = google_process_content(mock_state)

    # Assert function returns updated state and logs the exceptions
    assert result == mock_state
    assert mock_state['logger'].warning.call_count == 2
    mock_state['logger'].warning.assert_any_call('Failed to load content from url1: Test exception')
    mock_state['logger'].warning.assert_any_call('Failed to load content from url2: Test exception')

def test_logging_empty_urls(mock_state):
    # Test case: No URLs provided in state
    mock_state['unique_urls'] = []
    
    google_process_content(mock_state)
    
    # Assert logging of no content processed
    mock_state['logger'].info.assert_called_once_with('Processed content from 0 Google documents.')

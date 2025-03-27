import pytest
import os
import sys
import logging

# Add the project root to the Python path for importing modules
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # Disable logging during tests
    logging.basicConfig(level=logging.ERROR)
    
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Add to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    yield

# Create a fixture for a sample API key file
@pytest.fixture
def sample_api_key_file(tmp_path):
    # Create a temporary API key file
    api_key_path = tmp_path / "test_openai.key"
    api_key_path.write_text("sk-testkey123456789")
    
    return str(api_key_path)

# Create a fixture for a sample dataframe
@pytest.fixture
def sample_dataframe():
    import pandas as pd
    
    # Create a sample DataFrame with typical transcription data
    df = pd.DataFrame({
        'starttime': ['0:00:00,000', '0:00:05,000', '0:00:10,000'],
        'endtime': ['0:00:05,000', '0:00:10,000', '0:00:15,000'],
        'speaker': ['SPEAKER_1', 'SPEAKER_2', 'SPEAKER_1'],
        'text': ['Hello world', 'How are you?', 'I am fine.'],
    })
    
    return df

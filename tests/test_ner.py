import sys
import os
import pytest
import json
from src.extract_entities import entity_extraction_factory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def detect_agent():
    """Fixture to create a detect_agent function."""
    return entity_extraction_factory()

def load_queries_from_json(file_path):
    """Load queries from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)
    
def test_factory_loads(detect_agent):
    """Test that the factory function loads the model and returns a callable."""
    assert callable(detect_agent), "The factory should return a callable function."

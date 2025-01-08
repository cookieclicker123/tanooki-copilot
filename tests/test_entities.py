import sys
import os
import pytest
import json
from src.extract_entities import entity_extraction_factory
from src.data_types import AvailableEntities, ExtractedEntities

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def extract_entities():
    """Fixture to create an extract_entities function."""
    return entity_extraction_factory()

def load_queries_from_json(file_path):
    """Load queries from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def test_factory_loads(extract_entities):
    """Test that the factory function loads the model and returns a callable."""
    assert callable(extract_entities), "The factory should return a callable function."

def test_query_extracts_entities(extract_entities):
    """Test that specific queries extract the correct entities."""
    queries = load_queries_from_json("tests/fixtures/json_entity_files/clip_search.json")
    
    available_entities = AvailableEntities(
        project_id="project_123",
        contributors={
            "john_id": "John",
            "david_id": "David",
            "sarah_id": "Sarah"
        },
        locations={
            "beach_id": "Beach",
            "kitchen_id": "Kitchen"
        },
        cameras={
            "sony_id": "Sony A7S",
            "canon_id": "Canon 5D"
        },
        clip_types={"rush", "review"},
        shoot_dates={"2023-10-01", "2023-10-02"}
    )

    with open("tmp/entity_logs/entity_extraction_log.txt", "w") as log_file:
        for query in queries:
            extracted_entities: ExtractedEntities = extract_entities(query, available_entities.model_dump())
            log_file.write(f"\nQuery: '{query}'\n")
            log_file.write("Extracted Entities:\n")
            for entity_type, entity_list in extracted_entities.entities.items():
                log_file.write(f"  {entity_type}: {entity_list}\n")
            
            # Example assertion: Check if at least one entity is extracted
            assert any(extracted_entities.entities.values()), f"Query '{query}' did not extract any entities." 
import sys
import os
import pytest
import json
from src.intent import intent_recognition
from src.data_types import AgentType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def detect_agent():
    """Fixture to create a detect_agent function."""
    return intent_recognition()

def load_queries_from_json(file_path):
    """Load queries from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)
    
def test_factory_loads(detect_agent):
    """Test that the factory function loads the model and returns a callable."""
    assert callable(detect_agent), "The factory should return a callable function."


def test_query_selects_production_agent(detect_agent):
    """Test that specific queries select the PRODUCTION agent."""
    queries = load_queries_from_json("tests/fixtures/json_intent_files/production_queries.json")
    
    with open("tmp/intent_logs/index_agent_log.txt", "w") as log_file:
        for query in queries:
            agent_result = detect_agent(query)
            log_file.write(f"\nQuery: '{query}'\n")
            log_file.write("Probabilities:\n")
            for agent, probability in agent_result.all_scores.items():
                log_file.write(f"  {agent}: {probability:.2f}\n")
            
            assert AgentType.PRODUCTION in agent_result.agents, f"Query '{query}' did not select the PRODUCTION agent."

def test_query_selects_tv_post_production_agent(detect_agent):
    """Test that specific queries select the TV_POST_PRODUCTION agent."""
    queries = load_queries_from_json("tests/fixtures/json_intent_files/tv_post_production_queries.json")
    
    with open("tmp/intent_logs/tv_post_production_agent_log.txt", "w") as log_file:
        for query in queries:
            agent_result = detect_agent(query)
            log_file.write(f"\nQuery: '{query}'\n")
            log_file.write("Probabilities:\n")
            for agent, probability in agent_result.all_scores.items():
                log_file.write(f"  {agent}: {probability:.2f}\n")

            assert AgentType.TV_POST_PRODUCTION in agent_result.agents, f"Query '{query}' did not select the TV_POST_PRODUCTION agent."

def test_query_selects_tanooki_agent(detect_agent):
    """Test that specific queries select the TANOOKI agent."""
    queries = load_queries_from_json("tests/fixtures/json_intent_files/tanooki_queries.json")
    
    with open("tmp/intent_logs/tanooki_agent_log.txt", "w") as log_file:
        for query in queries:
            agent_result = detect_agent(query)
            log_file.write(f"\nQuery: '{query}'\n")
            log_file.write("Probabilities:\n")
            for agent, probability in agent_result.all_scores.items():
                log_file.write(f"  {agent}: {probability:.2f}\n")
            
            assert AgentType.TANOOKI in agent_result.agents, f"Query '{query}' did not select the TANOOKI agent."

def test_query_selects_multi_agent(detect_agent):
    """Test that specific queries select both the PRODUCTION and TV_POST_PRODUCTION agents."""
    queries = load_queries_from_json("tests/fixtures/json_intent_files/multi_agent_queries.json")
    
    with open("tmp/intent_logs/multi_agent_log.txt", "w") as log_file:
        for query in queries:
            agent_result = detect_agent(query)
            log_file.write(f"\nQuery: '{query}'\n")
            log_file.write("Probabilities:\n")
            for agent, probability in agent_result.all_scores.items():
                log_file.write(f"  {agent}: {probability:.2f}\n")
            
            assert AgentType.PRODUCTION in agent_result.agents, f"Query '{query}' did not select the PRODUCTION agent."
            assert AgentType.TV_POST_PRODUCTION in agent_result.agents, f"Query '{query}' did not select the TV_POST_PRODUCTION agent."


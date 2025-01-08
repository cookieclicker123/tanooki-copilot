import spacy
from datetime import datetime
from src.data_types import AgentType, AgentResult, IntentFn

def intent_recognition(model_path: str = "./tmp/models/agent_model", 
                          confidence_threshold: float = 0.6) -> IntentFn:
    """Factory function to create a detect_agent function with preloaded model and threshold."""
    nlp = spacy.load(model_path)
    
    def detect_agent(text: str) -> AgentResult:
        """Detect which agent should handle the query and return an AgentResult."""
        doc = nlp(text)
        scores = doc.cats
        
        if not scores:
            return AgentResult(
                text=text,
                timestamp=datetime.now(),
                agents=[],
                confidence=0.0,
                all_scores=scores
            )
        
        # Collect all agents with confidence above the threshold
        agents = [AgentType(agent_str) for agent_str, confidence in scores.items() if confidence > confidence_threshold]
        
        return AgentResult(
            text=text,
            timestamp=datetime.now(),
            agents=agents,
            confidence=max(scores.values()) if agents else 0.0,
            all_scores=scores
        )
    
    return detect_agent

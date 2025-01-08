from enum import Enum
from typing import Callable, NamedTuple, Dict, List, Any, Optional, Set
from pydantic import BaseModel
from datetime import datetime

class AgentType(str, Enum):
    TV_POST_PRODUCTION = "TV_POST_PRODUCTION"
    TANOOKI = "TANOOKI"
    PRODUCTION = "PRODUCTION"
    INVALID_QUERY = "INVALID_QUERY"
    
    @property
    def description(self) -> str:
        return {
            AgentType.TV_POST_PRODUCTION: "Handles TV post-production related queries",
            AgentType.TANOOKI: "Handles Tanooki app specific queries",
            AgentType.PRODUCTION: "Handles production and scene related queries",
            AgentType.INVALID_QUERY: "Handles unauthorized or invalid queries"
        }[self]
    
    def __str__(self):
        return self.value

class AgentResult(BaseModel):
    text: str
    timestamp: datetime
    agents: List[AgentType]
    confidence: float
    all_scores: Dict[str, float]

class SearchSubject(str, Enum):
    SUMMARY = "summary"
    CONTRIBUTOR = "contributor"
    CLIP = "clip"
    LOCATION = "location"
    CAMERA = "camera"
    SHOOT_DATE = "shootDate"

class ValidationError(Exception):
    def __init__(self, message: str, invalid_ids: List[str]):
        self.message = message
        self.invalid_ids = invalid_ids
        super().__init__(message)

# Available Entities
class AvailableEntities(BaseModel):
    project_id: str
    contributors: Dict[str, str]
    locations: Dict[str, str]
    cameras: Dict[str, str]
    clip_types: Set[str]
    shoot_dates: Set[str]

# Search Models
class ContributorFilter(BaseModel):
    contributors: List[str]
    match_type: str = "any"
    seen: Optional[bool] = None
    spoke: Optional[bool] = None

class LocationFilter(BaseModel):
    locations: List[str]
    match_type: str = "any"

class CameraFilter(BaseModel):
    cameras: List[str]
    match_type: str = "any"

class SearchFilters(BaseModel):
    clip_id: Optional[str] = None
    contributors: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    cameras: Optional[List[str]] = None
    clip_types: Optional[List[str]] = None
    shoot_dates: Optional[List[str]] = None
    type: Optional[List[str]] = None

class SearchParams(BaseModel):
    project_id: str
    subject: SearchSubject
    filters: SearchFilters

class EntityRef(BaseModel):
    type: SearchSubject
    id: str
    duration: Optional[int] = None
    metadata: Optional[Dict] = None

class SearchResponse(BaseModel):
    request: SearchParams
    entities: List[EntityRef]
    total: int

class SearchDB(NamedTuple):
    get_available_entities: Callable[[str], AvailableEntities]
    search_contributors: Callable[[str, Optional[str]], List[EntityRef]]
    search_clips: Callable[[], List[EntityRef]]
    
class SearchAPI(NamedTuple):
    get_available_entities: Callable[[str], AvailableEntities]
    search: Callable[[SearchParams], SearchResponse]

class ExtractedEntities(BaseModel):
    entities: Dict[str, List[str]]
    normalized: Dict[str, List[str]]
    confidence: Dict[str, float]
    explanation: str | None

class LLMRequest(BaseModel):
    query: str
    prompt: str
    as_json: bool

class LLMResponse(BaseModel):
    generated_at: str
    agents: List[AgentType] | None
    search_result: SearchResponse | None
    request: LLMRequest
    raw_response: str | Dict[str, Any]
    model_name: str
    model_provider: str
    time_in_seconds: float

class ConversationMemory(BaseModel):
    history: List[LLMResponse]
    token_count: int = 0
    token_limit: int = 128000

class EntityExtractionResult(BaseModel):
    query: str
    llm_response: LLMResponse
    entities: ExtractedEntities | None
    error: str | None

class WorkflowResult(BaseModel):
    query: str
    entity_result: EntityExtractionResult | None
    error: str | None

# The callback types for text and JSON updates
OnTextFn = Callable[[str], None]
OnJsonFn = Callable[[str], None]

# Function signatures
IntentFn = Callable[[str], AgentResult]
LLMGenerateFn = Callable[[str, OnTextFn], LLMResponse]
EntityExtractionFn = Callable[[str], ExtractedEntities]
Workflow = Callable[[str], WorkflowResult]
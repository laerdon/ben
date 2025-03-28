from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from ..api.models import NotionBlock


class LogEntry(BaseModel):
    date: datetime
    blocks: List[NotionBlock]
    raw_text: Optional[str] = None
    summary: Optional[str] = None
    importance: float = 0.5  # Default importance score
    id: Optional[str] = None  # Add ID field


class MemoryEntry(BaseModel):
    embedding: List[float]
    metadata: dict
    id: str  # Unique identifier for the entry


class SearchResult(BaseModel):
    entry: LogEntry
    similarity_score: float
    final_score: float  # After applying recency bias

from pydantic import BaseModel
from typing import List, Dict, Any


class SummarizeResponse(BaseModel):
    summary: str


class QuestionResponse(BaseModel):
    similar_comments: List[Dict[str, Any]]
    answer: str


class HealthResponse(BaseModel):
    status: str
    redis: str


class SummarizeRequest(BaseModel):
    youtube_url: str


class QuestionRequest(BaseModel):
    question: str

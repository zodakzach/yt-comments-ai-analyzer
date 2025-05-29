from pydantic import BaseModel, Field
from typing import List, Dict, Any


class SummarizeResponse(BaseModel):
    summary: str


class QuestionResponse(BaseModel):
    similar_comments: List[Dict[str, Any]]
    answer: str


class HealthResponse(BaseModel):
    status: str
    redis: str


class QuestionRequest(BaseModel):
    question: str


class Comment(BaseModel):
    author: str = Field(..., description="Display name of the comment author")
    text: str = Field(..., description="Original text of the comment")
    likeCount: int = Field(0, description="Number of likes on the comment")
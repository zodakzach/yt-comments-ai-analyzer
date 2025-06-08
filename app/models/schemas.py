from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    redis: str


class Comment(BaseModel):
    author: str = Field(..., description="Display name of the comment author")
    text: str = Field(..., description="Original text of the comment")
    likeCount: int = Field(0, description="Number of likes on the comment")
    sentiment: Dict[str, float] = Field(
        default_factory=dict,
        description="VADER sentiment scores: neg, neu, pos, compound",
    )

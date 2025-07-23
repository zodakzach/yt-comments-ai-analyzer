from typing import Dict, List
from pydantic import BaseModel, Field, HttpUrl


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


class VideoInfo(BaseModel):
    title: str
    likeCount: int
    viewCount: int
    publishedAt: str
    url: HttpUrl
    thumbnailUrl: HttpUrl


class Session(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    video_info: VideoInfo
    summary: str = Field(..., description="Generated summary of the video")
    comments: List[Comment] = Field(
        ..., description="Top‐N comments with sentiment attached"
    )
    total_comments: int = Field(..., description="Count of all comments fetched")
    sentiment_stats: Dict[str, float] = Field(
        ..., description="Aggregate sentiment stats (e.g. avg, min, max)"
    )

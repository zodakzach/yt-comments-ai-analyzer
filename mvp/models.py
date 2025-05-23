from sqlalchemy import Column, String, LargeBinary, Text
from app.db import Base


class VideoComments(Base):
    __tablename__ = "video_comments"

    video_id = Column(String, primary_key=True, index=True)
    summary = Column(Text, nullable=False)
    embeddings = Column(LargeBinary, nullable=False)  # Store embeddings as binary

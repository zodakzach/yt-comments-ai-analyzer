import uuid
import json
from fastapi import APIRouter, HTTPException
from app.core.redis_client import redis_client
from app.models.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    QuestionRequest,
    QuestionResponse,
    HealthResponse,
)
from app.services.summarize import summarize_comments
from app.services.vectorize import vectorize_comments
from app.services.search import search_similar_comments, generate_answer

router = APIRouter()


@router.post("/summarize/", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    try:
        # Summarize comments (async)
        summary, sorted_comments = await summarize_comments(request.youtube_url)

        # Take top 500 comments and vectorize (async)
        top_comments = sorted_comments[:500]
        embeddings = await vectorize_comments(top_comments)

        # Create session and store in Redis (async)
        session_id = str(uuid.uuid4())
        await redis_client.setex(
            session_id,
            3600,  # 1 hour TTL
            json.dumps(
                {
                    "summary": summary,
                    "comments": top_comments,
                    "embeddings": [e.tolist() for e in embeddings],
                }
            ),
        )

        return {
            "session_id": session_id,
            "summary": summary,
            "message": "You can now ask questions using this session. Session expires in 1 hour.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question/", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    try:
        # Retrieve session data (async)
        data_raw = await redis_client.get(request.session_id)
        if not data_raw:
            raise HTTPException(status_code=404, detail="Session expired or not found")

        data = json.loads(data_raw)

        # Perform semantic search and answer generation (async)
        similar = await search_similar_comments(
            question=request.question,
            embeddings=data["embeddings"],
            comments=data["comments"],
        )
        answer = await generate_answer(request.question, similar, data["summary"])

        return {"similar_comments": similar, "answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        # Ping Redis to confirm connectivity
        pong = await redis_client.ping()
        return {"status": "ok", "redis": pong}
    except Exception:
        return {"status": "error", "redis": "unreachable"}

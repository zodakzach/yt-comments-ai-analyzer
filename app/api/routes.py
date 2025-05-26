import uuid
import json
import logging
from fastapi import APIRouter, HTTPException, Response, Request
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
from app.services.errors import (
    EmbeddingError,
    OpenAIInteractionError,
    CommentFetchError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/summarize/", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest, response: Response):
    # 1️⃣ Summarize comments
    try:
        summary, sorted_comments = await summarize_comments(request.youtube_url)
    except CommentFetchError as e:
        logger.error("Fetch-comments failure: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    except OpenAIInteractionError as e:
        logger.error("OpenAI failure during summarization: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        logger.exception("Unexpected summarization error")
        raise HTTPException(
            status_code=500, detail="Internal error summarizing comments."
        )

    # 2️⃣ Vectorize top comments
    try:
        top_comments = sorted_comments[:500]
        embeddings = await vectorize_comments(top_comments)

    except ValueError as ve:
        # No valid text to embed → client sent unusable comments
        logger.warning("No valid comments to embed: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))

    except EmbeddingError as ee:
        # Upstream rate‐limit or API error → treat as bad gateway
        logger.error("Embedding error: %s", ee)
        raise HTTPException(status_code=502, detail=str(ee))

    # 3️⃣ Store in Redis
    try:
        session_id = str(uuid.uuid4())
        await redis_client.setex(
            session_id,
            3600,
            json.dumps(
                {
                    "summary": summary,
                    "comments": top_comments,
                    "embeddings": [e.tolist() for e in embeddings],
                }
            ),
        )
    except Exception as e:
        logger.error("Redis error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error storing session.")

    # 🟢 Set session_id as a secure cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=3600,
        httponly=True,
        secure=True,
        samesite="lax",  # Consider "strict" if your frontend doesn't need to send the cookie on GETs from external sites
    )

    return {
        "summary": summary,
    }


@router.post("/question/", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest, http_request: Request):
    try:
        # 1️⃣ Get session ID from cookie
        session_id = http_request.cookies.get("session_id")
        if not session_id:
            raise HTTPException(status_code=401, detail="Missing session cookie")

        # 2️⃣ Fetch session from Redis
        data_raw = await redis_client.get(session_id)
        if not data_raw:
            raise HTTPException(status_code=404, detail="Session expired or not found")

        try:
            data = json.loads(data_raw)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in Redis for session_id=%s", session_id)
            raise HTTPException(status_code=500, detail="Corrupted session data")

        # 3️⃣ Search similar comments and generate answer
        similar = await search_similar_comments(
            question=request.question,
            embeddings=data["embeddings"],
            comments=data["comments"],
        )
        answer = await generate_answer(request.question, similar, data["summary"])

        return {"similar_comments": similar, "answer": answer}

    except OpenAIInteractionError as oe:
        logger.error("OpenAI API error while answering question: %s", oe)
        raise HTTPException(status_code=502, detail="OpenAI service unavailable")

    except HTTPException:
        raise  # Let FastAPI handle these as-is

    except Exception:
        logger.exception("Unhandled error during /question processing")
        raise HTTPException(status_code=500, detail="Unexpected error during Q&A")


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        # Ping Redis to confirm connectivity
        pong = await redis_client.ping()
        return {"status": "ok", "redis": pong}
    except Exception:
        return {"status": "error", "redis": "unreachable"}

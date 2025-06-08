import base64
import gzip
import json
import logging
from fastapi import APIRouter, Query, Response, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.redis_client import redis_client
from app.core.config import TEMPLATES_DIR
from app.models.schemas import HealthResponse
from app.services.sentiment import (
    annotate_comments_with_sentiment,
    compute_sentiment_stats,
)
from app.services.summarize import summarize_comments, extract_youtube_id
from app.services.session import (
    create_full_session,
    fetch_summary_and_comments,
    get_or_compute_embeddings,
)
from app.services.qa import search_similar_comments, generate_answer
from app.services.errors import (
    EmbeddingError,
    OpenAIInteractionError,
    CommentFetchError,
    DataCorruptionError,
    SessionExpiredError,
    SessionStorageError,
)


router = APIRouter()
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

COMMENT_EMBEDDING_LIMIT = 500  # Max comments to store in redis and embed for Q&A


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def handle_summarization_error(
    request: Request, error_msg: str, status_code: int = 400
):
    return templates.TemplateResponse(
        "error_partial.html",
        {"request": request, "error": error_msg},
        status_code=status_code,
    )


@router.post("/summarize/", response_class=HTMLResponse)
async def summarize(
    request: Request,
    youtube_url: str = Form(...),
):
    # 1. Extract video ID from URL
    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        logger.warning("Invalid YouTube URL provided: %s", youtube_url)
        return handle_summarization_error(
            request, "Invalid YouTube URL.", status_code=422
        )

    # 2. Summarize comments
    try:
        summary, sorted_comments = await summarize_comments(video_id)
    except CommentFetchError as e:
        logger.error("Fetch-comments failure: %s", e)
        return handle_summarization_error(
            request, f"Failed to fetch comments: {e}", status_code=502
        )
    except OpenAIInteractionError as e:
        logger.error("OpenAI failure during summarization: %s", e)
        return handle_summarization_error(
            request, f"AI error during summarization: {e}", status_code=502
        )
    except Exception:
        logger.exception("Unexpected summarization error")
        return handle_summarization_error(
            request, "Internal error summarizing comments.", status_code=500
        )

    # Annotate **all** comments with VADER
    try:
        sorted_comments = annotate_comments_with_sentiment(sorted_comments)
    except Exception as e:
        logger.warning("Sentiment analysis error, proceeding without it: %s", e)

    # Compute aggregate sentiment stats
    sentiment_stats = compute_sentiment_stats(sorted_comments)

    # Take only the top N for embedding & storage
    top_comments = sorted_comments[:COMMENT_EMBEDDING_LIMIT]

    try:
        session_id = await create_full_session(
            summary=summary,
            comments=top_comments,
            total_comments=len(sorted_comments),
            sentiment_stats=sentiment_stats,
        )
    except DataCorruptionError as e:
        logger.error("Session serialization error: %s", e)
        # This means something drove error preparing the blob
        return handle_summarization_error(
            request, "Internal error storing session.", 500
        )
    except SessionStorageError as e:
        logger.error("Session storage error: %s", e)
        # This means Redis calls failed
        return handle_summarization_error(
            request, "Internal error storing session.", 500
        )

    template_resp = templates.TemplateResponse(
        "summary_partial.html",
        {
            "request": request,
            "summary": summary,
            "top_comments": top_comments[:5],
            "total_comments": len(sorted_comments),
            "sentiment_stats": sentiment_stats,
            "session_id": session_id,
        },
        status_code=200,
    )

    return template_resp


@router.get("/session/", response_class=HTMLResponse)
async def get_session(
    request: Request,
    session_id: str = Query(None, description="Per-tab session ID"),
):
    # 0) Ensure we got a session_id
    if not session_id:
        return handle_summarization_error(
            request, "Missing session_id query parameter.", status_code=422
        )

    # 1) Fetch the blob
    try:
        raw = await redis_client.get(f"{session_id}:session")
    except Exception as err:
        logger.error("Redis error fetching session %s: %s", session_id, err)
        return handle_summarization_error(
            request, "Internal error fetching session.", status_code=500
        )

    # 2) Check existence
    if raw is None:
        return handle_summarization_error(
            request,
            "Session not found or expired. Please summarize a video first.",
            status_code=404,
        )

    # 3) Decode & decompress
    try:
        compressed = base64.b64decode(raw)
        data = json.loads(gzip.decompress(compressed).decode("utf-8"))
    except Exception as err:
        logger.error("Error decoding session payload %s: %s", session_id, err)
        return handle_summarization_error(
            request,
            "Corrupted session data. Please try summarizing again.",
            status_code=500,
        )

    # 4) Render the same summary_partial.html
    return templates.TemplateResponse(
        "summary_partial.html",
        {
            "request": request,
            "summary": data["summary"],
            "top_comments": data["comments"][:5],
            "total_comments": data["total_comments"],
            "sentiment_stats": data["sentiment_stats"],
            "error": None,
            "session_id": session_id,
        },
        status_code=200,
    )


@router.post("/question/", response_class=HTMLResponse)
async def answer_question(
    request: Request,
    session_id: str = Query(None, description="Per-tab session ID"),
    question: str = Form(...),
):
    # 1️⃣ Session + data
    try:
        summary, comments = await fetch_summary_and_comments(session_id)
    except (SessionExpiredError, DataCorruptionError) as e:
        return handle_chat_error(request, str(e))

    # 2️⃣ Embeddings
    try:
        embeddings = await get_or_compute_embeddings(session_id, comments)
    except EmbeddingError as e:
        return handle_chat_error(request, str(e))

    # 3️⃣ Q&A
    try:
        similar = await search_similar_comments(
            question=question,
            embeddings=embeddings,
            comments=comments,
        )
        answer = await generate_answer(question, similar, summary)
    except EmbeddingError as e:
        return handle_chat_error(request, f"Embedding error: {e}")
    except OpenAIInteractionError:
        return handle_chat_error(
            request, "AI service unavailable. Please try again later."
        )
    except Exception:
        return handle_chat_error(
            request, "Unexpected error during Q&A. Please try again."
        )

    # 4️⃣ Success render
    return templates.TemplateResponse(
        "message_partial.html",
        {
            "request": request,
            "error": None,
            "question": question,
            "answer": answer,
            "similar_comments": similar,
        },
        status_code=200,
    )


def handle_chat_error(request: Request, msg: str):
    return templates.TemplateResponse(
        "error_partial.html",
        {
            "request": request,
            "error": msg,
            "answer": None,
            "similar_comments": None,
        },
        status_code=200,
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    session_id: str = Query(None, description="Per-tab session ID"),
):
    # pick up the session ID from the param
    sid = session_id
    if not sid:
        return handle_chat_error(request, "Session missing or expired")

    return templates.TemplateResponse(
        "chat_partial.html",
        {
            "request": request,
            "session_id": sid,
        },
        status_code=200,
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        # Ping Redis to confirm connectivity
        pong = await redis_client.ping()
        return {"status": "ok", "redis": pong}
    except Exception:
        return {"status": "error", "redis": "unreachable"}

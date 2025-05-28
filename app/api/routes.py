import base64
import gzip
import uuid
import json
import logging
from fastapi import APIRouter, Response, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.redis_client import redis_client
from app.core.config import TEMPLATES_DIR
from app.models.schemas import (
    QuestionRequest,
    HealthResponse,
)
from app.services.summarize import summarize_comments, extract_youtube_id
from app.services.session import fetch_summary_and_comments, get_or_compute_embeddings
from app.services.search import search_similar_comments, generate_answer
from app.services.errors import (
    EmbeddingError,
    OpenAIInteractionError,
    CommentFetchError,
    DataCorruptionError,
    SessionExpiredError,
)


router = APIRouter()
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def handle_summarization_error(request: Request, error_msg: str):
    return templates.TemplateResponse(
        "summary.html",
        {"request": request, "error": error_msg, "summary": None},
        status_code=200,
    )


@router.post("/summarize/", response_class=HTMLResponse)
async def summarize(
    request: Request,
    response: Response,
    youtube_url: str = Form(...),
):
    # 1. Extract video ID from URL
    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        logger.warning("Invalid YouTube URL provided: %s", youtube_url)
        return handle_summarization_error(request, "Invalid YouTube URL.")

    # 2. Summarize comments
    try:
        summary, sorted_comments = await summarize_comments(video_id)
    except CommentFetchError as e:
        logger.error("Fetch-comments failure: %s", e)
        return handle_summarization_error(request, f"Failed to fetch comments: {e}")
    except OpenAIInteractionError as e:
        logger.error("OpenAI failure during summarization: %s", e)
        return handle_summarization_error(
            request, f"AI error during summarization: {e}"
        )
    except Exception:
        logger.exception("Unexpected summarization error")
        return handle_summarization_error(
            request, "Internal error summarizing comments."
        )

    # 3. Store session data in Redis
    session_id = str(uuid.uuid4())
    try:
        # Store summary
        await redis_client.set(f"{session_id}:summary", summary, ex=3600)

        # Compress and encode top comments
        top_comments = sorted_comments[:500]
        comments_json = json.dumps(top_comments)
        compressed = gzip.compress(comments_json.encode("utf-8"))
        encoded = base64.b64encode(compressed).decode("utf-8")
        await redis_client.set(f"{session_id}:comments", encoded, ex=3600)
    except Exception as e:
        logger.error("Redis storage error: %s", e)
        return handle_summarization_error(request, "Internal error storing session.")

    # 4. Set session cookie and render summary
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=3600,
        httponly=True,
        secure=True,
        samesite="strict",
    )
    return templates.TemplateResponse(
        "summary.html", {"request": request, "summary": summary, "error": None}
    )


def handle_chat_error(request: Request, msg: str):
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "error": msg,
            "answer": None,
            "similar_comments": None,
        },
        status_code=200,
    )


@router.post("/question/", response_class=HTMLResponse)
async def answer_question(
    data: QuestionRequest,
    request: Request,
    response: Response,
):
    # 1️⃣ Session + data
    try:
        session_id = request.cookies.get("session_id", "")
        summary, comments = await fetch_summary_and_comments(session_id)
    except (SessionExpiredError, DataCorruptionError) as e:
        return handle_chat_error(str(e))

    # 2️⃣ Embeddings
    try:
        embeddings = await get_or_compute_embeddings(session_id, comments)
    except EmbeddingError as e:
        return handle_chat_error(str(e))

    # 3️⃣ Q&A
    try:
        similar = await search_similar_comments(
            question=data.question,
            embeddings=embeddings,
            comments=comments,
        )
        answer = await generate_answer(data.question, similar, summary)
    except EmbeddingError as e:
        return handle_chat_error(f"Embedding error: {e}")
    except OpenAIInteractionError:
        return handle_chat_error("AI service unavailable. Please try again later.")
    except Exception:
        return handle_chat_error("Unexpected error during Q&A. Please try again.")

    # 4️⃣ Success render
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "error": None,
            "answer": answer,
            "similar_comments": similar,
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

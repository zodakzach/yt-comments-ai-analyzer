# app/api/summarizer_service.py
import logging
from openai import OpenAIError, RateLimitError

from app.services.fetch_comments import fetch_all_comments
from app.core.openai_client import async_client
from app.services.errors import CommentFetchError, OpenAIInteractionError

logger = logging.getLogger(__name__)


async def summarize_comments(video_id: str) -> tuple[str, list[dict]]:
    # 1️⃣ Fetch comments
    try:
        comments = await fetch_all_comments(video_id)
    except CommentFetchError as err:
        logger.error("Failed to fetch comments for %s: %s", video_id, err)
        # bubble up the domain error
        raise

    # 2️⃣ Sort & build prompt
    comments_sorted = sorted(comments, key=lambda x: x["likeCount"], reverse=True)
    prompt = "Summarize the following YouTube comments…\n\n" + "\n".join(
        f"- [{c['likeCount']} likes] {c['text']}" for c in comments_sorted[:50]
    )

    # 3️⃣ Call OpenAI
    try:
        response = await async_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": prompt},
            ],
        )
        summary = response.choices[0].message.content
    except RateLimitError as rl:
        logger.error("OpenAI rate limit for video %s: %s", video_id, rl)
        raise OpenAIInteractionError(f"Rate limit exceeded: {rl}") from rl
    except OpenAIError as oe:
        logger.error("OpenAI API error for video %s: %s", video_id, oe)
        raise OpenAIInteractionError(f"OpenAI chat completion error: {oe}") from oe

    except Exception:
        logger.exception("Unexpected error summarizing %s", video_id)
        raise

    return summary, comments_sorted

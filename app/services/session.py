# app/services/session.py
import base64, gzip, json
from typing import Tuple, List
import numpy as np

from app.core.redis_client import redis_client
from app.services.vectorize import vectorize_comments
from app.services.errors import (
    SessionExpiredError,
    DataCorruptionError,
    EmbeddingError,
)


async def fetch_summary_and_comments(session_id: str) -> Tuple[str, List[dict]]:
    raw_summary = await redis_client.get(f"{session_id}:summary")
    raw_comments = await redis_client.get(f"{session_id}:comments")
    if not raw_summary or not raw_comments:
        raise SessionExpiredError(
            "Session expired or not found. Please summarize a video first."
        )
    try:
        summary = raw_summary
        blob = base64.b64decode(raw_comments)
        comments = json.loads(gzip.decompress(blob).decode("utf-8"))
        return summary, comments
    except Exception:
        raise DataCorruptionError(
            "Corrupted session data. Please try summarizing again."
        )


async def get_or_compute_embeddings(
    session_id: str, comments: list, ttl_seconds: int = 3600
) -> List[np.ndarray]:
    key = f"{session_id}:embeddings"
    raw = await redis_client.get(key)
    if raw:
        try:
            arr = json.loads(gzip.decompress(base64.b64decode(raw)).decode("utf-8"))
            return [np.array(v) for v in arr]
        except Exception:
            # fall through to recompute
            pass

    # compute fresh
    try:
        embeddings = await vectorize_comments(comments)
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}")

    # cache
    try:
        arr = [e.tolist() for e in embeddings]
        blob = gzip.compress(json.dumps(arr).encode("utf-8"))
        await redis_client.set(key, base64.b64encode(blob), ex=ttl_seconds)
    except Exception:
        # caching failure is non-fatal
        pass

    return embeddings

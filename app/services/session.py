# app/services/session.py
import base64
import gzip
import json
import uuid
from typing import Dict, Tuple, List
import numpy as np
import logging

from app.core.redis_client import redis_client
from app.models.schemas import Comment
from app.services.vectorize import vectorize_comments
from app.services.errors import (
    SessionExpiredError,
    DataCorruptionError,
    EmbeddingError,
    SessionStorageError,
)

logger = logging.getLogger(__name__)
REDIS_EXPIRATION_SECONDS = 3600


async def fetch_summary_and_comments(session_id: str) -> Tuple[str, List[Comment]]:
    """
    Retrieve the summary and comments for a given session from Redis.

    The comments are stored as a base64-encoded, gzipped JSON blob.
    This function decodes, decompresses, and parses the comments,
    and returns them as validated Comment objects.

    Args:
        session_id (str): The session identifier.

    Raises:
        SessionExpiredError: If the session data is missing in Redis.
        DataCorruptionError: If the comments data is corrupted or cannot be parsed.

    Returns:
        Tuple[str, List[Comment]]: The summary string and a list of Comment objects.
    """
    raw_summary = await redis_client.get(f"{session_id}:summary")
    raw_comments = await redis_client.get(f"{session_id}:comments")

    if raw_summary is None or raw_comments is None:
        raise SessionExpiredError(
            "Session expired or not found. Please summarize a video first."
        )

    # raw_summary is already a str
    summary: str = raw_summary

    try:
        # 1) Base64-decode the blob (raw_comments is a str)
        gzipped_bytes = base64.b64decode(raw_comments)

        # 2) Decompress to get the original JSON bytes
        json_bytes = gzip.decompress(gzipped_bytes)

        # 3) Parse into Python list of dicts
        comment_dicts = json.loads(json_bytes.decode("utf-8"))

        # 4) Instantiate Comment models (will validate types)
        comments: List[Comment] = [Comment(**c) for c in comment_dicts]

        return summary, comments

    except Exception as e:
        raise DataCorruptionError(
            f"Corrupted session data. Please try summarizing again. ({e})"
        )


async def get_or_compute_embeddings(
    session_id: str,
    comments: List[Comment],
    ttl_seconds: int = REDIS_EXPIRATION_SECONDS,
) -> List[np.ndarray]:
    """
    Retrieve or compute embeddings for a list of comments, caching the result in Redis.

    If embeddings are cached, they are loaded, decompressed, and deserialized.
    If not, embeddings are computed using the vectorize_comments function,
    then serialized, compressed, and cached in Redis for future use.

    Args:
        session_id (str): The session identifier (used as Redis key).
        comments (List[Comment]): List of Comment objects to embed.
        ttl_seconds (int, optional): Time-to-live for the cache in seconds. Defaults to 3600.

    Raises:
        EmbeddingError: If embedding generation fails.

    Returns:
        List[np.ndarray]: List of embedding vectors as numpy arrays.
    """
    key = f"{session_id}:embeddings"

    # 1) Try loading from cache (raw is a str or None)
    raw: str | None = await redis_client.get(key)
    if raw is not None:
        try:
            # a) Base64-decode the ASCII string → compressed bytes
            compressed = base64.b64decode(raw)
            # b) GZIP-decompress → JSON bytes
            json_bytes = gzip.decompress(compressed)
            # c) Parse JSON → list of lists
            arr = json.loads(json_bytes.decode("utf-8"))
            # d) Rehydrate to numpy arrays
            return [np.array(v) for v in arr]
        except Exception:
            # If anything goes wrong, fall through and recompute
            pass

    # 2) Compute fresh embeddings
    try:
        embeddings = await vectorize_comments(comments)
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}")

    # 3) Cache them
    try:
        # a) Convert each array to a plain Python list
        arr = [e.tolist() for e in embeddings]
        # b) JSON-serialize
        json_payload = json.dumps(arr).encode("utf-8")
        # c) Compress
        compressed = gzip.compress(json_payload)
        # d) Base64-encode to an ASCII bytes object, then decode to str
        b64_str = base64.b64encode(compressed).decode("utf-8")
        # e) Store the string in Redis
        await redis_client.set(key, b64_str, ex=ttl_seconds)
    except Exception:
        # Cache failures are non-fatal
        pass

    return embeddings


async def create_full_session(
    summary: str,
    comments: List[Comment],
    total_comments: int,
    sentiment_stats: Dict[str, float],
    expiration: int = REDIS_EXPIRATION_SECONDS,
) -> str:
    """
    Stores everything needed to re-render the summary UI in one Redis blob:
      - summary text
      - list of Comment dicts (with sentiment)
      - total_comments count
      - sentiment_stats (percent positive/neutral/negative)

    Raises:
      - DataCorruptionError: on serialization/compression failure
      - SessionStorageError: on any Redis failure
    """
    session_id = uuid.uuid4().hex
    payload = {
        "summary": summary,
        "comments": [c.model_dump() for c in comments],
        "total_comments": total_comments,
        "sentiment_stats": sentiment_stats,
    }

    # Serialize, compress, encode
    try:
        raw = json.dumps(payload).encode("utf-8")
        compressed = gzip.compress(raw)
        blob = base64.b64encode(compressed).decode("utf-8")
    except (TypeError, ValueError, OSError, UnicodeError) as err:
        logger.error("Serialization error preparing session %s: %s", session_id, err)
        raise DataCorruptionError("Failed to prepare session payload") from err

    # Store single blob under one key
    try:
        await redis_client.set(f"{session_id}:session", blob, ex=expiration)
    except Exception as err:
        logger.error("Error storing full session %s: %s", session_id, err)
        raise SessionStorageError("Could not persist full session data") from err

    return session_id

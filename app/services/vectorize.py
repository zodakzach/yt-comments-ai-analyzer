import numpy as np
import re
import tiktoken
from openai import RateLimitError, OpenAIError
from app.core.openai_client import async_client
from app.models.schemas import Comment
from app.services.errors import EmbeddingError


def clean_text(text: str) -> str:
    """
    Remove non-ASCII characters from the input text.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string containing only ASCII characters.
    """
    return re.sub(r"[^\x00-\x7F]+", "", text)


def is_valid_comment(
    text: str, encoder, max_tokens: int = 8192, max_chars: int = 10000
) -> bool:
    """
    Validate a comment string for embedding.

    Args:
        text (str): The comment text to validate.
        encoder: The tokenizer/encoder for counting tokens.
        max_tokens (int, optional): Maximum allowed tokens. Defaults to 8192.
        max_chars (int, optional): Maximum allowed characters. Defaults to 10000.

    Returns:
        bool: True if the comment is valid, False otherwise.
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text or len(text) > max_chars:
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if len(encoder.encode(text)) > max_tokens:
        return False
    return True


async def vectorize_comments(comments: list[Comment]) -> list[np.ndarray]:
    """
    Generate embeddings for a list of comments using OpenAI's embedding API.

    This function cleans and validates each comment, batches them according to
    API constraints, and asynchronously requests embeddings. Returns a list of
    numpy arrays, one per valid comment.

    Args:
        comments (list[Comment]): List of Comment objects to embed.

    Raises:
        ValueError: If no valid comments are found.
        EmbeddingError: If the OpenAI API returns an error or no embeddings.

    Returns:
        list[np.ndarray]: List of embedding vectors as numpy arrays.
    """
    encoder = tiktoken.encoding_for_model("text-embedding-3-small")

    # 1) Clean & filter
    texts: list[str] = []
    for comment in comments:
        raw = comment.text
        cleaned = clean_text(raw).strip()
        if is_valid_comment(cleaned, encoder):
            texts.append(cleaned)

    if not texts:
        raise ValueError("No valid comment text to embed.")

    # 2) Batch constraints
    MAX_TOKENS_PER_BATCH = 300_000
    MAX_TEXTS_PER_BATCH = 2048

    current_batch: list[str] = []
    current_token_count = 0
    embeddings: list[np.ndarray] = []

    # 3) Loop & send batches
    try:
        for text in texts:
            token_count = len(encoder.encode(text))

            # flush if over limits
            if (current_token_count + token_count > MAX_TOKENS_PER_BATCH) or (
                len(current_batch) >= MAX_TEXTS_PER_BATCH
            ):
                resp = await async_client.embeddings.create(
                    input=current_batch, model="text-embedding-3-small"
                )
                embeddings.extend(np.array(item.embedding) for item in resp.data)

                current_batch = []
                current_token_count = 0

            current_batch.append(text)
            current_token_count += token_count

        # 4) Final batch
        if current_batch:
            resp = await async_client.embeddings.create(
                input=current_batch, model="text-embedding-3-small"
            )
            embeddings.extend(np.array(item.embedding) for item in resp.data)

    except RateLimitError as rl:
        raise EmbeddingError(f"Embedding rate limit exceeded: {rl}") from rl
    except OpenAIError as oe:
        raise EmbeddingError(f"OpenAI embedding error: {oe}") from oe

    if not embeddings:
        raise EmbeddingError("No embeddings were returned from OpenAI.")

    return embeddings

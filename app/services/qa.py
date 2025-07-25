import logging
from typing import List
import numpy as np
from openai import OpenAIError, RateLimitError
from app.core.openai_client import async_client
from app.models.schemas import Comment, Session
from app.services.errors import OpenAIInteractionError, EmbeddingError

logger = logging.getLogger(__name__)


async def search_similar_comments(
    question: str,
    embeddings: List[np.ndarray],
    comments: List[Comment],
    top_k: int = 5,
) -> List[Comment]:
    """
    Find the most similar comments to a question using cosine similarity.

    This function embeds the question, computes cosine similarity between the question
    and each comment embedding, and returns the top_k most similar comments.

    Args:
        question (str): The user's question.
        embeddings (List[np.ndarray]): List of comment embedding vectors.
        comments (List[Comment]): List of Comment objects corresponding to embeddings.
        top_k (int, optional): Number of top similar comments to return. Defaults to 5.

    Raises:
        EmbeddingError: If embedding the question fails.

    Returns:
        List[Comment]: The top_k most similar Comment objects.
    """
    try:
        response = await async_client.embeddings.create(
            input=[question], model="text-embedding-3-small"
        )
        question_vector = np.array(response.data[0].embedding)

    except RateLimitError as rl:
        logger.warning("OpenAI rate limit while embedding question: %s", rl)
        raise EmbeddingError(f"Rate limit exceeded: {rl}") from rl

    except OpenAIError as oe:
        logger.error("OpenAI embedding error for question '%s': %s", question, oe)
        raise EmbeddingError(f"Failed to embed question: {oe}") from oe

    # Similarity logic
    q_norm = question_vector / np.linalg.norm(question_vector)
    comment_vectors = [e / np.linalg.norm(e) for e in embeddings]
    sims = [(i, np.dot(q_norm, e)) for i, e in enumerate(comment_vectors)]
    sims.sort(key=lambda x: x[1], reverse=True)

    return [comments[i] for i, _ in sims[:top_k]]


async def generate_answer(
    question: str, relevant_comments: List[Comment], session: Session
) -> str:
    """
    Generate an answer to a question using a video summary and relevant comments.

    This function builds a prompt including the video summary, relevant comments,
    and the user's question, then calls the OpenAI API to generate an answer.

    Args:
        question (str): The user's question.
        relevant_comments (List[Comment]): List of comments most relevant to the question.
        session (Session): The session containing video metadata and summary.

    Raises:
        OpenAIInteractionError: If the OpenAI API call fails.

    Returns:
        str: The generated answer from the language model.
    """
    related_text = "\n".join(f"- {c.text}" for c in relevant_comments)
    prompt = f"""
    You are an intelligent assistant that answers questions about YouTube videos comments section. You are provided with the video’s metadata, an AI-generated summary, selected related comments, and audience sentiment insights. Use all relevant information to generate accurate, concise, and helpful answers to user questions.

    Video Information:
    - Title: {session.video_info.title}
    - Published At: {session.video_info.publishedAt}
    - Views: {session.video_info.viewCount}
    - Likes: {session.video_info.likeCount}
    - URL: {session.video_info.url}
    - Thumbnail: {session.video_info.thumbnailUrl}

    Video Summary:
    {session.summary}

    Related Comments:
    {related_text}

    Comment Insights:
    - Total Comments Fetched: {session.total_comments}
    - Sentiment Stats: {session.sentiment_stats}

    Question:
    {question}

    Based on the video information, summary, related comments, and sentiment insights, provide a clear, concise, and informed answer.
    """

    try:
        response = await async_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    except RateLimitError as rl:
        logger.warning("OpenAI rate limit during answer generation: %s", rl)
        raise OpenAIInteractionError(
            f"Rate limit exceeded during answer generation: {rl}"
        ) from rl

    except OpenAIError as oe:
        logger.error("OpenAI chat error for question '%s': %s", question, oe)
        raise OpenAIInteractionError(f"Failed to generate answer: {oe}") from oe

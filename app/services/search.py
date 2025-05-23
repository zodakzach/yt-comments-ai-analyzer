from typing import List, Dict, Any
import numpy as np
from app.core.openai_client import async_client


async def search_similar_comments(
    question: str,
    embeddings: List[np.ndarray],
    comments: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    # Embed the question
    response = await async_client.embeddings.create(
        input=[question], model="text-embedding-3-small"
    )
    question_vector = np.array(response.data[0].embedding)

    # Normalize and compute cosine similarities
    q_norm = question_vector / np.linalg.norm(question_vector)
    comment_vectors = [e / np.linalg.norm(e) for e in embeddings]
    sims = [(i, np.dot(q_norm, e)) for i, e in enumerate(comment_vectors)]
    sims.sort(key=lambda x: x[1], reverse=True)

    # Return the actual comment dicts
    return [comments[i] for i, _ in sims[:top_k]]


async def generate_answer(
    question: str, relevant_comments: List[Dict[str, Any]], summary: str
) -> str:
    # Build the prompt
    related_text = "\n".join(f"- {c['text']}" for c in relevant_comments)
    prompt = f"""
Video Summary:
{summary}

Related Comments:
{related_text}

Question: {question}

Based on the summary and related comments, please provide a clear, concise answer.
"""

    # Call the chat completion API
    response = await async_client.chat.completions.create(
        model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAIError, RateLimitError

from app.core.openai_client import async_client
from app.models.schemas import Comment, Session
from app.services.errors import OpenAIInteractionError, EmbeddingError
from numpy.typing import NDArray, ArrayLike

logger = logging.getLogger(__name__)

Float32Mat = NDArray[np.float32]

# --------------------------- Planner Data Model -----------------------------


@dataclass
class RetrievalPlan:
    need_comments: bool = True
    need_summary: bool = True
    prefer_recent: bool = False
    top_k: int = 8
    per_query_k: int = 5
    rerank: bool = True
    query_rewrites: List[str] = field(default_factory=list)
    min_keywords: List[str] = field(default_factory=list)
    answer_instructions: str = ""
    rationale: str = ""


# ----------------------------- Planner Step --------------------------------


async def plan_retrieval(question: str, session: Session) -> RetrievalPlan:
    """
    Ask the LLM to plan retrieval: do we need comments, how many,
    query rewrites, and any answer instructions.
    """
    system = (
        "You are a retrieval planner for a YouTube comments QA agent. "
        "Return only valid JSON with fields: "
        "{need_comments: bool, need_summary: bool, prefer_recent: bool, "
        " top_k: int, per_query_k: int, rerank: bool, "
        " query_rewrites: string[], min_keywords: string[], "
        " answer_instructions: string, rationale: string}."
    )

    user = (
        f"Video Title: {session.video_info.title}\n"
        f"Summary: {session.summary[:1200]}\n\n"
        f"User Question: {question}\n\n"
        "Guidelines:\n"
        "- If the question is opinion/consensus-based, prefer comments.\n"
        "- If it's factual about the video content, summary may suffice.\n"
        "- Provide 2-5 query_rewrites that will best retrieve relevant comments.\n"
        "- Set per_query_k small (3-7) and top_k to final merged size.\n"
        "- Set rerank true when the question is nuanced/long.\n"
        "- answer_instructions should guide the final answer style "
        "(e.g., 'cite top 3 comments', 'summarize consensus')."
    )

    try:
        resp = await async_client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        plan = RetrievalPlan(
            need_comments=bool(data.get("need_comments", True)),
            need_summary=bool(data.get("need_summary", True)),
            prefer_recent=bool(data.get("prefer_recent", False)),
            top_k=int(data.get("top_k", 8)),
            per_query_k=int(data.get("per_query_k", 5)),
            rerank=bool(data.get("rerank", True)),
            query_rewrites=list(data.get("query_rewrites", [])),
            min_keywords=list(data.get("min_keywords", [])),
            answer_instructions=str(data.get("answer_instructions", "")),
            rationale=str(data.get("rationale", "")),
        )
        return plan
    except RateLimitError as rl:
        logger.warning("Plan retrieval rate limit: %s", rl)
        # Fall back to a default, simple plan
        return RetrievalPlan(
            query_rewrites=[question],
            answer_instructions="Be concise and cite 2-3 comments if used.",
            rationale="Fallback plan due to rate limit",
        )
    except (OpenAIError, json.JSONDecodeError) as e:
        logger.error("Planning failed: %s", e)
        # Safe default
        return RetrievalPlan(
            query_rewrites=[question],
            answer_instructions="Be concise and cite 2-3 comments if used.",
            rationale=f"Fallback plan due to error: {e}",
        )


# ------------------------- Embedding Utilities -----------------------------


def _normalize_matrix(vectors: ArrayLike) -> Float32Mat:
    """
    Normalize rows to unit length. Accepts list[list[float]], list[np.ndarray],
    or a 2D np.ndarray. Returns a (N, D) float32 ndarray.
    """
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


async def _embed_texts(
    texts: List[str], model: str = "text-embedding-3-small"
) -> np.ndarray:
    """
    Embed multiple texts in a single API call. Returns a 2D array of
    normalized vectors (rows).
    """
    try:
        resp = await async_client.embeddings.create(input=texts, model=model)
        vecs = [d.embedding for d in resp.data]
        return _normalize_matrix(vecs)
    except RateLimitError as rl:
        logger.warning("OpenAI rate limit while embedding: %s", rl)
        raise EmbeddingError(f"Rate limit exceeded: {rl}") from rl
    except OpenAIError as oe:
        logger.error("OpenAI embedding error: %s", oe)
        raise EmbeddingError(f"Failed to embed texts: {oe}") from oe


# -------------------------- Multi-query Retrieval --------------------------


async def search_similar_comments_multi(
    queries: List[str],
    embeddings: List[np.ndarray],
    comments: List[Comment],
    per_query_k: int,
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    Embed multiple queries at once, compute cosine similarities to all
    comment embeddings, take top per query, then merge via MaxSim and
    return top_k (index, score) pairs.
    """
    if not comments or not embeddings or not queries:
        return []

    # Normalize comment embeddings once
    comment_mat = _normalize_matrix(embeddings)
    if comment_mat.shape[0] == 0:
        return []

    # Embed queries (normalized)
    query_mat = await _embed_texts(queries)

    # Cosine similarity since all rows are normalized: dot product
    # query_mat: (Q, D), comment_mat.T: (D, N) => sims: (Q, N)
    sims = np.dot(query_mat, comment_mat.T)

    # Optionally do a keyword filter to bias candidates
    # (Simple heuristic: boost sims if min_keywords appear)
    # This can be moved into reranking for better accuracy.

    # Take top per query
    per_query = min(per_query_k, max(1, len(comments)))
    candidates: Dict[int, float] = {}
    for qi in range(sims.shape[0]):
        row = sims[qi]
        top_idx = np.argpartition(row, -per_query)[-per_query:]
        # Exact sort of these top candidates
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        for idx in top_idx:
            score = float(row[idx])
            # MaxSim merge
            if idx not in candidates or score > candidates[idx]:
                candidates[idx] = score

    # Take global top_k
    k = min(top_k, len(candidates))
    best = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k]
    return best


# ------------------------------ Reranking ----------------------------------


async def rerank_with_llm(
    question: str,
    candidates: List[Tuple[int, float]],
    comments: List[Comment],
    limit: int,
) -> List[Tuple[int, float]]:
    """
    Ask the LLM to rerank candidate comments by relevance to the question.
    Returns a list of (index, score) with higher score being better.
    """
    if not candidates:
        return []

    # Prepare a small pack for the model (cap length to control tokens)
    max_considered = min(50, len(candidates))
    pack = []
    for rank, (idx, score) in enumerate(candidates[:max_considered], start=1):
        text = comments[idx].text[:400]
        pack.append({"idx": idx, "text": text, "pre_score": score})

    sys = (
        "You are a reranker. Given a question and candidate comments, "
        "assign each a relevance score 0..10 and return JSON with "
        "scores: [{idx: int, score: number}] sorted descending."
    )
    usr = f"Question: {question}\nCandidates:\n" + "\n".join(
        f"- idx={c['idx']} pre={c['pre_score']:.3f} text={c['text']}" for c in pack
    )

    try:
        resp = await async_client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        scores = data.get("scores", [])
        scored = []
        for item in scores:
            idx = int(item.get("idx"))
            sc = float(item.get("score", 0.0))
            if any(idx == ci for ci, _ in candidates):
                scored.append((idx, sc))
        # Fallback: if the model returned nothing usable, just return original
        if not scored:
            return candidates[:limit]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
    except (RateLimitError, OpenAIError, json.JSONDecodeError) as e:
        logger.warning("Rerank failed, using original scores: %s", e)
        return candidates[:limit]


# --------------------------- Reflection / Check ----------------------------


async def coverage_check_and_refine(
    question: str, selected_comments: List[Comment]
) -> Dict:
    """
    Ask the LLM if the current comments are sufficient to answer the question.
    Returns JSON with: {need_more: bool, reason: str, new_queries: string[]}
    """
    examples = "\n".join(f"- {c.text[:300]}" for c in selected_comments[:8])
    sys = (
        "You are a coverage checker. Decide if the comments shown are "
        "sufficient to answer the question. Return JSON: "
        "{need_more: bool, reason: string, new_queries: string[]}. "
        "Only add new_queries if something key is missing."
    )
    usr = f"Question: {question}\nCurrent Comments (sample):\n{examples}\n"

    try:
        resp = await async_client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        need_more = bool(data.get("need_more", False))
        reason = str(data.get("reason", ""))
        new_queries = list(data.get("new_queries", []))
        return {
            "need_more": need_more,
            "reason": reason,
            "new_queries": new_queries,
        }
    except (RateLimitError, OpenAIError, json.JSONDecodeError) as e:
        logger.warning("Coverage check failed, assuming sufficient: %s", e)
        return {"need_more": False, "reason": str(e), "new_queries": []}


# ------------------------ Answer Generation (updated) ----------------------


async def generate_answer(
    question: str,
    relevant_comments: List[Comment],
    session: Session,
    answer_instructions: str = "",
) -> str:
    """
    Generate an answer using video summary and relevant comments.
    Added 'answer_instructions' to let the planner steer the style.
    """
    related_text = "\n".join(f"- {c.text}" for c in relevant_comments)
    prompt = f"""
You are an intelligent assistant that answers questions about YouTube videos'
comments section. Use video metadata, the summary, selected comments, and
sentiment insights to answer accurately and concisely. If information is
insufficient, say so and suggest what else is needed.

Answer Instructions (from planner):
{answer_instructions or "None"}

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
{related_text or "(none selected)"}

Comment Insights:
- Total Comments Fetched: {session.total_comments}
- Sentiment Stats: {session.sentiment_stats}

Question:
{question}

Provide a clear, concise, and grounded answer. Prefer consensus from comments
when relevant. If citing comments, do so naturally (e.g., “Several viewers
mentioned …”). Avoid fabricating details.
""".strip()

    try:
        response = await async_client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
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


# --------------------------- Agent Orchestration ---------------------------


@dataclass
class AgentResult:
    answer: str
    used_comments: List[Comment]


async def agent_answer(
    question: str,
    session: Session,
    comment_embeddings: List[np.ndarray],
    comments: List[Comment],
    max_loops: int = 2,
) -> AgentResult:
    plan = await plan_retrieval(question, session)
    logger.debug("Retrieval plan: %s", plan)

    selected: List[Comment] = []
    if plan.need_comments:
        queries = [question] + [q for q in plan.query_rewrites if q.strip()]

        cand = await search_similar_comments_multi(
            queries=queries,
            embeddings=comment_embeddings,
            comments=comments,
            per_query_k=plan.per_query_k,
            top_k=max(plan.top_k, 5),
        )

        if plan.rerank and cand:
            cand = await rerank_with_llm(
                question=question,
                candidates=cand,
                comments=comments,
                limit=plan.top_k,
            )
        else:
            cand = cand[: plan.top_k]

        selected_idx = [i for i, _ in cand]
        selected = [comments[i] for i in selected_idx]

        loops = 0
        while loops < max_loops:
            loops += 1
            check = await coverage_check_and_refine(question, selected)
            if not check.get("need_more"):
                break

            new_queries = [q for q in check.get("new_queries", []) if q.strip()]
            if not new_queries:
                break

            more = await search_similar_comments_multi(
                queries=new_queries,
                embeddings=comment_embeddings,
                comments=comments,
                per_query_k=max(3, plan.per_query_k // 2),
                top_k=plan.top_k * 2,
            )

            merged = {i: s for i, s in cand}
            for i, s in more:
                if i not in merged or s > merged[i]:
                    merged[i] = s

            merged_list = sorted(merged.items(), key=lambda x: x[1], reverse=True)

            if plan.rerank:
                merged_list = await rerank_with_llm(
                    question=question,
                    candidates=merged_list,
                    comments=comments,
                    limit=plan.top_k,
                )
            else:
                merged_list = merged_list[: plan.top_k]

            selected_idx = [i for i, _ in merged_list]
            selected = [comments[i] for i in selected_idx]

    answer = await generate_answer(
        question=question,
        relevant_comments=selected,
        session=session,
        answer_instructions=plan.answer_instructions,
    )
    return AgentResult(answer=answer, used_comments=selected)

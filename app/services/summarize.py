from app.services.fetch_comments import fetch_comments
from app.core.openai_client import async_client  # AsyncOpenAI instance


async def summarize_comments(video_id: str) -> tuple[str, list[dict]]:
    # Fetch and sort comments
    comments = await fetch_comments(video_id)
    comments_sorted = sorted(comments, key=lambda x: x["likeCount"], reverse=True)

    # Build prompt from top 50 comments
    prompt_lines = [
        f"- [{c['likeCount']} likes] {c['text']}" for c in comments_sorted[:50]
    ]
    prompt = (
        "Summarize the following YouTube comments. Comments with more likes are more important:\n\n"
        + "\n".join(prompt_lines)
    )

    # Call OpenAI asynchronously
    response = await async_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt},
        ],
    )

    summary = response.choices[0].message.content
    return summary, comments_sorted

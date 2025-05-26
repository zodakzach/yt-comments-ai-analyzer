import httpx
from app.core.config import settings  # Use centralized config
from app.services.errors import CommentFetchError

BASE_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


async def fetch_all_comments(video_id: str) -> list[dict]:
    YT_API_KEY = settings.YOUTUBE_API_KEY
    comments = []
    next_page_token = None

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "key": YT_API_KEY,
                "pageToken": next_page_token,
            }
            try:
                response = await client.get(BASE_URL, params=params)
                response.raise_for_status()
            except httpx.RequestError as exc:
                # network-level (DNS, timeout, etc.)
                raise CommentFetchError(
                    f"Network error fetching comments: {exc}"
                ) from exc
            except httpx.HTTPStatusError as exc:
                # non-2xx status code
                text = exc.response.text
                code = exc.response.status_code
                raise CommentFetchError(f"YouTube API returned {code}: {text}") from exc

            # at this point we know status_code == 200
            try:
                data = response.json()
            except ValueError as exc:
                raise CommentFetchError("Invalid JSON in YouTube response") from exc

            for item in data.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(
                    {
                        "author": top["authorDisplayName"],
                        "text": top["textOriginal"],
                        "likeCount": top.get("likeCount", 0),
                        "publishedAt": top["publishedAt"],
                    }
                )

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

    return comments

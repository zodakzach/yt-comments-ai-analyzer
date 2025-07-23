import httpx
from app.core.config import settings
from typing import Literal
from app.models.schemas import VideoInfo

ThumbnailQuality = Literal[
    "default", "mqdefault", "hqdefault", "sddefault", "maxresdefault"
]


async def fetch_video_info(video_id: str) -> dict:
    """
    Asynchronously fetch YouTube video information (title, like count, view count, published date).

    Parameters:
        video_id (str): The YouTube video ID.

    Returns:
        dict: {
            "title": str,
            "likeCount": int,
            "viewCount": int,
            "publishedAt": str
        }
    """
    api_key = settings.YOUTUBE_API_KEY
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet,statistics", "id": video_id, "key": api_key}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    items = data.get("items", [])
    if not items:
        raise ValueError(f"No video found with ID {video_id}")

    snippet = items[0]["snippet"]
    stats = items[0]["statistics"]

    return {
        "title": snippet.get("title"),
        "likeCount": int(stats.get("likeCount", 0)),
        "viewCount": int(stats.get("viewCount", 0)),
        "publishedAt": snippet.get("publishedAt"),
    }


def get_youtube_thumbnail_url(video_id: str, quality: str = "hqdefault") -> str:
    """
    Construct a YouTube video thumbnail URL.

    Parameters:
        video_id (str): The YouTube video ID.
        quality (str): One of 'default', 'mqdefault', 'hqdefault', 'sddefault', 'maxresdefault'.
                       Defaults to 'hqdefault'.

    Returns:
        str: The URL to the thumbnail image.
    """
    valid_qualities = {
        "default",
        "mqdefault",
        "hqdefault",
        "sddefault",
        "maxresdefault",
    }
    if quality not in valid_qualities:
        raise ValueError(
            f"Invalid quality '{quality}'. Choose one of {valid_qualities}."
        )
    return f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"


async def build_video_object(
    video_id: str,
    thumbnail_quality: ThumbnailQuality = "hqdefault",
) -> VideoInfo:
    """
    Fetch full video info and wrap it in a VideoInfo model,
    with the YouTube watch URL and a validated thumbnail URL.

    Parameters:
        video_id (str): The YouTube video ID.
        thumbnail_quality (str): One of 'default', 'mqdefault', 'hqdefault', 'sddefault', 'maxresdefault'.

    Returns:
        VideoInfo: Pydantic model with:
          - title: str
          - likeCount: int
          - viewCount: int
          - publishedAt: str
          - url: HttpUrl
          - thumbnailUrl: HttpUrl
    """
    # 1) Grab the basic metadata (dict with keys title, likeCount, viewCount, publishedAt, etc.)
    info: dict[str, str | int] = await fetch_video_info(video_id)

    # 2) Build the watch URL
    info["url"] = f"https://www.youtube.com/watch?v={video_id}"

    # 3) Attach a thumbnail URL (may raise ValueError if quality invalid)
    info["thumbnailUrl"] = get_youtube_thumbnail_url(
        video_id, quality=thumbnail_quality
    )

    # 4) Validate & return as VideoInfo
    #    In Pydantic v2, use model_validate to coerce/validate types.
    return VideoInfo.model_validate(info)

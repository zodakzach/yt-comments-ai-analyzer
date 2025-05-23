import httpx
from app.core.config import settings  # Use centralized config

BASE_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


async def fetch_all_comments(video_id):
    YT_API_KEY = settings.YOUTUBE_API_KEY  # From Pydantic settings

    comments = []
    next_page_token = None

    async with httpx.AsyncClient() as client:
        while True:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "key": YT_API_KEY,
                "pageToken": next_page_token,
            }
            response = await client.get(BASE_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("items", []):
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(
                        {
                            "author": top_comment["authorDisplayName"],
                            "text": top_comment["textOriginal"],
                            "likeCount": top_comment.get("likeCount", 0),
                            "publishedAt": top_comment["publishedAt"],
                        }
                    )
                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break
            else:
                raise Exception(
                    f"Failed to fetch comments: {response.status_code} - {response.text}"
                )

    return comments

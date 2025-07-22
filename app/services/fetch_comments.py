import asyncio
from typing import List
import httpx
from app.core.config import settings
from app.services.errors import CommentFetchError
from app.models.schemas import Comment

BASE_THREADS = "https://www.googleapis.com/youtube/v3/commentThreads"
BASE_COMMENTS = "https://www.googleapis.com/youtube/v3/comments"
SEM = asyncio.Semaphore(8)  # limit concurrency

async def fetch_all_comments(video_id: str) -> List[Comment]:
    api_key = settings.YOUTUBE_API_KEY
    comments: List[Comment] = []
    page_token: str | None = None

    async with httpx.AsyncClient(timeout=10.0, http2=True) as client:
        while True:
            # 1) grab one page of threads (with first 5 replies inline)
            resp = await client.get(
                BASE_THREADS,
                params={
                    "part": "snippet,replies",
                    "videoId": video_id,
                    "maxResults": 100,
                    "key": api_key,
                    "pageToken": page_token,
                },
            )
            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                raise CommentFetchError(f"YT page error: {e}") from e

            items = data.get("items", [])
            # collect parentIds that actually have more replies
            tasks = []
            for item in items:
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(
                    Comment(
                        author=top["authorDisplayName"],
                        text=top["textOriginal"],
                        likeCount=top.get("likeCount", 0),
                    )
                )

                # inline replies (up to 5)
                for rep in item.get("replies", {}).get("comments", []):
                    sn = rep["snippet"]
                    comments.append(
                        Comment(
                            author=sn["authorDisplayName"],
                            text=sn["textOriginal"],
                            likeCount=sn.get("likeCount", 0),
                        )
                    )

                # schedule a full‐fetch only if there are more replies
                if item["snippet"].get("totalReplyCount", 0) > len(item.get("replies", {}).get("comments", [])):
                    parent_id = item["snippet"]["topLevelComment"]["id"]
                    tasks.append(_fetch_all_replies(parent_id, client, api_key))

            # 2) fire off all the “extra replies” jobs in parallel
            if tasks:
                results = await asyncio.gather(*tasks)
                for reply_list in results:
                    comments.extend(reply_list)

            # 3) next page?
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return comments


async def _fetch_all_replies(
    parent_id: str, client: httpx.AsyncClient, api_key: str
) -> List[Comment]:
    replies: List[Comment] = []
    token: str | None = None

    # bound concurrency
    async with SEM:
        while True:
            resp = await client.get(
                BASE_COMMENTS,
                params={
                    "part": "snippet",
                    "parentId": parent_id,
                    "maxResults": 100,
                    "key": api_key,
                    "pageToken": token,
                },
            )
            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                raise CommentFetchError(f"Reply fetch error: {e}") from e

            for item in data.get("items", []):
                sn = item["snippet"]
                replies.append(
                    Comment(
                        author=sn["authorDisplayName"],
                        text=sn["textOriginal"],
                        likeCount=sn.get("likeCount", 0),
                    )
                )

            token = data.get("nextPageToken")
            if not token:
                break

    return replies

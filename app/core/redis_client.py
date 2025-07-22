from upstash_redis.asyncio import Redis
from app.core.config import settings

redis_client = Redis(
    url=str(settings.UPSTASH_REDIS_REST_URL), token=str(settings.UPSTASH_REDIS_REST_TOKEN)
)

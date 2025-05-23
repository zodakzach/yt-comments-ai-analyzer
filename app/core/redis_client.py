from dotenv import load_dotenv
load_dotenv()   # ⬅️ load .env into os.environ

from upstash_redis.asyncio import Redis

# Now UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are available
redis_client = Redis.from_env()

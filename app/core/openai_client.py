from openai import OpenAI
from openai import AsyncOpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.THREAD_OPENAI_API_KEY)

async_client = AsyncOpenAI(api_key=settings.THREAD_OPENAI_API_KEY)

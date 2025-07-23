from pathlib import Path
from dotenv import load_dotenv
import os

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
ENV_PATH = BASE_DIR / ".env"

# ─── Load .env ──────────────────────────────────────────────────────────────────
if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")
# override=True makes sure any existing os.environ value
# is replaced by what's in your .env
load_dotenv(dotenv_path=str(ENV_PATH), encoding="utf-8", override=True)


# ─── Settings class ────────────────────────────────────────────────────────────
class Settings:
    def __init__(self):
        # Will KeyError if missing; fail fast
        self.YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]
        self.THREAD_OPENAI_API_KEY = os.environ["THREAD_OPENAI_API_KEY"]
        self.UPSTASH_REDIS_REST_URL = os.environ["UPSTASH_REDIS_REST_URL"]
        self.UPSTASH_REDIS_REST_TOKEN = os.environ["UPSTASH_REDIS_REST_TOKEN"]

        # sanity‐check empties
        for name in (
            "YOUTUBE_API_KEY",
            "THREAD_OPENAI_API_KEY",
            "UPSTASH_REDIS_REST_URL",
            "UPSTASH_REDIS_REST_TOKEN",
        ):
            val = getattr(self, name)
            if not val:
                raise RuntimeError(f"Env var {name!r} is empty")


# single shared instance
settings = Settings()

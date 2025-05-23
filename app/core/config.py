from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    YOUTUBE_API_KEY: str
    THREAD_OPENAI_API_KEY: str

    # Tell BaseSettings to load .env and ignore any other env vars
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"           # ← ignore UPSTASH_* and any other extras
    )

settings = Settings()

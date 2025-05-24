from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    YOUTUBE_API_KEY: str
    THREAD_OPENAI_API_KEY: str

    # Add your Upstash credentials here
    UPSTASH_REDIS_REST_URL: str
    UPSTASH_REDIS_REST_TOKEN: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore anything else
    )


settings = Settings()

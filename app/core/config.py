from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()  # Optional if using Pydantic’s auto-loading


class Settings(BaseSettings):
    YOUTUBE_API_KEY: str
    THREAD_OPENAI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

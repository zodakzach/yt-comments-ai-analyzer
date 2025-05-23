from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite database file
DATABASE_URL = "sqlite+aiosqlite:///./comments.db"

# Create async database engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create sessionmaker
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Base model for ORM
Base = declarative_base()


# Dependency for FastAPI routes
async def get_db():
    async with async_session() as session:
        yield session

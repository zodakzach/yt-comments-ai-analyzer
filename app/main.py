# app/main.py
from app.core.logging import init_logging

# 1️⃣ Initialize logging first
init_logging()

import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

logger = logging.getLogger(__name__)  # now logs as "app.main"

# ─── App & CORS ───────────────────────────────────────────────────────────────
app = FastAPI(title="YouTube Comment Summarizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Timing & Logging Middleware ──────────────────────────────────────────────
@app.middleware("http")
async def log_and_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(
        "%s %s → %d in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    response.headers["X-Process-Time"] = f"{duration:.3f}"
    return response


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Welcome to the YouTube Comment Summarizer API"}


app.include_router(api_router)

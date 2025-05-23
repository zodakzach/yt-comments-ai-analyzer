# app/main.py
import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("thread-summarizer")

# ─── App & CORS ───────────────────────────────────────────────────────────────
app = FastAPI(title="Thread Summarizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down in prod
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

    # Log method, path, status code, and duration
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} in {duration:.3f}s"
    )

    # Optionally add the timing header
    response.headers["X-Process-Time"] = f"{duration:.3f}"
    return response

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Welcome to the Thread Summarizer API"}

app.include_router(api_router)

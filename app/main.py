import time
import logging
from fastapi import FastAPI, Request
from app.core.logging import init_logging
from fastapi.staticfiles import StaticFiles
from app.core.config import STATIC_DIR
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

# ─── App & CORS ───────────────────────────────────────────────────────────────
app = FastAPI(title="YouTube Comment Summarizer API", version="0.1.0")
# Mount the static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Timing & Logging Middleware ──────────────────────────────────────────────
# Initialize logging
init_logging()

logger = logging.getLogger(__name__)


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
app.include_router(api_router)

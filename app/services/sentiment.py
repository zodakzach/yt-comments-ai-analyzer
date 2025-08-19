import os
from pathlib import Path
from typing import Dict, List

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from app.core.config import BASE_DIR
from app.models.schemas import Comment

# - If NLTK_DATA is set in the environment, use that.
# - Otherwise, default to <project_root>/nltk_data.
NLTK_DATA_DIR = Path(os.environ.get("NLTK_DATA", BASE_DIR / "nltk_data"))


def ensure_vader(download_dir: Path) -> None:
    """
    Ensure the VADER lexicon is available in download_dir.
    If missing, download it to that directory.
    """
    # Make sure NLTK searches our directory
    if str(download_dir) not in nltk.data.path:
        nltk.data.path.append(str(download_dir))

    # Check if VADER is already present (zipped or unzipped)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        return
    except LookupError:
        pass

    try:
        nltk.data.find("sentiment/vader_lexicon")
        return
    except LookupError:
        pass

    # Not found: download to our directory
    download_dir.mkdir(parents=True, exist_ok=True)
    nltk.download("vader_lexicon", download_dir=str(download_dir), quiet=True)

    # Re-verify after download (some environments need a second path append)
    if str(download_dir) not in nltk.data.path:
        nltk.data.path.append(str(download_dir))

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError as e_unzipped:
            raise RuntimeError(
                f"Failed to locate VADER lexicon in {download_dir}. "
                "Ensure write permissions or set NLTK_DATA to a writable dir."
            ) from e_unzipped


# Ensure VADER is present at startup in the chosen directory
ensure_vader(NLTK_DATA_DIR)

_sia = SentimentIntensityAnalyzer()


def analyze_comment_sentiment(text: str) -> Dict[str, float]:
    """
    Returns VADER scores for a single piece of text:
      - neg, neu, pos, compound
    """
    return _sia.polarity_scores(text)


def annotate_comments_with_sentiment(comments: List[Comment]) -> List[Comment]:
    """
    Adds a .sentiment dict to each Comment.
    """
    for c in comments:
        c.sentiment = analyze_comment_sentiment(c.text)
    return comments


def compute_sentiment_stats(comments: List[Comment]) -> Dict[str, float]:
    """
    Computes percent of comments that are positive (compound >= .05),
    negative (compound <= -.05), and neutral (-.05 < compound < .05).
    """
    total = len(comments)
    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    pos = sum(1 for c in comments if c.sentiment["compound"] >= 0.05)
    neg = sum(1 for c in comments if c.sentiment["compound"] <= -0.05)
    neu = total - pos - neg

    return {
        "positive": pos / total * 100,
        "negative": neg / total * 100,
        "neutral": neu / total * 100,
    }

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import List, Dict
from app.models.schemas import Comment

# Download once at startup
nltk.download("vader_lexicon", quiet=True)

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

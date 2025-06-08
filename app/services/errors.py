"""app/services/errors.py

Custom exception types for session handling, data decoding, and AI interactions.
"""


class SessionExpiredError(Exception):
    """Raised when a session is missing or expired."""

    pass


class DataCorruptionError(Exception):
    """Raised when session data is corrupted or cannot be decoded."""

    pass


class CommentFetchError(Exception):
    def __init__(self, message="Failed to fetch comments"):
        super().__init__(message)


class OpenAIInteractionError(Exception):
    """Raised when OpenAI API call fails."""

    pass


class EmbeddingError(Exception):
    """Raised when something goes wrong generating embeddings."""

    pass


class SessionStorageError(Exception):
    """Raised when storing or retrieving a session fails."""

    pass

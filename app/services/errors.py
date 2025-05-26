class CommentFetchError(Exception):
    """Something went wrong retrieving YouTube comments."""

    pass


class OpenAIInteractionError(Exception):
    """Raised when OpenAI API call fails."""

    pass


class EmbeddingError(Exception):
    """Raised when something goes wrong generating embeddings."""

    pass

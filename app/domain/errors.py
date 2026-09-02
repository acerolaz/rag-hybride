class UnparsableDocumentError(Exception):
    """Raised when a source document cannot be normalized into the pivot schema."""


class UnsupportedFormatError(Exception):
    """Raised when no parser is registered for a document's file extension."""


class EmbeddingServiceError(Exception):
    """Raised when the embedding provider fails to return a vector."""

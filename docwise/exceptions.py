"""Domain exceptions with messages suitable for CLI and UI display."""


class DocWiseError(Exception):
    """Base exception for application errors."""


class UnsupportedDocumentError(DocWiseError):
    """Raised when a document type is not supported."""


class InvalidDocumentError(DocWiseError):
    """Raised when a document cannot be parsed safely."""


class EmptyDocumentError(DocWiseError):
    """Raised when a document contains no extractable text."""


class DocumentTooLargeError(DocWiseError):
    """Raised when an upload exceeds the configured size limit."""


class DocumentNotFoundError(DocWiseError):
    """Raised when an indexed document cannot be found."""


class ConfigurationError(DocWiseError):
    """Raised when required application configuration is missing."""

"""DocWise AI - a local-first document retrieval augmented generation system."""

from .config import Settings
from .models import DocumentRecord, IndexResult, RAGResponse, RetrievedChunk

__all__ = ["DocumentRecord", "IndexResult", "RAGResponse", "RetrievedChunk", "Settings"]

__version__ = "1.0.0"

"""
Package RAG - Retrieval-Augmented Generation pour les messages.
"""

from .engine import RAGEngine, get_rag_engine
from .chunking import TextChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .llm_client import OllamaClient

__all__ = [
    "RAGEngine",
    "get_rag_engine",
    "TextChunker",
    "EmbeddingManager",
    "VectorStore",
    "OllamaClient",
]

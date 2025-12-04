"""
Package RAG - Retrieval-Augmented Generation pour les messages.

Fonctionnalités:
- Ingestion multi-format (JSON, CSV, TXT)
- Chunking intelligent (fenêtre de conversation)
- Recherche hybride (Vector + BM25)
- Re-ranking avec cross-encoder
- Évaluation RAGAS-like (Faithfulness, Relevancy, Precision, Recall)
"""

from .engine import RAGEngine, get_rag_engine
from .chunking import TextChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .llm_client import OllamaClient
from .retriever import HybridRetriever, BM25Retriever, CrossEncoderReranker
from .ingestion import DocumentIngester, DataCleaner, JSONMessageParser
from .evaluation import (
    RAGEvaluator,
    EvaluationReport,
    EvaluationResult,
    quick_evaluate,
    TestDatasetGenerator
)

__all__ = [
    # Core
    "RAGEngine",
    "get_rag_engine",
    
    # Chunking & Embeddings
    "TextChunker",
    "EmbeddingManager",
    "VectorStore",
    
    # LLM
    "OllamaClient",
    
    # Retrieval avancé
    "HybridRetriever",
    "BM25Retriever",
    "CrossEncoderReranker",
    
    # Ingestion
    "DocumentIngester",
    "DataCleaner",
    "JSONMessageParser",
    
    # Évaluation
    "RAGEvaluator",
    "EvaluationReport",
    "EvaluationResult",
    "quick_evaluate",
    "TestDatasetGenerator",
]

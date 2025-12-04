"""
Moteur RAG principal - orchestre chunking, embeddings, retrieval hybride et évaluation.
"""

import pandas as pd
from typing import Dict, Optional, List

from .chunking import TextChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .llm_client import OllamaClient
from .retriever import HybridRetriever, BM25Retriever
from .ingestion import DocumentIngester, DataCleaner
from .evaluation import RAGEvaluator, quick_evaluate, EvaluationReport


class RAGEngine:
    """
    Moteur RAG avancé pour indexer et interroger les messages.
    
    Fonctionnalités:
    - Ingestion intelligente multi-format
    - Chunking avec fenêtre de conversation
    - Recherche hybride (Vector + BM25)
    - Re-ranking avec cross-encoder
    - Évaluation intégrée (RAGAS-like)
    """
    
    def __init__(
        self,
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_conversation_windows: bool = True,
        window_size: int = 5,
        # Nouvelles options
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        Initialise le moteur RAG.
        
        Args:
            ollama_model: Modèle Ollama à utiliser
            ollama_base_url: URL du serveur Ollama
            chunk_size: Taille des chunks
            chunk_overlap: Chevauchement entre chunks
            embedding_model: Modèle pour les embeddings
            use_conversation_windows: Utiliser le chunking par fenêtre
            window_size: Taille de la fenêtre de messages
            use_hybrid_search: Activer la recherche hybride (Vector + BM25)
            use_reranking: Activer le re-ranking cross-encoder
            reranker_model: Modèle de re-ranking
            vector_weight: Poids recherche vectorielle (0-1)
            bm25_weight: Poids recherche BM25 (0-1)
        """
        self.use_conversation_windows = use_conversation_windows
        self.window_size = window_size
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        
        # Composants de base
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.vector_store = VectorStore(
            collection_name="messages",
            embedding_function=self.embedding_manager.embedding_function
        )
        self.llm_client = OllamaClient(
            base_url=ollama_base_url,
            model=ollama_model
        )
        
        # Composants avancés
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            use_reranking=use_reranking,
            reranker_model=reranker_model,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        # Ingestion
        self.ingester = DocumentIngester()
        
        # Évaluation
        self.evaluator = RAGEvaluator(llm_client=self.llm_client)
        
        self.messages_df: Optional[pd.DataFrame] = None
        self.indexing_stats: Dict = {}
        self._indexed_documents: List[str] = []
        self._indexed_metadatas: List[Dict] = []
    
    @property
    def ollama_model(self) -> str:
        """Retourne le modèle Ollama actuel."""
        return self.llm_client.model
    
    @ollama_model.setter
    def ollama_model(self, value: str):
        """Change le modèle Ollama."""
        self.llm_client.model = value
    
    def index_messages(self, messages_df: pd.DataFrame) -> int:
        """
        Indexe les messages avec chunking intelligent.
        
        Args:
            messages_df: DataFrame contenant les messages
            
        Returns:
            Nombre de chunks indexés
        """
        self.messages_df = messages_df.copy()
        
        # Réinitialiser le store
        self.vector_store.reset_collection(metadata={
            "embedding_model": self.embedding_manager.model_name,
            "chunking_strategy": "conversation_window" if self.use_conversation_windows else "individual",
            "hybrid_search": self.use_hybrid_search,
            "reranking": self.use_reranking
        })
        
        documents, metadatas, ids = self._prepare_documents(messages_df)
        
        # Stocker pour le retriever hybride
        self._indexed_documents = documents
        self._indexed_metadatas = metadatas
        
        indexed_count = self.vector_store.add_documents(documents, metadatas, ids)
        
        # Indexer aussi pour BM25 (recherche hybride)
        if self.use_hybrid_search:
            self.hybrid_retriever.index_documents(documents, metadatas)
        
        # Stats
        self.indexing_stats = {
            "total_messages": len(messages_df),
            "total_chunks": indexed_count,
            "chunking_strategy": "conversation_window" if self.use_conversation_windows else "individual",
            "embedding_model": self.embedding_manager.model_name,
            "embedding_dimension": self.embedding_manager.get_embedding_dimension(),
            "hybrid_search_enabled": self.use_hybrid_search,
            "reranking_enabled": self.use_reranking
        }
        
        return indexed_count
    
    def _prepare_documents(self, messages_df: pd.DataFrame):
        """Prépare les documents pour l'indexation."""
        documents = []
        metadatas = []
        ids = []
        
        if self.use_conversation_windows:
            documents, metadatas, ids = self._prepare_window_chunks(messages_df)
        else:
            documents, metadatas, ids = self._prepare_individual_chunks(messages_df)
        
        return documents, metadatas, ids
    
    def _prepare_window_chunks(self, messages_df: pd.DataFrame):
        """Prépare les chunks par fenêtre de conversation."""
        documents, metadatas, ids = [], [], []
        
        messages_list = []
        for _, row in messages_df.iterrows():
            content = str(row.get('content', ''))
            if content and content != 'nan':
                messages_list.append({
                    'sender': row.get('sender_name', 'Inconnu'),
                    'content': content,
                    'date': str(row.get('date', '')),
                    'timestamp': row.get('timestamp_ms', 0)
                })
        
        conversation_chunks = self.chunker.chunk_messages_window(
            messages_list,
            window_size=self.window_size
        )
        
        for i, chunk in enumerate(conversation_chunks):
            documents.append(chunk['text'])
            metadatas.append({
                "chunk_type": "conversation_window",
                "start_idx": chunk['start_idx'],
                "end_idx": chunk['end_idx'],
                "message_count": chunk['message_count'],
                "senders": ", ".join(chunk['senders']),
                "start_date": chunk['start_date'],
                "end_date": chunk['end_date'],
                "chunk_index": i
            })
            ids.append(VectorStore.generate_id(chunk['text'], i))
        
        return documents, metadatas, ids
    
    def _prepare_individual_chunks(self, messages_df: pd.DataFrame):
        """Prépare les chunks par message individuel."""
        documents, metadatas, ids = [], [], []
        chunk_index = 0
        
        for _, row in messages_df.iterrows():
            content = str(row.get('content', ''))
            if not content or content == 'nan':
                continue
            
            sender = row.get('sender_name', 'Inconnu')
            date = str(row.get('date', ''))
            timestamp = row.get('timestamp_ms', 0)
            
            full_text = f"[{sender}] ({date}): {content}"
            text_chunks = self.chunker.chunk_text(full_text)
            
            for j, text_chunk in enumerate(text_chunks):
                documents.append(text_chunk)
                metadatas.append({
                    "chunk_type": "message",
                    "sender": sender,
                    "date": date,
                    "timestamp": int(timestamp) if pd.notna(timestamp) else 0,
                    "original_content": content[:500],
                    "chunk_index": chunk_index,
                    "sub_chunk": j,
                    "total_sub_chunks": len(text_chunks)
                })
                ids.append(VectorStore.generate_id(text_chunk, chunk_index))
                chunk_index += 1
        
        return documents, metadatas, ids
    
    def search(self, query: str, n_results: int = 5, use_hybrid: bool = None) -> List[Dict]:
        """
        Recherche dans les messages indexés.
        
        Args:
            query: Requête de recherche
            n_results: Nombre de résultats
            use_hybrid: Forcer/désactiver la recherche hybride
            
        Returns:
            Liste de documents pertinents
        """
        should_use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid_search
        
        if should_use_hybrid and self._indexed_documents:
            # Recherche hybride avec re-ranking
            return self.hybrid_retriever.search(
                query,
                n_results=n_results,
                n_candidates=n_results * 4,  # Plus de candidats pour le re-ranking
                use_reranking=self.use_reranking
            )
        else:
            # Recherche vectorielle simple
            return self.vector_store.search(query, n_results)
    
    def chat(self, question: str, n_context: int = 5) -> Dict:
        """
        Répond à une question en utilisant le RAG.
        
        Args:
            question: La question de l'utilisateur
            n_context: Nombre de chunks de contexte
            
        Returns:
            Dict avec la réponse et les sources
        """
        relevant_messages = self.search(question, n_results=n_context)
        
        if not relevant_messages:
            return {
                "answer": "Je n'ai pas encore de messages indexés. Veuillez d'abord charger un fichier JSON.",
                "sources": [],
                "retrieval_method": "none"
            }
        
        context = "\n".join([f"- {msg['content']}" for msg in relevant_messages])
        
        system_prompt = """Tu es un assistant qui analyse des conversations de messagerie.
Tu réponds toujours en français.
Tu dois répondre aux questions en te basant UNIQUEMENT sur les messages fournis dans le contexte.
Si tu ne peux pas répondre avec les informations disponibles, dis-le clairement.
Sois concis et précis dans tes réponses.
Ne fabrique pas d'informations qui ne sont pas dans le contexte."""

        user_prompt = f"""Voici des messages de conversation pertinents pour répondre à la question:

{context}

Question: {question}

Réponds à la question en te basant uniquement sur ces messages. Cite les passages pertinents."""

        answer = self.llm_client.generate(user_prompt, system_prompt)
        
        # Déterminer la méthode de retrieval utilisée
        retrieval_method = "hybrid" if self.use_hybrid_search else "vector"
        if self.use_reranking:
            retrieval_method += "+rerank"
        
        return {
            "answer": answer,
            "sources": relevant_messages,
            "retrieval_method": retrieval_method,
            "contexts": [msg['content'] for msg in relevant_messages]
        }
    
    def evaluate(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> 'EvaluationReport':
        """
        Évalue la qualité du RAG sur un ensemble de questions.
        
        Args:
            questions: Liste de questions de test
            ground_truths: Réponses attendues (optionnel)
            
        Returns:
            Rapport d'évaluation avec métriques RAGAS-like
        """
        samples = []
        
        for i, question in enumerate(questions):
            result = self.chat(question)
            
            samples.append({
                'question': question,
                'answer': result['answer'],
                'contexts': result.get('contexts', []),
                'ground_truth': ground_truths[i] if ground_truths and i < len(ground_truths) else None
            })
        
        return self.evaluator.evaluate_dataset(
            samples,
            model_name=self.ollama_model
        )
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Évalue une seule réponse RAG.
        
        Returns:
            Dict avec scores de faithfulness, relevancy, precision, recall
        """
        result = self.evaluator.evaluate_sample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        return {
            'faithfulness': result.faithfulness,
            'answer_relevancy': result.answer_relevancy,
            'context_precision': result.context_precision,
            'context_recall': result.context_recall,
            'overall_score': result.overall_score,
            'hallucination_detected': result.hallucination_detected,
            'hallucination_details': result.hallucination_details
        }
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du RAG."""
        stats = {
            "total_indexed": self.vector_store.count(),
            "model": self.llm_client.model,
            "status": "ready" if self.vector_store.count() > 0 else "empty",
            "embedding_model": self.embedding_manager.model_name,
            "embedding_dimension": self.embedding_manager.get_embedding_dimension(),
            "chunking_strategy": "conversation_window" if self.use_conversation_windows else "individual",
            "window_size": self.window_size if self.use_conversation_windows else None,
            "hybrid_search": self.use_hybrid_search,
            "reranking": self.use_reranking,
        }
        
        if self.indexing_stats:
            stats.update(self.indexing_stats)
        
        return stats
    
    def check_ollama_status(self) -> Dict:
        """Vérifie le statut d'Ollama."""
        return self.llm_client.check_status()


# Singleton
_rag_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Retourne l'instance singleton du RAG."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance

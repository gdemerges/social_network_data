"""
Moteur RAG principal - orchestre chunking, embeddings, retrieval hybride et évaluation.
"""

import pandas as pd
import threading
from typing import Dict, Optional, List

from .chunking import TextChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .llm_client import OllamaClient, get_stream_buffer
from .retriever import HybridRetriever, BM25Retriever
from .ingestion import DocumentIngester, DataCleaner
from .evaluation import RAGEvaluator, quick_evaluate, EvaluationReport
from .cache import get_query_cache


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
        bm25_weight: float = 0.4,
        use_cache: bool = True
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
            use_cache: Activer le cache des requêtes
        """
        self.use_conversation_windows = use_conversation_windows
        self.window_size = window_size
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.use_cache = use_cache
        
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

        # Cache
        self.cache = get_query_cache() if use_cache else None

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
    
    def _detect_statistical_question(self, question: str) -> Optional[Dict]:
        """
        Détecte si la question nécessite une analyse statistique.
        
        Returns:
            Dict avec le type d'analyse et les paramètres, ou None
        """
        question_lower = question.lower()
        
        # Patterns de questions statistiques
        statistical_patterns = [
            ("plus parlé", "message_count"),
            ("le plus de messages", "message_count"),
            ("combien de messages", "message_count"),
            ("qui parle le plus", "message_count"),
            ("le plus actif", "message_count"),
            ("la plus active", "message_count"),
            ("nombre de messages", "message_count"),
        ]
        
        # Détection de période temporelle
        months = {
            "janvier": 1, "février": 2, "mars": 3, "avril": 4,
            "mai": 5, "juin": 6, "juillet": 7, "août": 8,
            "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
        }
        
        detected_month = None
        for month_name, month_num in months.items():
            if month_name in question_lower:
                detected_month = (month_name, month_num)
                break
        
        for pattern, analysis_type in statistical_patterns:
            if pattern in question_lower:
                return {
                    "type": analysis_type,
                    "month": detected_month,
                    "question": question
                }
        
        return None
    
    def _analyze_statistics(self, analysis_request: Dict) -> str:
        """
        Analyse les statistiques sur les messages bruts.
        """
        if self.messages_df is None or len(self.messages_df) == 0:
            return "Aucune donnée n'est chargée pour effectuer cette analyse."
        
        df = self.messages_df.copy()
        
        # Filtrer par mois si demandé
        month_info = analysis_request.get("month")
        if month_info:
            month_name, month_num = month_info
            # S'assurer que la colonne date existe
            if 'date' in df.columns:
                df['month'] = pd.to_datetime(df['date']).dt.month
                df_filtered = df[df['month'] == month_num]
                
                if len(df_filtered) == 0:
                    # Vérifier quels mois sont disponibles
                    available_months = df['month'].unique()
                    month_names = {1: "janvier", 2: "février", 3: "mars", 4: "avril",
                                   5: "mai", 6: "juin", 7: "juillet", 8: "août",
                                   9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"}
                    available = [month_names.get(m, str(m)) for m in sorted(available_months)]
                    return f"❌ Aucun message trouvé pour {month_name}.\n\n📅 Mois disponibles dans les données : {', '.join(available)}"
                
                df = df_filtered
                period_text = f"en {month_name}"
            else:
                period_text = "(toute la période)"
        else:
            period_text = "(toute la période)"
        
        # Analyse selon le type
        if analysis_request["type"] == "message_count":
            # Compter les messages par personne
            sender_counts = df['sender_name'].value_counts()
            
            if len(sender_counts) == 0:
                return "Aucun message à analyser."
            
            top_sender = sender_counts.index[0]
            top_count = sender_counts.iloc[0]
            total_messages = len(df)
            
            # Construire la réponse
            response = f"📊 **Statistiques des messages {period_text}**\n\n"
            response += f"🏆 **{top_sender}** a le plus parlé avec **{top_count} messages** ({top_count/total_messages*100:.1f}% du total)\n\n"
            response += f"**Classement complet :**\n"
            
            for i, (sender, count) in enumerate(sender_counts.head(10).items(), 1):
                emoji = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}."))
                percentage = count / total_messages * 100
                response += f"{emoji} **{sender}**: {count} messages ({percentage:.1f}%)\n"
            
            response += f"\n📈 **Total**: {total_messages} messages {period_text}"
            
            return response
        
        return "Type d'analyse non reconnu."
    
    def chat(self, question: str, n_context: int = 5) -> Dict:
        """
        Répond à une question en utilisant le RAG.

        Args:
            question: La question de l'utilisateur
            n_context: Nombre de chunks de contexte

        Returns:
            Dict avec la réponse et les sources
        """
        # Vérifier le cache d'abord (si activé)
        if self.cache:
            cached_result = self.cache.get(question, n_context)
            if cached_result is not None:
                # Ajouter une indication que c'est du cache
                cached_result['from_cache'] = True
                return cached_result

        # Vérifier si c'est une question statistique
        stat_analysis = self._detect_statistical_question(question)
        if stat_analysis:
            answer = self._analyze_statistics(stat_analysis)
            result = {
                "answer": answer,
                "sources": [],
                "retrieval_method": "statistical_analysis",
                "from_cache": False
            }
            return result
        
        relevant_messages = self.search(question, n_results=n_context)
        
        if not relevant_messages:
            return {
                "answer": "Je n'ai pas encore de messages indexés. Veuillez d'abord charger un fichier JSON.",
                "sources": [],
                "retrieval_method": "none"
            }
        
        # Formater le contexte selon le type de chunk
        context_parts = []
        for i, msg in enumerate(relevant_messages, 1):
            metadata = msg['metadata']
            content = msg['content']
            
            if metadata.get('chunk_type') == 'conversation_window':
                # Chunk de fenêtre de conversation - le contenu contient déjà les [Nom]: message
                senders = metadata.get('senders', 'Inconnu')
                context_parts.append(f"--- Extrait {i} (Participants: {senders}) ---\n{content}")
            else:
                # Message individuel
                sender = metadata.get('sender_name', 'Inconnu')
                context_parts.append(f"[{sender}]: {content}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """Tu es un assistant qui analyse des conversations de messagerie.
Tu réponds toujours en français.

RÈGLES STRICTES:
1. Base-toi UNIQUEMENT sur les messages fournis dans le contexte
2. Fais TRÈS attention à QUI dit QUOI - ne confonds JAMAIS les personnes
3. Cite EXACTEMENT les passages pertinents avec le nom de la personne qui parle
4. Si une information n'est pas claire ou absente, dis-le explicitement
5. Ne fais AUCUNE inférence ou supposition au-delà de ce qui est écrit
6. Quand tu mentionnes une personne, vérifie DEUX FOIS que c'est bien elle qui a dit/fait cette chose

IMPORTANT:
- Les extraits peuvent contenir plusieurs messages de différentes personnes
- Chaque message est au format [Nom]: contenu
- Réponds de manière claire et structurée
- NE liste PAS tous les messages - fais une SYNTHÈSE pertinente

Format de réponse:
- Réponds directement et de manière concise
- Cite uniquement les messages les plus pertinents (2-3 max)
- Utilise des bullet points si plusieurs éléments de réponse"""

        user_prompt = f"""Voici des extraits de conversation pertinents pour répondre à la question.

EXTRAITS:
{context}

QUESTION: {question}

Réponds de manière synthétique et structurée. Ne recopie pas tous les messages, extrais seulement l'information pertinente."""

        answer = self.llm_client.generate(user_prompt, system_prompt, max_tokens=3000)

        # Déterminer la méthode de retrieval utilisée
        retrieval_method = "hybrid" if self.use_hybrid_search else "vector"
        if self.use_reranking:
            retrieval_method += "+rerank"

        result = {
            "answer": answer,
            "sources": relevant_messages,
            "retrieval_method": retrieval_method,
            "contexts": [msg['content'] for msg in relevant_messages],
            "from_cache": False
        }

        # Mettre en cache le résultat (si activé)
        if self.cache:
            self.cache.set(question, result, n_context)

        return result

    def chat_stream(self, question: str, session_id: str, n_context: int = 5):
        """
        Répond à une question en utilisant le RAG avec streaming.

        Cette méthode lance la génération en arrière-plan et stocke
        les tokens dans un buffer accessible via session_id.

        Args:
            question: La question de l'utilisateur
            session_id: ID unique de la session de streaming
            n_context: Nombre de chunks de contexte
        """
        def _generate_in_background():
            """Fonction exécutée en arrière-plan pour générer la réponse."""
            buffer = get_stream_buffer()

            try:
                # Vérifier le cache d'abord
                if self.cache:
                    cached_result = self.cache.get(question, n_context)
                    if cached_result is not None:
                        # Mettre la réponse cachée directement
                        buffer.append(session_id, "⚡ **(Réponse en cache)**\n\n")
                        buffer.append(session_id, cached_result['answer'])
                        buffer.mark_complete(session_id)
                        return

                # Vérifier questions statistiques
                stat_analysis = self._detect_statistical_question(question)
                if stat_analysis:
                    answer = self._analyze_statistics(stat_analysis)
                    buffer.append(session_id, answer)
                    buffer.mark_complete(session_id)
                    return

                # Recherche normale
                relevant_messages = self.search(question, n_results=n_context)

                if not relevant_messages:
                    buffer.append(session_id, "Je n'ai pas encore de messages indexés.")
                    buffer.mark_complete(session_id)
                    return

                # Préparer le contexte
                context_parts = []
                for i, msg in enumerate(relevant_messages, 1):
                    metadata = msg['metadata']
                    content = msg['content']

                    if metadata.get('chunk_type') == 'conversation_window':
                        senders = metadata.get('senders', 'Inconnu')
                        context_parts.append(f"--- Extrait {i} (Participants: {senders}) ---\n{content}")
                    else:
                        sender = metadata.get('sender_name', 'Inconnu')
                        context_parts.append(f"[{sender}]: {content}")

                context = "\n\n".join(context_parts)

                # Prompts
                system_prompt = """Tu es un assistant qui analyse des conversations de messagerie.
Tu réponds toujours en français.

RÈGLES STRICTES:
1. Base-toi UNIQUEMENT sur les messages fournis dans le contexte
2. Fais TRÈS attention à QUI dit QUOI - ne confonds JAMAIS les personnes
3. Cite EXACTEMENT les passages pertinents avec le nom de la personne qui parle
4. Si une information n'est pas claire ou absente, dis-le explicitement
5. Ne fais AUCUNE inférence ou supposition au-delà de ce qui est écrit
6. Réponds de manière claire et structurée"""

                user_prompt = f"""Voici des extraits de conversation pertinents pour répondre à la question.

EXTRAITS:
{context}

QUESTION: {question}

Réponds de manière synthétique et structurée."""

                # Générer avec streaming
                for token in self.llm_client.generate_stream(
                    user_prompt,
                    system_prompt,
                    max_tokens=3000
                ):
                    buffer.append(session_id, token)

                # Marquer comme terminé
                buffer.mark_complete(session_id)

                # Mettre en cache la réponse complète
                if self.cache:
                    buffer_state = buffer.get(session_id)
                    if buffer_state:
                        result = {
                            "answer": buffer_state['content'],
                            "sources": relevant_messages,
                            "retrieval_method": "hybrid" if self.use_hybrid_search else "vector",
                            "contexts": [msg['content'] for msg in relevant_messages],
                            "from_cache": False
                        }
                        self.cache.set(question, result, n_context)

            except Exception as e:
                buffer.set_error(session_id, f"Erreur: {str(e)}")

        # Créer le buffer et lancer en arrière-plan
        buffer = get_stream_buffer()
        buffer.create(session_id)

        # Lancer la génération dans un thread séparé
        thread = threading.Thread(target=_generate_in_background, daemon=True)
        thread.start()

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
            "cache_enabled": self.use_cache,
        }

        if self.indexing_stats:
            stats.update(self.indexing_stats)

        # Ajouter stats du cache si activé
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats['cache_stats'] = cache_stats

        return stats
    
    def check_ollama_status(self) -> Dict:
        """Vérifie le statut d'Ollama."""
        return self.llm_client.check_status()


# Singleton avec thread-safety
_rag_instance: Optional[RAGEngine] = None
_rag_lock = threading.Lock()


def get_rag_engine() -> RAGEngine:
    """
    Retourne l'instance singleton du RAG (thread-safe).

    Utilise le pattern double-checked locking pour optimiser les performances
    tout en garantissant la thread-safety.
    """
    global _rag_instance

    # Premier check (sans lock pour performance)
    if _rag_instance is None:
        # Acquérir le lock uniquement si l'instance n'existe pas
        with _rag_lock:
            # Deuxième check (avec lock pour thread-safety)
            if _rag_instance is None:
                _rag_instance = RAGEngine()

    return _rag_instance

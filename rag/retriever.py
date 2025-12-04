"""
Module de Retrieval Avancé - Recherche hybride avec re-ranking.
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import math


class BM25Retriever:
    """Retriever basé sur BM25 pour la recherche lexicale (keyword)."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialise BM25.
        
        Args:
            k1: Paramètre de saturation du terme (1.2-2.0)
            b: Paramètre de normalisation de longueur (0-1)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize le texte en mots."""
        text = text.lower()
        # Supprimer la ponctuation et découper en mots
        tokens = re.findall(r'\b[a-zàâäéèêëïîôùûüç]+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """
        Indexe les documents pour BM25.
        
        Args:
            documents: Liste des documents texte
        """
        self.documents = documents
        self.doc_freqs = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.vocab.update(freq.keys())
        
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.documents else 0
        
        # Calculer IDF pour chaque terme
        n_docs = len(documents)
        for term in self.vocab:
            doc_containing = sum(1 for freq in self.doc_freqs if term in freq)
            # IDF avec smoothing
            self.idf[term] = math.log((n_docs - doc_containing + 0.5) / (doc_containing + 0.5) + 1)
    
    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calcule le score BM25 pour un document."""
        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        for term in query_tokens:
            if term not in doc_freq:
                continue
            
            tf = doc_freq[term]
            idf = self.idf.get(term, 0)
            
            # Formule BM25
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Recherche les documents les plus pertinents.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de tuples (index_document, score)
        """
        query_tokens = self._tokenize(query)
        
        scores = []
        for idx in range(len(self.documents)):
            score = self._score(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))
        
        # Trier par score décroissant
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class CrossEncoderReranker:
    """
    Re-ranker utilisant un cross-encoder pour scorer query-document pairs.
    Utilise sentence-transformers avec un modèle de cross-encoding.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialise le re-ranker.
        
        Args:
            model_name: Nom du modèle cross-encoder
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Charge le modèle à la demande."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                print("⚠️ sentence-transformers non installé. Re-ranking désactivé.")
                return None
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-classe les documents selon leur pertinence réelle.
        
        Args:
            query: La question utilisateur
            documents: Liste de documents avec 'content' 
            top_k: Nombre de documents à retourner
            
        Returns:
            Documents re-classés avec score de pertinence
        """
        if not documents:
            return []
        
        if self.model is None:
            # Fallback: retourner les documents sans re-ranking
            return documents[:top_k]
        
        # Préparer les paires query-document
        pairs = [(query, doc.get('content', '')) for doc in documents]
        
        # Scorer avec le cross-encoder
        scores = self.model.predict(pairs)
        
        # Ajouter les scores aux documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # Trier par score de re-ranking
        reranked = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return reranked[:top_k]


class HybridRetriever:
    """
    Retriever hybride combinant recherche vectorielle et lexicale.
    
    Architecture:
    1. Recherche vectorielle (sémantique) via ChromaDB
    2. Recherche lexicale (BM25) pour les termes exacts
    3. Fusion des résultats (Reciprocal Rank Fusion)
    4. Re-ranking final avec cross-encoder
    """
    
    def __init__(
        self,
        vector_store,
        use_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60
    ):
        """
        Initialise le retriever hybride.
        
        Args:
            vector_store: VectorStore pour la recherche sémantique
            use_reranking: Activer le re-ranking
            reranker_model: Modèle de cross-encoder
            vector_weight: Poids de la recherche vectorielle
            bm25_weight: Poids de la recherche BM25
            rrf_k: Paramètre RRF (60 est standard)
        """
        self.vector_store = vector_store
        self.bm25 = BM25Retriever()
        self.reranker = CrossEncoderReranker(reranker_model) if use_reranking else None
        
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        
        self._documents_indexed: List[str] = []
        self._documents_metadata: List[Dict] = []
    
    def index_documents(self, documents: List[str], metadatas: List[Dict]):
        """
        Indexe les documents pour la recherche hybride.
        
        Args:
            documents: Liste des textes
            metadatas: Métadonnées associées
        """
        self._documents_indexed = documents
        self._documents_metadata = metadatas
        
        # Indexer pour BM25
        self.bm25.fit(documents)
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Fusionne les résultats avec Reciprocal Rank Fusion.
        
        RRF score = Σ 1/(k + rank_i)
        
        Args:
            vector_results: Résultats de la recherche vectorielle (idx, score)
            bm25_results: Résultats de BM25 (idx, score)
            
        Returns:
            Liste fusionnée (idx, rrf_score)
        """
        rrf_scores: Dict[int, float] = {}
        
        # Ajouter scores vectoriels
        for rank, (idx, _) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + self.vector_weight / (self.rrf_k + rank + 1)
        
        # Ajouter scores BM25
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + self.bm25_weight / (self.rrf_k + rank + 1)
        
        # Trier par score RRF
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        n_candidates: int = 20,
        use_reranking: Optional[bool] = None
    ) -> List[Dict]:
        """
        Recherche hybride avec re-ranking optionnel.
        
        Args:
            query: Requête utilisateur
            n_results: Nombre de résultats finaux
            n_candidates: Nombre de candidats avant re-ranking
            use_reranking: Override pour le re-ranking
            
        Returns:
            Liste de documents pertinents avec scores
        """
        should_rerank = use_reranking if use_reranking is not None else (self.reranker is not None)
        
        # 1. Recherche vectorielle (sémantique)
        vector_results = self.vector_store.search(query, n_results=n_candidates)
        
        # Convertir en format (idx, score) - trouver l'index dans nos documents
        vector_idx_scores = []
        for i, result in enumerate(vector_results):
            content = result.get('content', '')
            try:
                idx = self._documents_indexed.index(content)
                distance = result.get('distance', 1.0)
                score = 1 / (1 + distance)  # Convertir distance en similarité
                vector_idx_scores.append((idx, score))
            except ValueError:
                # Document non trouvé dans l'index local
                vector_idx_scores.append((i, 0.5))
        
        # 2. Recherche BM25 (lexicale)
        bm25_results = self.bm25.search(query, top_k=n_candidates)
        
        # 3. Fusion RRF
        fused_results = self._reciprocal_rank_fusion(vector_idx_scores, bm25_results)
        
        # 4. Construire les documents résultats
        candidates = []
        seen_idx = set()
        
        for idx, rrf_score in fused_results[:n_candidates]:
            if idx in seen_idx or idx >= len(self._documents_indexed):
                continue
            seen_idx.add(idx)
            
            candidates.append({
                'content': self._documents_indexed[idx],
                'metadata': self._documents_metadata[idx] if idx < len(self._documents_metadata) else {},
                'rrf_score': rrf_score,
                'retrieval_method': 'hybrid'
            })
        
        # 5. Re-ranking (si activé)
        if should_rerank and self.reranker:
            candidates = self.reranker.rerank(query, candidates, top_k=n_results)
        else:
            candidates = candidates[:n_results]
        
        return candidates
    
    def get_config(self) -> Dict:
        """Retourne la configuration du retriever."""
        return {
            'type': 'hybrid',
            'vector_weight': self.vector_weight,
            'bm25_weight': self.bm25_weight,
            'rrf_k': self.rrf_k,
            'reranking_enabled': self.reranker is not None,
            'reranker_model': self.reranker.model_name if self.reranker else None,
            'documents_indexed': len(self._documents_indexed)
        }

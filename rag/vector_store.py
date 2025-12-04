"""
Module de stockage vectoriel avec ChromaDB.
"""

import chromadb
from typing import List, Dict, Optional
import hashlib


class VectorStore:
    """Gère le stockage et la recherche vectorielle avec ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = "messages",
        embedding_function=None
    ):
        """
        Initialise le store vectoriel.
        
        Args:
            collection_name: Nom de la collection
            embedding_function: Fonction d'embedding à utiliser
        """
        self.collection_name = collection_name
        self.client = chromadb.Client()
        self.embedding_function = embedding_function
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    def reset_collection(self, metadata: Optional[Dict] = None):
        """Réinitialise la collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=metadata or {}
        )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
        batch_size: int = 100
    ) -> int:
        """
        Ajoute des documents à la collection par batches.
        
        Args:
            documents: Liste des textes
            metadatas: Liste des métadonnées
            ids: Liste des IDs uniques
            batch_size: Taille des batches
            
        Returns:
            Nombre de documents indexés
        """
        indexed_count = 0
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            if batch_docs:
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                indexed_count += len(batch_docs)
        
        return indexed_count
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Recherche les documents les plus similaires.
        
        Args:
            query: Requête de recherche
            n_results: Nombre de résultats
            
        Returns:
            Liste des résultats avec contenu et métadonnées
        """
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                "content": doc,
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results.get('distances') else None
            })
        
        return formatted_results
    
    def count(self) -> int:
        """Retourne le nombre de documents indexés."""
        return self.collection.count()
    
    @staticmethod
    def generate_id(text: str, index: int) -> str:
        """Génère un ID unique pour un document."""
        content = f"{text}_{index}"
        return hashlib.md5(content.encode()).hexdigest()

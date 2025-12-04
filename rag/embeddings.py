"""
Module de gestion des embeddings pour le RAG.
"""

from chromadb.utils import embedding_functions


class EmbeddingManager:
    """Gère les embeddings avec différents modèles."""
    
    # Dimensions des modèles connus
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise le gestionnaire d'embeddings.
        
        Args:
            model_name: Nom du modèle sentence-transformers à utiliser
                       Options: all-MiniLM-L6-v2 (rapide), all-mpnet-base-v2 (qualité)
        """
        self.model_name = model_name
        self._embedding_function = None
    
    @property
    def embedding_function(self):
        """Lazy loading de la fonction d'embedding."""
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        return self._embedding_function
    
    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings selon le modèle."""
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)

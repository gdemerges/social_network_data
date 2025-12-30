"""
Module de cache pour les requêtes RAG.
"""

import hashlib
import json
import time
from typing import Dict, Optional, Any
from collections import OrderedDict
import threading


class QueryCache:
    """
    Cache LRU thread-safe pour les résultats de requêtes RAG.

    Caractéristiques:
    - LRU (Least Recently Used) éviction
    - Thread-safe avec locks
    - TTL (Time To Live) configurable
    - Statistiques de hit/miss
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600  # 1 heure par défaut
    ):
        """
        Initialise le cache.

        Args:
            max_size: Nombre maximum d'entrées en cache
            ttl_seconds: Durée de vie des entrées en secondes
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

        # Statistiques
        self._hits = 0
        self._misses = 0

    def _generate_key(self, query: str, context_size: int = 5) -> str:
        """
        Génère une clé de cache unique pour une requête.

        Args:
            query: La question de l'utilisateur
            context_size: Nombre de chunks de contexte

        Returns:
            Clé de cache (hash MD5)
        """
        # Normaliser la query (lower case, strip)
        normalized_query = query.lower().strip()

        # Créer une clé unique
        key_data = f"{normalized_query}:{context_size}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, context_size: int = 5) -> Optional[Dict[str, Any]]:
        """
        Récupère un résultat du cache.

        Args:
            query: La question de l'utilisateur
            context_size: Nombre de chunks de contexte

        Returns:
            Le résultat en cache, ou None si pas trouvé/expiré
        """
        key = self._generate_key(query, context_size)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Vérifier l'expiration
            if time.time() - entry['timestamp'] > self.ttl_seconds:
                # Entrée expirée, la supprimer
                del self._cache[key]
                self._misses += 1
                return None

            # Hit ! Déplacer en fin (LRU)
            self._cache.move_to_end(key)
            self._hits += 1

            return entry['data']

    def set(self, query: str, result: Dict[str, Any], context_size: int = 5):
        """
        Ajoute un résultat au cache.

        Args:
            query: La question de l'utilisateur
            result: Le résultat à mettre en cache
            context_size: Nombre de chunks de contexte
        """
        key = self._generate_key(query, context_size)

        with self._lock:
            # Si la clé existe déjà, la supprimer pour la réinsérer en fin
            if key in self._cache:
                del self._cache[key]

            # Ajouter la nouvelle entrée
            self._cache[key] = {
                'data': result,
                'timestamp': time.time()
            }

            # Éviction LRU si nécessaire
            if len(self._cache) > self.max_size:
                # Supprimer l'entrée la plus ancienne (première dans OrderedDict)
                self._cache.popitem(last=False)

    def clear(self):
        """Vide complètement le cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.

        Returns:
            Dict avec hits, misses, hit_rate, size, etc.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds
            }

    def invalidate_old_entries(self):
        """
        Supprime toutes les entrées expirées.

        Utile pour nettoyer le cache périodiquement.
        """
        current_time = time.time()

        with self._lock:
            # Créer une liste des clés à supprimer
            keys_to_delete = [
                key for key, entry in self._cache.items()
                if current_time - entry['timestamp'] > self.ttl_seconds
            ]

            # Supprimer les entrées expirées
            for key in keys_to_delete:
                del self._cache[key]

        return len(keys_to_delete)


# Singleton global pour le cache
_global_cache: Optional[QueryCache] = None
_cache_lock = threading.Lock()


def get_query_cache() -> QueryCache:
    """
    Retourne l'instance singleton du cache (thread-safe).

    Returns:
        Instance du QueryCache
    """
    global _global_cache

    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = QueryCache()

    return _global_cache

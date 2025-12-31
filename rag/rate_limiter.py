"""
Module de rate limiting pour protéger contre les abus.
"""

import time
import threading
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimitConfig:
    """Configuration du rate limiting."""

    max_requests: int = 10  # Nombre max de requêtes
    window_seconds: int = 60  # Fenêtre de temps en secondes
    cleanup_interval: int = 300  # Nettoyage toutes les 5 minutes

    # Circuit breaker
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5  # Nombre d'échecs avant ouverture
    recovery_timeout: int = 30  # Secondes avant tentative de récupération


@dataclass
class CircuitBreakerState:
    """État du circuit breaker."""

    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0


class RateLimiter:
    """
    Rate limiter thread-safe avec circuit breaker.

    Caractéristiques:
    - Limite le nombre de requêtes par fenêtre de temps
    - Circuit breaker pour protection contre services défaillants
    - Nettoyage automatique des entrées anciennes
    - Thread-safe avec locks
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialise le rate limiter.

        Args:
            config: Configuration du rate limiter
        """
        self.config = config or RateLimitConfig()

        # Stockage des timestamps par identifiant (IP, user_id, etc.)
        self._requests: Dict[str, deque] = {}
        self._lock = threading.Lock()

        # Circuit breaker
        self._circuit_breaker = CircuitBreakerState()
        self._cb_lock = threading.Lock()

        # Stats
        self._total_requests = 0
        self._total_blocked = 0
        self._last_cleanup = time.time()

    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si une requête est autorisée.

        Args:
            identifier: Identifiant unique (IP, session, user_id)

        Returns:
            Tuple (allowed, error_message)
                - allowed: True si autorisé, False sinon
                - error_message: Message d'erreur si bloqué
        """
        # Vérifier le circuit breaker d'abord
        if self.config.enable_circuit_breaker:
            cb_allowed, cb_error = self._check_circuit_breaker()
            if not cb_allowed:
                return False, cb_error

        current_time = time.time()

        with self._lock:
            self._total_requests += 1

            # Créer la queue si elle n'existe pas
            if identifier not in self._requests:
                self._requests[identifier] = deque()

            request_times = self._requests[identifier]

            # Supprimer les requêtes hors de la fenêtre
            cutoff_time = current_time - self.config.window_seconds
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()

            # Vérifier la limite
            if len(request_times) >= self.config.max_requests:
                self._total_blocked += 1

                # Calculer le temps d'attente
                oldest_request = request_times[0]
                wait_time = int(self.config.window_seconds - (current_time - oldest_request))

                error_msg = (
                    f"⚠️ Rate limit dépassé. "
                    f"Maximum {self.config.max_requests} requêtes par "
                    f"{self.config.window_seconds}s. "
                    f"Réessayez dans {wait_time}s."
                )
                return False, error_msg

            # Ajouter la requête actuelle
            request_times.append(current_time)

            # Nettoyage périodique
            if current_time - self._last_cleanup > self.config.cleanup_interval:
                self._cleanup_old_entries(current_time)
                self._last_cleanup = current_time

        return True, None

    def _check_circuit_breaker(self) -> Tuple[bool, Optional[str]]:
        """
        Vérifie l'état du circuit breaker.

        Returns:
            Tuple (allowed, error_message)
        """
        with self._cb_lock:
            current_time = time.time()

            # État OPEN (circuit ouvert, requêtes bloquées)
            if self._circuit_breaker.state == "open":
                time_since_failure = current_time - self._circuit_breaker.last_failure_time

                # Tenter de récupérer après timeout
                if time_since_failure >= self.config.recovery_timeout:
                    self._circuit_breaker.state = "half_open"
                    self._circuit_breaker.success_count = 0
                else:
                    wait_time = int(self.config.recovery_timeout - time_since_failure)
                    return False, (
                        f"🔴 Service temporairement indisponible. "
                        f"Circuit breaker ouvert. Réessayez dans {wait_time}s."
                    )

            # État HALF_OPEN (test de récupération)
            elif self._circuit_breaker.state == "half_open":
                # On laisse passer pour tester
                pass

        return True, None

    def record_success(self):
        """Enregistre une requête réussie (pour circuit breaker)."""
        if not self.config.enable_circuit_breaker:
            return

        with self._cb_lock:
            if self._circuit_breaker.state == "half_open":
                self._circuit_breaker.success_count += 1

                # Si succès, fermer le circuit
                if self._circuit_breaker.success_count >= 2:
                    self._circuit_breaker.state = "closed"
                    self._circuit_breaker.failure_count = 0

            elif self._circuit_breaker.state == "closed":
                # Reset du compteur d'échecs sur succès
                self._circuit_breaker.failure_count = 0

    def record_failure(self):
        """Enregistre un échec de requête (pour circuit breaker)."""
        if not self.config.enable_circuit_breaker:
            return

        with self._cb_lock:
            self._circuit_breaker.failure_count += 1
            self._circuit_breaker.last_failure_time = time.time()

            # Ouvrir le circuit si trop d'échecs
            if self._circuit_breaker.failure_count >= self.config.failure_threshold:
                self._circuit_breaker.state = "open"

    def _cleanup_old_entries(self, current_time: float):
        """Nettoie les entrées anciennes."""
        cutoff_time = current_time - self.config.window_seconds * 2

        # Supprimer les identifiants qui n'ont plus de requêtes récentes
        identifiers_to_remove = []
        for identifier, request_times in self._requests.items():
            # Nettoyer les timestamps anciens
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()

            # Si plus de requêtes, marquer pour suppression
            if not request_times:
                identifiers_to_remove.append(identifier)

        # Supprimer les identifiants vides
        for identifier in identifiers_to_remove:
            del self._requests[identifier]

    def get_stats(self) -> Dict:
        """
        Retourne les statistiques du rate limiter.

        Returns:
            Dict avec stats (total, blocked, rate, circuit_breaker, etc.)
        """
        with self._lock:
            total = self._total_requests
            blocked = self._total_blocked
            block_rate = (blocked / total * 100) if total > 0 else 0.0

            active_identifiers = len(self._requests)
            total_active_requests = sum(len(q) for q in self._requests.values())

        with self._cb_lock:
            cb_state = self._circuit_breaker.state
            cb_failures = self._circuit_breaker.failure_count

        return {
            "total_requests": total,
            "total_blocked": blocked,
            "block_rate": round(block_rate, 2),
            "active_identifiers": active_identifiers,
            "active_requests": total_active_requests,
            "circuit_breaker": {
                "state": cb_state,
                "failure_count": cb_failures,
                "enabled": self.config.enable_circuit_breaker
            },
            "config": {
                "max_requests": self.config.max_requests,
                "window_seconds": self.config.window_seconds,
                "failure_threshold": self.config.failure_threshold
            }
        }

    def reset(self, identifier: Optional[str] = None):
        """
        Réinitialise le rate limiter.

        Args:
            identifier: Si fourni, reset seulement cet identifiant.
                       Sinon, reset complet.
        """
        with self._lock:
            if identifier:
                if identifier in self._requests:
                    del self._requests[identifier]
            else:
                self._requests.clear()
                self._total_requests = 0
                self._total_blocked = 0

        # Reset circuit breaker
        if not identifier:
            with self._cb_lock:
                self._circuit_breaker.state = "closed"
                self._circuit_breaker.failure_count = 0
                self._circuit_breaker.success_count = 0


# Singleton global
_rate_limiter: Optional[RateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """
    Retourne l'instance singleton du rate limiter (thread-safe).

    Args:
        config: Configuration (utilisée seulement à la première création)

    Returns:
        Instance du RateLimiter
    """
    global _rate_limiter

    if _rate_limiter is None:
        with _limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = RateLimiter(config)

    return _rate_limiter

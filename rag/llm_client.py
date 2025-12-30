"""
Client pour communiquer avec Ollama (LLM local).
"""

import requests
import threading
import json
from typing import Dict, List, Generator, Optional


class OllamaClient:
    """Client pour l'API Ollama."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
        timeout: int = 120
    ):
        """
        Initialise le client Ollama.
        
        Args:
            base_url: URL de base du serveur Ollama
            model: Modèle par défaut à utiliser
            timeout: Timeout en secondes pour les requêtes
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Génère une réponse avec le LLM.
        
        Args:
            prompt: Le prompt utilisateur
            system_prompt: Le prompt système
            temperature: Température de génération
            max_tokens: Nombre maximum de tokens à générer
            
        Returns:
            La réponse du LLM
        """
        url = f"{self.base_url}/api/generate"
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("response", "Erreur: pas de réponse")
        except requests.exceptions.ConnectionError:
            return "❌ Erreur: Impossible de se connecter à Ollama. Assurez-vous qu'Ollama est lancé (`ollama serve`)."
        except requests.exceptions.Timeout:
            return "❌ Erreur: Timeout - le modèle met trop de temps à répondre."
        except Exception as e:
            return f"❌ Erreur lors de l'appel à Ollama: {str(e)}"
    
    def check_status(self) -> Dict:
        """
        Vérifie si Ollama est disponible et liste les modèles.
        
        Returns:
            Dict avec le statut et les modèles disponibles
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return {
                    "status": "online",
                    "models": model_names,
                    "current_model": self.model,
                    "model_available": any(self.model in m for m in model_names)
                }
        except Exception:
            pass
        
        return {
            "status": "offline",
            "models": [],
            "current_model": self.model,
            "model_available": False
        }
    
    def list_models(self) -> List[str]:
        """Liste les modèles disponibles."""
        status = self.check_status()
        return status.get("models", [])

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Generator[str, None, None]:
        """
        Génère une réponse avec streaming (yield progressif des tokens).

        Args:
            prompt: Le prompt utilisateur
            system_prompt: Le prompt système
            temperature: Température de génération
            max_tokens: Nombre maximum de tokens à générer

        Yields:
            Chunks de texte au fur et à mesure de la génération
        """
        url = f"{self.base_url}/api/generate"

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,  # Activer le streaming
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=True  # Important pour le streaming
            )
            response.raise_for_status()

            # Lire la réponse ligne par ligne (Server-Sent Events)
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token

                        # Vérifier si c'est la fin
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError:
            yield "❌ Erreur: Impossible de se connecter à Ollama."
        except requests.exceptions.Timeout:
            yield "❌ Erreur: Timeout."
        except Exception as e:
            yield f"❌ Erreur: {str(e)}"


# Buffer global pour le streaming (thread-safe)
class StreamBuffer:
    """Buffer thread-safe pour stocker les réponses en streaming."""

    def __init__(self):
        self._buffers: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str):
        """Crée un nouveau buffer pour une session."""
        with self._lock:
            self._buffers[session_id] = {
                'content': '',
                'is_complete': False,
                'error': None
            }

    def append(self, session_id: str, text: str):
        """Ajoute du texte au buffer."""
        with self._lock:
            if session_id in self._buffers:
                self._buffers[session_id]['content'] += text

    def mark_complete(self, session_id: str):
        """Marque le streaming comme terminé."""
        with self._lock:
            if session_id in self._buffers:
                self._buffers[session_id]['is_complete'] = True

    def set_error(self, session_id: str, error: str):
        """Définit une erreur."""
        with self._lock:
            if session_id in self._buffers:
                self._buffers[session_id]['error'] = error
                self._buffers[session_id]['is_complete'] = True

    def get(self, session_id: str) -> Optional[Dict]:
        """Récupère l'état actuel du buffer."""
        with self._lock:
            return self._buffers.get(session_id, None)

    def delete(self, session_id: str):
        """Supprime un buffer."""
        with self._lock:
            if session_id in self._buffers:
                del self._buffers[session_id]


# Singleton global
_stream_buffer: Optional[StreamBuffer] = None
_buffer_lock = threading.Lock()


def get_stream_buffer() -> StreamBuffer:
    """Retourne l'instance singleton du buffer (thread-safe)."""
    global _stream_buffer

    if _stream_buffer is None:
        with _buffer_lock:
            if _stream_buffer is None:
                _stream_buffer = StreamBuffer()

    return _stream_buffer

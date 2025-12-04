"""
Client pour communiquer avec Ollama (LLM local).
"""

import requests
from typing import Dict, List


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
        max_tokens: int = 1000
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

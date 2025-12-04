"""
Module de chunking pour découper les textes en segments optimisés pour le RAG.
"""

from typing import List, Dict


class TextChunker:
    """Gère le découpage intelligent des textes en chunks."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50
    ):
        """
        Initialise le chunker.
        
        Args:
            chunk_size: Taille maximale d'un chunk en caractères
            chunk_overlap: Chevauchement entre chunks pour maintenir le contexte
            min_chunk_size: Taille minimale d'un chunk (éviter les chunks trop petits)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Découpe un texte en chunks avec chevauchement.
        
        Args:
            text: Le texte à découper
            
        Returns:
            Liste des chunks
        """
        if not text or len(text) < self.min_chunk_size:
            return [text] if text else []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                last_break = self._find_break_point(text, start, end)
                if last_break > start:
                    end = last_break
            
            chunk = text[start:end].strip()
            
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks
    
    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Trouve un bon point de coupure (fin de phrase ou espace)."""
        search_text = text[start:end]
        
        for pattern in ['. ', '! ', '? ', '\n', ', ', ' ']:
            last_pos = search_text.rfind(pattern)
            if last_pos > len(search_text) * 0.3:
                return start + last_pos + len(pattern)
        
        return end
    
    def chunk_messages_window(
        self,
        messages: List[Dict],
        window_size: int = 5
    ) -> List[Dict]:
        """
        Crée des chunks par fenêtre glissante de messages.
        
        Args:
            messages: Liste de messages avec sender, content, date
            window_size: Nombre de messages par fenêtre
            
        Returns:
            Liste de chunks avec métadonnées
        """
        if not messages:
            return []
        
        chunks = []
        
        for i in range(0, len(messages), window_size - 1):
            window = messages[i:i + window_size]
            
            combined_text = "\n".join([
                f"[{m.get('sender', 'Inconnu')}]: {m.get('content', '')}"
                for m in window
                if m.get('content')
            ])
            
            if combined_text:
                chunks.append({
                    "text": combined_text,
                    "start_idx": i,
                    "end_idx": min(i + window_size, len(messages)),
                    "message_count": len(window),
                    "senders": list(set(m.get('sender', '') for m in window)),
                    "start_date": window[0].get('date', ''),
                    "end_date": window[-1].get('date', '')
                })
        
        return chunks

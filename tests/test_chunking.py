"""Tests pour le module de chunking."""

import pytest
from rag.chunking import TextChunker


class TestTextChunker:
    """Tests pour la classe TextChunker."""

    def test_init_default_params(self):
        """Test l'initialisation avec paramètres par défaut."""
        chunker = TextChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_size == 50

    def test_init_custom_params(self):
        """Test l'initialisation avec paramètres personnalisés."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20
        assert chunker.min_chunk_size == 10

    def test_chunk_empty_text(self):
        """Test avec un texte vide."""
        chunker = TextChunker()
        result = chunker.chunk_text("")
        assert result == []

    def test_chunk_short_text(self):
        """Test avec un texte court qui tient dans un chunk."""
        chunker = TextChunker(chunk_size=100)
        text = "Ceci est un texte court."
        result = chunker.chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_text_below_min_size(self):
        """Test avec un texte en dessous de la taille minimale."""
        chunker = TextChunker(min_chunk_size=50)
        text = "Court"
        result = chunker.chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_long_text(self, long_text):
        """Test avec un texte long nécessitant plusieurs chunks."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        result = chunker.chunk_text(long_text)

        # Doit créer plusieurs chunks
        assert len(result) > 1

        # Chaque chunk doit respecter la taille max (avec marge pour les break points)
        for chunk in result:
            assert len(chunk) <= chunker.chunk_size + 100  # Marge pour les break points

    def test_chunk_overlap(self):
        """Test que le chevauchement fonctionne correctement."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100  # Texte de 100 caractères
        result = chunker.chunk_text(text)

        # Doit y avoir chevauchement entre chunks
        assert len(result) >= 2

    def test_find_break_point_sentence(self):
        """Test que le break point préfère les fins de phrase."""
        chunker = TextChunker()
        text = "Première phrase. Deuxième phrase. Troisième phrase."

        # Le break point devrait être après un point
        break_point = chunker._find_break_point(text, 0, 30)
        assert break_point > 0
        # Devrait tomber après "Première phrase."
        assert text[break_point - 2] == '.'

    def test_chunk_messages_window_empty(self):
        """Test avec liste de messages vide."""
        chunker = TextChunker()
        result = chunker.chunk_messages_window([])
        assert result == []

    def test_chunk_messages_window_basic(self, sample_messages):
        """Test le chunking par fenêtre avec messages de base."""
        chunker = TextChunker()
        result = chunker.chunk_messages_window(sample_messages, window_size=3)

        # Doit créer des chunks
        assert len(result) > 0

        # Chaque chunk doit avoir les bonnes métadonnées
        for chunk in result:
            assert 'text' in chunk
            assert 'start_idx' in chunk
            assert 'end_idx' in chunk
            assert 'message_count' in chunk
            assert 'senders' in chunk
            assert 'start_date' in chunk
            assert 'end_date' in chunk

            # Le texte doit contenir le format [Sender]: message
            assert '[' in chunk['text']
            assert ']:' in chunk['text']

    def test_chunk_messages_window_size(self, sample_messages):
        """Test que la taille de fenêtre est respectée."""
        chunker = TextChunker()
        window_size = 2
        result = chunker.chunk_messages_window(sample_messages, window_size=window_size)

        # Chaque chunk (sauf peut-être le dernier) devrait avoir window_size messages
        for i, chunk in enumerate(result[:-1]):
            assert chunk['message_count'] <= window_size

    def test_chunk_messages_window_senders(self, sample_messages):
        """Test que les senders sont correctement extraits."""
        chunker = TextChunker()
        result = chunker.chunk_messages_window(sample_messages, window_size=5)

        # Le premier chunk devrait contenir tous les senders
        chunk = result[0]
        senders = chunk['senders']
        assert isinstance(senders, list)
        assert len(senders) >= 1
        # Devrait contenir Alice, Bob, Charlie
        assert any('Alice' in s or 'Bob' in s or 'Charlie' in s for s in senders)

    def test_chunk_messages_skip_empty_content(self):
        """Test que les messages sans contenu sont ignorés."""
        chunker = TextChunker()
        messages = [
            {'sender': 'Alice', 'content': 'Hello', 'date': '2024-01-01'},
            {'sender': 'Bob', 'content': '', 'date': '2024-01-01'},  # Vide
            {'sender': 'Charlie', 'content': None, 'date': '2024-01-01'},  # None
            {'sender': 'Dave', 'content': 'World', 'date': '2024-01-01'}
        ]

        result = chunker.chunk_messages_window(messages, window_size=4)

        # Doit créer des chunks mais sans les messages vides
        assert len(result) > 0
        # Le texte ne devrait contenir que Alice et Dave
        text = result[0]['text']
        assert 'Hello' in text
        assert 'World' in text

    def test_chunk_messages_window_overlap(self, sample_messages):
        """Test que les fenêtres glissantes créent un chevauchement."""
        chunker = TextChunker()
        window_size = 3
        result = chunker.chunk_messages_window(sample_messages, window_size=window_size)

        # Avec une fenêtre de 3 et un glissement de window_size-1,
        # il doit y avoir chevauchement
        if len(result) > 1:
            # Vérifier que les indices se chevauchent
            chunk1_end = result[0]['end_idx']
            chunk2_start = result[1]['start_idx']
            # Avec window_size=3, le glissement est de 2, donc chevauchement de 1
            assert chunk2_start < chunk1_end

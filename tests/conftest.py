"""Configuration pytest et fixtures partagées."""

import pytest
import pandas as pd
from datetime import datetime
import tempfile
import shutil


@pytest.fixture
def sample_messages():
    """Fixture avec des messages de test."""
    return [
        {
            'sender': 'Alice',
            'content': 'Bonjour tout le monde!',
            'date': '2024-01-01',
            'timestamp': 1704067200000
        },
        {
            'sender': 'Bob',
            'content': 'Salut Alice! Comment ça va?',
            'date': '2024-01-01',
            'timestamp': 1704067260000
        },
        {
            'sender': 'Alice',
            'content': 'Très bien merci! Et toi?',
            'date': '2024-01-01',
            'timestamp': 1704067320000
        },
        {
            'sender': 'Charlie',
            'content': 'Hey les amis!',
            'date': '2024-01-02',
            'timestamp': 1704153600000
        },
        {
            'sender': 'Bob',
            'content': 'Ça va super! On se voit demain?',
            'date': '2024-01-02',
            'timestamp': 1704153660000
        }
    ]


@pytest.fixture
def sample_dataframe():
    """Fixture avec un DataFrame de messages."""
    return pd.DataFrame({
        'sender_name': ['Alice', 'Bob', 'Charlie'],
        'content': [
            'Premier message',
            'Deuxième message',
            'Troisième message'
        ],
        'timestamp_ms': [1704067200000, 1704067260000, 1704067320000],
        'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-01'])
    })


@pytest.fixture
def sample_json_data():
    """Fixture avec des données JSON au format Facebook."""
    return {
        'messages': [
            {
                'sender_name': 'Alice',
                'timestamp_ms': 1704067200000,
                'content': 'Test message 1'
            },
            {
                'sender_name': 'Bob',
                'timestamp_ms': 1704067260000,
                'content': 'Test message 2'
            }
        ]
    }


@pytest.fixture
def temp_chroma_dir():
    """Fixture qui crée et nettoie un répertoire temporaire pour ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup après le test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def long_text():
    """Fixture avec un texte long pour tester le chunking."""
    return """
    Ceci est un texte assez long qui sera utilisé pour tester le chunking.
    Il contient plusieurs phrases. Chaque phrase apporte un peu de contexte.
    Le chunking doit découper ce texte de manière intelligente.
    Il doit respecter les limites de taille tout en gardant le sens.
    Les chunks doivent se chevaucher pour maintenir le contexte.
    C'est important pour le RAG de ne pas perdre d'information.
    """ * 10  # Répéter pour avoir un texte vraiment long

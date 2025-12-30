"""Tests pour le module de stockage vectoriel."""

import pytest
from rag.vector_store import VectorStore


class TestVectorStore:
    """Tests pour la classe VectorStore."""

    def test_init_default_params(self, temp_chroma_dir):
        """Test initialisation avec paramètres par défaut."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        assert store.collection_name == "messages"
        assert store.persist_directory == temp_chroma_dir
        assert store.client is not None
        assert store.collection is not None

    def test_init_custom_collection_name(self, temp_chroma_dir):
        """Test initialisation avec nom de collection personnalisé."""
        custom_name = "test_collection"
        store = VectorStore(
            collection_name=custom_name,
            persist_directory=temp_chroma_dir
        )

        assert store.collection_name == custom_name

    def test_add_documents_single(self, temp_chroma_dir):
        """Test ajout d'un seul document."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        documents = ["Ceci est un document de test"]
        metadatas = [{"source": "test"}]
        ids = ["doc1"]

        count = store.add_documents(documents, metadatas, ids)

        assert count == 1
        assert store.count() == 1

    def test_add_documents_multiple(self, temp_chroma_dir):
        """Test ajout de plusieurs documents."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        documents = [
            "Premier document",
            "Deuxième document",
            "Troisième document"
        ]
        metadatas = [
            {"source": "test1"},
            {"source": "test2"},
            {"source": "test3"}
        ]
        ids = ["doc1", "doc2", "doc3"]

        count = store.add_documents(documents, metadatas, ids)

        assert count == 3
        assert store.count() == 3

    def test_add_documents_batch(self, temp_chroma_dir):
        """Test ajout par batches."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        # Créer 250 documents pour tester le batching (batch_size=100)
        num_docs = 250
        documents = [f"Document {i}" for i in range(num_docs)]
        metadatas = [{"index": i} for i in range(num_docs)]
        ids = [f"doc{i}" for i in range(num_docs)]

        count = store.add_documents(
            documents, metadatas, ids, batch_size=100
        )

        assert count == num_docs
        assert store.count() == num_docs

    def test_search_empty_store(self, temp_chroma_dir):
        """Test recherche dans un store vide."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        results = store.search("test query")

        assert results == []

    def test_search_basic(self, temp_chroma_dir):
        """Test recherche basique."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        # Ajouter des documents
        documents = [
            "Python est un langage de programmation",
            "Java est aussi un langage de programmation",
            "Les chats sont des animaux domestiques"
        ]
        metadatas = [{"topic": "programming"}, {"topic": "programming"}, {"topic": "animals"}]
        ids = ["doc1", "doc2", "doc3"]

        store.add_documents(documents, metadatas, ids)

        # Rechercher quelque chose lié à la programmation
        results = store.search("langage programmation", n_results=2)

        assert len(results) <= 2
        assert all('content' in r for r in results)
        assert all('metadata' in r for r in results)

    def test_search_n_results(self, temp_chroma_dir):
        """Test que n_results est respecté."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        # Ajouter 5 documents
        documents = [f"Document numéro {i}" for i in range(5)]
        metadatas = [{"index": i} for i in range(5)]
        ids = [f"doc{i}" for i in range(5)]

        store.add_documents(documents, metadatas, ids)

        # Demander 3 résultats
        results = store.search("document", n_results=3)

        assert len(results) == 3

    def test_search_more_results_than_available(self, temp_chroma_dir):
        """Test recherche avec plus de résultats demandés que disponibles."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        # Ajouter 2 documents
        documents = ["Doc 1", "Doc 2"]
        metadatas = [{"id": 1}, {"id": 2}]
        ids = ["doc1", "doc2"]

        store.add_documents(documents, metadatas, ids)

        # Demander 10 résultats alors qu'il n'y en a que 2
        results = store.search("doc", n_results=10)

        # Doit retourner seulement 2
        assert len(results) == 2

    def test_reset_collection(self, temp_chroma_dir):
        """Test réinitialisation de la collection."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        # Ajouter des documents
        documents = ["Test 1", "Test 2"]
        metadatas = [{"id": 1, "type": "test"}, {"id": 2, "type": "test"}]
        ids = ["doc1", "doc2"]

        store.add_documents(documents, metadatas, ids)
        initial_count = store.count()
        assert initial_count >= 2  # Au moins 2 documents

        # Réinitialiser avec metadata non vide
        store.reset_collection(metadata={"reset": "true"})

        # La collection doit être vide
        assert store.count() == 0

    def test_reset_collection_with_metadata(self, temp_chroma_dir):
        """Test réinitialisation avec métadonnées."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        metadata = {"version": "1.0", "created_at": "2024-01-01"}
        store.reset_collection(metadata=metadata)

        # La collection devrait être créée (vérification implicite)
        assert store.collection is not None

    def test_count_empty(self, temp_chroma_dir):
        """Test comptage sur collection vide."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        assert store.count() == 0

    def test_count_after_additions(self, temp_chroma_dir):
        """Test comptage après ajouts."""
        store = VectorStore(persist_directory=temp_chroma_dir)

        initial_count = store.count()

        # Ajouter 3 documents
        documents = ["Document A unique", "Document B unique", "Document C unique"]
        metadatas = [{"idx": 0}, {"idx": 1}, {"idx": 2}]
        ids = ["unique1", "unique2", "unique3"]

        added = store.add_documents(documents, metadatas, ids)

        # Vérifier que des documents ont été ajoutés
        assert added == 3
        assert store.count() == initial_count + 3

    def test_generate_id_deterministic(self):
        """Test que generate_id est déterministe."""
        text = "Test text"
        index = 42

        id1 = VectorStore.generate_id(text, index)
        id2 = VectorStore.generate_id(text, index)

        # Les IDs doivent être identiques
        assert id1 == id2

    def test_generate_id_different_inputs(self):
        """Test que des entrées différentes génèrent des IDs différents."""
        id1 = VectorStore.generate_id("Text 1", 0)
        id2 = VectorStore.generate_id("Text 2", 0)
        id3 = VectorStore.generate_id("Text 1", 1)

        # Tous doivent être différents
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_persistence(self, temp_chroma_dir):
        """Test que les données persistent entre les instances."""
        # Créer une instance et ajouter des documents
        store1 = VectorStore(persist_directory=temp_chroma_dir)
        documents = ["Document persistant"]
        metadatas = [{"persistent": True}]
        ids = ["persist1"]

        store1.add_documents(documents, metadatas, ids)
        assert store1.count() == 1

        # Créer une nouvelle instance avec le même répertoire
        store2 = VectorStore(persist_directory=temp_chroma_dir)

        # Les documents doivent toujours être là
        assert store2.count() == 1

        # La recherche doit fonctionner
        results = store2.search("persistant")
        assert len(results) == 1

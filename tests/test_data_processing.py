"""Tests pour le module de traitement des données."""

import pytest
import pandas as pd
import json
import base64
from dashboard.data_processing import (
    parse_whatsapp_format,
    decode_upload_content,
    process_messages,
    filter_messages,
    compute_statistics,
    validate_file_size,
    MAX_FILE_SIZE_BYTES
)


class TestParseWhatsappFormat:
    """Tests pour le parsing du format WhatsApp."""

    def test_parse_basic_format(self):
        """Test parsing d'un format WhatsApp basique."""
        text = """[14/09/2024 12:08:15] Alice: Bonjour
[14/09/2024 12:09:20] Bob: Salut Alice!"""

        result = parse_whatsapp_format(text)

        assert len(result) == 2
        assert result[0]['sender_name'] == 'Alice'
        assert result[0]['content'] == 'Bonjour'
        assert result[1]['sender_name'] == 'Bob'
        assert result[1]['content'] == 'Salut Alice!'

    def test_parse_multiline_message(self):
        """Test parsing d'un message multiligne."""
        text = """[14/09/2024 12:08:15] Alice: Ceci est un message
qui continue sur plusieurs lignes
et se termine ici
[14/09/2024 12:09:20] Bob: Message suivant"""

        result = parse_whatsapp_format(text)

        assert len(result) == 2
        # Le premier message devrait contenir toutes les lignes
        assert 'plusieurs lignes' in result[0]['content']

    def test_parse_empty_text(self):
        """Test avec un texte vide."""
        result = parse_whatsapp_format("")
        assert result == []

    def test_parse_system_messages_filtered(self):
        """Test que les messages système courts sont filtrés."""
        text = """[14/09/2024 12:08:15] Alice: Message normal
[14/09/2024 12:09:20] ‎ """

        result = parse_whatsapp_format(text)

        # Ne devrait avoir que le message d'Alice
        assert len(result) == 1
        assert result[0]['sender_name'] == 'Alice'


class TestValidateFileSize:
    """Tests pour la validation de taille de fichier."""

    def test_validate_small_file(self):
        """Test avec un petit fichier."""
        # Créer un petit contenu base64
        content = base64.b64encode(b"Hello World").decode()
        full_content = f"data:text/plain;base64,{content}"

        is_valid, error_msg = validate_file_size(full_content)

        assert is_valid is True
        assert error_msg == ""

    def test_validate_large_file(self):
        """Test avec un fichier trop gros."""
        # Créer un contenu qui dépasse 100MB
        large_data = b"X" * (MAX_FILE_SIZE_BYTES + 1000)
        content = base64.b64encode(large_data).decode()
        full_content = f"data:text/plain;base64,{content}"

        is_valid, error_msg = validate_file_size(full_content)

        assert is_valid is False
        assert "trop volumineux" in error_msg
        assert "100" in error_msg  # Devrait mentionner la limite

    def test_validate_edge_case_near_limit(self):
        """Test avec un fichier proche de la limite."""
        # Créer un contenu proche mais en dessous de 100MB
        safe_data = b"X" * (MAX_FILE_SIZE_BYTES - 10000)
        content = base64.b64encode(safe_data).decode()
        full_content = f"data:text/plain;base64,{content}"

        is_valid, error_msg = validate_file_size(full_content)

        # Devrait être valide
        assert is_valid is True


class TestDecodeUploadContent:
    """Tests pour le décodage du contenu uploadé."""

    def test_decode_json_format(self, sample_json_data):
        """Test décodage d'un fichier JSON."""
        json_str = json.dumps(sample_json_data)
        encoded = base64.b64encode(json_str.encode()).decode()
        content = f"data:application/json;base64,{encoded}"

        result = decode_upload_content(content, "test.json")

        assert 'messages' in result
        assert len(result['messages']) == 2
        assert result['messages'][0]['sender_name'] == 'Alice'

    def test_decode_csv_format(self):
        """Test décodage d'un fichier CSV."""
        csv_data = "sender_name,timestamp_ms,content\nAlice,1704067200000,Hello\nBob,1704067260000,Hi"
        encoded = base64.b64encode(csv_data.encode()).decode()
        content = f"data:text/csv;base64,{encoded}"

        result = decode_upload_content(content, "test.csv")

        assert 'messages' in result
        assert len(result['messages']) == 2

    def test_decode_file_too_large(self):
        """Test que les fichiers trop gros sont rejetés."""
        # Créer un gros fichier
        large_data = b"X" * (MAX_FILE_SIZE_BYTES + 1000)
        encoded = base64.b64encode(large_data).decode()
        content = f"data:text/plain;base64,{encoded}"

        with pytest.raises(ValueError) as exc_info:
            decode_upload_content(content, "large.txt")

        assert "trop volumineux" in str(exc_info.value)


class TestProcessMessages:
    """Tests pour le traitement des messages."""

    def test_process_basic_messages(self, sample_json_data):
        """Test traitement de base des messages."""
        result = process_messages(sample_json_data)

        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'sentiment' in result.columns
        assert len(result) == 2

    def test_process_creates_date_column(self, sample_json_data):
        """Test que la colonne date est créée."""
        result = process_messages(sample_json_data)

        assert 'date' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_process_adds_sentiment(self, sample_json_data):
        """Test que l'analyse de sentiment est ajoutée."""
        result = process_messages(sample_json_data)

        assert 'sentiment' in result.columns
        # Les sentiments doivent être des nombres
        assert all(isinstance(s, (int, float)) for s in result['sentiment'])


class TestFilterMessages:
    """Tests pour le filtrage des messages."""

    def test_filter_no_filters(self, sample_dataframe):
        """Test sans aucun filtre."""
        result = filter_messages(sample_dataframe)

        # Doit retourner tous les messages
        assert len(result) == len(sample_dataframe)

    def test_filter_by_date(self, sample_dataframe):
        """Test filtrage par dates."""
        start_date = '2024-01-01'
        end_date = '2024-01-01'

        result = filter_messages(sample_dataframe, start_date, end_date)

        # Tous les messages sont du 01/01/2024
        assert len(result) == 3

    def test_filter_by_sender(self, sample_dataframe):
        """Test filtrage par expéditeur."""
        result = filter_messages(sample_dataframe, senders=['Alice'])

        # Doit retourner seulement Alice
        assert len(result) == 1
        assert result.iloc[0]['sender_name'] == 'Alice'

    def test_filter_multiple_senders(self, sample_dataframe):
        """Test filtrage avec plusieurs expéditeurs."""
        result = filter_messages(sample_dataframe, senders=['Alice', 'Bob'])

        # Doit retourner Alice et Bob
        assert len(result) == 2
        assert set(result['sender_name']) == {'Alice', 'Bob'}

    def test_filter_combined(self, sample_dataframe):
        """Test filtrage combiné date + expéditeur."""
        result = filter_messages(
            sample_dataframe,
            start_date='2024-01-01',
            end_date='2024-01-01',
            senders=['Alice']
        )

        # Doit retourner seulement Alice du 01/01
        assert len(result) == 1
        assert result.iloc[0]['sender_name'] == 'Alice'


class TestComputeStatistics:
    """Tests pour le calcul des statistiques."""

    def test_compute_basic_stats(self, sample_dataframe):
        """Test calcul des statistiques de base."""
        # Ajouter sentiment pour le test
        df = sample_dataframe.copy()
        df['sentiment'] = [0.5, -0.2, 0.8]

        result = compute_statistics(df)

        assert 'messages_by_day' in result
        assert 'sentiment_by_day' in result
        assert 'sender_counts' in result
        assert 'unique_senders' in result

    def test_compute_sender_counts(self, sample_dataframe):
        """Test que les compteurs de senders sont corrects."""
        # Ajouter colonne sentiment manquante
        df = sample_dataframe.copy()
        df['sentiment'] = [0.0, 0.0, 0.0]

        result = compute_statistics(df)

        sender_counts = result['sender_counts']
        assert sender_counts['Alice'] == 1
        assert sender_counts['Bob'] == 1
        assert sender_counts['Charlie'] == 1

    def test_compute_unique_senders(self, sample_dataframe):
        """Test extraction des senders uniques."""
        # Ajouter colonne sentiment manquante
        df = sample_dataframe.copy()
        df['sentiment'] = [0.0, 0.0, 0.0]

        result = compute_statistics(df)

        unique_senders = result['unique_senders']
        assert isinstance(unique_senders, list)
        assert set(unique_senders) == {'Alice', 'Bob', 'Charlie'}

    def test_compute_messages_by_day(self, sample_dataframe):
        """Test comptage des messages par jour."""
        # Ajouter colonne sentiment manquante
        df = sample_dataframe.copy()
        df['sentiment'] = [0.0, 0.0, 0.0]

        result = compute_statistics(df)

        messages_by_day = result['messages_by_day']
        # Tous les messages sont du même jour
        assert len(messages_by_day) == 1
        assert messages_by_day.iloc[0] == 3

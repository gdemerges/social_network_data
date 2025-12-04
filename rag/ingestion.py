"""
Module d'ingestion avancée de données.
Support multi-formats avec parsing intelligent.
"""

import json
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib


@dataclass
class ParsedDocument:
    """Document parsé avec métadonnées."""
    content: str
    source: str
    doc_type: str
    metadata: Dict[str, Any]
    chunks: List[str] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


class DataCleaner:
    """Nettoie et normalise les données textuelles."""
    
    # Patterns de nettoyage
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    EMOJI_PATTERN = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    WHITESPACE_PATTERN = re.compile(r'\s+')
    SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s\.\,\!\?\;\:\'\"\-\(\)\[\]àâäéèêëïîôùûüç]')
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_emojis: bool = False,
        normalize_whitespace: bool = True,
        lowercase: bool = False,
        min_content_length: int = 10
    ):
        """
        Initialise le cleaner.
        
        Args:
            remove_urls: Supprimer les URLs
            remove_emails: Supprimer les emails
            remove_emojis: Supprimer les emojis
            normalize_whitespace: Normaliser les espaces
            lowercase: Convertir en minuscules
            min_content_length: Longueur minimale du contenu
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_emojis = remove_emojis
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.min_content_length = min_content_length
    
    def clean(self, text: str) -> str:
        """Nettoie un texte."""
        if not text:
            return ""
        
        result = str(text)
        
        if self.remove_urls:
            result = self.URL_PATTERN.sub('[URL]', result)
        
        if self.remove_emails:
            result = self.EMAIL_PATTERN.sub('[EMAIL]', result)
        
        if self.remove_emojis:
            result = self.EMOJI_PATTERN.sub('', result)
        
        if self.normalize_whitespace:
            result = self.WHITESPACE_PATTERN.sub(' ', result).strip()
        
        if self.lowercase:
            result = result.lower()
        
        return result
    
    def is_valid_content(self, text: str) -> bool:
        """Vérifie si le contenu est valide."""
        if not text:
            return False
        cleaned = self.clean(text)
        return len(cleaned) >= self.min_content_length


class BaseParser(ABC):
    """Classe de base pour les parsers de documents."""
    
    @abstractmethod
    def can_parse(self, source: Union[str, Path, Dict]) -> bool:
        """Vérifie si ce parser peut traiter la source."""
        pass
    
    @abstractmethod
    def parse(self, source: Union[str, Path, Dict]) -> ParsedDocument:
        """Parse la source et retourne un document."""
        pass


class JSONMessageParser(BaseParser):
    """Parser pour les exports JSON de messageries (Facebook, Instagram, etc.)."""
    
    # Formats connus de messageries
    KNOWN_FORMATS = {
        'facebook': ['messages', 'participants'],
        'instagram': ['messages', 'participants'],
        'whatsapp': ['messages'],
        'telegram': ['messages'],
    }
    
    def __init__(self, cleaner: Optional[DataCleaner] = None):
        """Initialise le parser."""
        self.cleaner = cleaner or DataCleaner()
    
    def can_parse(self, source: Union[str, Path, Dict]) -> bool:
        """Vérifie si c'est un JSON de messagerie."""
        if isinstance(source, dict):
            return 'messages' in source or 'data' in source
        
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if path.suffix.lower() == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return 'messages' in data or isinstance(data, list)
        except Exception:
            pass
        return False
    
    def _detect_format(self, data: Dict) -> str:
        """Détecte le format de messagerie."""
        for format_name, required_keys in self.KNOWN_FORMATS.items():
            if all(key in data for key in required_keys):
                return format_name
        
        if 'messages' in data:
            return 'generic'
        return 'unknown'
    
    def _decode_facebook_encoding(self, text: str) -> str:
        """Décode l'encodage Facebook (Latin-1 mal interprété en UTF-8)."""
        if not text:
            return ""
        try:
            return text.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text
    
    def _extract_messages(self, data: Dict) -> List[Dict]:
        """Extrait les messages selon le format détecté."""
        format_type = self._detect_format(data)
        messages = []
        
        raw_messages = data.get('messages', data.get('data', []))
        if isinstance(raw_messages, list):
            for msg in raw_messages:
                extracted = self._normalize_message(msg, format_type)
                if extracted:
                    messages.append(extracted)
        
        return messages
    
    def _normalize_message(self, msg: Dict, format_type: str) -> Optional[Dict]:
        """Normalise un message vers un format commun."""
        # Champs possibles selon les formats
        sender_fields = ['sender_name', 'from', 'author', 'sender', 'name']
        content_fields = ['content', 'text', 'message', 'body']
        timestamp_fields = ['timestamp_ms', 'timestamp', 'date', 'time', 'created_at']
        
        # Extraire l'expéditeur
        sender = None
        for field in sender_fields:
            if field in msg:
                sender = msg[field]
                if format_type == 'facebook':
                    sender = self._decode_facebook_encoding(sender)
                break
        
        # Extraire le contenu
        content = None
        for field in content_fields:
            if field in msg and msg[field]:
                content = msg[field]
                if format_type == 'facebook':
                    content = self._decode_facebook_encoding(content)
                break
        
        # Pas de contenu textuel = ignorer
        if not content or not self.cleaner.is_valid_content(content):
            return None
        
        # Extraire le timestamp
        timestamp = None
        for field in timestamp_fields:
            if field in msg:
                timestamp = msg[field]
                break
        
        return {
            'sender': sender or 'Unknown',
            'content': self.cleaner.clean(content),
            'timestamp': timestamp,
            'type': msg.get('type', 'message'),
            'raw': msg  # Garder les données brutes
        }
    
    def parse(self, source: Union[str, Path, Dict]) -> ParsedDocument:
        """Parse un fichier JSON de messagerie."""
        if isinstance(source, dict):
            data = source
            source_name = "dict_input"
        else:
            path = Path(source)
            source_name = path.name
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Extraire les messages
        messages = self._extract_messages(data)
        
        # Construire le contenu global
        full_content = "\n".join([
            f"[{m['sender']}]: {m['content']}"
            for m in messages
        ])
        
        # Métadonnées
        format_type = self._detect_format(data)
        participants = data.get('participants', [])
        if participants:
            participant_names = [
                self._decode_facebook_encoding(p.get('name', '')) 
                if format_type == 'facebook' else p.get('name', '')
                for p in participants
            ]
        else:
            participant_names = list(set(m['sender'] for m in messages))
        
        metadata = {
            'format': format_type,
            'message_count': len(messages),
            'participants': participant_names,
            'messages': messages,  # Messages normalisés
            'title': data.get('title', source_name),
        }
        
        if messages:
            timestamps = [m['timestamp'] for m in messages if m['timestamp']]
            if timestamps:
                metadata['first_message'] = min(timestamps)
                metadata['last_message'] = max(timestamps)
        
        return ParsedDocument(
            content=full_content,
            source=source_name,
            doc_type='messaging',
            metadata=metadata
        )


class TextFileParser(BaseParser):
    """Parser pour les fichiers texte simples."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.rst', '.log'}
    
    def __init__(self, cleaner: Optional[DataCleaner] = None):
        self.cleaner = cleaner or DataCleaner()
    
    def can_parse(self, source: Union[str, Path, Dict]) -> bool:
        if isinstance(source, (str, Path)):
            return Path(source).suffix.lower() in self.SUPPORTED_EXTENSIONS
        return False
    
    def parse(self, source: Union[str, Path, Dict]) -> ParsedDocument:
        path = Path(source)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = self.cleaner.clean(content)
        
        return ParsedDocument(
            content=cleaned_content,
            source=path.name,
            doc_type='text',
            metadata={
                'file_size': path.stat().st_size,
                'extension': path.suffix,
            }
        )


class CSVMessageParser(BaseParser):
    """Parser pour les exports CSV de messages."""
    
    def __init__(self, cleaner: Optional[DataCleaner] = None):
        self.cleaner = cleaner or DataCleaner()
    
    def can_parse(self, source: Union[str, Path, Dict]) -> bool:
        if isinstance(source, (str, Path)):
            return Path(source).suffix.lower() == '.csv'
        return False
    
    def parse(self, source: Union[str, Path, Dict]) -> ParsedDocument:
        import csv
        
        path = Path(source)
        messages = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Détecter les colonnes de contenu/sender
                content = (
                    row.get('content') or row.get('message') or 
                    row.get('text') or row.get('body', '')
                )
                sender = (
                    row.get('sender') or row.get('from') or 
                    row.get('author') or row.get('name', 'Unknown')
                )
                
                if self.cleaner.is_valid_content(content):
                    messages.append({
                        'sender': sender,
                        'content': self.cleaner.clean(content),
                        'timestamp': row.get('timestamp') or row.get('date'),
                        'raw': row
                    })
        
        full_content = "\n".join([
            f"[{m['sender']}]: {m['content']}"
            for m in messages
        ])
        
        return ParsedDocument(
            content=full_content,
            source=path.name,
            doc_type='csv_messages',
            metadata={
                'message_count': len(messages),
                'messages': messages,
                'participants': list(set(m['sender'] for m in messages))
            }
        )


class DocumentIngester:
    """
    Pipeline d'ingestion de documents.
    Gère automatiquement le parsing selon le type de fichier.
    """
    
    def __init__(
        self,
        cleaner: Optional[DataCleaner] = None,
        custom_parsers: Optional[List[BaseParser]] = None
    ):
        """
        Initialise l'ingester.
        
        Args:
            cleaner: Instance de DataCleaner
            custom_parsers: Parsers personnalisés additionnels
        """
        self.cleaner = cleaner or DataCleaner()
        
        # Parsers par défaut
        self.parsers: List[BaseParser] = [
            JSONMessageParser(self.cleaner),
            TextFileParser(self.cleaner),
            CSVMessageParser(self.cleaner),
        ]
        
        if custom_parsers:
            self.parsers.extend(custom_parsers)
        
        self.ingestion_stats: Dict[str, Any] = {
            'documents_processed': 0,
            'messages_extracted': 0,
            'errors': []
        }
    
    def _get_parser(self, source: Union[str, Path, Dict]) -> Optional[BaseParser]:
        """Trouve le parser approprié pour la source."""
        for parser in self.parsers:
            if parser.can_parse(source):
                return parser
        return None
    
    def ingest(self, source: Union[str, Path, Dict]) -> Optional[ParsedDocument]:
        """
        Ingère un document.
        
        Args:
            source: Chemin du fichier, dict JSON ou données brutes
            
        Returns:
            Document parsé ou None si échec
        """
        parser = self._get_parser(source)
        
        if parser is None:
            error_msg = f"Aucun parser trouvé pour: {source}"
            self.ingestion_stats['errors'].append(error_msg)
            return None
        
        try:
            document = parser.parse(source)
            
            # Mettre à jour les stats
            self.ingestion_stats['documents_processed'] += 1
            if 'message_count' in document.metadata:
                self.ingestion_stats['messages_extracted'] += document.metadata['message_count']
            
            return document
            
        except Exception as e:
            error_msg = f"Erreur parsing {source}: {str(e)}"
            self.ingestion_stats['errors'].append(error_msg)
            return None
    
    def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> List[ParsedDocument]:
        """
        Ingère tous les fichiers d'un répertoire.
        
        Args:
            directory: Chemin du répertoire
            recursive: Parcourir récursivement
            
        Returns:
            Liste des documents parsés
        """
        path = Path(directory)
        documents = []
        
        pattern = '**/*' if recursive else '*'
        
        for file_path in path.glob(pattern):
            if file_path.is_file():
                doc = self.ingest(file_path)
                if doc:
                    documents.append(doc)
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'ingestion."""
        return self.ingestion_stats.copy()
    
    def reset_stats(self):
        """Réinitialise les statistiques."""
        self.ingestion_stats = {
            'documents_processed': 0,
            'messages_extracted': 0,
            'errors': []
        }

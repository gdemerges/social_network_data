"""
Module de traitement des données de messages.
"""

import pandas as pd
import json
import base64
import re
from datetime import datetime
from textblob import TextBlob


# Limite de taille de fichier en bytes (100MB)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def parse_whatsapp_format(text: str) -> list:
    """
    Parse un export WhatsApp au format [date heure] Nom: message.
    
    Args:
        text: Contenu du fichier WhatsApp
        
    Returns:
        Liste de messages au format standardisé
    """
    messages = []
    
    # Pattern pour les messages WhatsApp: [14/09/2025 12:08:15] Nom: message
    # Supporte aussi le format sans crochets
    pattern = r'\[?(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\]?\s*([^:]+?):\s*(.+?)(?=\n\[?\d{2}/\d{2}/\d{4}|$)'
    
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        date_str, sender, content = match.groups()
        
        # Nettoyer le contenu
        content = content.strip()
        
        # Ignorer les messages système vides ou très courts
        if content.startswith('‎') or len(content) < 3:
            # Garder quand même certains messages système informatifs
            if not any(keyword in content for keyword in ['a créé le groupe', 'a ajouté', 'vous a ajouté']):
                continue
        
        try:
            # Parser la date au format DD/MM/YYYY HH:MM:SS
            dt = datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S')
            timestamp_ms = int(dt.timestamp() * 1000)
            
            messages.append({
                'sender_name': sender.strip(),
                'timestamp_ms': timestamp_ms,
                'content': content
            })
        except ValueError:
            # Si erreur de parsing de date, utiliser timestamp actuel
            import time
            messages.append({
                'sender_name': sender.strip(),
                'timestamp_ms': int(time.time() * 1000),
                'content': content
            })
    
    return messages


def validate_file_size(content: str) -> tuple[bool, str]:
    """
    Valide la taille du fichier uploadé.

    Args:
        content: Contenu base64 du fichier

    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        # Le contenu est au format "data:type;base64,content"
        if ',' in content:
            content_string = content.split(',', 1)[1]
        else:
            content_string = content

        # Calculer la taille approximative du fichier décodé
        # Base64 encode augmente la taille de ~33%, donc on divise par 1.33
        encoded_size = len(content_string)
        decoded_size = (encoded_size * 3) // 4

        if decoded_size > MAX_FILE_SIZE_BYTES:
            size_mb = decoded_size / (1024 * 1024)
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            return False, f"❌ Fichier trop volumineux: {size_mb:.1f}MB. Limite: {max_mb:.0f}MB"

        return True, ""
    except Exception as e:
        return False, f"❌ Erreur lors de la validation de la taille: {str(e)}"


def decode_upload_content(content: str, filename: str = None) -> dict:
    """
    Décode le contenu uploadé en base64.

    Args:
        content: Contenu base64 du fichier
        filename: Nom du fichier pour déterminer le format

    Returns:
        Données décodées (dict pour JSON, DataFrame converti en dict pour CSV/TXT)

    Raises:
        ValueError: Si le fichier est trop volumineux
    """
    # Valider la taille du fichier avant de le décoder
    is_valid, error_msg = validate_file_size(content)
    if not is_valid:
        raise ValueError(error_msg)

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    decoded_str = decoded.decode('utf-8')
    
    # Déterminer le format du fichier
    if filename:
        if filename.endswith('.json'):
            return json.loads(decoded_str)
        elif filename.endswith('.csv'):
            # Pour CSV, on attend les colonnes: sender_name, timestamp_ms, content
            import io
            df = pd.read_csv(io.StringIO(decoded_str))
            # Convertir en format messages
            messages = df.to_dict('records')
            return {'messages': messages}
        elif filename.endswith('.txt'):
            # Essayer de détecter si c'est un export WhatsApp
            if re.search(r'\[\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\]', decoded_str):
                messages = parse_whatsapp_format(decoded_str)
                if messages:
                    return {'messages': messages}
            
            # Sinon, traiter comme texte brut
            import time
            return {
                'messages': [{
                    'sender_name': 'Document',
                    'timestamp_ms': int(time.time() * 1000),
                    'content': decoded_str
                }]
            }
    
    # Par défaut, essayer JSON
    try:
        return json.loads(decoded_str)
    except json.JSONDecodeError:
        # Si échec, essayer format WhatsApp
        if re.search(r'\[\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\]', decoded_str):
            messages = parse_whatsapp_format(decoded_str)
            if messages:
                return {'messages': messages}
        
        # Sinon traiter comme texte
        import time
        return {
            'messages': [{
                'sender_name': 'Document',
                'timestamp_ms': int(time.time() * 1000),
                'content': decoded_str
            }]
        }


def process_messages(data: dict) -> pd.DataFrame:
    """
    Traite les messages JSON en DataFrame avec analyse de sentiment.
    
    Args:
        data: Données JSON contenant les messages
        
    Returns:
        DataFrame avec les messages traités
    """
    messages = pd.DataFrame(data['messages'])
    messages['date'] = pd.to_datetime(messages['timestamp_ms'], unit='ms')
    messages['sentiment'] = messages['content'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    return messages


def filter_messages(
    messages: pd.DataFrame,
    start_date: str = None,
    end_date: str = None,
    senders: list = None
) -> pd.DataFrame:
    """
    Filtre les messages par date et expéditeur.
    
    Args:
        messages: DataFrame des messages
        start_date: Date de début
        end_date: Date de fin
        senders: Liste des expéditeurs à inclure
        
    Returns:
        DataFrame filtré
    """
    filtered = messages.copy()
    
    if start_date and end_date:
        mask = (filtered['date'] >= start_date) & (filtered['date'] <= end_date)
        filtered = filtered.loc[mask]
    
    if senders:
        filtered = filtered[filtered['sender_name'].isin(senders)]
    
    return filtered


def compute_statistics(messages: pd.DataFrame) -> dict:
    """
    Calcule les statistiques des messages.
    
    Args:
        messages: DataFrame des messages
        
    Returns:
        Dict avec les statistiques agrégées
    """
    return {
        "messages_by_day": messages.groupby(messages['date'].dt.date).size(),
        "sentiment_by_day": messages.groupby(messages['date'].dt.date)['sentiment'].mean(),
        "sender_counts": messages['sender_name'].value_counts(),
        "unique_senders": messages['sender_name'].unique().tolist()
    }

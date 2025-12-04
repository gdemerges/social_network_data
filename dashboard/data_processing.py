"""
Module de traitement des données de messages.
"""

import pandas as pd
import json
import base64
from textblob import TextBlob


def decode_upload_content(content: str) -> dict:
    """
    Décode le contenu uploadé en base64.
    
    Args:
        content: Contenu base64 du fichier
        
    Returns:
        Données JSON décodées
    """
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    return json.loads(decoded.decode('utf-8'))


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

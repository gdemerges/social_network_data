"""
Module d'analytics avancées pour l'analyse de conversations.

Fonctionnalités:
- Word Cloud / Analyse de fréquence
- Topic Modeling (LDA)
- Network Graph des interactions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re
from datetime import datetime


class AdvancedAnalytics:
    """
    Classe pour analytics avancées sur les conversations.
    """

    def __init__(self):
        """Initialise les stopwords français courants."""
        self.stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou',
            'mais', 'donc', 'car', 'ni', 'or', 'je', 'tu', 'il', 'elle',
            'nous', 'vous', 'ils', 'elles', 'ce', 'ca', 'ça', 'cela',
            'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
            'qui', 'que', 'quoi', 'dont', 'où', 'si', 'y', 'en',
            'à', 'au', 'aux', 'avec', 'sans', 'pour', 'par', 'dans',
            'sur', 'sous', 'entre', 'vers', 'chez', 'être', 'avoir',
            'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
            'falloir', 'vouloir', 'devoir', 'croire', 'rendre',
            'pas', 'plus', 'comme', 'tout', 'tous', 'toute', 'toutes',
            'très', 'bien', 'aussi', 'encore', 'déjà', 'jamais',
            'est', 'sont', 'ai', 'as', 'a', 'avons', 'avez', 'ont',
            'suis', 'es', 'sommes', 'êtes', 'était', 'étaient',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'should', 'could', 'can',
            'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }

    def extract_words(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extrait les mots significatifs d'un texte.

        Args:
            text: Texte à analyser
            min_length: Longueur minimale des mots

        Returns:
            Liste de mots nettoyés
        """
        if not text or not isinstance(text, str):
            return []

        # Nettoyer et découper
        text = text.lower()
        # Garder lettres, chiffres et espaces
        text = re.sub(r'[^a-zà-ÿ0-9\s]', ' ', text)
        words = text.split()

        # Filtrer
        words = [
            w for w in words
            if len(w) >= min_length and w not in self.stopwords
        ]

        return words

    def compute_word_frequencies(
        self,
        df: pd.DataFrame,
        top_n: int = 50,
        sender_filter: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Calcule les fréquences des mots.

        Args:
            df: DataFrame avec colonne 'message' et 'sender'
            top_n: Nombre de mots à retourner
            sender_filter: Filtrer par expéditeur (optionnel)

        Returns:
            Dict {mot: fréquence} des top_n mots
        """
        if df.empty or 'message' not in df.columns:
            return {}

        # Filtrer par expéditeur si demandé
        if sender_filter and 'sender' in df.columns:
            df = df[df['sender'] == sender_filter]

        # Extraire tous les mots
        all_words = []
        for msg in df['message']:
            words = self.extract_words(msg)
            all_words.extend(words)

        # Compter
        counter = Counter(all_words)
        return dict(counter.most_common(top_n))

    def compute_word_cloud_data(
        self,
        df: pd.DataFrame,
        top_n: int = 100
    ) -> List[Dict[str, any]]:
        """
        Prépare les données pour un word cloud.

        Args:
            df: DataFrame des messages
            top_n: Nombre de mots

        Returns:
            Liste de dicts avec {word, frequency, size}
        """
        frequencies = self.compute_word_frequencies(df, top_n=top_n)

        if not frequencies:
            return []

        # Normaliser les tailles pour visualisation (10-100)
        max_freq = max(frequencies.values())
        min_freq = min(frequencies.values())
        freq_range = max_freq - min_freq if max_freq > min_freq else 1

        data = []
        for word, freq in frequencies.items():
            normalized_size = 10 + (freq - min_freq) / freq_range * 90
            data.append({
                'word': word,
                'frequency': freq,
                'size': normalized_size
            })

        return data

    def compute_topic_distribution(
        self,
        df: pd.DataFrame,
        n_topics: int = 5,
        keywords_per_topic: int = 5
    ) -> List[Dict]:
        """
        Analyse simple de topics basée sur co-occurrences.

        Note: Implémentation simple sans LDA pour éviter dépendances lourdes.
        Utilise clustering de mots basé sur fréquences.

        Args:
            df: DataFrame des messages
            n_topics: Nombre de topics
            keywords_per_topic: Mots-clés par topic

        Returns:
            Liste de topics avec keywords et scores
        """
        if df.empty or 'message' not in df.columns:
            return []

        # Extraire mots de tous les messages
        all_words = []
        for msg in df['message']:
            words = self.extract_words(msg)
            all_words.extend(words)

        if not all_words:
            return []

        # Compter fréquences globales
        counter = Counter(all_words)
        top_words = dict(counter.most_common(n_topics * keywords_per_topic))

        # Créer des "topics" en groupant les mots par fréquence
        # (simplifié, pas de vrai LDA)
        words_sorted = sorted(top_words.items(), key=lambda x: x[1], reverse=True)

        topics = []
        chunk_size = len(words_sorted) // n_topics

        for i in range(n_topics):
            start = i * chunk_size
            end = start + chunk_size if i < n_topics - 1 else len(words_sorted)

            topic_words = words_sorted[start:end][:keywords_per_topic]

            if topic_words:
                keywords = [w[0] for w in topic_words]
                scores = [w[1] for w in topic_words]
                total_score = sum(scores)

                topics.append({
                    'topic_id': i + 1,
                    'keywords': keywords,
                    'scores': scores,
                    'weight': total_score / sum(top_words.values()) if sum(top_words.values()) > 0 else 0
                })

        return topics

    def compute_interaction_network(
        self,
        df: pd.DataFrame,
        time_window_minutes: int = 5
    ) -> Dict[str, any]:
        """
        Construit un réseau d'interactions entre participants.

        Une interaction = réponse dans une fenêtre de temps.

        Args:
            df: DataFrame avec 'sender', 'timestamp', 'message'
            time_window_minutes: Fenêtre de temps pour considérer une interaction

        Returns:
            Dict avec nodes et edges pour graphe
        """
        if df.empty or 'sender' not in df.columns:
            return {'nodes': [], 'edges': []}

        # S'assurer que timestamp est datetime
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Trier par temps
            df = df.sort_values('timestamp')
        else:
            # Pas de timestamp, utiliser l'ordre des messages
            df = df.copy()
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')

        # Compter messages par sender
        message_counts = df['sender'].value_counts().to_dict()

        # Détecter interactions (messages successifs dans fenêtre de temps)
        interactions = Counter()

        for i in range(1, len(df)):
            current_sender = df.iloc[i]['sender']
            prev_sender = df.iloc[i-1]['sender']

            if current_sender != prev_sender:
                # Vérifier fenêtre de temps
                time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds() / 60

                if time_diff <= time_window_minutes:
                    # Créer edge (ordre alphabétique pour éviter doublons)
                    edge = tuple(sorted([prev_sender, current_sender]))
                    interactions[edge] += 1

        # Créer nodes
        nodes = []
        for sender, count in message_counts.items():
            nodes.append({
                'id': sender,
                'label': sender,
                'size': count,
                'messages': count
            })

        # Créer edges
        edges = []
        for (source, target), weight in interactions.items():
            edges.append({
                'source': source,
                'target': target,
                'weight': weight
            })

        return {
            'nodes': nodes,
            'edges': edges
        }

    def compute_activity_heatmap(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List]:
        """
        Calcule une heatmap d'activité (jour de semaine x heure).

        Args:
            df: DataFrame avec 'timestamp'

        Returns:
            Dict avec days, hours, et matrix de counts
        """
        if df.empty or 'timestamp' not in df.columns:
            return {'days': [], 'hours': [], 'data': []}

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Extraire jour de semaine et heure
        df['weekday'] = df['timestamp'].dt.dayofweek  # 0=Lundi
        df['hour'] = df['timestamp'].dt.hour

        # Compter messages par (weekday, hour)
        heatmap = df.groupby(['weekday', 'hour']).size().reset_index(name='count')

        # Créer matrice 7x24
        days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        hours = list(range(24))

        matrix = []
        for day in range(7):
            row = []
            for hour in range(24):
                count = heatmap[
                    (heatmap['weekday'] == day) & (heatmap['hour'] == hour)
                ]['count'].sum()
                row.append(int(count))
            matrix.append(row)

        return {
            'days': days,
            'hours': hours,
            'data': matrix
        }


# Singleton
_analytics: Optional[AdvancedAnalytics] = None


def get_analytics() -> AdvancedAnalytics:
    """
    Retourne l'instance singleton d'analytics.

    Returns:
        Instance AdvancedAnalytics
    """
    global _analytics
    if _analytics is None:
        _analytics = AdvancedAnalytics()
    return _analytics

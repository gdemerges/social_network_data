"""
Pipeline d'évaluation RAG - Métriques de qualité et détection d'hallucinations.

Implémente des métriques inspirées de RAGAS:
- Faithfulness: Le LLM répond-il fidèlement au contexte?
- Answer Relevancy: La réponse est-elle pertinente à la question?
- Context Precision: Les chunks récupérés sont-ils pertinents?
- Context Recall: A-t-on récupéré tous les chunks nécessaires?
"""

import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import re


@dataclass
class EvaluationSample:
    """Un échantillon d'évaluation avec question/réponse/ground truth."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Résultat d'évaluation pour un échantillon."""
    sample_id: str
    question: str
    answer: str
    
    # Scores (0-1)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    # Score global
    overall_score: float = 0.0
    
    # Détails
    hallucination_detected: bool = False
    hallucination_details: List[str] = field(default_factory=list)
    
    # Métadonnées
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluation_time_ms: float = 0.0


@dataclass
class EvaluationReport:
    """Rapport complet d'évaluation."""
    results: List[EvaluationResult]
    
    # Métriques agrégées
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_overall_score: float = 0.0
    
    # Stats globales
    total_samples: int = 0
    hallucination_count: int = 0
    hallucination_rate: float = 0.0
    
    # Métadonnées
    evaluation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return asdict(self)
    
    def save(self, path: str):
        """Sauvegarde le rapport en JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class FaithfulnessEvaluator:
    """
    Évalue si la réponse est fidèle au contexte fourni.
    Détecte les hallucinations (affirmations non supportées par le contexte).
    """
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Client LLM pour l'évaluation (optionnel)
        """
        self.llm_client = llm_client
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extrait les affirmations d'un texte."""
        # Simplification: chaque phrase est une affirmation
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def _check_claim_in_context(self, claim: str, contexts: List[str]) -> Tuple[bool, float]:
        """
        Vérifie si une affirmation est supportée par le contexte.
        
        Returns:
            (is_supported, confidence)
        """
        claim_lower = claim.lower()
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))
        
        if len(claim_words) < 3:
            return True, 1.0  # Trop court pour juger
        
        # Mots importants (exclure les stop words basiques)
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'à', 'et', 
                      'est', 'sont', 'a', 'ont', 'que', 'qui', 'dans', 'pour', 'sur',
                      'avec', 'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles',
                      'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall'}
        
        important_words = claim_words - stop_words
        
        if not important_words:
            return True, 1.0
        
        # Chercher les mots dans le contexte
        full_context = ' '.join(contexts).lower()
        context_words = set(re.findall(r'\b\w+\b', full_context))
        
        matched_words = important_words & context_words
        coverage = len(matched_words) / len(important_words) if important_words else 1.0
        
        # Si moins de 50% des mots importants sont dans le contexte
        # c'est probablement une hallucination
        is_supported = coverage >= 0.5
        
        return is_supported, coverage
    
    def evaluate(self, sample: EvaluationSample) -> Tuple[float, List[str]]:
        """
        Évalue la fidélité d'une réponse.
        
        Returns:
            (score_faithfulness, liste_hallucinations)
        """
        claims = self._extract_claims(sample.answer)
        
        if not claims:
            return 1.0, []
        
        supported_count = 0
        hallucinations = []
        
        for claim in claims:
            is_supported, confidence = self._check_claim_in_context(
                claim, sample.contexts
            )
            
            if is_supported:
                supported_count += 1
            else:
                hallucinations.append(f"Non supporté (conf={confidence:.2f}): {claim[:100]}...")
        
        faithfulness = supported_count / len(claims) if claims else 1.0
        
        return faithfulness, hallucinations


class AnswerRelevancyEvaluator:
    """
    Évalue si la réponse est pertinente par rapport à la question.
    """
    
    def __init__(self):
        pass
    
    def _extract_keywords(self, text: str) -> set:
        """Extrait les mots-clés d'un texte."""
        words = re.findall(r'\b[a-zàâäéèêëïîôùûüç]{3,}\b', text.lower())
        
        # Filtrer les stop words
        stop_words = {
            'les', 'des', 'une', 'que', 'qui', 'dans', 'pour', 'sur', 'avec',
            'est', 'sont', 'ont', 'été', 'être', 'avoir', 'fait', 'faire',
            'the', 'and', 'that', 'this', 'with', 'from', 'have', 'has',
            'comment', 'quoi', 'quel', 'quelle', 'quels', 'quelles', 'pourquoi'
        }
        
        return set(words) - stop_words
    
    def evaluate(self, sample: EvaluationSample) -> float:
        """
        Évalue la pertinence de la réponse.
        
        Returns:
            Score de 0 à 1
        """
        question_keywords = self._extract_keywords(sample.question)
        answer_keywords = self._extract_keywords(sample.answer)
        
        if not question_keywords:
            return 1.0  # Question trop courte
        
        # Intersection des mots-clés
        common = question_keywords & answer_keywords
        
        # Score basé sur le coverage des mots-clés de la question
        relevancy = len(common) / len(question_keywords) if question_keywords else 1.0
        
        # Bonus si la réponse est substantielle
        if len(answer_keywords) > len(question_keywords):
            relevancy = min(1.0, relevancy + 0.1)
        
        return min(1.0, relevancy)


class ContextPrecisionEvaluator:
    """
    Évalue si les chunks récupérés sont pertinents pour la question.
    """
    
    def _calculate_chunk_relevance(self, question: str, chunk: str) -> float:
        """Calcule la pertinence d'un chunk par rapport à la question."""
        question_words = set(re.findall(r'\b[a-zàâäéèêëïîôùûüç]{3,}\b', question.lower()))
        chunk_words = set(re.findall(r'\b[a-zàâäéèêëïîôùûüç]{3,}\b', chunk.lower()))
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words & chunk_words)
        return overlap / len(question_words)
    
    def evaluate(self, sample: EvaluationSample) -> float:
        """
        Évalue la précision du contexte.
        
        Returns:
            Score de 0 à 1
        """
        if not sample.contexts:
            return 0.0
        
        relevance_scores = [
            self._calculate_chunk_relevance(sample.question, ctx)
            for ctx in sample.contexts
        ]
        
        # Weighted average: les premiers chunks comptent plus
        weights = [1.0 / (i + 1) for i in range(len(relevance_scores))]
        weighted_sum = sum(s * w for s, w in zip(relevance_scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class ContextRecallEvaluator:
    """
    Évalue si tous les chunks nécessaires ont été récupérés.
    Nécessite un ground truth pour être précis.
    """
    
    def evaluate(self, sample: EvaluationSample) -> float:
        """
        Évalue le recall du contexte.
        
        Returns:
            Score de 0 à 1
        """
        if not sample.ground_truth:
            # Sans ground truth, utiliser une heuristique
            # basée sur la couverture de la réponse par le contexte
            return self._estimate_recall(sample)
        
        # Avec ground truth: vérifier que les infos du GT sont dans le contexte
        gt_keywords = set(re.findall(r'\b[a-zàâäéèêëïîôùûüç]{4,}\b', sample.ground_truth.lower()))
        
        if not gt_keywords:
            return 1.0
        
        context_text = ' '.join(sample.contexts).lower()
        context_keywords = set(re.findall(r'\b[a-zàâäéèêëïîôùûüç]{4,}\b', context_text))
        
        covered = gt_keywords & context_keywords
        return len(covered) / len(gt_keywords) if gt_keywords else 1.0
    
    def _estimate_recall(self, sample: EvaluationSample) -> float:
        """Estime le recall sans ground truth."""
        answer_keywords = set(re.findall(r'\b[a-zàâäéèêëïîôùûüç]{4,}\b', sample.answer.lower()))
        context_text = ' '.join(sample.contexts).lower()
        context_keywords = set(re.findall(r'\b[a-zàâäéèêëïîôùûüç]{4,}\b', context_text))
        
        if not answer_keywords:
            return 1.0
        
        covered = answer_keywords & context_keywords
        return len(covered) / len(answer_keywords) if answer_keywords else 1.0


class RAGEvaluator:
    """
    Pipeline complet d'évaluation RAG.
    
    Combine toutes les métriques:
    - Faithfulness
    - Answer Relevancy  
    - Context Precision
    - Context Recall
    """
    
    def __init__(
        self,
        llm_client=None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            llm_client: Client LLM pour évaluation avancée
            weights: Poids des métriques pour le score global
        """
        self.faithfulness_eval = FaithfulnessEvaluator(llm_client)
        self.relevancy_eval = AnswerRelevancyEvaluator()
        self.precision_eval = ContextPrecisionEvaluator()
        self.recall_eval = ContextRecallEvaluator()
        
        # Poids par défaut
        self.weights = weights or {
            'faithfulness': 0.3,
            'answer_relevancy': 0.25,
            'context_precision': 0.25,
            'context_recall': 0.2
        }
    
    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        sample_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Évalue un échantillon unique.
        
        Args:
            question: Question posée
            answer: Réponse générée
            contexts: Chunks de contexte utilisés
            ground_truth: Réponse attendue (optionnel)
            sample_id: ID de l'échantillon
            
        Returns:
            Résultat d'évaluation
        """
        import time
        start_time = time.time()
        
        sample = EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        # Évaluer chaque métrique
        faithfulness, hallucinations = self.faithfulness_eval.evaluate(sample)
        relevancy = self.relevancy_eval.evaluate(sample)
        precision = self.precision_eval.evaluate(sample)
        recall = self.recall_eval.evaluate(sample)
        
        # Score global pondéré
        overall = (
            faithfulness * self.weights['faithfulness'] +
            relevancy * self.weights['answer_relevancy'] +
            precision * self.weights['context_precision'] +
            recall * self.weights['context_recall']
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return EvaluationResult(
            sample_id=sample_id or f"sample_{hash(question) % 10000}",
            question=question,
            answer=answer,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
            overall_score=overall,
            hallucination_detected=len(hallucinations) > 0,
            hallucination_details=hallucinations,
            evaluation_time_ms=elapsed_ms
        )
    
    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        model_name: str = "unknown"
    ) -> EvaluationReport:
        """
        Évalue un dataset complet.
        
        Args:
            samples: Liste de dicts avec 'question', 'answer', 'contexts', 'ground_truth'
            model_name: Nom du modèle évalué
            
        Returns:
            Rapport d'évaluation complet
        """
        results = []
        
        for i, sample in enumerate(samples):
            result = self.evaluate_sample(
                question=sample['question'],
                answer=sample['answer'],
                contexts=sample.get('contexts', []),
                ground_truth=sample.get('ground_truth'),
                sample_id=sample.get('id', f"sample_{i}")
            )
            results.append(result)
        
        # Calculer les moyennes
        n = len(results)
        if n == 0:
            return EvaluationReport(results=[])
        
        avg_faith = sum(r.faithfulness for r in results) / n
        avg_rel = sum(r.answer_relevancy for r in results) / n
        avg_prec = sum(r.context_precision for r in results) / n
        avg_rec = sum(r.context_recall for r in results) / n
        avg_overall = sum(r.overall_score for r in results) / n
        
        hallucination_count = sum(1 for r in results if r.hallucination_detected)
        
        return EvaluationReport(
            results=results,
            avg_faithfulness=avg_faith,
            avg_answer_relevancy=avg_rel,
            avg_context_precision=avg_prec,
            avg_context_recall=avg_rec,
            avg_overall_score=avg_overall,
            total_samples=n,
            hallucination_count=hallucination_count,
            hallucination_rate=hallucination_count / n if n > 0 else 0,
            model_name=model_name,
            config={'weights': self.weights}
        )


class TestDatasetGenerator:
    """
    Génère un dataset de test pour l'évaluation RAG.
    """
    
    def __init__(self, rag_engine):
        """
        Args:
            rag_engine: Instance du RAGEngine
        """
        self.rag_engine = rag_engine
    
    def generate_from_messages(
        self,
        n_samples: int = 20,
        question_templates: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Génère des échantillons de test à partir des messages indexés.
        
        Args:
            n_samples: Nombre d'échantillons à générer
            question_templates: Templates de questions
            
        Returns:
            Dataset de test
        """
        templates = question_templates or [
            "Qui a envoyé des messages sur {topic}?",
            "Quels messages parlent de {topic}?",
            "Combien de messages mentionnent {topic}?",
            "Que disent les messages à propos de {topic}?",
            "Résume les discussions sur {topic}.",
        ]
        
        # Extraire des topics depuis les messages indexés
        stats = self.rag_engine.get_stats()
        
        # Générer des questions basiques
        basic_questions = [
            "Qui sont les participants à cette conversation?",
            "Quel est le sujet principal de cette conversation?",
            "Combien de messages ont été échangés?",
            "Qui a envoyé le plus de messages?",
            "Quel est le ton général de la conversation?",
        ]
        
        samples = []
        for i, question in enumerate(basic_questions[:n_samples]):
            # Obtenir la réponse du RAG
            result = self.rag_engine.chat(question)
            
            samples.append({
                'id': f"test_{i}",
                'question': question,
                'answer': result['answer'],
                'contexts': [s.get('content', '') for s in result.get('sources', [])],
                'ground_truth': None  # À remplir manuellement pour évaluation précise
            })
        
        return samples
    
    def save_dataset(self, samples: List[Dict], path: str):
        """Sauvegarde le dataset en JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    
    def load_dataset(self, path: str) -> List[Dict]:
        """Charge un dataset depuis JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Fonction utilitaire pour évaluation rapide
def quick_evaluate(
    rag_engine,
    questions: List[str],
    ground_truths: Optional[List[str]] = None
) -> EvaluationReport:
    """
    Évaluation rapide d'un RAG engine.
    
    Args:
        rag_engine: Instance du RAGEngine
        questions: Liste de questions à tester
        ground_truths: Réponses attendues (optionnel)
        
    Returns:
        Rapport d'évaluation
    """
    evaluator = RAGEvaluator()
    samples = []
    
    for i, question in enumerate(questions):
        result = rag_engine.chat(question)
        
        samples.append({
            'question': question,
            'answer': result['answer'],
            'contexts': [s.get('content', '') for s in result.get('sources', [])],
            'ground_truth': ground_truths[i] if ground_truths and i < len(ground_truths) else None
        })
    
    return evaluator.evaluate_dataset(
        samples,
        model_name=rag_engine.ollama_model
    )

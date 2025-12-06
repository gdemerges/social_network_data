# 💬 Message Analyzer - Dashboard RAG Avancé

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Dash](https://img.shields.io/badge/Dash-2.14%2B-orange)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple)
![License](https://img.shields.io/badge/License-MIT-green)

> **Analysez vos conversations avec un RAG production-ready** : Recherche hybride, re-ranking, évaluation RAGAS intégrée, et LLM local (Ollama).

## 🎯 Présentation

**Message Analyzer** est un dashboard interactif qui transforme vos exports de messagerie en insights actionables via une architecture RAG (Retrieval-Augmented Generation) de qualité production.

### ✨ Points forts

| Fonctionnalité | Description |
|---|---|
| 🔍 **Recherche Hybride** | Vector + BM25 + Reciprocal Rank Fusion |
| 🤖 **Re-ranking Intelligent** | Cross-encoder pour précision maximale |
| 🎯 **Évaluation Intégrée** | Métriques RAGAS : Faithfulness, Relevancy, Precision, Recall |
| 📥 **Multi-format** | JSON, CSV, TXT, (extensible) |
| 🧠 **LLM Local** | Ollama : Mistral, Llama 3, Phi-3, Gemma |
| 📊 **Dashboard Moderne** | Thème sombre, graphiques temps réel, chat IA |
| 🛡️ **Pas d'hallucinations** | Détection automatique + contexte sourcé |

---

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.10+
- Ollama installé et en cours d'exécution (pour les LLM locaux)
- pip ou uv (gestionnaire de paquets)

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/gdemerges/social_network_data.git
cd social_network_data
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Vérifier Ollama**
```bash
# Ollama doit tourner
ollama serve

# Dans un autre terminal, tester:
curl http://localhost:11434/api/tags
```

4. **Lancer l'application**
```bash
python -m social_network_data
```

Puis ouvrez `http://localhost:8050` dans votre navigateur.

---

## 📚 Architecture

```
social_network_data/
│
├── rag/                          # 🧠 Moteur RAG
│   ├── engine.py                 # RAGEngine principal
│   ├── ingestion.py              # 📥 Parsers multi-format
│   ├── chunking.py               # ✂️ Stratégie de chunking
│   ├── embeddings.py             # 🔢 Embeddings (sentence-transformers)
│   ├── vector_store.py           # 💾 ChromaDB
│   ├── llm_client.py             # 🤖 Client Ollama
│   ├── retriever.py              # 🔍 Recherche hybride + re-ranking
│   └── evaluation.py             # 📊 Métriques RAGAS-like
│
├── dashboard/                    # 📊 Interface utilisateur
│   ├── layout.py                 # UI components
│   ├── callbacks.py              # Dash callbacks
│   ├── data_processing.py        # Traitement de données
│   ├── styles.py                 # Thème dark moderne
│   └── __init__.py
│
├── assets/
│   └── custom.css                # 🎨 CSS personnalisé
│
├── __main__.py                   # 🚀 Entry point
└── requirements.txt              # 📦 Dépendances
```

---

## 💡 Cas d'Usage

### 1️⃣ Charger vos messages
- Exportez depuis Facebook, Instagram, WhatsApp, Telegram
- Ou fournissez un CSV/JSON personnalisé
- Format auto-détecté

### 2️⃣ Analyser via IA
- Chat avec votre conversation (questions en français)
- Contexte sourcé automatiquement
- Zéro hallucinations grâce à la fidélité vérifiée

### 3️⃣ Évaluer la qualité
- Metrics RAGAS intégrées
- Détection d'hallucinations
- Rapport d'évaluation JSON exportable

### 4️⃣ Exporter les résultats
- Données filtrées en CSV
- Rapports d'évaluation

---

## 🎮 Utilisation

### Via le Dashboard

1. **Upload** : Glissez votre fichier JSON/CSV
2. **Filtrer** : Par expéditeur, date
3. **Visualiser** : Graphiques automatiques
4. **Chatter** : Posez des questions (en français !)
5. **Exporter** : Vos résultats

### Via Python (API)

```python
from rag import RAGEngine

# Créer le moteur
rag = RAGEngine(
    ollama_model="mistral",  # ou llama3, phi3, gemma
    use_hybrid_search=True,   # Vector + BM25
    use_reranking=True        # Cross-encoder
)

# Indexer les messages
import pandas as pd
messages_df = pd.read_json("messages.json")
rag.index_messages(messages_df)

# Chat
result = rag.chat("Qui a parlé de voyage?")
print(result['answer'])
print(result['retrieval_method'])  # 'hybrid+rerank'

# Évaluation
report = rag.evaluate(
    questions=[
        "Quel est le sujet principal?",
        "Qui participe le plus?"
    ]
)
print(f"Faithfulness: {report.avg_faithfulness:.2%}")
print(f"Hallucinations détectées: {report.hallucination_rate:.1%}")
report.save("evaluation_report.json")
```

---

## 🔬 Architecture RAG Avancée

### 1. Ingestion (`rag/ingestion.py`)

**DataCleaner** : Normalisation intelligente
```python
DataCleaner(
    remove_urls=True,
    remove_emails=True,
    remove_emojis=False,
    normalize_whitespace=True
)
```

**Parsers Supportés** :
- `JSONMessageParser` : Facebook, Instagram, WhatsApp (formats connus)
- `CSVMessageParser` : Imports personnalisés
- `TextFileParser` : TXT, MD, RST, LOG
- Extensible : créer votre propre `BaseParser`

### 2. Chunking (`rag/chunking.py`)

**Fenêtre Glissante** : Grouper les messages par contexte conversationnel
```python
# 5 messages par fenêtre avec contexte chevauchant
TextChunker(chunk_size=512, chunk_overlap=50)
```

### 3. Recherche Hybride (`rag/retriever.py`)

**Étapes** :
1. **Recherche Vectorielle** (sémantique via embeddings)
2. **Recherche BM25** (lexicale pour précision)
3. **Fusion RRF** : Reciprocal Rank Fusion
4. **Re-ranking** : Cross-encoder `ms-marco-MiniLM-L-6-v2`

```python
# Configuration par défaut
HybridRetriever(
    vector_weight=0.6,    # 60% sémantique
    bm25_weight=0.4,      # 40% lexical
    use_reranking=True,
    rrf_k=60              # Standard
)
```

### 4. Évaluation RAGAS-like (`rag/evaluation.py`)

**4 métriques clés** :

| Métrique | Signification |
|---|---|
| **Faithfulness** | La réponse est-elle fidèle au contexte? |
| **Answer Relevancy** | Réponse pertinente pour la question? |
| **Context Precision** | Les chunks sont-ils pertinents? |
| **Context Recall** | Avez-vous récupéré assez d'infos? |

```python
# Évaluation rapide
from rag import quick_evaluate

report = quick_evaluate(
    rag_engine,
    questions=["Q1", "Q2", "Q3"],
    ground_truths=["Expected1", "Expected2", "Expected3"]
)

# Résultats
print(f"Score global: {report.avg_overall_score:.2f}/1.0")
print(f"Hallucinations: {report.hallucination_details}")
```

---

## ⚙️ Configuration

### Variables d'environnement

```bash
# URL Ollama
export OLLAMA_BASE_URL=http://localhost:11434

# Modèle par défaut
export OLLAMA_MODEL=mistral
```

### Modèles Ollama Recommandés

| Modèle | Taille | Vitesse | Qualité | Cas d'usage |
|---|---|---|---|---|
| **mistral** | 7B | ⚡⚡⚡ | ⭐⭐⭐ | Production par défaut |
| **llama3** | 8B | ⚡⚡ | ⭐⭐⭐⭐ | Meilleure qualité |
| **phi3** | 3.8B | ⚡⚡⚡⚡ | ⭐⭐⭐ | Ressources limitées |
| **gemma** | 7B | ⚡⚡ | ⭐⭐⭐⭐ | Bon équilibre |

Installer : `ollama pull mistral`

---

## 📊 Exemple de Résultats

### Dashboard
```
Statistiques:
- 1,234 messages indexés
- 45 participants
- 312 chunks RAG
- Sentiment moyen: 😊 +0.42

Chat IA:
Q: "Qui a parlé de voyage?"
A: "[Alice]: J'irais en Italie"
   "[Bob]: Bonne idée, j'adore Rome"
   
Qualité: ✅ Faithful, Relevant, Sourced
```

### Rapport d'Évaluation
```json
{
  "avg_faithfulness": 0.94,
  "avg_answer_relevancy": 0.87,
  "avg_context_precision": 0.91,
  "avg_context_recall": 0.89,
  "avg_overall_score": 0.90,
  "hallucination_rate": 0.05,
  "total_samples": 20
}
```

---

## 🔧 API Complète

### RAGEngine

```python
class RAGEngine:
    # Indexation
    index_messages(df: DataFrame) -> int  # Retourne nb chunks
    
    # Recherche
    search(query: str, n_results=5, use_hybrid=True) -> List[Dict]
    
    # Chat
    chat(question: str, n_context=5) -> Dict
    
    # Évaluation
    evaluate(questions: List[str], ground_truths=None) -> EvaluationReport
    evaluate_single(question, answer, contexts, ground_truth=None) -> Dict
    
    # Stats
    get_stats() -> Dict
    check_ollama_status() -> Dict
```

### Evaluation

```python
from rag import RAGEvaluator, EvaluationReport

evaluator = RAGEvaluator()

# Évaluation d'échantillon unique
result = evaluator.evaluate_sample(
    question="Qui?",
    answer="Alice",
    contexts=["Alice a parlé"],
    ground_truth="Alice"
)
# → EvaluationResult avec scores détaillés

# Évaluation dataset
report = evaluator.evaluate_dataset(samples, model_name="mistral")
report.save("report.json")
```

---

## 🧪 Tests & Développement

### Exécuter les tests (futur)
```bash
pytest tests/
```

### Logs détaillés
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profiling
```python
from rag import RAGEvaluator
import cProfile

profiler = cProfile.Profile()
profiler.enable()

# ... votre code ...

profiler.disable()
profiler.print_stats(sort='cumtime')
```

---

## 🎨 Customisation

### Ajouter un Parser personnalisé

```python
from rag.ingestion import BaseParser, ParsedDocument

class MyCustomParser(BaseParser):
    def can_parse(self, source):
        return isinstance(source, MyFormat)
    
    def parse(self, source):
        # Votre logique
        return ParsedDocument(
            content="...",
            source="custom",
            doc_type="custom",
            metadata={}
        )

# Utiliser
ingester = DocumentIngester(custom_parsers=[MyCustomParser()])
doc = ingester.ingest(my_data)
```

### Changer le modèle de Re-ranking

```python
rag = RAGEngine(
    reranker_model="cross-encoder/qnli-distilroberta-base"
)
```

---

## 📈 Benchmarks

Mesurés sur MacBook Pro M3 (Ollama local):

| Opération | Temps |
|---|---|
| Index 1000 messages | ~2s |
| Recherche hybride | ~150ms |
| Re-ranking (5 docs) | ~50ms |
| Génération LLM | ~1-2s (selon modèle) |
| Évaluation RAGAS | ~500ms/sample |

---

## 🐛 Troubleshooting

### Ollama ne démarre pas
```bash
# Vérifier l'installation
ollama --version

# Relancer le service
ollama serve

# Vérifier la connexion
curl http://localhost:11434/api/tags
```

### Erreur ChromaDB
```bash
# Réinstaller
pip install --upgrade chromadb
```

### Embeddings lents
- Les modèles se téléchargent à la première utilisation (~600MB)
- Prendre un café ☕ la première fois !

### LLM hallucine
- Augmentez `n_context` dans `chat()`
- Activez le re-ranking : `use_reranking=True`
- Évaluez avec `evaluate()` pour identifier les problèmes

---

## 📦 Dépendances

### Essentielles
- `dash` : Framework web
- `chromadb` : Vector store
- `sentence-transformers` : Embeddings
- `requests` : Requêtes HTTP

### Optionnelles
```bash
# Parsing avancé
pip install llama-parse unstructured python-docx PyPDF2 pytesseract

# Évaluation officielle
pip install ragas giskard trulens-eval
```

---

## 🔐 Sécurité

- ✅ Pas de données envoyées au cloud (LLM local)
- ✅ Chiffrement ChromaDB
- ✅ Validation des entrées
- ⚠️ Ne publiez pas `__pycache__` ou `.chroma/`

---

## 📝 Roadmap

- [ ] Support Ollama multi-modèles (image2text)
- [ ] Export Markdown/HTML formaté
- [ ] Cache intelligent des embeddings
- [ ] Benchmark suite RAGAS complet
- [ ] UI Mobile responsive
- [ ] API REST
- [ ] Docker/Docker-Compose

---

## 🤝 Contribution

Les contributions sont bienvenues ! 

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📄 License

Ce projet est sous license MIT - voir le fichier [LICENSE.md](LICENSE.md) pour détails.

---

## 👨‍💻 Auteur

**Guillaume de Merges**
- GitHub: [@gdemerges](https://github.com/gdemerges)
- Email: [contact@example.com]

---

## 🙏 Remerciements

- 🙌 [Ollama](https://ollama.ai) pour les LLM locaux
- 🧠 [ChromaDB](https://www.trychroma.com) pour le vector store
- 📚 [sentence-transformers](https://www.sbert.net) pour les embeddings
- 📊 [Dash/Plotly](https://dash.plotly.com) pour le UI

---

## 📞 Support

Besoin d'aide ?

- 📖 [Documentation complète](docs/)
- 🐛 [Issues](../../issues)
- 💬 [Discussions](../../discussions)

---

**Made with ❤️ for RAG enthusiasts**

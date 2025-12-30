# Workflows du Pipeline RAG

Ce document présente les différents scénarios d'architecture pour alimenter le chatbot RAG, depuis l'extraction des sources documentaires jusqu'à la génération de réponses.

---

## Scénario 1 : Pipeline batch simple (Actuel)

### Description
Indexation périodique des sources documentaires. Le chatbot interroge une base PostgreSQL + pgvector alimentée par des jobs batch.

### Architecture

```
                           INGESTION
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Sitemap.xml │────▶│  Crawler     │────▶│  Extracteur  │────▶│  Embedder    │
│  AdocMS      │     │              │     │  HTML → MD   │     │  BGE-m3      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                           SERVING                           ┌──────────────────┐
┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │   PostgreSQL     │
│ Utilisateur  │────▶│  FastAPI     │────▶│  Retriever   │──▶│   + pgvector     │
│              │     │              │     │              │   │                  │
└──────────────┘     └──────────────┘     └──────┬───────┘   └──────────────────┘
                                                 │
                     ┌──────────────┐     ┌──────▼───────-┐
                     │   Réponse    │◀────│  LLM Mistral  │
                     │   enrichie   │     │               │
                     └──────────────┘     └───────────────┘
```

### Mise à jour des données
- **Fréquence** : Manuelle ou planifiée (cron)
- **Détection** : Checksum SHA-256

### Adapté si
- Sources mises à jour < 1x/jour
- Tolérance à un délai de fraîcheur (heures/jours)

### Limitations
- Pas de conservation des fichiers bruts
- Re-téléchargement nécessaire pour ré-indexer
- Pas de traçabilité des versions

---

## Scénario 2 : Pipeline avec cache des sources

### Description
Ajout d'un stockage objet (MinIO/S3) pour conserver les fichiers bruts. Permet ré-indexation sans re-téléchargement.

### Architecture

```
                           INGESTION
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Sitemap     │────▶│  Crawler     │────▶│  Downloader  │
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌───────────────────────────┼────────────────────────────-┐
                     │                           ▼                             │
                     │   ┌──────────────┐   ┌──────────────┐                   │
                     │   │  Object Store│   │  Extracteur  │                   │
                     │   │  (raw files) │──▶│              │                   │
                     │   └──────────────┘   └──────┬───────┘                   │
                     │                             │                           │
                     │   ┌──────────────┐   ┌──────▼───────┐                   │
                     │   │  PostgreSQL  │◀──│  Embedder    │                   │
                     │   │              │   │              │                   │
                     │   └──────────────┘   └──────────────┘                   │
                     └─────────────────────────────────────────────────────────┘
                                                 │
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│ Chatbot user │────▶│  RAG API     │────▶│  Retriever   │
│              │     │              │     │  + Reranker  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌──────────────┐     ┌──────▼-───────┐
                     │   Réponse    │◀────│  LLM Mistral  │
                     └──────────────┘     └───────────────┘
```

### Avantages pour le RAG
- **Ré-indexation rapide** : Changement de modèle d'embedding sans re-crawl
- **Debug qualité** : Comparer chunk vs source originale
- **Audit** : Traçabilité des sources utilisées par le chatbot
- **Fallback** : Si le site source est down, on peut toujours indexer depuis le cache

### Structure Object Store
```
bucket: rag-sources/
├── raw/                          # Fichiers téléchargés
│   └── adocms/
│       └── 2024-12-16/
│           ├── page1.html
│           └── doc.docx
├── extracted/                    # Markdown généré
│   └── adocms/
│       └── page1.md
└── snapshots/                    # Backups d'embeddings (optionnel)
    └── 2024-12-16.parquet
```

### Adapté si
- Besoin de ré-indexer fréquemment (changement de modèle, tuning)
- Exigences d'audit/traçabilité
- Sources parfois indisponibles

---


## Scénario 3 : Multi-sources avec orchestration

### Description
Plusieurs sources documentaires (AdocMS, SharePoint, PDFs...) alimentent un RAG unifié via un orchestrateur.

### Architecture

```
                           SOURCES MULTIPLES
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  AdocMS      │   │  SharePoint  │   │  Adoc-CSU    │   │     PDFs     │
│              │   │              │   │              │   │              │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       └──────────────────┼──────────────────┼──────────────────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────────────────────────┐
                   │        ORCHESTRATEUR            │
                   │          (Airflow)              │
                   │   - Scheduling par source       │
                   │   - Retry & alerting            │
                   │   - Monitoring                  │
                   └──────────────┬──────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  Extractor   │         │  Extractor   │         │  Extractor   │
│  AdocMS      │         │  SharePoint  │         │  PDF         │
└──────┬───────┘         └──────┬───────┘         └──────┬───────┘
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                │
                                ▼
                   ┌─────────────────────────────────┐
                   │         PIPELINE UNIFIÉ         │
                   │   Chunking → Embedding → Store  │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │     PostgreSQL + pgvector       │
                   │     (toutes sources confondues) │
                   └──────────────┬──────────────────┘
                                  │
                                  │
                                  ▼
┌──────────────┐     ┌─────────────────────────────────┐     ┌──────────────┐
│ Chatbot user │────▶│  RAG API avec filtrage source   │────▶│  LLM         │
│              │     │  (ex: "cherche dans adocMS")    │     │              │
└──────────────┘     └─────────────────────────────────┘     └──────────────┘
```


### Adapté si
- Multiples bases documentaires à unifier
- Besoin de recherche cross-source

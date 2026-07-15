# ShopFloor GenAI Engine

**GenAI-Powered Manufacturing Knowledge Assistant** — Production RAG + CAG backend for intelligent shop floor operations.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/fastapi-0.115+-green)](https://fastapi.tiangolo.com/) [![ChromaDB](https://img.shields.io/badge/chromadb-0.5.23-purple)](https://www.trychroma.com/) [![Live Demo](https://img.shields.io/badge/demo-Railway-brightgreen)](https://shopfloor-dryrun-genai-engine-production.up.railway.app/docs)

## Overview

The **ShopFloor GenAI Engine** is a production-grade retrieval-augmented generation (RAG) and cached-generation (CAG) system built for manufacturing environments.

**🎬 See it in action:** [View the animated demo (49 sec, 2.5x speed)](https://github.com/rah00l/shopfloor-dryrun-genai-web#-demo) showing Q&A, source traceability, and SOP generation in the paired Rails web app. It ingests shop floor documentation (SOPs, work instructions, maintenance manuals, quality alerts, engineering change notices, root cause reports) and provides:

- **Natural-language Q&A** over a manufacturing corpus with full source traceability
- **Cached-generation (CAG)** for repeated questions — reducing latency and cost
- **Confidence scoring** derived from real retrieval distance, not guessed labels
- **Auto-SOP generation** — merges engineering changes into complete updated procedures
- **Automated evaluation harness** — measures retrieval accuracy across known-good test cases
- **Production-ready architecture** — handles chunking, embedding, retrieval, and LLM generation at scale

Built and deployed in **24 hours** as part of GOH-UC-020 (GenAI-Powered Intelligent Manufacturing Knowledge Assistant for Shop Floor Operations) hackathon, selected from a pool of 100+ use cases based on 65–75% architectural reuse from ReconPilot AI.

---

## Key Features

### 🎯 Retrieval + Generation (RAG)
- **Semantic search** over 8-document manufacturing corpus using ChromaDB embeddings
- **Chunking at logic boundaries** (section-aware, not arbitrary size) — preserves context
- **OpenAI embeddings** for high-quality vector representations
- **Confidence thresholds** tuned from 0.7 → 0.5 after real eval harness testing

### 📦 Caching for Repeated Questions (CAG)
- **Prompt-level caching** — identical questions served from cache without re-computing embeddings or LLM calls
- **Cache badge** visible to end users — shows whether an answer came from cache (CAG) or fresh lookup (RAG)
- **Reduces latency and cost** on repeat questions

### ✍️ Auto-SOP Generation
- **Merges engineering changes** into complete, updated procedures
- **Preserves unaffected steps** — only updates relevant sections
- **Flags when no baseline exists** — never invents procedures
- **Generates draft** ready for human review

### 🔍 Source Traceability + Confidence
- **Every answer shows**: citation status (Cited/Not-found), retrieval mode (RAG/CAG), confidence tier (High/Medium/Low)
- **Expandable source cards** with exact document, section, and match strength
- **Thumbs up/down feedback loop** — operators flag wrong answers on the spot, logged for review
- **No false confidence claims** — 100% accuracy is not claimed; instead, work is shown at every layer

### 🧪 Automated Evaluation
- **Eval harness** runs a fixed set of test questions against known-correct expected sources
- **Pass/fail count** — independently verifiable evidence
- **9/9 test questions passing** in current build
- **Real-world validation** — confidence thresholds tuned after running against actual corpus

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   FastAPI Application                  │
│                     (app.py, 8000)                     │
│                                                        │
│  ┌─────────────────────────────────────────────────────┤
│  │              /analyze (Q&A Endpoint)                │
│  │  - Input: question, session_id, context             │
│  │  - Output: answer, sources, confidence, mode (RAG/CAG)
│  └─────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────────────────────────────────────────┤
│  │         /generate-sop (SOP Generation Endpoint)     │
│  │  - Input: change_text (engineering change)          │
│  │  - Output: draft_sop, reference_docs, grounded flag │
│  └─────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────────────────────────────────────────┤
│  │  Health / Ready / Info Endpoints                    │
│  │  - /health, /ready, /info (startup checks)          │
│  └─────────────────────────────────────────────────────┤
│                                                        │
└────────────────────────────────────────────────────────┘
         │               │               │
         ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   RAG Pipe   │ │   Chunker    │ │   Embedder   │
│ (retriever)  │ │ (chunker.py) │ │(embedder.py) │
│              │ │              │ │              │
│ - Query text │ │ - Ingest all │ │ - OpenAI API │
│ - Retrieve K │ │   docs       │ │ - Generate   │
│   chunks     │ │ - Split by   │ │   vectors    │
│ - Re-rank    │ │   section    │ │ - Store with │
│ - LLM gen    │ │ - Skip empty │ │   metadata   │
│              │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
         │
         ▼
    ┌──────────────────────┐
    │   ChromaDB (Vector   │
    │   Store + Cache)     │
    │                      │
    │ - In-process (POC)   │
    │ - Persistent storage │
    │ - CAG prompt caching │
    │ - Cosine similarity  │
    │                      │
    └──────────────────────┘
         │
         ▼
    ┌──────────────────────┐
    │  rag_docs/           │
    │  (8 markdown files)  │
    │                      │
    │ - SOP docs (3)       │
    │ - Work instructions  │
    │ - Maintenance manual │
    │ - Quality alerts (2) │
    │ - ECN                │
    │ - RCA report         │
    │                      │
    └──────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI 0.115 | Async web server, OpenAPI docs |
| **Server** | Uvicorn | ASGI app server |
| **Embeddings** | OpenAI API | Semantic vector generation |
| **Vector Store** | ChromaDB 0.5.23 | In-process RAG index (POC) |
| **LLM** | OpenAI GPT-4o | Generation + reasoning |
| **Language** | Python 3.11 | Core backend |
| **Config** | python-dotenv | Environment management |
| **Container** | Docker + Railway | Production deployment |

---

## Quick Start

### Prerequisites
- Python 3.11+
- `pip` / `poetry`
- OpenAI API key (`OPENAI_API_KEY`)

### Local Development

1. **Clone & navigate:**
   ```bash
   git clone https://github.com/rah00l/shopfloor-dryrun-genai-engine.git
   cd shopfloor-dryrun-genai-engine
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   export OPENAI_API_KEY="sk-..."
   export CHROMA_PATH="./chroma_db"
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Verify startup:**
   - Health: `GET http://localhost:8000/health`
   - OpenAPI docs: `http://localhost:8000/docs`

### Docker (Local with docker-compose)

From the parent directory (if running alongside `shopfloor-dryrun-genai-web`):

```bash
docker-compose up -d engine
# Check logs: docker-compose logs -f engine
# API available at: http://localhost:8000
```

---

## API Reference

### POST `/analyze`
Retrieve + generate an answer to a question about shop floor documentation.

**Request:**
```json
{
  "question": "What torque spec should I use for door hinge bracket mounting?",
  "session_id": "optional-session-id",
  "context": {}
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "status": "ok",
  "explanation": "According to WI-4.2, the door hinge bracket mounting bolts should be torqued to 45 Nm...",
  "sources": [
    {
      "document": "02-WI-4.2-torque-specifications.md",
      "section": "Torque Specifications",
      "confidence": "high",
      "match_distance": 0.15
    }
  ],
  "from_cache": false,
  "timestamp": "2026-07-15T12:34:56.789Z"
}
```

### POST `/generate-sop`
Generate an updated SOP by merging an engineering change with the existing baseline procedure.

**Request:**
```json
{
  "change_text": "Door hinge bracket thickness increased from 2mm to 2.5mm. Fixture requires alignment adjustment."
}
```

**Response:**
```json
{
  "status": "ok",
  "draft_sop": "## SOP-3.1 Bracket Installation, Rev C\n\n1. Confirm incoming bracket matches...",
  "reference_docs": ["05-ECN-2291-bracket-thickness.md", "01-SOP-3.1-bracket-installation.md"],
  "grounded": true,
  "timestamp": "2026-07-15T12:35:20.123Z"
}
```

### GET `/health`
Service health check.

**Response:**
```json
{
  "status": "ok",
  "service": "ai-analyst-engine",
  "version": "2.0.0-alpha",
  "timestamp": "2026-07-15T12:34:00.000Z"
}
```

---

## Production Scaling

This POC is intentionally kept lean for a 24-hour hackathon build. Here's the path to production scale:

### Large-File Ingestion
- **Streaming + chunking** — files never loaded fully into memory
- **Object storage** (S3) — raw files land here; app stores only processed text + pointer
- **Background jobs** — ingestion runs async, never blocks user-facing requests
- **Connector ecosystem** — one connector per source system (Jira, Confluence, SharePoint, HR platforms)

### Chunking at Scale
- **Parallel worker pool** — chunk across many files at once, not sequentially
- **Hash-based dedup** — skip re-chunking unchanged sections since last sync
- **Oversized auto-split** — sections exceeding max-length auto-split for consistent quality

### Faster Retrieval Beyond CAG
- **Approximate nearest-neighbor (HNSW)** — fast search across millions of vectors
- **Read replicas** — query load spread across multiple machines
- **Hybrid search** — vector similarity + keyword search for exact-term questions
- **Result re-ranking** — improve relevance without re-running full search

### Vector Store Evolution
- **Current**: ChromaDB in-process (POC)
- **Scale**: Distributed vector database (Pinecone, Qdrant, pgvector)
- **Architecture**: Embedding generation and query serving scale independently
- **Sharding**: Index distributed across nodes as corpus grows

### Observability & Recovery
- **Structured logging** — question, retrieval distance, confidence, response time, cache hit/miss
- **Metrics dashboard** — retrieval accuracy, average latency, cache hit rate over time
- **Distributed tracing** — trace requests across connector → ingestion → retrieval → LLM
- **Alerting** — confidence-score drift, spike in "not found" responses (early signal to update corpus)
- **Retries + circuit breakers** — transient failures don't cascade
- **Ingestion checkpointing** — interrupted sync resumes, doesn't restart from zero
- **Regular backups** — vector store backed up with documented restore path

---

## Evaluation & Benchmarks

**Automated Eval Harness:**
- **9/9 test questions passing** — independently verifiable evidence of retrieval accuracy
- **Real-world tuning** — confidence thresholds calibrated against actual corpus performance (0.7 too strict → 0.5 optimal)
- **No one-time-wins** — eval harness runs on every build, not a one-off demo

**Key Insights:**
1. **Domain-specific threshold tuning matters** — 0.7 confidence cutoff incorrectly rejected valid, grounded content from the manufacturing corpus
2. **CAG is a feature, not an implementation detail** — surfacing mode badges and match strength to users builds trust and transparency
3. **Honest failure modes** — saying "I don't know" directly is stronger than inventing an answer

---

## Project Structure

```
shopfloor-dryrun-genai-engine/
├── app.py                    # FastAPI application + endpoints
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production container image
├── .env.example              # Environment template
│
├── rag/                       # RAG pipeline modules
│   ├── __init__.py
│   ├── pipeline.py           # Orchestrates initialization and query flow
│   ├── chunker.py            # Document chunking (section-aware)
│   ├── embedder.py           # Vector generation (OpenAI API)
│   ├── retriever.py          # ChromaDB search + re-ranking
│   ├── cache.py              # CAG prompt-level caching logic
│   ├── sop_chain.py          # SOP generation (change merging)
│   ├── store.py              # Vector store operations
│   └── eval.py               # Automated evaluation harness
│
└── rag_docs/                 # Sample manufacturing corpus (8 files)
    ├── 01-SOP-3.1-bracket-installation.md
    ├── 02-WI-4.2-torque-specifications.md
    ├── 03-MM-station4-fixture-maintenance.md
    ├── 04-QA-2026-009-quality-alert.md
    ├── 05-ECN-2291-bracket-thickness.md
    ├── 06-RCA-2026-014-bracket-flex.md
    ├── 07-SOP-7.4-conveyor-tensioner-inspection.md
    └── 08-QA-2026-021-conveyor-slippage.md
```

---

## Deployment

### Railway (Production)

```bash
# Connect Railway project:
railway link

# Deploy:
railway up

# View logs:
railway logs
```

**Environment Variables (set in Railway):**
```
OPENAI_API_KEY=sk-...
CHROMA_PATH=/app/chroma_db
RAILS_ENV=production
PORT=8000
```

**Live Endpoint:**
```
https://shopfloor-dryrun-genai-engine-production.up.railway.app
```

---

## Key Bugs Found & Fixed During Build

1. **Chunker preamble bug** — First sections from `##`-header-starting documents were silently dropped. Fixed: explicit chunk ID generation for all sections.
2. **Confidence threshold tuning** — Initial 0.7 threshold too aggressive, incorrectly flagging valid grounded content. Tuned to 0.5 after eval harness testing.
3. **Port configuration** — Stale Sinatra references in initializers; switched cleanly to FastAPI.

---

## Contributing

This is a reference build from a 24-hour hackathon. If you'd like to extend it:

1. **Add new manufacturing corpus** — drop markdown files in `rag_docs/`, re-run `pipeline.initialize()`
2. **Tune confidence thresholds** — modify `retriever.py`, run eval harness
3. **Add connectors** — build intake adapters for Jira, Confluence, SharePoint in a new `connectors/` module
4. **Implement production vector store** — swap ChromaDB for Pinecone/Qdrant in `store.py`

---

## Contact & Questions

- **Build**: GOH-UC-020 (GenAI-Powered Intelligent Manufacturing Knowledge Assistant)
- **Hackathon**: 24-hour build, July 2026
- **Author**: [Rahul Patil](https://github.com/rah00l)
- **Live Demo**: [shopfloor-dryrun-genai-engine-production.up.railway.app](https://shopfloor-dryrun-genai-engine-production.up.railway.app/docs)
- **Paired Web App**: [shopfloor-dryrun-genai-web](https://github.com/rah00l/shopfloor-dryrun-genai-web)

---

## License

MIT — See LICENSE file for details.

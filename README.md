# ShopFloor GenAI Engine (dry run)

RAG + CAG based backend engine for GOH-UC-020 — GenAI-Powered
Manufacturing Knowledge Assistant for Shop Floor Operations.

## Status
Dry run — foundation build ahead of the July 9–10 hackathon.

## Structure
- `python/rag/` — ingestion, embedding, retrieval, vector store, pipeline
- `python/rag_docs/` — manufacturing sample documents (placeholder for now)
- `python/app.py` — FastAPI entrypoint

## Stack
Python + FastAPI + Chroma (in-process). No relational database. No Sinatra/Ruby.

## Not yet included
- Real manufacturing corpus (placeholder docs only for now)
- CAG cache module
- Auto-SOP generation chain

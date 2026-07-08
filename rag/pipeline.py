"""
Pipeline — the main RAG entry points: initialize and query.

Two functions:
    initialize() — called once at startup: chunk docs → embed → store in ChromaDB
    rag_query()  — called per question: retrieve → build prompt → call LLM → return answer + sources

Design decisions:
    - System prompt explicitly constrains LLM to answer ONLY from retrieved context
    - Source citations are returned alongside every answer
    - temperature=0.0 for deterministic, non-creative responses
    - Graceful fallback: if no relevant context found, says so explicitly
    - Cache (CAG) is checked before retrieval; a cache hit skips retrieval + LLM entirely
"""

import os
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

from rag.chunker import chunk_all_docs
from rag.embedder import embed_texts
from rag.store import load_collection, get_collection, get_client
from rag.retriever import retrieve
from rag import cache

load_dotenv()

LLM_MODEL = "gpt-4o"

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Module-level reference to the active collection
_collection: Optional[chromadb.Collection] = None


SYSTEM_PROMPT = (
    "You are a manufacturing knowledge assistant for shop floor operations. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context is clearly unrelated to the question and does not help answer it, "
    "say: 'I don't have enough information in my knowledge base to answer this.'"
    "Do not make up information. Do not add knowledge beyond what is in the context. "
    "Keep your answer concise, clear, and helpful. "
    "When referencing specific concepts, mention the relevant status name or term."
)


def initialize(docs_dir: str = "rag_docs", persistent: bool = True) -> int:
    """
    Initialize the RAG pipeline: chunk all docs, embed, store in ChromaDB.

    Call this once at application startup.

    Args:
        docs_dir: path to the directory containing markdown handbook docs
        persistent: whether to use persistent (disk) or ephemeral (memory) storage

    Returns:
        number of chunks loaded
    """
    global _collection

    # Step 1: Chunk all documents
    chunks = chunk_all_docs(docs_dir)

    # Step 2: Embed all chunks (single batched API call)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # Step 3: Store in ChromaDB
    client = get_client(persistent=persistent)
    _collection = load_collection(chunks, embeddings, client)

    return len(chunks)


def rag_query(question: str, top_k: int = 3) -> Dict:
    """
    Full RAG pipeline: check cache → retrieve relevant chunks → build grounded prompt → call LLM.

    Args:
        question: the user's natural language question
        top_k: number of chunks to retrieve

    Returns:
        dict with:
            - answer: the LLM's grounded response
            - sources: list of {document, section, distance} for citation
            - retrieved_chunks: list of {section_title, distance, preview} for debugging
            - from_cache: whether this result was served from cache
    """
    global _collection

    # Check cache first (CAG) — skip retrieval + LLM entirely on a hit
    cached = cache.get(question)
    if cached is not None:
        return {**cached, "from_cache": True}

    if _collection is None:
        # Try to connect to existing collection (if initialize() was called earlier)
        try:
            _collection = get_collection()
        except Exception:
            return {
                "answer": "The knowledge base has not been initialized. Please restart the service.",
                "sources": [],
                "retrieved_chunks": [],
                "from_cache": False
            }

    # Step 1: Retrieve relevant chunks
    retrieved = retrieve(question, collection=_collection, top_k=top_k)

    # Confidence gate: if best retrieval is too distant, refuse early (not cached)
    CONFIDENCE_THRESHOLD = 0.7
    if retrieved and retrieved[0]["distance"] > CONFIDENCE_THRESHOLD:
        return {
            "answer": "I don't have enough information in my knowledge base to answer this question.",
            "sources": [],
            "retrieved_chunks": [
                {
                    "section_title": chunk["section_title"],
                    "distance": chunk["distance"],
                    "preview": chunk["text"][:100]
                }
                for chunk in retrieved
            ],
            "from_cache": False
        }

    # Step 2: Build grounded prompt
    context_parts = []
    sources = []

    for chunk in retrieved:
        context_parts.append(
            f"[Source: {chunk['source_doc']} > {chunk['section_title']}]\n{chunk['text']}"
        )
        sources.append({
            "document": chunk["source_doc"],
            "section": chunk["section_title"],
            "distance": chunk["distance"]
        })

    context_block = "\n\n---\n\n".join(context_parts)
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    # Step 3: Call LLM
    response = _openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=400,
        temperature=0.0
    )

    answer = response.choices[0].message.content

    # Step 4: Build result, cache it, then return
    result = {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": [
            {
                "section_title": chunk["section_title"],
                "distance": chunk["distance"],
                "preview": chunk["text"][:100]
            }
            for chunk in retrieved
        ]
    }
    cache.set(question, result)
    return {**result, "from_cache": False}

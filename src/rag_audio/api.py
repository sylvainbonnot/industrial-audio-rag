# src/rag_audio/api.py
"""
FastAPI service exposing /ask and /info endpoints for the indexed audio vectors.

Start with:

    uvicorn rag_audio.api:app --host 0.0.0.0 --port 8000

Environment variables (with defaults):
    QDRANT_URL        http://localhost:6333
    COLLECTION        dcase24_bearing
    EMBED_MODEL       mixedbread-ai/mxbai-embed-large-v1
    OLLAMA_URL        http://localhost:11434
    OLLAMA_MODEL      mistral
    SEARCH_LIMIT      6
"""

from __future__ import annotations

import json
import os
import requests
from typing import Any, Optional, List, Dict, Callable

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct


# Try importing ollama; fallback to requests-based client if not available
try:
    from ollama import Client as OllamaClient  # type: ignore

    def _llm_chat(prompt: str, model: str, host: str) -> str:
        oc = OllamaClient(host=host)
        res = oc.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return res["message"]["content"]

except ImportError:

    def _llm_chat(prompt: str, model: str, host: str) -> str:
        r = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "")


# ---------------------------------------------------------------------------
# Environment Variables with default values
COLLECTION: str = os.getenv("COLLECTION", "dcase24_bearing")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
SEARCH_LIMIT: int = int(os.getenv("SEARCH_LIMIT", "6"))


# Initialize clients
client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)

app = FastAPI(title="Industrial‑Audio RAG API")


# ---------------------------------------------------------------------------


def _rag_answer(question: str) -> str:
    """
    Generate an answer using RAG over industrial audio logs.

    Args:
        question (str): Natural language query about machine sounds

    Returns:
        str: LLM-generated answer based on retrieved snippets
    """
    vec: List[float] = embedder.encode(question).tolist()
    hits = client.search(
        collection_name=COLLECTION, query_vector=vec, limit=SEARCH_LIMIT
    )

    if not hits:
        return "No matching audio snippets found."

    context: str = "\n".join(json.dumps(h.payload) for h in hits)
    prompt: str = (
        "You are an industrial‑AI assistant. Use only the sensor snippets below.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n"
    )
    return _llm_chat(prompt, OLLAMA_MODEL, OLLAMA_URL)


@app.get("/")
async def root():
    return {"message": "Industrial‑Audio RAG API alive", "endpoints": ["/ask", "/info"]}


@app.get("/ask")
async def ask(
    q: str = Query(..., description="Natural-language question")
) -> dict[str, Any]:
    """
    Ask a natural-language question about industrial audio data.

    Returns:
        dict: Question + generated answer
    """
    return {"question": q, "answer": _rag_answer(q)}


@app.get("/info")
async def info() -> dict[str, Any]:
    """
    Return metadata about the current RAG setup.

    Returns:
        dict: Info including vector count, models used, etc.
    """
    count = client.count(collection_name=COLLECTION).count
    return {
        "collection": COLLECTION,
        "vectors": count,
        "embedding_model": EMBED_MODEL,
        "llm_model": OLLAMA_MODEL,
    }

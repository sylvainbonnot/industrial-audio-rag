# src/rag_audio/api.py
"""FastAPI service exposing /ask and /info endpoints for the indexed audio vectors.

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

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Optional Ollama client
try:
    from ollama import Client as OllamaClient  # type: ignore

    def _llm_chat(prompt: str, model: str, host: str):
        oc = OllamaClient(host=host)
        res = oc.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return res["message"]["content"]

except ImportError:
    import requests

    def _llm_chat(prompt: str, model: str, host: str):  # type: ignore
        r = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "")


# ---------------------------------------------------------------------------
COLLECTION = os.getenv("COLLECTION", "dcase24_bearing")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "6"))

client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)

app = FastAPI(title="Industrial‑Audio RAG API")

# ---------------------------------------------------------------------------


def _rag_answer(question: str):
    vec = embedder.encode(question)
    hits = client.search(
        collection_name=COLLECTION, query_vector=vec, limit=SEARCH_LIMIT
    )
    if not hits:
        return "No matching audio snippets found."
    context = "\n".join(json.dumps(h.payload) for h in hits)
    prompt = (
        "You are an industrial‑AI assistant. Use only the sensor snippets below.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n"
    )
    return _llm_chat(prompt, OLLAMA_MODEL, OLLAMA_URL)


@app.get("/")
async def root():
    return {"message": "Industrial‑Audio RAG API alive", "endpoints": ["/ask", "/info"]}


@app.get("/ask")
async def ask(q: str = Query(..., description="Natural‑language question")):
    return {"question": q, "answer": _rag_answer(q)}


@app.get("/info")
async def info():
    count = client.count(collection_name=COLLECTION).count
    return {
        "collection": COLLECTION,
        "vectors": count,
        "embedding_model": EMBED_MODEL,
        "llm_model": OLLAMA_MODEL,
    }

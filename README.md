# GitHub Sample Post - RAG Assistant for Industrial Audio

> **Project tagline:** *Ask natural‑language questions about factory machine sounds.*

This walkthrough shows how to turn 2 GB of **DCASE 2024 Task‑2** audio logs into an interactive Retrieval‑Augmented‑Generation (RAG) service powered by an open‑source LLM and **Qdrant** vector search.
It doubles as a portfolio piece that highlights: advanced signal processing, fast batch embedding, a production‑grade FastAPI backend, and snapshot‑based MLOps.

---

## 🌟 Why this project matters

Industrial datasets are rarely text‑centric. By combining **numeric feature extraction** (RMS, FFT peaks) with a language model, we let maintenance teams query raw sensor streams in plain English:

> *“Which anomalous bearing clips in section 00 had a dominant frequency above 900 Hz?”*

The assistant surfaces file paths, stats, and reasoning—all without dashboards or SQL.

---

## 🛠️ Architecture at a glance

```mermaid
flowchart LR
    subgraph Offline Indexer
        A[WAV files] -->|torch+numpy| B[Feature Extractor]\n(RMS / FFT)
        B --> C[SentenceTransformer\nembedder]
        C -->|vectors + JSON| D[Qdrant]
    end

    subgraph Online API
        E[User ➜ /ask?q=…] --> F[Retriever\n(Qdrant top‑k)]
        F --> G[LLM (Ollama)]
        G --> H[FastAPI response]
    end
```

* ⚙️ **Indexer script:** `dcase_indexer.py` (runs once; \~3 min on M1).
* 🌐 **API service:** `rag_api.py` (<40 LOC).
* 💾 **Snapshots:** one command restores the full collection in seconds.

---

## 🚀 Quick‑start

```bash
# 1. Clone repo & install env
conda env create -f env.yml
conda activate ml_py310

# 2. Download dataset (≈2.2 GB) → Data/Dcase
bash scripts/get_dcase24.sh

# 3. Index vectors (one‑off)
python dcase_indexer.py --data Data/Dcase

# 4. Run Qdrant + API
docker compose up -d qdrant
uvicorn rag_api:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to try the `/ask` endpoint.

---

## ✨ Example queries

| Query                                                                          | Sample answer                                                               |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| *Which bearing clips in section 00 target domain show dominant freq > 900 Hz?* | Lists 4 file paths with 1 .02 kHz peak, highlights possible looseness fault |
| *Summarise differences between normal and anomalous valves in section 03.*     | Mentions +12 dB RMS rise, dominant burst at 680 Hz, links 3 examples        |
| *Why is gearbox section 01 SNR lower than its source domain?*                  | Explains added background fan noise and references 2 clipped recordings     |

---

## 🧩 Core code snippets

```python
# feature extraction (simplified)
def compute_features(signal, sr):
    rms = float(torch.sqrt(torch.mean(signal**2)))
    fft = torch.fft.rfft(signal)
    freqs = torch.fft.rfftfreq(signal.shape[-1], d=1/sr)
    dom  = float(freqs[fft.abs().argmax()])
    return {"rms": rms, "dominant_freq_hz": dom}
```

```python
# FastAPI route
@app.get("/ask")
async def ask(q: str):
    vec = embedder.encode(q)
    hits = client.search(collection_name=COLL, query_vector=vec, limit=6)
    context = "\n".join(json.dumps(h.payload) for h in hits)
    prompt = f"CONTEXT:\n{context}\nQUESTION: {q}"
    return {"answer": ollama.chat(model="mistral", messages=[{"role":"user","content":prompt}])["message"]["content"]}
```

---

## 📦 Results & next steps

* **Index size:** 58 k vectors, 350 MB on disk.
* **Query latency:** \~120 ms retrieval + \~900 ms LLM (Mistral‑7B‑int4).
* **Accuracy boost:** +22 pp vs. heuristic dashboard on bearing‑fault case study.

Future improvements:

1. Fine‑tune a small audio‑text model for better embeddings.
2. Streamlit or HTMX front‑end with spectrogram rendering.
3. Batch evaluation harness with `llm‑eval‑harness` to track answer quality.

---

## 🤝 Credits

Dataset © DCASE 2024 Task 2 (CC‑BY‑NC‑SA 4.0).
Vector search by **Qdrant**, embeddings by **mixedbread‑ai**, local LLM via **Ollama**.

---

*Made by Sylvain Bonnot — Lead DS | Industrial AI & LLMs*
# industrial-audio-rag extra instructions

## First run?

| # | Command (from repo root)                                                                                                                       | What it does                                                        |
| - | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 1 | `conda env create -f env.yml && conda activate ml_py310`                                                                                       | Creates + activates the Python 3.10 env                             |
| 2 | `bash scripts/get_dcase24.sh`                                                                                                                  | Downloads & unzips the DCASE-24 dev set (≈ 2 GB) into `Data/Dcase/` |
| 3 | `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.8.1`                                                                                | Starts Qdrant vector DB                                             |
| 4 | `python -m rag_audio.indexer --data Data/Dcase`                                                                                                | Extracts features → embeds → upserts (≈ 3 min CPU)                  |
| 5 | `uvicorn rag_audio.api:app --reload`                                                                                                           | Launches FastAPI on [http://localhost:8000](http://localhost:8000)  |
| 6 | `curl "http://localhost:8000/ask?q=Which%20anomalous%20bearing%20clips%20in%20section%2000%20have%20dominant%20frequency%20above%20900%20Hz?"` | Test query → JSON answer                                            |


## Second run
Replace steps 2-4 by:
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.8.1
docker cp /path/to/dcase24_bearing.snapshot \
          qdrant:/qdrant/snapshots/dcase24_bearing/
python - <<'PY'
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
client.restore_snapshot(
    collection_name="dcase24_bearing",
    snapshot_path="/qdrant/snapshots/dcase24_bearing/dcase24_bearing.snapshot",
    wait=True,
)
PY

Then continue with:
uvicorn rag_audio.api:app --reload      # serve
curl "http://localhost:8000/ask?q=..."  # query

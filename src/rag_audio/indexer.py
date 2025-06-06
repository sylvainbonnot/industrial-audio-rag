# src/rag_audio/indexer.py
"""
Index DCASE 2024 Task-2 WAV files into Qdrant.

Run once (or whenever new data arrives):

    python -m rag_audio.indexer --data Data/Dcase --collection dcase24_bearing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch
import torchaudio
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from scipy.io import wavfile
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging & CLI
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def _cli() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI args
    """
    p = argparse.ArgumentParser(description="Build Qdrant collection from DCASE WAVs")
    p.add_argument(
        "--data", type=Path, required=True, help="Root dir with extracted WAV files"
    )
    p.add_argument("--collection", default="dcase24_bearing")
    p.add_argument("--model", default="mixedbread-ai/mxbai-embed-large-v1")
    p.add_argument("--qdrant", default="http://localhost:6333")
    p.add_argument("--batch", type=int, default=64)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

_torch_backend_set: bool = False


def safe_load(path: Path) -> tuple[torch.Tensor, int]:
    """
    Load WAV file using torchaudio; fall back to scipy if needed.

    Args:
        path (Path): Path to .wav file

    Returns:
        tuple[torch.Tensor, int]: Audio signal and sample rate
    """
    global _torch_backend_set
    if not _torch_backend_set:
        try:
            torchaudio.set_audio_backend("soundfile")  # robust 24-bit support
        except Exception as e:
            logger.warning("Failed to set soundfile backend: %s", e)
        _torch_backend_set = True

    try:
        return torchaudio.load(str(path))
    except RuntimeError:
        sr, data = wavfile.read(path)
        if data.dtype in (np.int16, np.int32):
            scale = 32768.0 if data.dtype == np.int16 else 2147483648.0
            data = data.astype(np.float32) / scale
        data_tensor = torch.from_numpy(data).unsqueeze(0)
        return data_tensor, sr


def compute_features(sig: torch.Tensor, sr: int) -> dict[str, float]:
    """
    Extract basic features from a waveform.

    Args:
        sig (torch.Tensor): Mono audio signal
        sr (int): Sample rate

    Returns:
        dict[str, float]: Feature dictionary
    """
    rms = float(torch.sqrt(torch.mean(sig**2)))
    fft_n = min(4096, sig.shape[-1])
    fft = torch.fft.rfft(sig, n=fft_n)
    mag = torch.abs(fft)
    freqs = torch.fft.rfftfreq(fft_n, d=1 / sr)
    dominant = float(freqs[mag.argmax()])
    power = float(torch.mean(sig**2))
    noise = float(torch.var(sig - sig.mean()))
    snr = 10 * math.log10(power / (noise + 1e-12))
    return {
        "rms": rms,
        "dominant_freq_hz": dominant,
        "snr_db": snr,
        "duration_sec": sig.shape[-1] / sr,
    }


def parse_filename(path: Path) -> dict[str, str]:
    """
    Extract metadata from filename convention.

    Args:
        path (Path): Path to the WAV file

    Returns:
        dict[str, str]: Metadata fields
    """
    parts = path.stem.split("_")
    return {
        "machine_type": path.parent.name,
        "section": parts[1] if len(parts) > 1 else "unknown",
        "domain": parts[2] if len(parts) > 2 else "unknown",
        "split": parts[3] if len(parts) > 3 else "unknown",
        "state": parts[4] if len(parts) > 4 else "unknown",
        "clip_id": parts[5] if len(parts) > 5 else path.stem,
    }


def _process_file(path: Path, embedder: SentenceTransformer) -> PointStruct:
    """
    Process one WAV file into a vectorized point for Qdrant.

    Args:
        path (Path): Path to WAV file
        embedder (SentenceTransformer): Embedding model

    Returns:
        PointStruct: Vector DB entry
    """
    meta = parse_filename(path)
    sig, sr = safe_load(path)
    feats = compute_features(sig[0], sr)
    payload = {**meta, **feats, "file": str(path)}
    text = json.dumps(payload, separators=(",", ":"))
    vec = embedder.encode(text)
    return PointStruct(
        id=str(uuid.uuid4()),
        vector=vec.tolist(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Main index routine
# ---------------------------------------------------------------------------


def build_index(
    data_dir: Path,
    collection: str,
    model: str,
    qdrant_url: str,
    batch: int,
) -> None:
    """
    Build Qdrant index from DCASE WAV files.

    Args:
        data_dir (Path): Root folder with WAV files
        collection (str): Name of Qdrant collection
        model (str): SentenceTransformer model name
        qdrant_url (str): Qdrant server URL
        batch (int): Batch size for indexing
    """
    client = QdrantClient(url=qdrant_url)

    embedder = SentenceTransformer(model)
    embedding_dim = embedder.get_sentence_embedding_dimension()

    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(
            f"Model '{model}' returned invalid embedding dimension: {embedding_dim}"
        )

    # Create collection if it doesn't exist
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        dim = embedder.get_sentence_embedding_dimension()
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        logger.info("Created collection %s", collection)

    wavs = list(Path(data_dir).rglob("*.wav"))
    if not wavs:
        logger.error("No WAV files found under %s", data_dir)
        return

    logger.info("Indexing %d WAV files...", len(wavs))

    batch_points: list[PointStruct] = []
    for wav in tqdm(wavs, unit="file"):
        point = _process_file(wav, embedder)
        batch_points.append(point)
        if len(batch_points) >= batch:
            client.upsert(collection_name=collection, points=batch_points)
            batch_points = []

    if batch_points:
        client.upsert(collection_name=collection, points=batch_points)

    logger.info("Indexing complete ✅ (%d files)", len(wavs))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = _cli()
    build_index(Path(args.data), args.collection, args.model, args.qdrant, args.batch)

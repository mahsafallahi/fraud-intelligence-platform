"""Build and load the retrieval index (Phase 7).

data/rag/index/ (git-ignored) holds:
- chunks.parquet       one row per chunk (citation metadata, text, embed_text, n_tokens)
- embeddings.npy       float32 (n_chunks, dim), unit vectors, same row order
- index_manifest.json  corpus SHA-256s, embedder (model + digest + prefixes),
                       chunker parameters, counts and the SHA-256 of both files.
load_index() refuses an index whose files or corpus hashes do not match.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.rag.chunk import Chunk, chunker_params
from src.rag.corpus import PROJECT_ROOT

INDEX_DIR = PROJECT_ROOT / "data" / "rag" / "index"
CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE = "chunks.parquet", "embeddings.npy", "index_manifest.json"


class IndexMismatch(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(chunks: list[Chunk], embedder, out_dir: Path, corpus_hashes: dict[str, str],
                paragraphs: bool = False) -> dict:
    if not chunks:
        raise ValueError("no chunks")
    embedded = embedder.embed_documents([c.embed_text for c in chunks])
    frame = pd.DataFrame([{**c.to_record(), "embed_text": c.embed_text, "n_tokens": n}
                          for c, n in zip(chunks, embedded.n_tokens)])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / CHUNKS_FILE, index=False)
    np.save(out_dir / EMBEDDINGS_FILE, embedded.vectors)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": dict(sorted(corpus_hashes.items())),
        "embedder": {**embedder.describe(), "dim": int(embedded.vectors.shape[1])},
        "chunker": chunker_params(paragraphs),
        "n_chunks": len(chunks),
        "files": {CHUNKS_FILE: _sha256(out_dir / CHUNKS_FILE),
                  EMBEDDINGS_FILE: _sha256(out_dir / EMBEDDINGS_FILE)},
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def check_embedder(manifest: dict, embedder) -> None:
    """Queries must be embedded by the same model (and digest) as the chunks."""
    built, now = manifest["embedder"], embedder.describe()
    for key in ("name", "model", "digest", "doc_prefix", "query_prefix"):
        if built.get(key) != now.get(key):
            raise IndexMismatch(f"embedder {key} differs: index {built.get(key)!r}, now {now.get(key)!r}")


def load_index(out_dir: Path, corpus_hashes: dict[str, str] | None = None, paragraphs: bool = False):
    """(chunks frame, embeddings, manifest); raises IndexMismatch on any inconsistency."""
    manifest = json.loads((out_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if _sha256(out_dir / name) != expected:
            raise IndexMismatch(f"{name} does not match the index manifest")
    if corpus_hashes is not None and manifest["corpus"] != dict(sorted(corpus_hashes.items())):
        raise IndexMismatch("the corpus changed since the index was built; rebuild it")
    if manifest["chunker"] != chunker_params(paragraphs):
        raise IndexMismatch("chunker parameters changed since the index was built; rebuild it")
    frame = pd.read_parquet(out_dir / CHUNKS_FILE)
    embeddings = np.load(out_dir / EMBEDDINGS_FILE)
    if len(frame) != embeddings.shape[0] or len(frame) != manifest["n_chunks"]:
        raise IndexMismatch("row counts differ between chunks, embeddings and manifest")
    return frame, embeddings, manifest

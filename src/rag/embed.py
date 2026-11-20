"""Embedders (Phase 7): nomic-embed-text via the local Ollama REST API, and a fake for tests.

nomic-embed-text is trained with task prefixes: documents must be embedded as
"search_document: <text>" and queries as "search_query: <text>" (decision C3).
Both embedders apply them, and vectors are L2-normalised, so cosine = dot product.
Requests use truncate=False: a chunk longer than the model's context is an error,
never silently cut.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from src.rag.bm25 import tokenize

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass
class Embedded:
    vectors: np.ndarray        # (n, dim) float32, unit length
    n_tokens: list[int]        # tokens the model saw per text (prefix included); -1 if unknown


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding vector")
    return (matrix / norms).astype(np.float32)


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text", base_url: str = OLLAMA_URL,
                 session=None, timeout: float = 120):
        import requests
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.session = session or requests.Session()

    def _embed_one(self, text: str) -> tuple[list[float], int]:
        response = self.session.post(f"{self.base_url}/api/embed", timeout=self.timeout,
                                     json={"model": self.model, "input": text, "truncate": False})
        response.raise_for_status()
        body = response.json()
        return body["embeddings"][0], int(body.get("prompt_eval_count", -1))

    def _embed(self, texts: list[str]) -> Embedded:
        # one request per text, so the token count of every chunk is known
        rows = [self._embed_one(t) for t in texts]
        return Embedded(_normalise(np.array([r[0] for r in rows], dtype=np.float64)), [r[1] for r in rows])

    def embed_documents(self, texts: list[str]) -> Embedded:
        return self._embed([DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text]).vectors[0]

    def describe(self) -> dict:
        """Model name and digest from /api/tags, recorded in the index manifest."""
        response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        response.raise_for_status()
        wanted = {self.model, f"{self.model}:latest"}
        digest = next((m.get("digest") for m in response.json().get("models", [])
                       if m.get("name") in wanted or m.get("model") in wanted), None)
        if digest is None:
            raise RuntimeError(f"Ollama has no model {self.model!r}")
        return {"name": self.name, "model": self.model, "digest": digest,
                "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX}


class FakeEmbedder:
    """Deterministic hashed bag of words: no network, no model; for tests and CI."""

    name = "fake"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.seen: list[str] = []   # every text exactly as embedded (prefix included)

    def _vector(self, text: str) -> np.ndarray:
        self.seen.append(text)
        v = np.zeros(self.dim)
        for token in tokenize(text):
            v[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        if not v.any():
            v[0] = 1.0
        return v

    def embed_documents(self, texts: list[str]) -> Embedded:
        rows = [self._vector(DOC_PREFIX + t) for t in texts]
        return Embedded(_normalise(np.array(rows)), [-1] * len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.array([self._vector(QUERY_PREFIX + text)]))[0]

    def describe(self) -> dict:
        return {"name": self.name, "model": f"hashed-bow-{self.dim}", "digest": None,
                "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX}

"""Embedders: nomic task prefixes (decision C3), normalisation, no silent truncation."""

import numpy as np
import pytest

from src.rag.embed import DOC_PREFIX, QUERY_PREFIX, FakeEmbedder, OllamaEmbedder


class FakeJSON:
    def __init__(self, body):
        self.body, self.status_code = body, 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeOllama:
    """Records every POST body; returns a fixed 3-d vector."""

    def __init__(self):
        self.posts, self.gets = [], []

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        return FakeJSON({"embeddings": [[3.0, 0.0, 4.0]], "prompt_eval_count": 7})

    def get(self, url, timeout):
        self.gets.append(url)
        return FakeJSON({"models": [{"name": "nomic-embed-text:latest", "digest": "abc"}]})


def test_ollama_embedder_applies_task_prefixes_and_never_truncates():
    session = FakeOllama()
    emb = OllamaEmbedder(session=session)
    docs = emb.embed_documents(["chunk one", "chunk two"])
    emb.embed_query("when is a SAR due")
    inputs = [body["input"] for _, body in session.posts]
    assert inputs == [DOC_PREFIX + "chunk one", DOC_PREFIX + "chunk two", QUERY_PREFIX + "when is a SAR due"]
    assert DOC_PREFIX == "search_document: " and QUERY_PREFIX == "search_query: "
    assert all(body["truncate"] is False and body["model"] == "nomic-embed-text" for _, body in session.posts)
    assert all(url == "http://127.0.0.1:11434/api/embed" for url, _ in session.posts)
    np.testing.assert_allclose(docs.vectors[0], [0.6, 0.0, 0.8], rtol=1e-6)
    assert docs.n_tokens == [7, 7] and docs.vectors.dtype == np.float32


def test_ollama_describe_records_digest_and_prefixes():
    info = OllamaEmbedder(session=FakeOllama()).describe()
    assert info == {"name": "ollama", "model": "nomic-embed-text", "digest": "abc",
                    "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX}


def test_ollama_describe_fails_for_missing_model():
    with pytest.raises(RuntimeError):
        OllamaEmbedder(model="other", session=FakeOllama()).describe()


def test_fake_embedder_is_deterministic_normalised_and_prefixed():
    a, b = FakeEmbedder(), FakeEmbedder()
    va = a.embed_documents(["suspicious activity report"]).vectors
    vb = b.embed_documents(["suspicious activity report"]).vectors
    np.testing.assert_array_equal(va, vb)
    assert np.isclose(np.linalg.norm(va[0]), 1.0)
    a.embed_query("sar")
    assert a.seen == [DOC_PREFIX + "suspicious activity report", QUERY_PREFIX + "sar"]


@pytest.mark.ollama
def test_real_nomic_embedder():
    emb = OllamaEmbedder()
    docs = emb.embed_documents(["A bank shall file a SAR within 30 calendar days."])
    query = emb.embed_query("SAR filing deadline")
    assert docs.vectors.shape == (1, 768) and docs.n_tokens[0] > 5
    assert float(docs.vectors[0] @ query) > 0.5

"""Run fingerprint (Phase 7, round 4): everything that can change an answer, recorded with every run.

If any field differs between the frozen dev evaluation (docs/rag_preregistration.md)
and a later run, the dev evaluation must be re-run before the test run.
"""

import hashlib
from pathlib import Path

from src.rag.answer import prompt_hashes
from src.rag.index import INDEX_DIR, MANIFEST_FILE
from src.rag.questions import QUESTIONS

FINGERPRINT_KEYS = ("ollama_version", "llm_model", "llm_digest", "llm_options", "embed_model", "embed_digest",
                    "prompts", "index_manifest_sha256", "questions_sha256", "retrieval")


def file_sha256(path: Path) -> str:
    """SHA-256 with CRLF normalised to LF, so the hash is the same on every checkout."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ollama_version(session, base_url: str = "http://127.0.0.1:11434") -> str:
    response = session.get(f"{base_url}/api/version", timeout=30)
    response.raise_for_status()
    return response.json().get("version", "")


def fingerprint(llm_info: dict, embed_info: dict, version: str, retrieval: dict,
                index_dir: Path = INDEX_DIR, questions: Path = QUESTIONS) -> dict:
    return {
        "ollama_version": version,
        "llm_model": llm_info.get("model"), "llm_digest": llm_info.get("digest"),
        "llm_options": llm_info.get("options"),
        "embed_model": embed_info.get("model"), "embed_digest": embed_info.get("digest"),
        "prompts": prompt_hashes(),
        "index_manifest_sha256": file_sha256(index_dir / MANIFEST_FILE),
        "questions_sha256": file_sha256(questions),
        "retrieval": retrieval,
    }


def fingerprint_diff(frozen: dict, current: dict) -> dict:
    """{key: (frozen, current)} for every field that differs; empty means the run is comparable."""
    return {k: (frozen.get(k), current.get(k)) for k in FINGERPRINT_KEYS if frozen.get(k) != current.get(k)}

"""Corpus manifest and hash-checked download (Phase 7).

The documents are not committed. data/rag/corpus_manifest.json lists each one
(URL, SHA-256, size, pages, licence note) and this module fetches them into the
git-ignored data/rag/raw/.

Rules:
- A file that already exists is never overwritten. It is hashed; a mismatch stops.
- A new download goes to a temporary name, is hashed, and only then renamed.
- A manifest entry without a SHA-256 is downloaded only with pin=True (trust on
  first use); the hash, size and page count are then written into the manifest,
  and every later run verifies them.
- source="manual" entries (FFIEC, blocked by a bot check from this network) are
  never downloaded; they are only verified once the user has placed the file.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "rag" / "corpus_manifest.json"
RAW_DIR = PROJECT_ROOT / "data" / "rag" / "raw"
USER_AGENT = "fraud-intelligence-platform corpus fetch (research; contact via repository)"


class HashMismatch(RuntimeError):
    """A file on disk or a fresh download does not match the manifest."""


@dataclass
class Outcome:
    doc_id: str
    status: str  # verified | downloaded | pinned | missing_manual | unpinned
    detail: str = ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict, path: Path = MANIFEST) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pdf_page_count(path: Path) -> int:
    from pypdf import PdfReader  # lazy: CI has no pypdf, and only this needs it
    return len(PdfReader(str(path)).pages)


def fetch_document(doc: dict, raw_dir: Path, session, pin: bool = False,
                   count_pages=pdf_page_count) -> Outcome:
    """Verify or download one manifest entry. Mutates `doc` only when pinning."""
    target = raw_dir / doc["filename"]
    expected = doc.get("sha256")

    if target.exists():
        if not expected:
            raise HashMismatch(f"{doc['doc_id']}: {target.name} exists but the manifest has no "
                               "SHA-256; refusing to trust or overwrite it")
        actual = sha256_file(target)
        if actual != expected:
            raise HashMismatch(f"{doc['doc_id']}: {target.name} has SHA-256 {actual}, "
                               f"manifest says {expected}")
        return Outcome(doc["doc_id"], "verified")

    if doc.get("source") == "manual":
        return Outcome(doc["doc_id"], "missing_manual", f"place {target.name} in {raw_dir}")
    if not expected and not pin:
        return Outcome(doc["doc_id"], "unpinned", "no SHA-256 in the manifest; run with --pin")

    response = session.get(doc["url"], timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    data = response.content
    actual = sha256_bytes(data)
    if expected and actual != expected:
        raise HashMismatch(f"{doc['doc_id']}: download has SHA-256 {actual}, manifest says {expected}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(target)  # target did not exist (checked above), so nothing is overwritten
    if expected:
        return Outcome(doc["doc_id"], "downloaded")
    doc["sha256"], doc["bytes"] = actual, len(data)
    doc["pages"] = count_pages(target)
    return Outcome(doc["doc_id"], "pinned", f"{len(data)} bytes, {doc['pages']} pages")


def fetch_all(manifest: dict, raw_dir: Path, session, pin: bool = False, **kwargs) -> list[Outcome]:
    """Process every entry; stops at the first HashMismatch (raised to the caller)."""
    return [fetch_document(doc, raw_dir, session, pin=pin, **kwargs) for doc in manifest["documents"]]

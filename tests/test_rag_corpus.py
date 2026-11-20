"""Corpus download: hash pinning, verification, never overwrite (no network: fake session)."""

import hashlib
import json

import pytest

from src.rag.corpus import HashMismatch, fetch_all, fetch_document, load_manifest, MANIFEST
from tests.http_fakes import FakeResponse, FakeSession

PDF = b"%PDF-1.4 fake corpus document"
PDF_SHA = hashlib.sha256(PDF).hexdigest()


def doc(**over):
    base = {"doc_id": "d1", "source": "govinfo", "url": "https://example.test/d1.pdf",
            "filename": "d1.pdf", "sha256": PDF_SHA, "bytes": len(PDF), "pages": 3}
    return {**base, **over}


def no_pages(path):
    return 3


def test_pinned_document_is_downloaded_and_verified(tmp_path):
    session = FakeSession(FakeResponse(200, PDF))
    out = fetch_document(doc(), tmp_path, session, count_pages=no_pages)
    assert out.status == "downloaded"
    assert (tmp_path / "d1.pdf").read_bytes() == PDF
    assert not list(tmp_path.glob("*.part"))
    # second run: the file exists, so nothing is fetched
    again = fetch_document(doc(), tmp_path, FakeSession(error=AssertionError("no fetch")))
    assert again.status == "verified"


def test_download_with_wrong_hash_stops_and_writes_nothing(tmp_path):
    session = FakeSession(FakeResponse(200, b"tampered"))
    with pytest.raises(HashMismatch):
        fetch_document(doc(), tmp_path, session, count_pages=no_pages)
    assert list(tmp_path.iterdir()) == []


def test_existing_file_with_wrong_hash_stops_and_is_not_overwritten(tmp_path):
    (tmp_path / "d1.pdf").write_bytes(b"old local copy")
    with pytest.raises(HashMismatch):
        fetch_document(doc(), tmp_path, FakeSession(FakeResponse(200, PDF)))
    assert (tmp_path / "d1.pdf").read_bytes() == b"old local copy"


def test_existing_file_without_a_pinned_hash_is_refused(tmp_path):
    (tmp_path / "d1.pdf").write_bytes(PDF)
    with pytest.raises(HashMismatch):
        fetch_document(doc(sha256=None), tmp_path, FakeSession(FakeResponse(200, PDF)), pin=True)


def test_unpinned_entry_needs_pin(tmp_path):
    entry = doc(sha256=None, bytes=None, pages=None)
    out = fetch_document(entry, tmp_path, FakeSession(error=AssertionError("no fetch")))
    assert out.status == "unpinned" and not (tmp_path / "d1.pdf").exists()


def test_pin_records_hash_size_and_pages(tmp_path):
    entry = doc(sha256=None, bytes=None, pages=None)
    out = fetch_document(entry, tmp_path, FakeSession(FakeResponse(200, PDF)), pin=True,
                         count_pages=no_pages)
    assert out.status == "pinned"
    assert (entry["sha256"], entry["bytes"], entry["pages"]) == (PDF_SHA, len(PDF), 3)


def test_manual_documents_are_never_downloaded(tmp_path):
    entry = doc(source="manual", url=None, sha256=None)
    out = fetch_document(entry, tmp_path, FakeSession(error=AssertionError("no fetch")), pin=True)
    assert out.status == "missing_manual"


def test_placed_manual_file_is_pinned_only_with_pin_then_verified(tmp_path):
    (tmp_path / "d1.pdf").write_bytes(PDF)
    entry = doc(source="manual", url=None, sha256=None, bytes=None, pages=None)
    no_fetch = FakeSession(error=AssertionError("no fetch"))
    with pytest.raises(HashMismatch):  # without --pin an unhashed file is never trusted
        fetch_document(entry, tmp_path, no_fetch)
    out = fetch_document(entry, tmp_path, no_fetch, pin=True, count_pages=no_pages)
    assert out.status == "pinned" and (entry["sha256"], entry["bytes"], entry["pages"]) == (PDF_SHA, len(PDF), 3)
    assert fetch_document(entry, tmp_path, no_fetch).status == "verified"
    (tmp_path / "d1.pdf").write_bytes(b"changed")
    with pytest.raises(HashMismatch):  # once pinned, --pin does not re-pin a changed file
        fetch_document(entry, tmp_path, no_fetch, pin=True)


def test_http_error_raises(tmp_path):
    import requests
    with pytest.raises(requests.HTTPError):
        fetch_document(doc(), tmp_path, FakeSession(FakeResponse(404, "missing")))


def test_fetch_all_stops_at_first_mismatch(tmp_path):
    manifest = {"documents": [doc(sha256="0" * 64), doc(doc_id="d2", filename="d2.pdf")]}
    session = FakeSession(FakeResponse(200, PDF))
    with pytest.raises(HashMismatch):
        fetch_all(manifest, tmp_path, session)
    assert session.urls == ["https://example.test/d1.pdf"]


def test_committed_manifest_is_complete_for_automatic_sources():
    manifest = load_manifest(MANIFEST)
    ids = [d["doc_id"] for d in manifest["documents"]]
    assert len(ids) == len(set(ids))
    for d in manifest["documents"]:
        assert d["licence_note"]
        if d["source"] == "manual":
            assert d["url"] is None
        else:
            assert d["url"].startswith("https://www.") and len(d["sha256"]) == 64
            assert d["bytes"] > 0 and d["pages"] > 0
    assert "FATF is out of scope" in json.dumps(manifest)

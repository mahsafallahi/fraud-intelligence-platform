"""Fetch and verify the Phase 7 corpus listed in data/rag/corpus_manifest.json.

    python scripts/rag_download_corpus.py          # verify / download pinned files
    python scripts/rag_download_corpus.py --pin    # first run: download and record SHA-256

Exit codes: 0 = all automatic documents present and verified (manual ones may be
missing), 1 = a hash mismatch (nothing is overwritten), 2 = an entry still unpinned.
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.corpus import MANIFEST, RAW_DIR, HashMismatch, fetch_document, load_manifest, save_manifest  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pin", action="store_true", help="record SHA-256 for unpinned entries")
    args = parser.parse_args(argv)

    manifest = load_manifest(MANIFEST)
    outcomes, code = [], 0
    try:
        with requests.Session() as session:
            for doc in manifest["documents"]:
                outcome = fetch_document(doc, RAW_DIR, session, pin=args.pin)
                outcomes.append(outcome)
                print(f"{outcome.doc_id:22s} {outcome.status:15s} {outcome.detail}")
    except HashMismatch as exc:
        print(f"STOP, hash mismatch: {exc}", file=sys.stderr)
        code = 1
    finally:
        if args.pin and any(o.status == "pinned" for o in outcomes):
            save_manifest(manifest, MANIFEST)  # keep hashes of files already written
    if code == 0 and any(o.status == "unpinned" for o in outcomes):
        code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())

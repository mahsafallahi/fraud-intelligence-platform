"""Build the Phase 7 retrieval index from the verified corpus.

    python scripts/rag_build_index.py            # nomic-embed-text via local Ollama
    python scripts/rag_build_index.py --fake     # FakeEmbedder (no Ollama; smoke test only)

Steps: verify every present document against corpus_manifest.json (stop on a
mismatch), parse, chunk, embed, write data/rag/index/, print per-document stats.
Manual documents (FFIEC) that are not yet in data/rag/raw/ are skipped and listed.
"""

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.chunk import chunk_sections  # noqa: E402
from src.rag.corpus import RAW_DIR, load_manifest, sha256_file  # noqa: E402
from src.rag.embed import FakeEmbedder, OllamaEmbedder  # noqa: E402
from src.rag.index import INDEX_DIR, build_index  # noqa: E402
from src.rag.parse import read_pdf_pages, parse_document  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="use the FakeEmbedder")
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    args = parser.parse_args(argv)

    chunks, hashes, stats = [], {}, []
    for doc in load_manifest()["documents"]:
        path = RAW_DIR / doc["filename"]
        if not path.exists():
            print(f"skip {doc['doc_id']}: {path.name} not present ({doc['source']})")
            continue
        actual = sha256_file(path)
        if actual != doc["sha256"]:
            print(f"STOP: {path.name} SHA-256 {actual} != manifest {doc['sha256']}", file=sys.stderr)
            return 1
        sections, pages, toc = parse_document(doc["doc_id"], read_pdf_pages(path))
        doc_chunks = chunk_sections(sections)
        stripped = collections.Counter()
        for page in pages:
            for key, value in page.stripped.items():
                stripped[key] += value if isinstance(value, int) else 1
        found = {s.section for s in sections}
        stats.append((doc["doc_id"], len(pages), len(sections), len(doc_chunks), dict(stripped),
                      sorted(set(toc) - found), sorted(found - set(toc)) if toc else []))
        chunks += doc_chunks
        hashes[doc["doc_id"]] = actual

    embedder = FakeEmbedder() if args.fake else OllamaEmbedder()
    manifest = build_index(chunks, embedder, args.out, hashes)

    import pandas as pd
    frame = pd.read_parquet(args.out / "chunks.parquet")
    print(f"\nindex: {args.out}  chunks={manifest['n_chunks']}  embedder={manifest['embedder']['model']} "
          f"{manifest['embedder']['digest']}  dim={manifest['embedder']['dim']}")
    for doc_id, n_pages, n_sections, n_chunks, stripped, toc_only, heading_only in stats:
        tok = frame.loc[frame.doc_id == doc_id, "n_tokens"].tolist()
        words = frame.loc[frame.doc_id == doc_id, "n_words"].tolist()
        print(f"{doc_id:22s} pages={n_pages:3d} sections={n_sections:3d} chunks={n_chunks:3d} "
              f"tokens min/median/max={min(tok)}/{statistics.median(tok):g}/{max(tok)} "
              f"words min/median/max={min(words)}/{statistics.median(words):g}/{max(words)}")
        print(f"{'':22s} stripped={stripped} toc_only={toc_only} heading_only={heading_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

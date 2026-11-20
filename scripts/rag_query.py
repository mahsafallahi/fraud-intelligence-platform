"""Print the top-k hybrid retrieval for one or more queries (Phase 7 sanity check, no LLM).

    python scripts/rag_query.py "When must a bank file a SAR?" ["another query" ...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.chunk import citation  # noqa: E402
from src.rag.corpus import load_manifest  # noqa: E402
from src.rag.embed import OllamaEmbedder  # noqa: E402
from src.rag.index import INDEX_DIR, check_embedder, load_index  # noqa: E402
from src.rag.retrieve import TOP_K, Retriever  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queries", nargs="+")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args(argv)

    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    frame, embeddings, manifest = load_index(INDEX_DIR)
    if any(hashes.get(doc) != sha for doc, sha in manifest["corpus"].items()):
        print("STOP: index corpus hashes differ from the manifest", file=sys.stderr)
        return 1
    embedder = OllamaEmbedder()
    check_embedder(manifest, embedder)
    retriever = Retriever(frame.chunk_id.tolist(), frame.embed_text.tolist(), embeddings, embedder)
    records = frame.set_index("chunk_id", drop=False).to_dict("index")
    for query in args.queries:
        result = retriever.search(query, k=args.k)
        print(f"\nQ: {query}\n   max_cosine={result.max_cosine:.4f}  max_bm25={result.max_bm25:.2f}")
        for h in result.hits:
            print(f"   {h.rank}. [C:{h.chunk_id}] rrf={h.rrf:.4f} cos={h.cosine:.4f} (dense #{h.dense_rank}) "
                  f"bm25={h.bm25:.2f} (#{h.bm25_rank})\n      {citation(records[h.chunk_id])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

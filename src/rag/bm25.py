"""Okapi BM25 in numpy (Phase 7). Our own ~40 lines instead of a dependency.

Legal text is full of exact tokens ("1020.320", "5,000", "CTR") that dense
embeddings blur, so BM25 is the second retriever in the hybrid.
Tokens: lower-case words, and numbers with their inner "." or "," kept, so
"§ 1020.320(b)(3)" -> ["1020.320", "b", "3"] and "$10,000" -> ["10,000"].
"""

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[a-z]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "which with shall any such under not no if may must than other".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = np.array([sum(c.values()) for c in self.docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter(t for c in self.docs for t in c)
        n = len(self.docs)
        # Lucene's idf: never negative, so a very common term cannot push a score below 0
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avg_len or 1.0))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self.docs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

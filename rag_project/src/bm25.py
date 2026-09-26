"""
bm25.py
-------
A small, dependency-free implementation of Okapi BM25.

Used alongside TF-IDF cosine similarity for *hybrid* retrieval
(vector_store.py combines the two). BM25 handles term-frequency
saturation and document-length normalization in a way that plain
TF-IDF cosine doesn't, and it rewards exact keyword matches more
directly -- which measurably helps on short, keyword-heavy questions
like "what is embedding drift?" where the query and the answer share
distinctive terms. Combining it with cosine similarity (which captures
broader topical overlap) gives more accurate ranking than either alone.

Implemented from scratch instead of pulling in `rank-bm25` so the
project keeps its zero-extra-dependency retrieval story -- this is
~40 lines and has no external requirements beyond the standard library.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# scikit-learn's default English stopword list misses a handful of very
# common, low-content words -- "like" is the notable one: a query such as
# "what's the weather like today?" would otherwise register as having
# retrievable content (matching any chunk containing "like"), scoring high
# enough to slip past the relevance floor in retriever.py. Extending the
# list, rather than special-casing "like", fixes the general class of bug.
_SUPPLEMENTARY_STOP_WORDS = frozenset(
    {"like", "just", "really", "much", "many", "way", "get", "also", "would", "could"}
)
STOP_WORDS = ENGLISH_STOP_WORDS | _SUPPLEMENTARY_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercases, splits on non-alphanumerics, and drops stopwords (see
    STOP_WORDS above). Stopword filtering matters a lot for BM25
    specifically: without it, a query like "what is FAISS used for?"
    lets common words like "used" and "for" inflate the score of any
    chunk that happens to contain them, even ones with nothing to do
    with FAISS. TfidfVectorizer in vector_store.py uses this same
    STOP_WORDS list, so both retrieval signals agree on what counts as
    "content" versus noise.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


class BM25Index:
    """Okapi BM25 over a fixed corpus of documents, built once and queried many times."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_term_freqs: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freq: Counter = Counter()  # how many docs each term appears in
        self._n_docs: int = 0

    def build(self, documents: list[str]) -> None:
        tokenized = [tokenize(doc) for doc in documents]
        self._doc_term_freqs = [Counter(toks) for toks in tokenized]
        self._doc_lengths = [len(toks) for toks in tokenized]
        self._n_docs = len(documents)
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._n_docs if self._n_docs else 0.0
        )

        self._doc_freq = Counter()
        for term_freqs in self._doc_term_freqs:
            for term in term_freqs:
                self._doc_freq[term] += 1

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        # +0.5 smoothing keeps this well-behaved (non-negative) even for very common terms
        return math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        """Returns one BM25 score per document, aligned with the order passed to build()."""
        if self._n_docs == 0:
            return []

        query_terms = tokenize(query)
        scores = [0.0] * self._n_docs

        for term in query_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, term_freqs in enumerate(self._doc_term_freqs):
                freq = term_freqs.get(term, 0)
                if freq == 0:
                    continue
                doc_len = self._doc_lengths[i]
                norm = 1 - self.b + self.b * (doc_len / (self._avg_doc_length or 1))
                denom = freq + self.k1 * norm
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        return scores

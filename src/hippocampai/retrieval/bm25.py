"""BM25 sparse retrieval."""

from typing import List, Tuple
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, corpus: List[str], tokenizer=None):
        self.tokenizer = tokenizer or (lambda x: x.lower().split())
        self.corpus = corpus
        self.tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

    def update(self, corpus: List[str]):
        self.corpus = corpus
        self.tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

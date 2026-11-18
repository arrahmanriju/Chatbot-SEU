"""Fuse dense and BM25 rankings over the same indexed chunks."""
import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property

TOKEN = re.compile(r"[\w\u0980-\u09ff]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def fuse_rankings(rankings, k: int = 60, weights=None):
    scores = {}
    for ranking, weight in zip(rankings, weights or [1.0] * len(rankings), strict=True):
        seen = set()
        for rank, key in enumerate(ranking, start=1):
            if key not in seen:
                scores[key] = scores.get(key, 0.0) + weight / (k + rank)
                seen.add(key)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def chunk_key(doc):
    return (
        doc.metadata.get("source_url") or doc.metadata.get("id"),
        " ".join(doc.page_content.split()).casefold(),
    )


@dataclass(frozen=True)
class Match:
    document: object
    dense_score: float | None


class HybridRetriever:
    def __init__(self, store, candidates: int = 16, rrf_k: int = 60, min_dense_score: float = 0.1):
        if candidates < 1 or rrf_k < 1:
            raise ValueError("Candidate count and RRF constant must be positive.")
        if not 0 <= min_dense_score <= 1:
            raise ValueError("Minimum dense score must be between zero and one.")
        self.min_dense_score = min_dense_score
        self.store = store
        self.candidates = candidates
        self.rrf_k = rrf_k

    @cached_property
    def sparse(self):
        from rank_bm25 import BM25Okapi

        documents = [
            self.store.docstore.search(key)
            for key in self.store.index_to_docstore_id.values()
        ]
        tokens = [tokenize(doc.page_content) for doc in documents]
        titles = {}
        seen_titles = set()
        for doc in documents:
            title = str(doc.metadata.get("title") or "")
            words = tuple(tokenize(title))
            key = (words, doc.metadata.get("source_url") or doc.metadata.get("id"))
            if words and len(words) <= 8 and key not in seen_titles:
                titles.setdefault(words, []).append(doc)
                seen_titles.add(key)
        return documents, BM25Okapi(tokens), titles

    def search(self, question: str, limit: int = 5):
        dense = self.store.similarity_search_with_relevance_scores(question, k=self.candidates)
        dense = [(doc, score) for doc, score in dense if score >= self.min_dense_score]
        documents, bm25, titles = self.sparse
        terms = tokenize(question)
        scores = bm25.get_scores(terms)
        title_docs = []
        for width in range(min(8, len(terms)), 0, -1):
            for offset in range(len(terms) - width + 1):
                phrase = tuple(terms[offset:offset + width])
                if width == 1 and (len(phrase[0]) < 4 or phrase[0] in {"what", "about", "does", "this", "have", "with"}):
                    continue
                title_docs.extend(titles.get(phrase, ()))
        title_docs = title_docs[:self.candidates]
        sparse = sorted(
            ((score, index) for index, score in enumerate(scores) if score > 0),
            key=lambda item: item[0], reverse=True,
        )[:self.candidates]
        matches = {chunk_key(documents[index]): Match(documents[index], None) for _, index in sparse}
        for doc in title_docs:
            matches.setdefault(chunk_key(doc), Match(doc, None))
        for doc, score in reversed(dense):
            matches[chunk_key(doc)] = Match(doc, score)
        ranked = fuse_rankings(
            ([chunk_key(doc) for doc, _ in dense], [chunk_key(documents[i]) for _, i in sparse],
             [chunk_key(doc) for doc in title_docs]),
            k=self.rrf_k, weights=(1.0, 1.0, 1.5),
        )
        return [matches[key] for key in ranked[:limit]]

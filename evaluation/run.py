"""Report measured source retrieval metrics and latency."""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from types import SimpleNamespace

from src.hybrid import HybridRetriever, Match, tokenize


def metrics(expected, ranked):
    relevant = set(expected)
    ranks = [rank for rank, url in enumerate(ranked, start=1) if url in relevant]
    retrieved = set(ranked) & relevant
    return len(retrieved) / len(relevant), 1 / ranks[0] if ranks else 0


class Dense:
    def __init__(self, store):
        self.store = store

    def search(self, question, limit):
        return [Match(doc, 1.0 - float(distance)) for doc, distance in
                self.store.similarity_search_with_score(question, k=limit)]


def lexical_search():
    from rank_bm25 import BM25Okapi
    from src.loader import DEFAULT_DATA_PATH, read_records

    docs = [SimpleNamespace(page_content=f"{row['title']}\n\n{row['clean_text']}",
                            metadata={"source_url": row["source_url"]})
            for row in read_records(DEFAULT_DATA_PATH) if row.get("clean_text")]
    tokens = [tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokens)

    def search(question, limit):
        scores = bm25.get_scores(tokenize(question))
        top = sorted(range(len(docs)), key=lambda index: scores[index], reverse=True)[:limit]
        return [Match(docs[index], None) for index in top if scores[index] > 0]

    return search


def pipelines(lexical_only, reranker_name):
    if lexical_only:
        return {"bm25_archive_documents": (lexical_search(), None)}
    from src.config import embedding_model
    from src.embedder import DEFAULT_MODEL
    from src.loader import load_university_data
    from src.reranker import Reranker
    from src.service import ChatService
    from src.splitter import split_documents
    from src.store import index_directory, load_index
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_huggingface import HuggingFaceEmbeddings

    original_store = FAISS.from_documents(
        split_documents(load_university_data()),
        HuggingFaceEmbeddings(model_name=DEFAULT_MODEL),
        distance_strategy=DistanceStrategy.COSINE,
    )
    dense = Dense(original_store)
    hybrid = HybridRetriever(load_index(index_directory()))
    reranker = Reranker(reranker_name)
    rewrite_service = ChatService(hybrid)

    def reranked(question, limit):
        return reranker.rank(question, hybrid.search(question, limit * 3))[:limit]

    return {
        "original_dense": (dense.search, None),
        f"hybrid_{embedding_model()}": (hybrid.search, None),
        "hybrid_reranked": (reranked, None),
        "reranked_with_rewriting": (reranked, rewrite_service.rewrite),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--pipeline", action="append", help="Run only the named pipeline(s).")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    benchmark = json.loads(Path(__file__).with_name("benchmark.json").read_text(encoding="utf-8"))
    for name, (search, rewrite) in pipelines(args.lexical_only, args.reranker).items():
        if args.pipeline and name not in args.pipeline:
            continue
        recalls, reciprocals, latencies = [], [], []
        rewritten = 0
        for case in benchmark:
            if not case["sources"]:
                continue
            started = perf_counter()
            question = case["question"]
            if rewrite:
                updated = rewrite(question, case.get("history", []))
                rewritten += updated != question
                question = updated
            matches = search(question, args.k)
            latency = (perf_counter() - started) * 1000
            urls = [match.document.metadata.get("source_url") for match in matches]
            recall, reciprocal = metrics(case["sources"], urls)
            recalls.append(recall)
            reciprocals.append(reciprocal)
            latencies.append(latency)
        print(json.dumps({
            "pipeline": name, "k": args.k, "queries": len(recalls),
            "recall_at_k": mean(recalls), "mrr_at_k": mean(reciprocals),
            "mean_latency_ms": round(mean(latencies), 2), "rewritten_queries": rewritten,
        }))


if __name__ == "__main__":
    main()

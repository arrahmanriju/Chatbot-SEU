"""Shared retrieval and feedback functions for Streamlit and the HTTP API."""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from src.hybrid import HybridRetriever
from src.reranker import Reranker
from src.service import ChatService
from src.store import INDEX_FILE, build_vectorstore, load_index

ROOT = Path(__file__).resolve().parents[1]


def load_or_build_index(directory: Path):
    """Load the local artifact, building it from the bundled corpus on a cold start."""
    if (directory / INDEX_FILE).is_file():
        return load_index(directory)
    return build_vectorstore(directory)


def create_retriever(store):
    return HybridRetriever(
        store,
        candidates=int(os.getenv("RETRIEVAL_CANDIDATES", "16")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        min_dense_score=float(os.getenv("MIN_DENSE_SCORE", "0.1")),
    )


def create_service(retriever):
    rerank_model = os.getenv("RERANK_MODEL")
    return ChatService(
        retriever,
        limit=int(os.getenv("RETRIEVAL_LIMIT", "5")),
        reranker=Reranker(rerank_model) if rerank_model else None,
    )


def save_feedback(request_id, helpful):
    database = Path(os.getenv("FEEDBACK_DB") or ROOT / "vectorstore" / "feedback.sqlite3")
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS feedback "
                           "(request_id TEXT PRIMARY KEY, helpful INTEGER NOT NULL)")
        connection.execute("INSERT INTO feedback VALUES (?, ?) "
                           "ON CONFLICT(request_id) DO UPDATE SET helpful = excluded.helpful",
                           (str(request_id), helpful))

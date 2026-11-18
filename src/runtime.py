"""Shared retrieval and feedback functions for Streamlit and the HTTP API."""
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from threading import RLock

from src.hybrid import HybridRetriever
from src.reranker import Reranker
from src.service import ChatService
from src.config import embedding_model
from src.store import (DEFAULT_PERSIST_DIR, INDEX_FILE, build_vectorstore, corpus_fingerprint,
                       corpus_paths, file_revision, load_index, read_manifest)

ROOT = Path(__file__).resolve().parents[1]


def load_or_build_index(directory: Path, *, check_corpus=False):
    """Load the local artifact, building it from the bundled corpus on a cold start."""
    if (directory / INDEX_FILE).is_file():
        if check_corpus:
            manifest = read_manifest(directory)
            saved = manifest.get("corpus_sha256")
            if saved and (saved != corpus_fingerprint() or manifest.get("model") != embedding_model()):
                return build_vectorstore(directory)
            if "corpus_sha256" not in manifest and directory.resolve() == DEFAULT_PERSIST_DIR.resolve():
                # Older default artifacts are checked against the chunk fingerprint once per process.
                return build_vectorstore(directory)
        return load_index(directory)
    return build_vectorstore(directory)


class IndexUnavailable(RuntimeError):
    pass


class IndexedRetriever:
    """Share one refreshable index across sessions without loading on a blank rerun."""

    def __init__(self, directory):
        self.directory = directory
        self._lock = RLock()
        self._revision = None
        self._retriever = None

    def get(self):
        with self._lock:
            try:
                revision = (embedding_model(), *(file_revision(path) for path in corpus_paths()))
                artifact = file_revision(self.directory / INDEX_FILE)
                if (self.directory / ".knowledge-update.lock").exists():
                    if self._retriever is None:
                        self._retriever = create_retriever(load_index(self.directory))
                    return self._retriever
                key = (revision, artifact)
                if key != self._revision:
                    self._retriever = create_retriever(load_or_build_index(self.directory, check_corpus=True))
                    self._revision = (revision, file_revision(self.directory / INDEX_FILE))
                return self._retriever
            except ModuleNotFoundError:
                raise
            except Exception as exc:
                raise IndexUnavailable("The knowledge base could not be built or loaded.") from exc

    def search(self, question, limit):
        return self.get().search(question, limit)

    def search_support(self, topic):
        return self.get().search_support(topic)

    def search_program_fees(self, programs, fee_type=None):
        return self.get().search_program_fees(programs, fee_type)


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

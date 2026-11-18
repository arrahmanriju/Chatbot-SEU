"""Build and load the offline FAISS index with JSON document metadata."""
import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import BadZipFile, ZipFile

from src.config import embedding_model, load_environment
from src.embedder import configure_cpu, deferred_embeddings, get_embedding_model
from src.loader import DEFAULT_DATA_PATH, OFFICIAL_UPDATES_PATH, VERIFIED_STUDENT_PATH

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[1] / "vectorstore" / "faiss_index"
INDEX_FILE = "index.zip"
logger = logging.getLogger(__name__)


def index_directory():
    load_environment()
    return Path(os.getenv("SEU_INDEX_DIR") or DEFAULT_PERSIST_DIR)


def corpus_paths(overlay=OFFICIAL_UPDATES_PATH):
    return DEFAULT_DATA_PATH, VERIFIED_STUDENT_PATH, Path(overlay)


def file_revision(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def corpus_fingerprint(paths=None):
    result = hashlib.sha256()
    for path in corpus_paths() if paths is None else paths:
        result.update(hashlib.sha256(path.read_bytes()).digest() if path.exists() else b"missing")
    return result.hexdigest()


def cosine_relevance(squared_distance: float) -> float:
    # For unit vectors, squared L2 distance is 2 - 2*cosine.
    return max(0.0, min(1.0, 1.0 - squared_distance / 2.0))


def build_vectorstore(persist_dir: str | Path = DEFAULT_PERSIST_DIR, data_path=None, *,
                      updates_path=None, append_from=None):
    configure_cpu()
    started = perf_counter()
    import faiss
    from langchain_community.vectorstores import FAISS

    from src.loader import load_university_data
    from src.splitter import split_documents

    target = Path(persist_dir)
    model = embedding_model()
    overlay = Path(updates_path) if updates_path is not None else OFFICIAL_UPDATES_PATH
    bundled = data_path is None or Path(data_path).resolve() == DEFAULT_DATA_PATH.resolve()
    corpus_hash = corpus_fingerprint(corpus_paths(overlay)) if bundled else None
    documents = load_university_data(data_path or DEFAULT_DATA_PATH)
    if bundled:
        documents.extend(load_university_data(VERIFIED_STUDENT_PATH))
        if overlay.exists():
            documents.extend(load_university_data(overlay))
    superseded = {old: str(doc.metadata["id"]) for doc in documents
                  if not doc.metadata.get("superseded_by")
                  for old in doc.metadata.get("supersedes", ())}
    for doc in documents:
        replacement = superseded.get(str(doc.metadata.get("id")))
        if replacement:
            doc.metadata["superseded_by"] = replacement
    records = {}
    for chunk in split_documents(documents):
        record = {"content": chunk.page_content, "metadata": chunk.metadata}
        records.setdefault(json.dumps(record, sort_keys=True, ensure_ascii=False), chunk)
    if bundled and corpus_fingerprint(corpus_paths(overlay)) != corpus_hash:
        raise RuntimeError("Knowledge base changed during indexing. Retry with the updated corpus.")
    chunks = list(records.values())
    fingerprint = hashlib.sha256("\n".join(records).encode()).hexdigest()
    try:
        existing = read_manifest(target)
    except (OSError, ValueError, BadZipFile, KeyError):
        existing = {}
    if existing.get("model") == model and existing.get("fingerprint") == fingerprint:
        return load_index(target)
    if not chunks:
        raise ValueError("The university corpus produced no searchable chunks.")

    serialized = [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
    baseline = target if append_from is None else Path(append_from)
    try:
        previous = read_manifest(baseline)
    except (OSError, ValueError, BadZipFile, KeyError):
        previous = {}
    old_docs = previous.get("documents", [])
    if (previous.get("model") == model and old_docs and len(old_docs) < len(serialized)
            and serialized[:len(old_docs)] == old_docs):
        vectorstore = load_index(baseline)
        vectorstore.add_documents(chunks[len(old_docs):])
    else:
        vectorstore = FAISS.from_documents(
            chunks, deferred_embeddings(lambda: get_embedding_model(model)), relevance_score_fn=cosine_relevance
        )
    target.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target, suffix=".tmp", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        with ZipFile(temp_path, "w") as archive:
            archive.writestr("vectors.faiss", faiss.serialize_index(vectorstore.index).tobytes())
            archive.writestr("documents.json", json.dumps({
                "version": 1,
                "model": model,
                "fingerprint": fingerprint,
                "corpus_sha256": corpus_hash,
                "documents": serialized,
            }, ensure_ascii=False))
        os.replace(temp_path, target / INDEX_FILE)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("faiss_index_written chunks=%d elapsed_ms=%.1f", len(chunks), (perf_counter()-started)*1000)
    return vectorstore


def read_manifest(persist_dir: str | Path = DEFAULT_PERSIST_DIR):
    with ZipFile(Path(persist_dir) / INDEX_FILE) as archive:
        return json.loads(archive.read("documents.json"))


def load_index(persist_dir: str | Path = DEFAULT_PERSIST_DIR):
    artifact = Path(persist_dir) / INDEX_FILE
    if not artifact.is_file():
        raise FileNotFoundError("Build the index with python -m src.store.")

    configure_cpu()
    started = perf_counter()
    import faiss
    import numpy as np
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    # Load index artifacts produced by our offline build from a trusted volume.
    with ZipFile(artifact) as archive:
        manifest = json.loads(archive.read("documents.json"))
        if manifest.get("version") != 1 or manifest.get("model") != embedding_model():
            raise ValueError("Index configuration changed. Rebuild with python -m src.store.")
        index = faiss.deserialize_index(np.frombuffer(archive.read("vectors.faiss"), dtype=np.uint8))
    documents = {
        str(position): Document(page_content=record["content"], metadata=record["metadata"])
        for position, record in enumerate(manifest["documents"])
    }
    if index.ntotal != len(documents) or not documents:
        raise ValueError("Index is inconsistent. Rebuild with python -m src.store.")
    logger.info("faiss_index_loaded chunks=%d elapsed_ms=%.1f", len(documents), (perf_counter()-started)*1000)
    return FAISS(
        deferred_embeddings(lambda: get_embedding_model(manifest["model"])), index, InMemoryDocstore(documents),
        {position: key for position, key in enumerate(documents)},
        relevance_score_fn=cosine_relevance,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--index-dir", type=Path, default=index_directory())
    args = parser.parse_args()
    index = build_vectorstore(args.index_dir, args.data)
    print(f"Indexed {index.index.ntotal} chunks in {args.index_dir}")

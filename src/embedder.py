"""Share one native embedding model and bound repeated query work per index."""
import hashlib
import logging
import os
from functools import lru_cache
from threading import RLock
from time import perf_counter

from src.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)
_MODEL_LOCK = RLock()


@lru_cache(maxsize=1)
def configure_cpu():
    count = int(os.getenv("SEU_CPU_THREADS", "1"))
    if count < 1:
        raise ValueError("SEU_CPU_THREADS must be positive.")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(count)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return count


@lru_cache(maxsize=1)
def _load_model(model_name):
    threads = configure_cpu()
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # A caller may already have initialized PyTorch inter-op execution.
        pass
    started = perf_counter()
    model = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )
    logger.info("embedding_model_loaded elapsed_ms=%.1f threads=%d", (perf_counter() - started) * 1000, threads)
    return model


def get_embedding_model(model_name: str = DEFAULT_MODEL):
    with _MODEL_LOCK:
        return _load_model(model_name)


def deferred_embeddings(get_model):
    """Delay model import until a vector is needed and serialize native encoding."""
    from collections import OrderedDict
    from langchain_core.embeddings import Embeddings

    class DeferredEmbeddings(Embeddings):
        def __init__(self):
            self.vectors = OrderedDict()

        def embed_documents(self, texts):
            with _MODEL_LOCK:
                started = perf_counter()
                vectors = get_model().embed_documents(texts)
                logger.info("embedding_documents count=%d elapsed_ms=%.1f", len(texts), (perf_counter()-started)*1000)
                return vectors

        def embed_query(self, text):
            key = hashlib.sha256(text.encode("utf-8")).digest()
            started = perf_counter()
            with _MODEL_LOCK:
                vector = self.vectors.get(key)
                hit = vector is not None
                if hit:
                    self.vectors.move_to_end(key)
                else:
                    vector = tuple(get_model().embed_query(text))
                    self.vectors[key] = vector
                    if len(self.vectors) > 128:
                        self.vectors.popitem(last=False)
            logger.info("embedding_query cache_hit=%s elapsed_ms=%.1f", hit, (perf_counter()-started)*1000)
            return list(vector)

    return DeferredEmbeddings()

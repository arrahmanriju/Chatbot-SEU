from functools import lru_cache

from src.config import DEFAULT_MODEL


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = DEFAULT_MODEL):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs={"normalize_embeddings": True}
    )

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

try:
    from src.loader import load_university_data
    from src.splitter import split_documents
    from src.embedder import get_embedding_model
except ModuleNotFoundError:
    from loader import load_university_data
    from splitter import split_documents
    from embedder import get_embedding_model


def build_vectorstore(persist_dir: str = "vectorstore/faiss_index"):
    """
    Runs the full ingestion pipeline: load -> split -> embed -> store.
    Only needs to be run once (or whenever university_data.json changes).
    """
    print("Loading documents...")
    docs = load_university_data()

    print("Splitting into chunks...")
    chunks = split_documents(docs)

    print("Loading embedding model...")
    embeddings = get_embedding_model()

    print(f"Embedding and storing {len(chunks)} chunks in FAISS index...")
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )

    vectorstore.save_local(persist_dir)
    print(f"Done. Vector store saved to: {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
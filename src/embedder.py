# src/embedder.py

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads a Hugging Face embedding model.
    all-MiniLM-L6-v2 is small, fast, and good enough for a practice RAG project.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


if __name__ == "__main__":
    embedder = get_embedding_model()
    sample_text = "Southeast University is located in Dhaka, Bangladesh."
    vector = embedder.embed_query(sample_text)
    print(f"Embedding length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
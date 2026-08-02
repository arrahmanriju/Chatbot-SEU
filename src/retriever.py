try:
    from src.filter import load_vectorstore
except ModuleNotFoundError:
    from filter import load_vectorstore


def retrieve_top_chunks(question: str, vectorstore, k: int = 5):
    """
    Embeds the question and retrieves the top-k most relevant chunks
    from the FAISS index.
    """
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return results


if __name__ == "__main__":
    vectorstore = load_vectorstore()

    question = "What programs does SEU offer?"
    results = retrieve_top_chunks(question, vectorstore, k=5)

    print(f"Q: {question}\n")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- Chunk {i} (score: {score:.3f}) ---")
        print(doc.page_content[:200])
        print(f"Source: {doc.metadata.get('source_url')}\n")
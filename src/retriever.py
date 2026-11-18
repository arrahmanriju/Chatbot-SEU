"""Retrieve dense matches from the university knowledge base."""


def retrieve_top_chunks(question: str, vectorstore, k: int = 5):
    return vectorstore.similarity_search_with_relevance_scores(question, k=k)

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

try:
    from src.embedder import get_embedding_model
except ModuleNotFoundError:
    from embedder import get_embedding_model

GREETINGS = {
    "hi", "hello", "hey", "hii", "hiya", "yo",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye",
}

META_PHRASES = (
    "what information", "what can you", "what do you know",
    "who are you", "what are you", "help", "what can you do",
)


def load_vectorstore(persist_dir: str = "vectorstore/faiss_index"):
    embeddings = get_embedding_model()
    return FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.COSINE,
    )


def classify_message(question: str):
    """
    Returns one of: "greeting", "meta", "university", "off_topic"
    """
    normalized = question.strip().lower().rstrip("!?.")

    if normalized in GREETINGS:
        return "greeting"

    if any(phrase in normalized for phrase in META_PHRASES):
        return "meta"

    return None  # needs similarity check


def is_university_question(question: str, vectorstore, threshold: float = 0.1) -> bool:
    """
    Checks if a question is likely about the university using cosine similarity.
    """
    results = vectorstore.similarity_search_with_relevance_scores(question, k=1)
    if not results:
        return False

    top_doc, score = results[0]
    print(f"Score: {score:.3f} | Top match: {top_doc.page_content[:60]}...")
    return score >= threshold


if __name__ == "__main__":
    vectorstore = load_vectorstore()

    test_questions = [
        "hi",
        "What programs does SEU offer?",
        "What information do you have?",
        "What's the capital of France?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        category = classify_message(q)
        if category:
            print(f"-> Classified as: {category}")
        else:
            result = is_university_question(q, vectorstore)
            print(f"-> University question? {result}")
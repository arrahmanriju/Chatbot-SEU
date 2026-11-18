"""Lightweight conversational shortcuts and the compatibility index loader."""
from pathlib import Path

GREETINGS = {
    "hi", "hello", "hey", "hii", "hiya", "yo",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye",
}

META_PHRASES = {
    "what information do you have", "what can you do", "what do you know",
    "who are you", "what are you", "help",
}


def load_vectorstore(persist_dir: str | Path | None = None):
    from src.store import DEFAULT_PERSIST_DIR, load_index

    return load_index(DEFAULT_PERSIST_DIR if persist_dir is None else persist_dir)


def classify_message(question: str):
    normalized = " ".join(question.casefold().split()).rstrip("!?.")
    if normalized in GREETINGS:
        return "greeting"
    if normalized in META_PHRASES:
        return "meta"
    return None


def is_university_question(question: str, vectorstore, threshold: float = 0.1) -> bool:
    from src.retriever import retrieve_top_chunks

    chunks = retrieve_top_chunks(question, vectorstore, k=1)
    return bool(chunks and chunks[0][1] >= threshold)

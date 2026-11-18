"""Lightweight conversational shortcuts and the compatibility index loader."""
from pathlib import Path
import re

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


STUDENT_TOPIC = re.compile(
    r"\b(tuition|admission|enrollment|course|courses|registration|advising|cgpa|"
    r"waivers?|scholarships?|faculty|exams?|ums|student ledger|student portal|"
    r"transcripts?|convocation)\b|টিউশন|ভর্তি|কোর্স|নিবন্ধন|সিজিপিএ|ওয়েভার|"
    r"শিক্ষক|পরীক্ষা|ইউএমএস|vorti|porikkha", re.I,
)
NON_CAMPUS_TOPIC = re.compile(r"\b(golf course|driving (?:test|exam)|voter registration|vehicle registration)\b", re.I)

OTHER_INSTITUTION = re.compile(
    r"\b(?:buet|brac|nsu|iub|duet|mit|harvard|stanford)\b|"
    r"\b(?:at|of|from)\s+(?!the\b|southeast university\b)(?:[a-z]+\s+){1,4}university\b|"
    r"\buniversity of (?!southeast\b)[a-z]+|ঢাকা বিশ্ববিদ্যালয়|বুয়েট|ব্র্যাক বিশ্ববিদ্যালয়", re.I,
)


def mentions_other_institution(question: str) -> bool:
    return bool(OTHER_INSTITUTION.search(question))


def has_university_context(question: str) -> bool:
    """Recognize student topics in the context of a single-university assistant."""
    return bool(not mentions_other_institution(question) and (
        (STUDENT_TOPIC.search(question) and not NON_CAMPUS_TOPIC.search(question)) or re.search(r"\bseu\b|southeast university|সাউথইস্ট", question, re.I)
    ))
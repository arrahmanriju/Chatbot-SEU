"""Resolve a few narrowly scoped questions from reviewed official guidance."""
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

PATTERNS = (
    ("incomplete_advising", r"\badvis(?:ing|ement)\b.*\b(?:incomplete|unfinished|pending)\b|"
                            r"\b(?:incomplete|unfinished|pending)\b.*\badvis(?:ing|ement)\b"),
    ("course_registration", r"\bhow\b.*\bregister\b.*\bcourses?\b|\bhow\b.*\bcourse registration\b"),
    ("internship_forwarding_letter", r"\binternship\b.*\bforwarding letter\b|\bforwarding letter\b.*\binternship\b"),
    ("graduation_documents", r"\bgraduation\b.*\b(?:documents|checklist)\b|\b(?:documents|checklist)\b.*\bgraduation\b"),
    ("scholarships", r"\bwhat scholarships\b.*\bavailable\b|\bwhich scholarships\b.*\bavailable\b"),
)


def support_reply(retriever, question):
    topic = next((name for name, pattern in PATTERNS if re.search(pattern, question, re.I)), None)
    search = getattr(retriever, "search_support", None)
    if not topic or not callable(search):
        return None
    document = search(topic)
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, Mapping) or metadata.get("verification") != "official_public":
        return None
    url, text = metadata.get("source_url"), metadata.get("guidance_text")
    if not isinstance(url, str) or not isinstance(text, str):
        return None
    host = urlsplit(url).hostname or ""
    if host != "seu.edu.bd" and not host.endswith(".seu.edu.bd"):
        return None
    session = metadata.get("academic_session")
    years = re.findall(r"\b20\d{2}\b", question)
    gap = metadata.get("guidance_gap")
    if years and (not session or not all(year in session for year in years)):
        return None
    current = bool(re.search(r"\b(current|latest|today|now|upcoming)\b", question, re.I))
    message = "\n\n".join(part for part in (gap, text) if part) if gap else text
    route = "unverified" if gap and (topic in {"incomplete_advising", "internship_forwarding_letter",
                                               "graduation_documents"} or (topic == "scholarships" and current)) else "university"
    return message, {"id": 1, "source_id": metadata["id"], "title": metadata["title"], "url": url,
                     "admission_session": session, "potentially_outdated": bool(session)}, route

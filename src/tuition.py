"""Render fees only from the official schedule retrieved for the requested program."""
import re
from urllib.parse import urlsplit

from src.student import localized

PROGRAM = re.compile(r"\bcse\b|computer science|সিএসই|কম্পিউটার সায়েন্স", re.I)
PRICE = re.compile(r"\b(tuition|fees?|cost|per credit|khoroch)\b|ফি|খরচ", re.I)
SESSION = re.compile(r"\b(spring|summer|fall)\s+(20\d\d)\b", re.I)
YEAR = re.compile(r"\b20\d\d\b")


def requests_program_tuition(question):
    non_price = re.search(r"\b(waiver|scholarship|refund|deadline|installment)\b|ওয়েভার|বৃত্তি|সময়সীমা", question, re.I)
    return bool(PROGRAM.search(question) and PRICE.search(question) and not non_price)


def render_tuition(matches, question, language):
    normalized = question.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    for bangla, english in (("স্প্রিং", "Spring"), ("সামার", "Summer"), ("ফল", "Fall")):
        normalized = normalized.replace(bangla, english)
    requested = SESSION.search(normalized)
    requested_years = set(YEAR.findall(normalized))
    eligible = []
    for match in matches:
        metadata = match.document.metadata
        fees = metadata.get("fee_schedule")
        session = metadata.get("admission_session")
        url = metadata.get("source_url") or ""
        if (metadata.get("verification") != "official_public" or not isinstance(fees, dict)
                or not isinstance(session, str) or not SESSION.fullmatch(session) or fees.get("program") != "CSE"):
            continue
        if urlsplit(url).hostname not in {"seu.edu.bd", "www.seu.edu.bd"} or not url.startswith("https://"):
            continue
        if requested and requested.group().casefold() != session.casefold():
            continue
        if requested_years and set(YEAR.findall(session)) != requested_years:
            continue
        keys = ("per_credit_bdt", "credits", "total_tuition_bdt", "approximate_total_bdt")
        amounts = tuple(fees.get(key) for key in keys)
        if any(type(value) is not int or value <= 0 for value in amounts):
            continue
        per_credit, credits, tuition, total = amounts
        if per_credit * credits == tuition and total >= tuition:
            eligible.append((metadata, amounts))
    if not eligible:
        return None
    metadata, amounts = eligible[0]
    if any(other["admission_session"] != metadata["admission_session"] or values != amounts
           for other, values in eligible):
        return None

    session = metadata["admission_session"]
    per_credit, credits, tuition, total = (f"{value:,}" for value in amounts)
    answer = localized(
        language,
        f"**BSc in CSE — {session} admission**\n\n"
        f"- Tuition per credit: BDT {per_credit}\n- Required credits: {credits}\n"
        f"- Total tuition: BDT {tuition}\n- Approximate total program cost: BDT {total}\n\n"
        "These figures apply to the cited admission session. I could not verify rates for a newer intake.",
        f"**বিএসসি ইন সিএসই — {session} ভর্তি সেশন**\n\n"
        f"- প্রতি ক্রেডিটের টিউশন ফি: {per_credit} টাকা\n- মোট ক্রেডিট: {credits}\n"
        f"- মোট টিউশন ফি: {tuition} টাকা\n- আনুমানিক মোট প্রোগ্রাম খরচ: {total} টাকা\n\n"
        "এই হার উল্লেখিত ভর্তি সেশনের জন্য। পরবর্তী সেশনের হার যাচাই করা যায়নি।",
        f"**BSc in CSE — {session} admission**\n\n"
        f"- Per credit tuition: BDT {per_credit}\n- Mot credit: {credits}\n"
        f"- Mot tuition: BDT {tuition}\n- Anumanik total program cost: BDT {total}\n\n"
        "Ei fee cited admission session er jonno. Notun intake er rate verify kora jayni.",
    )
    citation = {"id": 1, "source_id": metadata["id"], "title": metadata["title"],
                "url": metadata["source_url"], "admission_session": session, "potentially_outdated": True}
    return answer, citation

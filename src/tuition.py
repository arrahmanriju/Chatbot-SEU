"""Build fee answers from official program rows in the active index."""
import re
from urllib.parse import urlsplit

from src.student import localized

SESSION = re.compile(r"\b(spring|summer|fall)\s+(20\d\d)\b", re.I)
YEAR = re.compile(r"\b20\d\d\b")
CURRENT = re.compile(r"\b(current|latest|now|today|upcoming|new intake|ekhon)\b|বর্তমান|এখন|সর্বশেষ", re.I)
SEASONS = {"spring": 0, "summer": 1, "fall": 2}
PROGRAM_FIELDS = ("per_credit_bdt", "credits", "total_tuition_bdt", "approximate_total_bdt")
EXTRA_FIELDS = ("student_activity_bdt", "lab_bdt", "noncredit_bdt")
ADMISSION_FIELDS = ("admission_bdt", "library_bdt", "admission_form_bdt", "annual_insurance_bdt")


def _fee_amounts(fees):
    if not isinstance(fees.get("program_name"), str):
        return None
    if fees.get("program") == "ADMISSION":
        charges = fees.get("charges_bdt")
        if not isinstance(charges, dict):
            return None
        amounts = tuple(charges.get(key) for key in ADMISSION_FIELDS)
        if all(type(value) is int and value >= 0 for value in amounts) and sum(amounts) == fees.get("total_bdt"):
            return (*amounts, fees["total_bdt"])
        return None

    amounts = tuple(fees.get(key) for key in PROGRAM_FIELDS)
    extras = fees.get("additional_fees_bdt")
    if any(type(value) is not int or value <= 0 for value in amounts) or not isinstance(extras, dict):
        return None
    additional = tuple(extras.get(key) for key in EXTRA_FIELDS)
    if any(value is not None and (type(value) is not int or value < 0) for value in additional):
        return None
    rate, credits, tuition, total = amounts
    if rate * credits != tuition or tuition + sum(value or 0 for value in additional) != total:
        return None
    return (*amounts, *additional)


def _session_key(session):
    season, year = session.split()
    return int(year), SEASONS[season.casefold()]


def _select_fees(matches, programs, question):
    normalized = question.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    for bangla, english in (("স্প্রিং", "Spring"), ("সামার", "Summer"), ("ফল", "Fall")):
        normalized = normalized.replace(bangla, english)
    exact = {m.group().casefold() for m in SESSION.finditer(normalized)}
    years = set(YEAR.findall(normalized))
    if len(exact) > 1 or len(years) > 1:
        return []

    schedules = {program: {} for program in programs}
    for match in matches:
        metadata = match.document.metadata
        fees = metadata.get("fee_schedule")
        session = metadata.get("admission_session")
        url = metadata.get("source_url") or ""
        if (metadata.get("verification") != "official_public" or not isinstance(fees, dict)
                or not isinstance(session, str) or not SESSION.fullmatch(session)
                or fees.get("program") not in schedules or not metadata.get("id")):
            continue
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in {"seu.edu.bd", "www.seu.edu.bd"}:
            continue
        if (exact and session.casefold() not in exact) or (years and not years.intersection(YEAR.findall(session))):
            continue
        amounts = _fee_amounts(fees)
        if amounts:
            schedules[fees["program"]].setdefault(session, []).append((metadata, fees, amounts))

    common = set.intersection(*(set(rows) for rows in schedules.values())) if schedules else set()
    if not common:
        return []
    selected_session = max(common, key=_session_key)
    selected = []
    for by_session in schedules.values():
        first, *others = by_session[selected_session]
        if any(row[2] != first[2] for row in others):
            return []
        selected.append(first)
    return selected


def render_tuition(matches, programs, question, language):
    selected = _select_fees(matches, programs, question)
    if not selected:
        return None
    current = bool(CURRENT.search(question))
    note = (localized(language,
                      "I could not verify current rates. Here are the published figures for the cited session.",
                      "বর্তমান ফি যাচাই করা যায়নি। উল্লেখিত সেশনের প্রকাশিত ফি নিচে দেওয়া হলো।",
                      "Current fee verify kora jayni. Cited admission session er published fee niche.")
            if current else localized(language, "These figures apply to the listed admission session. Newer fees may differ.",
                                      "এই ফি উল্লেখিত ভর্তি সেশনের জন্য। পরবর্তী সেশনের ফি ভিন্ন হতে পারে।",
                                      "Ei fee listed admission session er jonno. Notun intake er fee alada hote pare."))
    blocks, citations = [], []
    for position, (metadata, fees, amounts) in enumerate(selected, start=1):
        session = metadata["admission_session"]
        program = fees["program_name"]
        if fees["program"] == "ADMISSION":
            labels = localized(language,
                               ("Admission fee", "Library fee", "Admission form", "Annual insurance", "Listed charges"),
                               ("ভর্তি ফি", "লাইব্রেরি ফি", "ভর্তি ফরম", "বার্ষিক বীমা", "উল্লিখিত মোট ফি"),
                               ("Admission fee", "Library fee", "Admission form", "Annual insurance", "Mot listed fee"))
            extra_note = localized(language, "The initial tuition deposit is separate.",
                                   "প্রাথমিক টিউশন জমা আলাদা।", "Initial tuition deposit alada.")
        else:
            labels = localized(language,
                               ("Tuition per credit", "Required credits", "Total tuition", "Approximate program cost",
                                "Student activity fee", "Laboratory fee", "Non-credit subject fee"),
                               ("প্রতি ক্রেডিটের টিউশন", "মোট ক্রেডিট", "মোট টিউশন", "আনুমানিক প্রোগ্রাম খরচ",
                                "শিক্ষার্থী কার্যক্রম ফি", "ল্যাব ফি", "নন-ক্রেডিট বিষয়ের ফি"),
                               ("Per-credit tuition", "Mot credits", "Mot tuition", "Anumanik program cost",
                                "Student activity fee", "Lab fee", "Non-credit subject fee"))
            extra_note = ""
        fields = [
            f"- {label}: {'' if fees['program'] != 'ADMISSION' and index == 1 else 'BDT '}{amount:,}"
            for index, (label, amount) in enumerate(zip(labels, amounts, strict=True)) if amount is not None
        ]
        if extra_note:
            fields.append(extra_note)
        blocks.append(f"**{program} — {session}**\n\n" + "\n".join(fields))
        citations.append({"id": position, "source_id": metadata["id"], "title": metadata["title"],
                          "url": metadata["source_url"], "admission_session": session, "potentially_outdated": True})
    return "\n\n".join((note, *blocks) if current else (*blocks, note)), citations, current

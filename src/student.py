"""Student intent, language and account-access rules shared by chat and evaluation."""
import json
import re
from pathlib import Path

from src.hybrid import tokenize
from src.programs import tuition_programs

DATA = Path(__file__).resolve().parents[1] / "data"
CATEGORIES = json.loads((DATA / "student_categories.json").read_text(encoding="utf-8"))
PATTERNS = {
    category: [tuple(tokenize(term)) for term in terms.split("|")]
    for category, terms in CATEGORIES.items()
}
BANGLISH = re.compile(r"\b(amar|kothay|kivabe|kobe|hoise|ase|korbo|korba|lagbe|niye|pabo|koto|shesh|age|kore|acha|hobe)\b", re.I)
PERSONAL = re.compile(r"\b(my|mine|amar|am i|to me|myself|tell me my)\b|আমার|আমাকে|আমার নামে")
LOOKUP = re.compile(
    r"\b(cgpa|gpa|grades?|result|marks?|attendance|balance|ledger|account details|application status|complaint status|"
    r"approved|applied|ready|assigned|registered|owe|issued|percentage|enrolled|first class|status|"
    r"member|approval|baki|hoise|hoyeche|assign)\b|সিজিপিএ|জিপিএ|গ্রেড|উপস্থিতি|বাকি|অনুমোদিত|"
    r"নির্ধারিত|অবস্থা|অ্যাকাউন্টের তথ্য|সাময়িক সনদ|আমার নামে|প্রথম ক্লাস|ধার আছে|সদস্য"
)
GENERAL_NAVIGATION = re.compile(
    r"\b(where can i (view|see|find)|how (can|do) i (view|see|check)|kothay dekha|kivabe dekhbo)\b|কোথায়( দেখা| দেখব| পাব)"
)
SECRET = re.compile(r"\b(?:password|passcode|otp|verification code|পাসওয়ার্ড)\s*(?:is\s+|[:=]\s*)\S+", re.I)
PRIVATE_LINK = "https://status.seu.edu.bd/"
CLARIFY = re.compile(r"^(?:how much (?:is )?(?:the )?(?:tuition|fee)s?|when (?:does )?(?:my |the )?(?:next )?semester start|ফি কত|সেমিস্টার কবে শুরু|fee koto)[?.!।]*$", re.I)


def language_of(question):
    if re.search(r"[\u0980-\u09ff]", question):
        return "bn"
    return "banglish" if BANGLISH.search(question) else "en"


def classify_intents(question):
    terms = tokenize(question)
    candidates = []
    for category, phrases in PATTERNS.items():
        score = sum(len(phrase) ** 2 for phrase in phrases if any(
            tuple(terms[i:i + len(phrase)]) == phrase for i in range(len(terms))
        ))
        if score:
            candidates.append((score, category))
    ranked = [category for _, category in sorted(candidates, key=lambda item: -item[0])]
    if tuition_programs(question):
        first = "Tuition Fees and Payment"
        ranked = [first, *(category for category in ranked if category != first)]
    return ranked[:3]


def private_request(question):
    normalized = question.casefold()
    if SECRET.search(normalized):
        return True
    if GENERAL_NAVIGATION.search(normalized):
        return False
    return bool(LOOKUP.search(normalized) and (PERSONAL.search(normalized)
                or re.search(r"\bstudent\s+(?:id\s*[:#]?\s*)?[0-9]{3,}\b|ছাত্র আইডি", normalized)))


def requires_clarification(question):
    return bool(CLARIFY.fullmatch(question.strip()))


def safe_question(question):
    return "[credential removed]" if SECRET.search(question) else question


def localized(language, english, bangla, banglish):
    return {"bn": bangla, "banglish": banglish}.get(language, english)


def navigation_question(question):
    """Answer unmistakable requests for a public site without model calls."""
    norm = question.casefold()
    if not re.search(r"\b(where|who|email|link|kothay|kake)\b|কোথায়|লিংক|ইমেইল", norm):
        return None
    candidates = (
        (r"(?:email.*(?:admission|applying)|(?:admission|applying).*email)|ভর্তির বিষয়ে.*ইমেইল", "admission",
         ("Email admission@seu.edu.bd for admission questions.", "ভর্তি প্রশ্নের জন্য admission@seu.edu.bd-এ ইমেইল করুন।", "Admission question er jonno admission@seu.edu.bd e email korun.")),
        (r"\b(apply for admission|vorti application)\b|ভর্তির আবেদন", "admission",
         ("Apply through the SEU Online Admission portal.", "এসইউর অনলাইন ভর্তি পোর্টালে আবেদন করুন।", "SEU Online Admission portal diye apply korun.")),
        (r"\b(ums is down|ums down|ums status)\b|ইউএমএস বন্ধ", "status",
         ("Check the official SEU Software Services Status page for current UMS availability.", "ইউএমএসের বর্তমান অবস্থা অফিসিয়াল সফটওয়্যার স্ট্যাটাস পেজে দেখুন।", "UMS er current obostha official software status page e dekhun.")),
        (r"\b(library books online|boi online|library catalog)\b|লাইব্রেরির বই অনলাইনে", "library",
         ("Search for books in the SEU Library catalog.", "এসইউ লাইব্রেরি ক্যাটালগে বই খুঁজুন।", "SEU Library catalog e boi khujun.")),
    )
    hits = [(key, messages) for pattern, key, messages in candidates if re.search(pattern, norm)]
    return hits[0] if len(hits) == 1 else None


def replacement_certificate_request(question):
    normalized = question.casefold()
    return bool(re.search(r"\b(replace|replacement|lost|duplicate|harano)\b|হারানো|হারিয়ে|প্রতিস্থাপন", normalized)
                and re.search(r"\b(certificate|shonod)\b|সনদ", normalized))

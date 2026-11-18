"""Recognize program names separately from the intent in a student question."""
import re

PROGRAM_PATTERNS = (
    (("CSE",), r"cse|computer science(?: and engineering)?|সিএসই"),
    (("BBA",), r"bba|bachelor of business administration|বিবিএ"),
    (("EEE",), r"eee|electrical (?:and|&) electronic engineering|ইইই"),
    (("EMBA",), r"e\.?m\.?b\.?a\.?|executive mba|mba executive|এমবিএ এক্সিকিউটিভ|ইএমবিএ"),
    (("MBA_ONE_YEAR",), r"mba\s*(?:\(\s*)?(?:one[ -]year|1[ -]year)|(?:one[ -]year|1[ -]year) mba"),
    (("MBA_REGULAR",), r"mba\s*(?:\(\s*)?regular|regular mba"),
    (("MBA_REGULAR", "MBA_ONE_YEAR"), r"mba|master of business administration|এমবিএ"),
    (("LLB",), r"ll\.?b\.?|bachelor of laws|এলএলবি"),
    (("LLM",), r"ll\.?m\.?|master of laws|এলএলএম"),
    (("MDS",), r"mds|master of development studies"),
    (("PHARM",), r"b\.?\s*pharm(?:acy)?|pharmacy|ফার্মেসি|বি\.?\s*ফার্ম"),
    (("ARCH",), r"(?:b\.?\s*)?arch(?:itecture)?|আর্কিটেকচার|স্থাপত্য"),
    (("TEXTILE",), r"textile(?: engineering)?|টেক্সটাইল(?: ইঞ্জিনিয়ারিং)?"),
    (("ECONOMICS",), r"economics|অর্থনীতি"),
    (("ENGLISH_MA",), r"m\.?a\.? (?:in )?english|এমএ ইংরেজি"),
    (("ENGLISH_BA",), r"b\.?a\.? (?:in )?english|বিএ ইংরেজি"),
    (("BANGLA_MA",), r"m\.?a\.? (?:in )?(?:bangla|bengali)|এমএ বাংলা"),
    (("BANGLA_BA",), r"b\.?a\.? (?:in )?(?:bangla|bengali)|বিএ বাংলা"),
    (("ENGLISH_BA", "ENGLISH_MA"), r"english|ইংরেজি"),
    (("BANGLA_BA", "BANGLA_MA"), r"bangla|bengali|বাংলা"),
)
PATTERNS = tuple((codes, re.compile(r"(?<![a-z0-9])(?:" + text + r")(?![a-z0-9])", re.I))
                 for codes, text in PROGRAM_PATTERNS)
PRICE = re.compile(r"\b(tuition|fees?|cost|per[ -]credit|khoroch)\b|ফি|খরচ", re.I)
OTHER_TOPIC = re.compile(
    r"\b(waivers?|scholarships?|refunds?|deadline|installments?|salary|jobs?|certifications?|"
    r"software|interview|bootcamp|online course|coaching)\b|ওয়েভার|বৃত্তি|সময়সীমা|বেতন", re.I,
)
ACADEMIC_TOPIC = re.compile(
    r"\b(admission|entry requirements?|registrations?|register|advising|prerequisites?|"
    r"exams?|grading|cgpa|faculty|departments?|curriculum|credits?|scholarships?|waivers?)\b"
    r"|ভর্তি|নিবন্ধন|অ্যাডভাইজিং|পরীক্ষা|গ্রেড|শিক্ষক|বিভাগ|ওয়েভার|বৃত্তি", re.I,
)


def identify_programs(question):
    found = []
    occupied = []
    for codes, pattern in PATTERNS:
        for match in pattern.finditer(question):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            occupied.append(match.span())
            found.append((match.start(), codes))
    return tuple(dict.fromkeys(code for _, codes in sorted(found) for code in codes))


JOINING = re.compile(
    r"\b(?:when joining|to join|when enrolling|at (?:the time of )?admission|"
    r"upfront|initial payment|pay when (?:i )?enroll|vorti\s+(?:hote|howar somoy)|"
    r"bhorti\s+(?:hote|howar somoy))\b|ভর্তি\s*(?:হতে|হওয়ার সময়|সময়)", re.I,
)
PAYMENT_QUESTION = re.compile(r"\b(how much|pay|payment|koto|taka|lagbe)\b|কত|টাকা|দিতে", re.I)
FEE_TYPES = (
    ("tuition_deposit", r"\b(?:initial\s+)?tuition deposit\b|টিউশন জমা|tuition deposit"),
    ("application_fee", r"\b(?:application|admission form) (?:fees?|charges?|cost)\b|আবেদন ফি|ভর্তি ফরমের ফি"),
    ("admission_fee", r"\b(?:one[ -]time )?(?:admission|enrollment|vorti|bhorti) fees?\b|ভর্তি\s*ফি"),
    ("tuition_per_credit", r"\bper[ -]credit\b|প্রতি\s*ক্রেডিট"),
    ("semester_tuition", r"\b(?:semester(?:'s)? tuition|tuition (?:per|for (?:a|one|this)) semester|"
     r"(?:per|one) semester|semester (?:fees?|cost)|(?:fees?|cost) (?:in|for) (?:a|one|this) semester)\b|সেমিস্টারের টিউশন"),
    ("total_tuition", r"\b(?:total (?:program )?tuition|entire tuition)\b|মোট টিউশন"),
    ("total_cost", r"\b(?:total (?:program )?cost|(?:full|overall) (?:program )?cost|cost of studying)\b|মোট খরচ"),
)
TYPED_FEES = tuple((kind, re.compile(pattern, re.I)) for kind, pattern in FEE_TYPES)


def classify_fee(question):
    """Return the requested fee concept and program codes, or an empty request."""
    if OTHER_TOPIC.search(question):
        return None, ()
    programs = identify_programs(question)
    for kind, pattern in TYPED_FEES:
        if pattern.search(question):
            if programs or kind in {"admission_fee", "application_fee"}:
                return kind, programs
    if programs and JOINING.search(question) and (PRICE.search(question) or PAYMENT_QUESTION.search(question)):
        return "initial_payment", programs
    if programs and PRICE.search(question):
        return "program_overview", programs
    return None, ()


def fee_record_programs(kind, programs):
    if kind in {"admission_fee", "application_fee"}:
        return ("ADMISSION",)
    if kind == "initial_payment":
        return (*programs, "ADMISSION")
    return programs


def tuition_programs(question):
    kind, programs = classify_fee(question)
    return fee_record_programs(kind, programs)

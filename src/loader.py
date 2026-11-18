"""Load archive records and optional local, text-based PDF material."""
import json
import re
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "old_seu_website_cleaned.jsonl"


VERIFIED_STUDENT_PATH = DEFAULT_DATA_PATH.with_name("verified_student_sources.jsonl")
OFFICIAL_UPDATES_PATH = DEFAULT_DATA_PATH.with_name("official_updates.jsonl")


def read_records(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        if Path(path).suffix.lower() == ".json":
            records = json.load(source)
            if not isinstance(records, list):
                raise ValueError("JSON sources must contain an array of records.")
            yield from records
        else:
            for line_number, line in enumerate(source, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"Invalid corpus JSON at line {line_number}") from exc


def load_university_data(json_path: str | Path = DEFAULT_DATA_PATH):
    from langchain_core.documents import Document

    path = Path(json_path)
    documents = []
    for number, record in enumerate(read_records(path), start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Corpus record {number} must be an object.")
        metadata = {key: record.get(key) for key in
                    ("id", "source_url", "title", "category", "updated_at", "retrieved_at", "verification", "search_aliases", "admission_session")}
        if any(value is not None and not isinstance(value, (str, int)) for value in metadata.values()):
            raise ValueError(f"Invalid metadata for corpus record {number}.")
        if isinstance(record.get("fee_schedule"), dict):
            metadata["fee_schedule"] = record["fee_schedule"]
        if isinstance(record.get("program"), str):
            metadata["program"] = record["program"]
        if isinstance(record.get("fee_type"), list) and all(isinstance(value, str) for value in record["fee_type"]):
            metadata["fee_type"] = record["fee_type"]
        for key in ("academic_session", "effective_date", "document_type", "support_topic", "guidance_gap",
                    "superseded_by"):
            if isinstance(record.get(key), str):
                metadata[key] = record[key]
        if metadata.get("fee_schedule") and not metadata.get("admission_session") and metadata.get("academic_session"):
            metadata["admission_session"] = metadata["academic_session"]
        if isinstance(record.get("supersedes"), list):
            metadata["supersedes"] = [str(value) for value in record["supersedes"]]
        title = metadata["title"]
        pdf_path = record.get("pdf_path")
        if pdf_path:
            from pypdf import PdfReader

            local = (path.parent / pdf_path).resolve()
            if not local.is_relative_to(path.parent.resolve()) or local.suffix.lower() != ".pdf":
                raise ValueError(f"Invalid PDF path for corpus record {number}.")
            bodies = [(page.extract_text() or "", page_number)
                      for page_number, page in enumerate(PdfReader(local).pages, start=1)]
        else:
            content = record.get("clean_text")
            if not isinstance(content, str):
                raise ValueError(f"Corpus record {number} requires clean_text or pdf_path.")
            bodies = [(content, None)]
            if metadata.get("support_topic") and metadata.get("verification") == "official_public":
                metadata["guidance_text"] = content
        for text, page in bodies:
            clean = re.sub(r"[^\S\r\n]+", " ", text).strip()
            if not clean:
                continue
            fields = {**metadata, "page": page} if page else metadata
            documents.append(Document(
                page_content=f"{title}\n\n{clean}" if title else clean, metadata=fields,
            ))
    return documents

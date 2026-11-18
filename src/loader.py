"""Load archive records and optional local, text-based PDF material."""
import json
import re
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "old_seu_website_cleaned.jsonl"


VERIFIED_STUDENT_PATH = DEFAULT_DATA_PATH.with_name("verified_student_sources.jsonl")


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
                    ("id", "source_url", "title", "category", "updated_at", "retrieved_at", "verification", "search_aliases")}
        if any(value is not None and not isinstance(value, (str, int)) for value in metadata.values()):
            raise ValueError(f"Invalid metadata for corpus record {number}.")
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
        for text, page in bodies:
            clean = re.sub(r"[^\S\r\n]+", " ", text).strip()
            if not clean:
                continue
            fields = {**metadata, "page": page} if page else metadata
            documents.append(Document(
                page_content=f"{title}\n\n{clean}" if title else clean, metadata=fields,
            ))
    return documents

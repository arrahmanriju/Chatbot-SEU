from datetime import date
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.freshness import potentially_outdated
from src.loader import load_university_data, read_records


class IngestionTests(unittest.TestCase):
    def test_archive_has_real_source_metadata(self):
        rows = list(read_records(Path(__file__).parents[1] / "data" / "old_seu_website_cleaned.jsonl"))
        self.assertTrue(any(row.get("source_url") == "https://seu.edu.bd/vision" for row in rows))

    def test_json_and_jsonl_preserve_metadata(self):
        record = {"title": "Academic Calendar", "category": "academic_calendar", "updated_at": "2026-10-01"}
        with TemporaryDirectory() as directory:
            for extension, value in (("json", [record]), ("jsonl", record)):
                path = Path(directory, f"calendar.{extension}")
                path.write_text(json.dumps(value), encoding="utf-8")
                self.assertEqual(list(read_records(path)), [record])

    def test_missing_and_old_dates_are_flagged(self):
        today = date(2026, 11, 18)
        self.assertTrue(potentially_outdated(None, today))
        self.assertTrue(potentially_outdated("2024-01-01", today))
        self.assertFalse(potentially_outdated("2026-10-01", today))

    @unittest.skipUnless(find_spec("pypdf") and find_spec("langchain_core"), "PDF dependencies unavailable")
    def test_pdf_text_and_page_metadata(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=200)
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            text = DecodedStreamObject()
            text.set_data(b"BT /F1 12 Tf 20 40 Td (Teaching starts after orientation.) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(text)
            writer.write(root / "calendar.pdf")

            manifest = root / "sources.json"
            manifest.write_text(json.dumps([{
                "pdf_path": "calendar.pdf", "title": "Calendar", "category": "academic_calendar",
            }]), encoding="utf-8")
            documents = load_university_data(manifest)
            self.assertIn("Teaching starts after orientation.", documents[0].page_content)
            self.assertEqual(documents[0].metadata["page"], 1)
            self.assertEqual(documents[0].metadata["category"], "academic_calendar")

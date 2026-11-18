import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.update_knowledge_base import approved_record, digest, merge, PageText
from src.support import support_reply


def record(key="seu-test", **values):
    return {"id": key, "title": "Official notice", "source_url": "https://seu.edu.bd/notice",
            "clean_text": "A reviewed official document with enough substantive text for a test.", **values}


class KnowledgeUpdateTests(unittest.TestCase):
    def test_repeated_addition_and_update_preserves_old_text(self):
        first = approved_record(record(), False, False)
        result, events = merge([], [], [first])
        self.assertEqual(events[0][1], "added")
        revised = {**first, "clean_text": "A genuinely revised statement from the same official documentation."}
        result, events = merge(result, [], [revised])
        self.assertEqual(events[0][1], "updated")
        self.assertEqual(result[1]["clean_text"], first["clean_text"])
        self.assertEqual(result[1]["superseded_by"], first["id"])
        same, events = merge(result, [], [revised])
        self.assertEqual(events[0][1], "skipped")
        self.assertEqual(same, result)

    def test_duplicate_and_immutable_archive(self):
        first = approved_record(record(), False, False)
        result, events = merge([], [first], [first, {**first, "id": "seu-test-copy"}])
        self.assertEqual([event[1] for event in events], ["failed", "skipped"])
        self.assertEqual(result, [])
        revised = approved_record(record("seu-replacement", supersedes=[first["id"]]), False, False)
        result, events = merge([], [first], [revised])
        self.assertEqual(events[0][1], "added")
        self.assertEqual(result[0]["supersedes"], [first["id"]])

    def test_rejects_nonofficial_url(self):
        row = {**record(), "source_url": "https://seu.edu.bd.example.com/notice"}
        with self.assertRaisesRegex(ValueError, "official"):
            approved_record(row, False, False)

    def test_clean_html_excludes_forms_and_scripts(self):
        parser = PageText()
        parser.feed("<nav>menu</nav><main><p>Approved text</p><form><input>private field</form>"
                    "<script>code</script><p>Second paragraph</p></main>")
        text = "".join(parser.parts)
        self.assertIn("Approved text", text)
        self.assertIn("Second paragraph", text)
        for omitted in ("menu", "private field", "code"):
            self.assertNotIn(omitted, text)

    def test_scoped_guidance_preserves_uncertainty(self):
        retriever = Mock()
        retriever.search_support.return_value = SimpleNamespace(metadata={
            **record("seu-test-guide"), "verification": "official_public",
            "guidance_text": "UMS lists a recommendation-letter option.",
            "guidance_gap": "The forwarding-letter process could not be verified.",
        })
        answer, citation, route = support_reply(retriever, "How can I apply for an internship forwarding letter?")
        self.assertEqual(route, "unverified")
        self.assertIn("could not be verified", answer)
        self.assertTrue(citation["url"].startswith("https://seu.edu.bd/"))
        retriever.search_support.assert_called_once_with("internship_forwarding_letter")



    def test_download_must_match_reviewed_digest_on_apply(self):
        row = record(fetch=True)
        content = "Reviewed public page content with a verified topic and sufficient text."
        with patch("scripts.update_knowledge_base.collect", return_value=content):
            preview = approved_record(row, True, False)
            self.assertEqual(preview["clean_text"], content)
            with self.assertRaisesRegex(ValueError, "expected_sha256"):
                approved_record(row, True, True)
            accepted = approved_record({**row, "expected_sha256": digest(content)}, True, True)
            self.assertEqual(accepted["clean_text"], content)

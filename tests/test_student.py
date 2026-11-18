from collections import Counter
import unittest
from unittest.mock import Mock

from src.loader import read_records, VERIFIED_STUDENT_PATH
from src.service import ChatService
from src.student import DATA, language_of, private_request, safe_question


class StudentTests(unittest.TestCase):
    def test_generated_coverage_and_split_boundaries(self):
        rows = list(read_records(DATA / "student_questions.jsonl"))
        self.assertEqual(sorted(Counter(row["category"] for row in rows).values()), [15] * 15)
        self.assertEqual(len({row["id"] for row in rows}), 225)
        known_sources = {row["id"] for row in read_records(VERIFIED_STUDENT_PATH)}
        self.assertTrue(all(set(row["source_ids"]) <= known_sources for row in rows))
        development = {row["group_id"] for row in rows if row["split"] == "development"}
        test = {row["group_id"] for row in rows if row["split"] == "test"}
        self.assertTrue(development.isdisjoint(test))
        evaluated = list(read_records(DATA / "evaluation_dataset.jsonl"))
        self.assertTrue(all(row["reference_answer"] is None or row["expected_source_ids"] for row in evaluated))

    def test_private_lookup_and_secret_bypass_llm(self):
        search, generate = Mock(), Mock()
        service = ChatService(search, generate)
        for question in ("What is my latest CGPA?", "আমার উপস্থিতির হার কত?",
                         "Amar payment baki koto?", "password: example-credential"):
            self.assertEqual(service.chat(question).route, "private_record")
        generate.assert_not_called()
        search.search.assert_not_called()
        self.assertNotIn("example-credential", safe_question("password: example-credential"))
        self.assertFalse(private_request("Where can I view my grade?"))

    def test_navigation_has_official_citation_without_llm(self):
        search, generate = Mock(), Mock()
        reply = ChatService(search, generate).chat("Vorti application kothay korbo?")
        self.assertEqual(language_of("Vorti application kothay korbo?"), "banglish")
        self.assertEqual(reply.citations[0]["source_id"], "seu-admission")
        generate.assert_not_called()
        search.search.assert_not_called()

    def test_clarifies_unspecified_semester(self):
        service = ChatService(Mock(), Mock())
        self.assertEqual(service.chat("When does my next semester start?").route, "clarification")

    def test_lost_certificate_does_not_invent_replacement_rules(self):
        search, generate = Mock(), Mock()
        reply = ChatService(search, generate).chat("Which documents replace a lost certificate?")
        self.assertEqual(reply.route, "unverified")
        self.assertEqual(reply.citations[0]["source_id"], "seu-degree")
        generate.assert_not_called()
        search.search.assert_not_called()

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.filter import has_university_context
from src.loader import VERIFIED_STUDENT_PATH, read_records
from src.service import ChatService


class TuitionTests(unittest.TestCase):
    def setUp(self):
        self.record = next(row for row in read_records(VERIFIED_STUDENT_PATH)
                           if row["id"] == "seu-cse-tuition-fall-2025")
        self.search = Mock()
        self.search.search.return_value = [SimpleNamespace(
            document=SimpleNamespace(metadata=self.record, page_content=self.record["clean_text"])
        )]
        self.generate = Mock()
        self.chat = ChatService(self.search, self.generate)

    def test_fee_paraphrases_use_retrieved_figures(self):
        questions = (
            "Can you share the tuition fee of CSE?",
            "What is the tuition fee of CSE?",
            "How much does CSE cost at SEU?",
            "CSE per credit fee koto?",
            "CSE এর টিউশন ফি কত?",
            "What is the total cost of studying CSE?",
        )
        for question in questions:
            with self.subTest(question=question):
                reply = self.chat.chat(question)
                self.assertEqual(reply.route, "university")
                for figure in ("4,750", "150", "712,500", "777,500"):
                    self.assertIn(figure, reply.answer)
                self.assertIn("Fall 2025", reply.answer)
                self.assertEqual(reply.citations[0]["url"], self.record["source_url"])
                self.assertEqual(reply.citations[0]["admission_session"], "Fall 2025")
        self.assertEqual(self.search.search.call_count, len(questions))
        self.generate.assert_not_called()

    def test_no_verified_fee_means_no_figure(self):
        self.search.search.return_value = []
        reply = self.chat.chat("What is the tuition fee of CSE?")
        self.assertEqual(reply.route, "unverified")
        self.assertNotIn("4,750", reply.answer)
        self.generate.assert_not_called()

    def test_other_sessions_cannot_inherit_fall_2025_figures(self):
        for question in ("CSE fees for Fall 2026?", "CSE tuition in Spring 2025?",
                         "স্প্রিং ২০২৫ এর সিএসই ফি কত?"):
            with self.subTest(question=question):
                reply = self.chat.chat(question)
                self.assertEqual(reply.route, "unverified")
                self.assertNotIn("4,750", reply.answer)

    def test_other_university_cannot_inherit_seu_price(self):
        reply = self.chat.chat("What is the CSE tuition fee at BRAC University?")
        self.assertEqual(reply.route, "out_of_scope")
        self.search.search.assert_not_called()

    def test_student_topics_do_not_require_university_name(self):
        for question in ("Admission documents?", "Can I register for courses?", "How is CGPA calculated?",
                         "How do waivers work?", "Where is the faculty directory?",
                         "When are exams?", "Where is the UMS portal?"):
            self.assertTrue(has_university_context(question), question)
        self.assertFalse(has_university_context("Who won the cricket match?"))

    def test_student_question_with_weak_evidence_is_not_off_topic(self):
        self.generate.return_value = json.dumps({"status": "out_of_scope"})
        reply = self.chat.chat("How is CGPA calculated?")
        self.assertEqual(reply.route, "unverified")
        self.generate.assert_called_once()

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.service import ChatService


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.retriever.search.return_value = [
            SimpleNamespace(document=SimpleNamespace(
                page_content="Archived text",
                metadata={"title": "SEU", "source_url": "https://seu.edu.bd/"},
            ))
        ]
        self.generate = Mock()
        self.service = ChatService(self.retriever, self.generate)

    def test_invalid_source_cannot_produce_an_answer(self):
        self.generate.return_value = '{"status":"answered","answer":"Invented","source_ids":[9]}'
        self.assertEqual(self.service.chat("SEU admissions?").route, "unverified")

    def test_valid_source_preserves_url(self):
        self.generate.return_value = '{"status":"answered","answer":"An answer","source_ids":[1]}'
        reply = self.service.chat("SEU admissions?")
        self.assertEqual(reply.citations[0]["url"], "https://seu.edu.bd/")

    def test_follow_up_is_rewritten_before_retrieval(self):
        self.generate.side_effect = [
            '{"question":"What are the CSE fees?"}',
            '{"status":"unverified"}',
            '{"status":"unverified"}',
        ]
        self.service.chat("And the fees?", [("Tell me about CSE", "CSE information")])
        self.retriever.search.assert_called_once_with("What are the CSE fees?", 15)

    def test_bangla_greeting_skips_inference(self):
        self.assertEqual(self.service.chat("হ্যালো!").route, "conversation")
        self.generate.assert_not_called()
        self.retriever.search.assert_not_called()

    def test_no_evidence_can_still_route_unrelated_questions(self):
        self.retriever.search.return_value = []
        self.generate.return_value = '{"status":"out_of_scope"}'
        result = self.service.chat("How does soccer offside work?")
        self.assertEqual(result.route, "out_of_scope")
        self.assertFalse(result.citations)

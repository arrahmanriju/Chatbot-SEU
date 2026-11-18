from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.chain import ERROR_REPLY, GREETING_REPLY, UNVERIFIED_REPLY, run_chatbot
from src.filter import classify_message
from src.store import cosine_relevance, load_index


class ChatTests(unittest.TestCase):
    def test_greeting_skips_retrieval(self):
        store = Mock()
        self.assertEqual(run_chatbot(" Hello! ", store), GREETING_REPLY)
        store.similarity_search_with_relevance_scores.assert_not_called()

    def test_university_question_with_help_is_not_a_greeting(self):
        self.assertIsNone(classify_message("Can you help me with admissions?"))

    def test_missing_evidence_skips_groq(self):
        store = Mock()
        store.similarity_search_with_relevance_scores.return_value = []
        with patch("src.chain.ask_llm") as ask:
            self.assertEqual(run_chatbot("Which SEU degrees are available?", store), UNVERIFIED_REPLY)
        store.similarity_search_with_relevance_scores.assert_called_once()
        ask.assert_not_called()

    def test_retrieval_failure_is_safe(self):
        store = Mock()
        store.similarity_search_with_relevance_scores.side_effect = RuntimeError("secret")
        with self.assertLogs("src.chain", level="ERROR") as logs:
            self.assertEqual(run_chatbot("SEU degrees?", store), ERROR_REPLY)
        self.assertNotIn("secret", "\n".join(logs.output))

    def test_success_reuses_retrieved_documents(self):
        store = Mock()
        doc = Mock(page_content="A source excerpt", metadata={"title": "SEU"})
        store.similarity_search_with_relevance_scores.return_value = [(doc, 0.8)]
        with patch("src.chain.ask_llm", return_value="An answer") as ask:
            self.assertEqual(run_chatbot("SEU degrees?", store), "An answer")
        store.similarity_search_with_relevance_scores.assert_called_once()
        self.assertIn("A source excerpt", ask.call_args.args[0])


class IndexTests(unittest.TestCase):
    def test_legacy_pickle_requires_rebuild(self):
        with TemporaryDirectory() as directory:
            Path(directory, "index.pkl").write_bytes(b"legacy")
            with self.assertRaisesRegex(FileNotFoundError, "python -m src.store"):
                load_index(directory)

    def test_cosine_relevance_from_squared_l2(self):
        self.assertEqual(cosine_relevance(0), 1)
        self.assertEqual(cosine_relevance(1), 0.5)
        self.assertEqual(cosine_relevance(4), 0)

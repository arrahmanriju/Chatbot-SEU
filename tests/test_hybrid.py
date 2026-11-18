from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.hybrid import HybridRetriever, fuse_rankings, tokenize


class HybridTests(unittest.TestCase):
    def test_consensus_rises_above_single_rank(self):
        self.assertEqual(fuse_rankings([["a", "b"], ["c", "b"]]), ["b", "a", "c"])

    def test_duplicate_does_not_get_an_extra_vote(self):
        self.assertEqual(fuse_rankings([["a", "b", "b"], ["c"]]), ["a", "c", "b"])

    def test_bangla_combining_marks_are_preserved(self):
        self.assertEqual(tokenize("ভর্তি CSE?"), ["ভর্তি", "cse"])

    @unittest.skipUnless(find_spec("rank_bm25"), "rank-bm25 is unavailable")
    def test_named_page_is_retrieved_even_if_dense_search_misses_it(self):
        vision = SimpleNamespace(page_content="Vision: global recognition", metadata={"title": "Vision"})
        campus = SimpleNamespace(page_content="Campus location and student life", metadata={"title": "Campus"})
        store = Mock()
        store.index_to_docstore_id = {0: "vision", 1: "campus"}
        store.docstore.search.side_effect = {"vision": vision, "campus": campus}.get
        store.similarity_search_with_relevance_scores.return_value = [(campus, 0.8)]
        result = HybridRetriever(store).search("What is the university vision?", limit=1)
        self.assertIs(result[0].document, vision)

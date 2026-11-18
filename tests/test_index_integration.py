from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.store import build_vectorstore, load_index


def native_dependencies_available():
    if not all(find_spec(name) for name in ("faiss", "langchain_community", "langchain_text_splitters")):
        return False
    try:
        import faiss
    except ImportError:
        return False
    return True


@unittest.skipUnless(native_dependencies_available(), "Native FAISS dependencies are unavailable")
class IndexIntegrationTests(unittest.TestCase):
    def test_build_load_and_skip_unchanged_index(self):
        from langchain_core.embeddings import Embeddings

        class FixtureEmbeddings(Embeddings):
            builds = 0

            def embed_documents(self, texts):
                self.builds += 1
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                return [1.0, 0.0]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources.json"
            sources.write_text(json.dumps([{
                "title": "Calendar", "source_url": "https://archive.example.edu/calendar",
                "clean_text": "This archived calendar contains the teaching dates for the example semester.",
            }]), encoding="utf-8")
            embeddings = FixtureEmbeddings()
            with patch("src.store.get_embedding_model", return_value=embeddings):
                build_vectorstore(root, sources)
                build_vectorstore(root, sources)
                store = load_index(root)
                results = store.similarity_search_with_relevance_scores("calendar", k=1)
            self.assertEqual(embeddings.builds, 1)
            self.assertEqual(results[0][0].metadata["title"], "Calendar")
            self.assertEqual(results[0][1], 1.0)

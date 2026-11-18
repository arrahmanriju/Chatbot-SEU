from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.embedder import deferred_embeddings
from src.runtime import IndexedRetriever, load_or_build_index


class ResourceTests(unittest.TestCase):
    def test_concurrent_questions_encode_once_and_cache_is_bounded(self):
        model = Mock()
        model.embed_query.return_value = [0.5, 0.2]
        get_model = Mock(return_value=model)
        vectors = deferred_embeddings(get_model)
        get_model.assert_not_called()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(vectors.embed_query, ["same question"] * 8))
        model.embed_query.assert_called_once_with("same question")
        results[0][0] = 0
        self.assertEqual(vectors.embed_query("same question"), [0.5, 0.2])
        for number in range(130):
            vectors.embed_query(str(number))
        self.assertEqual(len(vectors.vectors), 128)

    def test_corpus_change_and_index_replacement_invalidate_once(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            corpus = directory / "sources.jsonl"
            corpus.write_text("old", encoding="utf8")
            resource = IndexedRetriever(directory)
            with (patch("src.runtime.corpus_paths", return_value=(corpus,)),
                  patch("src.runtime.load_or_build_index") as load,
                  patch("src.runtime.create_retriever", side_effect=lambda _: object())):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    snapshots = list(pool.map(lambda _: resource.get(), range(8)))
                self.assertTrue(all(item is snapshots[0] for item in snapshots))
                load.assert_called_once()
                corpus.write_text("new content", encoding="utf8")
                second = resource.get()
                self.assertIsNot(snapshots[0], second)
                self.assertIs(second, resource.get())
                self.assertEqual(load.call_count, 2)
                (directory / "index.zip").write_bytes(b"published")
                self.assertIsNot(second, resource.get())
                self.assertEqual(load.call_count, 3)

    def test_changed_bundled_fingerprint_rebuilds_existing_artifact(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "index.zip").touch()
            with (patch("src.runtime.read_manifest", return_value={"corpus_sha256": "old"}),
                  patch("src.runtime.corpus_fingerprint", return_value="new"),
                  patch("src.runtime.build_vectorstore") as build,
                  patch("src.runtime.load_index") as load):
                self.assertIs(load_or_build_index(directory, check_corpus=True), build.return_value)
                build.assert_called_once_with(directory)
                load.assert_not_called()

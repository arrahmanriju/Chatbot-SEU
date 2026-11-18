import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.runtime import load_or_build_index

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:
    AppTest = None

APP = Path(__file__).resolve().parents[1] / "app.py"


class IndexStartupTests(unittest.TestCase):
    def test_missing_directory_uses_existing_corpus_builder(self):
        with TemporaryDirectory() as temp:
            index = Path(temp) / "uncreated-index"
            with patch("src.runtime.build_vectorstore") as build:
                self.assertIs(load_or_build_index(index), build.return_value)
            build.assert_called_once_with(index)

    def test_existing_artifact_loads_without_building(self):
        with TemporaryDirectory() as temp:
            index = Path(temp)
            (index / "index.zip").touch()
            with patch("src.runtime.load_index") as load, patch("src.runtime.build_vectorstore") as build:
                self.assertIs(load_or_build_index(index), load.return_value)
            load.assert_called_once_with(index)
            build.assert_not_called()


@unittest.skipIf(AppTest is None, "Streamlit is unavailable")
class StreamlitStartupTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def test_streamlit_caches_service_and_reads_secret(self):
        with TemporaryDirectory() as temp:
            env = {"SEU_INDEX_DIR": str(Path(temp) / "missing"), "SEU_API_URL": "", "GROQ_API_KEY": ""}
            with (patch.dict(os.environ, env), patch.object(type(st.secrets), "get", return_value="test-key"),
                  patch("src.runtime.load_or_build_index", return_value=Mock()) as load):
                app = AppTest.from_file(str(APP)).run()
                self.assertFalse(app.exception)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(os.environ["GROQ_API_KEY"], "test-key")
                load.assert_called_once()

    def test_builder_failure_is_clear_and_redacted(self):
        with TemporaryDirectory() as temp:
            env = {"SEU_INDEX_DIR": temp, "SEU_API_URL": "", "GROQ_API_KEY": "test-key"}
            with (patch.dict(os.environ, env),
                  patch("src.runtime.load_or_build_index", side_effect=OSError("private value"))):
                app = AppTest.from_file(str(APP)).run()
            self.assertFalse(app.exception)
            self.assertIn("knowledge base could not be built or loaded", app.error[0].value)
            self.assertNotIn("private value", app.error[0].value)

    def test_optional_http_mode_does_not_load_local_index(self):
        with (patch.dict(os.environ, {"SEU_API_URL": "http://backend.example", "GROQ_API_KEY": ""}),
              patch("src.runtime.load_or_build_index") as load):
            app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        load.assert_not_called()

    def test_missing_dependency_points_to_cloud_requirements(self):
        with TemporaryDirectory() as temp:
            env = {"SEU_INDEX_DIR": temp, "SEU_API_URL": "", "GROQ_API_KEY": "test-key"}
            missing = ModuleNotFoundError(name="langchain_community")
            with (patch.dict(os.environ, env),
                  patch("src.runtime.load_or_build_index", side_effect=missing)):
                app = AppTest.from_file(str(APP)).run()
            self.assertFalse(app.exception)
            self.assertIn("root requirements.txt", app.error[0].value)
            self.assertIn("Python 3.11", app.error[0].value)

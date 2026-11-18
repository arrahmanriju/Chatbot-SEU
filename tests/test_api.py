import os
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

try:
    from fastapi.testclient import TestClient
    from src.api import app
except ModuleNotFoundError:
    TestClient = None

from src.service import Answer


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are unavailable")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_request_validation(self):
        self.assertEqual(self.client.post("/chat", json={"question": " "}).status_code, 422)

    def test_chat_response(self):
        service = Mock()
        service.chat.return_value = Answer("Hello", "conversation")
        with patch("src.api.get_service", return_value=service):
            reply = self.client.post("/chat", json={"question": "Hello"})
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.json()["route"], "conversation")
        self.assertIn("request_id", reply.json())

    def test_feedback_can_be_retried(self):
        with TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"FEEDBACK_DB": f"{directory}/feedback.sqlite"}):
                payload = {"request_id": str(uuid4()), "helpful": False}
                for _ in range(2):
                    self.assertEqual(self.client.post("/feedback", json=payload).json(), {"saved": True})

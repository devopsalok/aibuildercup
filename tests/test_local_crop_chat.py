import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from src.assistant.local_crop_chat import (
    CropChatError,
    ask_local_crop_assistant,
    retrieve_crop_guidance,
)


ROOT = Path(__file__).resolve().parents[1]


class LocalCropChatTests(unittest.TestCase):
    def test_retrieval_selects_local_crop_profile_and_relevant_guidance(self) -> None:
        guidance = retrieve_crop_guidance(
            "How should I manage irrigation during rice flowering?",
            crop="rice",
        )
        self.assertIn("rice:", guidance)
        self.assertIn("### Irrigation", guidance)

    @patch("src.assistant.local_crop_chat.urlopen")
    def test_chat_request_is_local_grounded_and_includes_chat_history(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = BytesIO(
            json.dumps({"message": {"content": "Check the soil and forecast first."}}).encode()
        )
        answer = ask_local_crop_assistant(
            "What should I do next?",
            crop="wheat",
            assessment_context="Water stress: High.",
            history=[{"role": "user", "content": "The field is dry."}],
            model="qwen2.5:3b",
        )
        self.assertEqual(answer, "Check the soil and forecast first.")
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][1]["content"], "The field is dry.")
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("Water stress: High.", system_prompt)
        self.assertIn("Crop profile", system_prompt)

    @patch("src.assistant.local_crop_chat.urlopen", side_effect=URLError("refused"))
    def test_connection_failure_explains_local_service_requirement(self, mocked_urlopen) -> None:
        with self.assertRaisesRegex(CropChatError, "Start Ollama"):
            ask_local_crop_assistant("How can I improve this field?")
        mocked_urlopen.assert_called_once()

    @patch("src.assistant.local_crop_chat.urlopen")
    def test_missing_model_response_gives_pull_command(self, mocked_urlopen) -> None:
        from urllib.error import HTTPError

        mocked_urlopen.side_effect = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            404,
            "missing model",
            {},
            BytesIO(b"model not found"),
        )
        with self.assertRaisesRegex(CropChatError, "ollama pull qwen2.5:3b"):
            ask_local_crop_assistant("What is the irrigation recommendation?")

    def test_invalid_question_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            ask_local_crop_assistant("   ")
        with self.assertRaisesRegex(ValueError, "2000 characters"):
            ask_local_crop_assistant("x" * 2001)


if __name__ == "__main__":
    unittest.main()

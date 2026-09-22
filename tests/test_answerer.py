import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.answerer.generate import build_messages, generate, load_settings, NoRedirect


class AnswererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"lm_studio_base_url": "http://127.0.0.1:1234/v1", "generation_model": "google/gemma-3-4b"}))
        self.data = {"question": "What happened?", "results": [
            {"source": "book.pdf", "page": 8, "text": "Zoë opened the door.\n"}
        ]}
        self.retrieval = self.root / "results.json"
        self.retrieval.write_text(json.dumps(self.data), encoding="utf-8")

    def test_exact_context_and_citation_mapping(self):
        messages, labels = build_messages(self.data)
        content = json.loads(messages[1]["content"])
        self.assertEqual(content["book_excerpts"][0]["text"], self.data["results"][0]["text"])
        self.assertEqual(labels, ["[book.pdf p. 8]"])
        with self.assertRaises(ValueError):
            build_messages({"question": "Why?", "results": []})

    def test_remote_url_and_redirect_rejected(self):
        self.config.write_text(json.dumps({"lm_studio_base_url": "https://example.com/v1", "generation_model": "x"}))
        with self.assertRaises(ValueError):
            load_settings(self.config)
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, None, None, "https://example.com")

    @patch("src.answerer.generate.post_completion")
    def test_saved_response_and_diagnostics(self, completion):
        completion.return_value = {"model": "google/gemma-3-4b", "choices": [{
            "message": {"content": "Someone arrived [book.pdf p. 99]."}, "finish_reason": "length"
        }]}
        output = self.root / "answer"
        report = generate(self.retrieval, self.config, output)
        self.assertEqual(len(report["warnings"]), 2)
        self.assertTrue((output / "request.json").exists())
        self.assertTrue((output / "response.json").exists())
        self.assertTrue((output / "answer.json").exists())
        with self.assertRaises(FileExistsError):
            generate(self.retrieval, self.config, output)

    @patch("src.answerer.generate.post_completion", side_effect=RuntimeError("offline"))
    def test_failure_preserves_request_without_answer(self, completion):
        output = self.root / "failure"
        with self.assertRaises(RuntimeError):
            generate(self.retrieval, self.config, output)
        self.assertTrue((output / "request.json").exists())
        self.assertFalse((output / "answer.json").exists())


if __name__ == "__main__":
    unittest.main()

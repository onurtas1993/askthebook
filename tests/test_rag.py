import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rag import ask


class RagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "lm_studio_base_url": "http://127.0.0.1:1234/v1",
            "generation_model": "google/gemma-3-4b",
            "rag_index_dir": "indexes/book", "rag_output_dir": "qa", "rag_top_k": 3,
        }))
        self.output = self.root / "run"
        self.load = patch("src.rag.pipeline.load_config").start()
        self.retrieve = patch("src.rag.pipeline.retrieve").start()
        self.generate = patch("src.rag.pipeline.generate").start()
        self.addCleanup(patch.stopall)
        self.retrieve.return_value = {"results": [{"page": 8, "text": "Evidence"}]}
        self.generate.return_value = {"answer": "Answer", "warnings": []}

    def test_composition_and_config_relative_paths(self):
        result = ask("Why?", config_path=self.config, output_dir=self.output)
        self.assertEqual(result["answer"], "Answer")
        self.assertEqual(result["sources"][0]["page"], 8)
        self.assertEqual(self.retrieve.call_args.args[1], self.root / "indexes/book")
        self.assertEqual(self.retrieve.call_args.args[4], 3)
        self.assertEqual(self.generate.call_args.args[0], self.output / "retrieval/results.json")
        self.assertEqual(json.loads((self.output / "run.json").read_text())["status"], "complete")
        self.assertTrue((self.output / "result.json").exists())

    def test_retrieval_failure_never_calls_generation(self):
        self.retrieve.side_effect = RuntimeError("model mismatch")
        with self.assertRaises(RuntimeError):
            ask("Why?", config_path=self.config, output_dir=self.output)
        self.generate.assert_not_called()
        status = json.loads((self.output / "run.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "retrieval"))
        self.assertFalse((self.output / "result.json").exists())

    def test_generation_failure_and_retry_protection(self):
        self.generate.side_effect = RuntimeError("LM Studio unavailable")
        with self.assertRaises(RuntimeError):
            ask("Why?", config_path=self.config, output_dir=self.output)
        status = json.loads((self.output / "run.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "generation"))
        with self.assertRaises(FileExistsError):
            ask("Why?", config_path=self.config, output_dir=self.output)
        self.assertFalse((self.output / "result.json").exists())

    def test_invalid_input_and_unique_default_runs(self):
        with self.assertRaises(ValueError):
            ask(" ", config_path=self.config)
        with self.assertRaises(ValueError):
            ask("Why?", config_path=self.config, top_k=0)
        first = ask("Why?", config_path=self.config)
        second = ask("Why?", config_path=self.config)
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertEqual(Path(first["output_dir"]).parent, self.root / "qa")


if __name__ == "__main__":
    unittest.main()

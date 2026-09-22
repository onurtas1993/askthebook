"""Small synthetic vectors make mapping and similarity errors easy to detect."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.indexer.index import build_index, load_index


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "embeddings"
        self.input.mkdir()
        self.output = self.root / "index"
        self.rows = [
            {"chunk_id": "book.pdf:page:2", "source": "book.pdf", "page": 2,
             "text": "Caf\u00e9", "embedding": [1.0, 0.0]},
            {"chunk_id": "book.pdf:page:5", "source": "book.pdf", "page": 5,
             "text": "Another page", "embedding": [0.0, 1.0]},
        ]
        (self.input / "manifest.json").write_text(json.dumps({
            "chunks": 2, "dimensions": 2, "normalized": True,
        }))
        self.write_rows()

    def write_rows(self):
        (self.input / "embeddings.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8",
        )

    def test_round_trip_and_label_mapping(self):
        build_index(self.input, self.output)
        index, mapping, manifest = load_index(self.output)
        scores, labels = index.search(np.array([[0.6, 0.8]], dtype=np.float32), 2)
        self.assertEqual(labels.tolist(), [[1, 0]])
        np.testing.assert_allclose(scores, [[0.8, 0.6]], atol=1e-6)
        self.assertEqual(mapping[labels[0, 0]]["page"], 5)
        self.assertEqual(mapping[0]["text"], "Caf\u00e9")
        self.assertEqual(manifest["vectors"], 2)

    def test_rejects_bad_vectors_and_duplicate_ids(self):
        cases = ([2.0, 0.0], [float("nan"), 0.0], [1.0], [0.0, 0.0])
        for vector in cases:
            with self.subTest(vector=vector):
                self.rows[0]["embedding"] = vector
                self.write_rows()
                with self.assertRaises(ValueError):
                    build_index(self.input, self.output)
                self.assertFalse(self.output.exists())
        self.rows[0]["embedding"] = [1.0, 0.0]
        self.rows[1]["chunk_id"] = self.rows[0]["chunk_id"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_index(self.input, self.output)

    def test_count_mismatch_and_incomplete_run(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "row count"):
            build_index(self.input, self.output)
        (self.input / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            build_index(self.input, self.output)

    def test_overwrite_and_mapping_corruption(self):
        build_index(self.input, self.output)
        original = (self.output / "index.faiss").read_bytes()
        with self.assertRaises(FileExistsError):
            build_index(self.input, self.output)
        self.assertEqual(original, (self.output / "index.faiss").read_bytes())
        with (self.output / "chunks.jsonl").open("a") as output:
            output.write("\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_index(self.output)


if __name__ == "__main__":
    unittest.main()

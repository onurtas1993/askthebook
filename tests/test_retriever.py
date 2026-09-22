import unittest
from unittest.mock import Mock

import faiss
import numpy as np

from src.retriever.search import QUERY_PREFIX, embed_question, search_vector


class RetrieverTests(unittest.TestCase):
    def test_ranking_and_k_larger_than_index(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.eye(2, dtype=np.float32))
        mapping = [{"chunk_id": "book:2", "page": 2}, {"chunk_id": "book:8", "page": 8}]
        results = search_vector(index, mapping, [0.6, 0.8], 10)
        self.assertEqual([r["page"] for r in results], [8, 2])
        self.assertEqual([r["rank"] for r in results], [1, 2])
        self.assertAlmostEqual(results[0]["score"], 0.8, places=6)
        for vector, k in [([0.6, 0.8], 0), ([1], 1), ([2, 0], 1), ([float('nan'), 0], 1)]:
            with self.assertRaises(ValueError):
                search_vector(index, mapping, vector, k)

    def test_query_prompt_end_token_and_normalization(self):
        client = Mock()
        client.tokens.return_value = [10, 20]
        client.request.side_effect = [
            {"tokens": [151643]},
            {"usage": {"prompt_tokens": 3}, "data": [{"embedding": [2.0] + [0.0] * 1023}]},
        ]
        vector, count = embed_question(client, "Why?", 100)
        client.tokens.assert_called_once_with(QUERY_PREFIX + "Why?")
        self.assertEqual(client.request.call_args.args[1]["input"], [10, 20, 151643])
        self.assertEqual(count, 3)
        self.assertEqual(vector, [1.0] + [0.0] * 1023)

    def test_oversize_rejected_before_embedding(self):
        client = Mock()
        client.request.return_value = {"tokens": [151643]}
        client.tokens.return_value = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "No truncation"):
            embed_question(client, "Question", 3)
        self.assertEqual(client.request.call_count, 1)
        with self.assertRaisesRegex(ValueError, "empty"):
            embed_question(client, "  ", 100)


if __name__ == "__main__":
    unittest.main()

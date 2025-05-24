"""
tests/test_core.py

Unit tests for chroma-curator core utilities.
- Tests cosine similarity math
- Tests loading and searching real vector data from test_vectors.json

Author: Mike Harris (mike.harris@tigerblue.tech)
Date: 2025-05-25

"""

import unittest
import json
import os

from chroma_curator import core

class TestCore(unittest.TestCase):
    def test_cosine_similarity(self):
        """
        Check that orthogonal vectors have zero similarity,
        and parallel vectors have maximum similarity.
        """
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        v3 = [2.0, 0.0]
        # Orthogonal vectors: similarity = 0
        sim_ortho = core.cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim_ortho, 0.0, places=5)
        # Parallel vectors: similarity = 1
        sim_parallel = core.cosine_similarity(v1, v3)
        self.assertAlmostEqual(sim_parallel, 1.0, places=5)

    def test_load_sample_vectors_and_self_similarity(self):
        """
        Load test_vectors.json and confirm:
        - Data is a list of dicts with 'vector' and 'id'
        - Each record is most similar to itself
        """
        sample_file = os.path.join(os.path.dirname(__file__), "test_vectors.json")
        with open(sample_file) as f:
            records = json.load(f)
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)
        self.assertTrue(all('vector' in rec and 'id' in rec for rec in records))

        # For first 3 vectors, check that each is most similar to itself
        for rec in records[:3]:
            sims = [core.cosine_similarity(rec['vector'], other['vector']) for other in records]
            max_idx = sims.index(max(sims))
            self.assertEqual(records[max_idx]['id'], rec['id'])

    def test_top_similar(self):
        """
        If your core.py implements a 'top_similar' or 'top_similar_vectors' function,
        test that it returns the right records.
        """
        # Uncomment and adjust if implemented
        """
        sample_file = os.path.join(os.path.dirname(__file__), "test_vectors.json")
        with open(sample_file) as f:
            records = json.load(f)
        query = records[0]['vector']
        # Find top 3 most similar (should include the query record itself)
        results = core.top_similar_vectors(query, records, top_k=3)
        ids = [rec['id'] for rec in results]
        self.assertIn(records[0]['id'], ids)
        """

if __name__ == "__main__":
    unittest.main()

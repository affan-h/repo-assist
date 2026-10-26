"""
Integration and unit tests for Qdrant Vector Store layer in repo-assist.
"""

import os
import shutil
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import vector_store
from config import DATA_DIR


class TestVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure a test-specific collection name to avoid touching production data
        cls.test_collection = "test_codebase_embeddings"
        cls.dim = 64  # lightweight dimension for fast unit tests

    def setUp(self):
        # Ensure fresh collection for each test
        client = vector_store.get_vector_client()
        try:
            if client.collection_exists(self.test_collection):
                client.delete_collection(self.test_collection)
        except Exception:
            pass
        vector_store.ensure_collection(dimension=self.dim, collection_name=self.test_collection)

    def tearDown(self):
        client = vector_store.get_vector_client()
        try:
            if client.collection_exists(self.test_collection):
                client.delete_collection(self.test_collection)
        except Exception:
            pass

    def test_deterministic_point_id(self):
        """Test UUID5 generation is deterministic and unique per composite key."""
        id1 = vector_store.generate_point_id("httpx", "summary", "Client::get::10")
        id2 = vector_store.generate_point_id("httpx", "summary", "Client::get::10")
        id3 = vector_store.generate_point_id("httpx", "summary", "Client::post::20")
        id4 = vector_store.generate_point_id("got", "summary", "Client::get::10")

        self.assertEqual(id1, id2, "Identical composite keys must produce identical UUIDs")
        self.assertNotEqual(id1, id3, "Different source_id must produce distinct UUIDs")
        self.assertNotEqual(id1, id4, "Different repo must produce distinct UUIDs")
        self.assertEqual(len(id1), 36, "Point ID should be a standard 36-char UUID string")

    def test_upsert_and_count(self):
        """Test inserting points and counting total / per-repo vectors."""
        vec1 = np.random.randn(self.dim).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        vec2 = np.random.randn(self.dim).astype(np.float32)
        vec2 /= np.linalg.norm(vec2)

        points = [
            {
                "repo": "test_repo_alpha",
                "source_type": "summary",
                "source_id": "file1.py::FuncA::1",
                "vector": vec1,
                "text": "Summary for FuncA in Alpha",
                "metadata": {"file_path": "file1.py"},
            },
            {
                "repo": "test_repo_alpha",
                "source_type": "pr",
                "source_id": "PR#101_chunk0",
                "vector": vec2,
                "text": "PR 101 chunk text",
                "metadata": {"pr_number": 101},
            },
        ]

        upserted = vector_store.upsert_points(points, collection_name=self.test_collection)
        self.assertEqual(upserted, 2)

        total_count = vector_store.count_vectors(collection_name=self.test_collection)
        self.assertEqual(total_count, 2)

        alpha_count = vector_store.count_vectors(repo="test_repo_alpha", collection_name=self.test_collection)
        self.assertEqual(alpha_count, 2)

        beta_count = vector_store.count_vectors(repo="test_repo_beta", collection_name=self.test_collection)
        self.assertEqual(beta_count, 0)

    def test_multi_tenant_isolation(self):
        """Test that searching in repo A never leaks points from repo B."""
        # Create a specific query vector
        base_vec = np.zeros(self.dim, dtype=np.float32)
        base_vec[0] = 1.0  # directional vector

        # Point for Repo A (identical vector, highest cosine similarity 1.0)
        p_alpha = {
            "repo": "tenant_alpha",
            "source_type": "summary",
            "source_id": "alpha_symbol",
            "vector": base_vec,
            "text": "Alpha secret code",
        }

        # Point for Repo B (also identical vector, highest cosine similarity 1.0)
        p_beta = {
            "repo": "tenant_beta",
            "source_type": "summary",
            "source_id": "beta_symbol",
            "vector": base_vec,
            "text": "Beta secret code",
        }

        vector_store.upsert_points([p_alpha, p_beta], collection_name=self.test_collection)

        # Search specifically within tenant_alpha
        results_alpha = vector_store.search_semantic(
            query_vector=base_vec,
            repo="tenant_alpha",
            top_k=5,
            collection_name=self.test_collection,
        )

        self.assertEqual(len(results_alpha), 1)
        self.assertEqual(results_alpha[0]["source_id"], "alpha_symbol")
        self.assertEqual(results_alpha[0]["text"], "Alpha secret code")
        # Ensure tenant_beta was NOT returned
        for hit in results_alpha:
            self.assertEqual(hit["payload"]["repo"], "tenant_alpha")

        # Search specifically within tenant_beta
        results_beta = vector_store.search_semantic(
            query_vector=base_vec,
            repo="tenant_beta",
            top_k=5,
            collection_name=self.test_collection,
        )

        self.assertEqual(len(results_beta), 1)
        self.assertEqual(results_beta[0]["source_id"], "beta_symbol")
        for hit in results_beta:
            self.assertEqual(hit["payload"]["repo"], "tenant_beta")

    def test_source_type_filter(self):
        """Test filtering by source_type (e.g. only PRs or only summaries)."""
        vec = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

        points = [
            {
                "repo": "filter_repo",
                "source_type": "summary",
                "source_id": "sym1",
                "vector": vec,
                "text": "Summary 1",
            },
            {
                "repo": "filter_repo",
                "source_type": "pr",
                "source_id": "pr1",
                "vector": vec,
                "text": "PR 1",
            },
            {
                "repo": "filter_repo",
                "source_type": "doc_chunk",
                "source_id": "doc1",
                "vector": vec,
                "text": "Doc 1",
            },
        ]
        vector_store.upsert_points(points, collection_name=self.test_collection)

        # Filter by PR only
        pr_results = vector_store.search_semantic(
            query_vector=vec,
            repo="filter_repo",
            source_type="pr",
            top_k=5,
            collection_name=self.test_collection,
        )
        self.assertEqual(len(pr_results), 1)
        self.assertEqual(pr_results[0]["source_type"], "pr")
        self.assertEqual(pr_results[0]["source_id"], "pr1")

    def test_delete_repo(self):
        """Test that delete_repo deletes only points belonging to the target repo."""
        vec = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

        points = [
            {"repo": "repo_to_delete", "source_type": "summary", "source_id": "s1", "vector": vec},
            {"repo": "repo_to_delete", "source_type": "pr", "source_id": "p1", "vector": vec},
            {"repo": "repo_to_keep", "source_type": "summary", "source_id": "s2", "vector": vec},
        ]
        vector_store.upsert_points(points, collection_name=self.test_collection)

        self.assertEqual(vector_store.count_vectors(collection_name=self.test_collection), 3)

        vector_store.delete_repo("repo_to_delete", collection_name=self.test_collection)

        self.assertEqual(vector_store.count_vectors(repo="repo_to_delete", collection_name=self.test_collection), 0)
        self.assertEqual(vector_store.count_vectors(repo="repo_to_keep", collection_name=self.test_collection), 1)
        self.assertEqual(vector_store.count_vectors(collection_name=self.test_collection), 1)

    def test_search_semantic_primary_qdrant(self):
        """Test that embeddings.search_semantic routes to vector_store and returns hits."""
        from unittest.mock import patch
        import embeddings

        mock_vec = np.zeros(embeddings.EMBEDDING_DIM, dtype=np.float32)
        mock_vec[0] = 1.0

        p = {
            "repo": "integration_test_repo",
            "source_type": "summary",
            "source_id": "test_file.py::TestClass::10",
            "vector": mock_vec,
            "text": "Integrated semantic search test text",
        }
        vector_store.upsert_points([p])

        try:
            with patch("embeddings.compute_embedding", return_value=mock_vec):
                hits = embeddings.search_semantic(db_path=":memory:", repo="integration_test_repo", query="dummy", top_k=3)
                self.assertGreaterEqual(len(hits), 1)
                self.assertEqual(hits[0]["source_type"], "summary")
                self.assertEqual(hits[0]["source_id"], "test_file.py::TestClass::10")
                self.assertIn("score", hits[0])
                self.assertAlmostEqual(hits[0]["score"], 1.0, places=4)
        finally:
            vector_store.delete_repo("integration_test_repo")


if __name__ == "__main__":
    unittest.main()

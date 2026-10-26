"""
Vector database layer for repo-assist powered by Qdrant.

Supports dual operating modes:
  1. Embedded Local Mode (Default):
     Runs in-process using Qdrant's Rust engine persisting directly to disk
     at `QDRANT_PATH` (DATA_DIR / "qdrant"). Zero external daemons required.
  2. Client-Server Mode:
     Connects to an external Qdrant instance if `QDRANT_URL` is set
     (e.g., http://qdrant:6333 in containerized clusters).

Key Capabilities:
  - HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index.
  - Multi-tenant payload filtering (where repo == target_repo) executed
    during graph traversal, preventing cross-tenant leakage.
  - Deterministic UUID5 point IDs for idempotent upserts.
"""

import atexit
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import DATA_DIR, QDRANT_PATH, QDRANT_URL

COLLECTION_NAME = "codebase_embeddings"
DEFAULT_VECTOR_DIM = 768

_client_instance: Optional[QdrantClient] = None


def get_vector_client() -> QdrantClient:
    """Returns a singleton or pooled QdrantClient instance."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    if QDRANT_URL:
        print(f"  [vector_store] Connecting to Qdrant server at {QDRANT_URL}...")
        _client_instance = QdrantClient(url=QDRANT_URL)
    else:
        path = Path(QDRANT_PATH)
        path.mkdir(parents=True, exist_ok=True)
        _client_instance = QdrantClient(path=str(path))
    return _client_instance


def close_vector_client() -> None:
    """Closes embedded client connections if open."""
    global _client_instance
    if _client_instance is not None:
        try:
            _client_instance.close()
        except Exception:
            pass
        _client_instance = None


atexit.register(close_vector_client)


def generate_point_id(repo: str, source_type: str, source_id: str) -> str:
    """Generates a deterministic UUID5 from the composite identity key."""
    namespace = uuid.NAMESPACE_DNS
    name = f"{repo}::{source_type}::{source_id}"
    return str(uuid.uuid5(namespace, name))


def ensure_collection(
    dimension: int = DEFAULT_VECTOR_DIM,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Ensures that the target vector collection and its payload indexes exist."""
    client = get_vector_client()
    exists = False
    try:
        exists = client.collection_exists(collection_name)
    except Exception:
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                on_disk=True,
            ),
        )

        # Create payload indexes on metadata fields for fast filtered traversals (remote server mode)
        if QDRANT_URL:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="repo",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="source_type",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )


def upsert_points(
    points: list[dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 100,
) -> int:
    """
    Upserts a list of vector points into Qdrant.
    Each point dict should contain:
      - repo: str
      - source_type: str ('summary', 'pr', 'doc_chunk')
      - source_id: str
      - vector: np.ndarray or list of floats
      - text: Optional[str]
      - metadata: Optional[dict]
    """
    if not points:
        return 0

    first_vec = points[0]["vector"]
    dim = len(first_vec) if hasattr(first_vec, "__len__") else DEFAULT_VECTOR_DIM
    ensure_collection(dimension=dim, collection_name=collection_name)

    client = get_vector_client()
    total_upserted = 0

    for i in range(0, len(points), batch_size):
        chunk = points[i : i + batch_size]
        qpoints = []
        for p in chunk:
            repo = p["repo"]
            source_type = p["source_type"]
            source_id = p["source_id"]
            vec = p["vector"]
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()

            pid = generate_point_id(repo, source_type, source_id)
            payload = {
                "repo": repo,
                "source_type": source_type,
                "source_id": source_id,
                "text": p.get("text", ""),
                "model_name": p.get("model_name", ""),
            }
            if "metadata" in p and isinstance(p["metadata"], dict):
                payload.update(p["metadata"])

            qpoints.append(
                qmodels.PointStruct(
                    id=pid,
                    vector=vec,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=qpoints,
            wait=True,
        )
        total_upserted += len(qpoints)

    return total_upserted


def search_semantic(
    query_vector: np.ndarray,
    repo: str,
    top_k: int = 5,
    source_type: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """
    Performs HNSW approximate nearest neighbor search filtered by repository.
    Returns results formatted identically to the legacy SQLite search_semantic:
      [{'source_type': ..., 'source_id': ..., 'score': float, 'text': ...}, ...]
    """
    client = get_vector_client()
    try:
        if not client.collection_exists(collection_name):
            return []
    except Exception:
        return []

    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    must_conditions = [
        qmodels.FieldCondition(
            key="repo",
            match=qmodels.MatchValue(value=repo),
        )
    ]
    if source_type:
        must_conditions.append(
            qmodels.FieldCondition(
                key="source_type",
                match=qmodels.MatchValue(value=source_type),
            )
        )

    query_filter = qmodels.Filter(must=must_conditions)

    try:
        res = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        hits = res.points
    except Exception as e:
        # Fallback to search method if query_points encounters compatibility issues
        try:
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            return []

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append({
            "source_type": payload.get("source_type", ""),
            "source_id": payload.get("source_id", ""),
            "score": float(hit.score),
            "text": payload.get("text", ""),
            "payload": payload,
        })

    return results


def delete_repo(repo: str, collection_name: str = COLLECTION_NAME) -> None:
    """Removes all indexed vectors associated with a repository."""
    client = get_vector_client()
    try:
        if not client.collection_exists(collection_name):
            return
        client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="repo",
                            match=qmodels.MatchValue(value=repo),
                        )
                    ]
                )
            ),
        )
    except Exception as e:
        print(f"  [vector_store] Error deleting points for repo '{repo}': {e}")


def count_vectors(repo: Optional[str] = None, collection_name: str = COLLECTION_NAME) -> int:
    """Counts indexed vector points, optionally filtered by repository."""
    client = get_vector_client()
    try:
        if not client.collection_exists(collection_name):
            return 0
        if repo:
            cnt = client.count(
                collection_name=collection_name,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="repo",
                            match=qmodels.MatchValue(value=repo),
                        )
                    ]
                ),
            )
        else:
            cnt = client.count(collection_name=collection_name)
        return cnt.count
    except Exception:
        return 0

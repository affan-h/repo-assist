import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from repo_assist import store, vectors
from repo_assist.config import get_settings
from repo_assist.ingest import git

FIXTURE_FILES = {
    "pkg/client.py": '''"""HTTP client."""
import os

DEFAULT_TIMEOUT = 5.0


class Client:
    """Sends requests through a transport."""

    def __init__(self, timeout=DEFAULT_TIMEOUT):
        self.timeout = timeout

    def get(self, url):
        return self.send(build_request("GET", url))

    def send(self, request):
        return self._transport_for_url(request.url).handle_request(request)

    def _transport_for_url(self, url):
        """Pick the mounted transport whose pattern matches the URL."""
        return self._transport


def build_request(method, url):
    return {"method": method, "url": url}
''',
    "src/retry.ts": '''export interface RetryOptions {
  limit: number;
}

export const calculateRetryDelay = (attempt: number): number => {
  return Math.min(1000 * 2 ** attempt, computeCap());
};

export class Retrier {
  run(fn: () => void) {
    fn();
  }
}
''',
    "docs/guide.md": '''# Guide

Intro text for the guide that explains what the library is for and how to install it.

## Timeouts

The client applies a default timeout of five seconds to every request. You can
override the timeout per request or disable it entirely by passing None.

## Retries

Retries use exponential backoff computed by calculateRetryDelay.
''',
    "tests/test_client.py": "def test_get():\n    assert True\n",
}


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    name = "test-hash-embedder"
    dim = 64

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for term in store.search_terms(text):
            v[int(hashlib.md5(term.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def embed_documents(self, texts):
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def backend():
    """Vector search backend; parametrize a test over ["exact", "qdrant"] to run it on both."""
    return "exact"


@pytest.fixture
def data_dir(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("REPO_ASSIST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    if backend == "qdrant":
        monkeypatch.setenv("REPO_ASSIST_QDRANT_URL", ":memory:")  # in-process Qdrant, same API as a server
    else:
        monkeypatch.delenv("REPO_ASSIST_QDRANT_URL", raising=False)
    monkeypatch.setattr(vectors, "_index", None)
    get_settings.cache_clear()
    store.init_db()
    yield tmp_path / "data"
    get_settings.cache_clear()


@pytest.fixture
def fixture_repo(tmp_path) -> Path:
    root = tmp_path / "fixture-repo"
    for rel, content in FIXTURE_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    env = {"GIT_AUTHOR_NAME": "Dev", "GIT_AUTHOR_EMAIL": "d@x", "GIT_COMMITTER_NAME": "Dev",
           "GIT_COMMITTER_EMAIL": "d@x", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, env=env)
    run("init", "-q")
    run("add", "-A")
    run("commit", "-q", "-m", "Add client with default timeout\n\nTimeouts default to 5s so hung servers fail fast.")
    return root


@pytest.fixture
def ingested_repo(data_dir, fixture_repo, monkeypatch):
    """The fixture repo registered as `acme/widget` and fully ingested (no network)."""
    from repo_assist import ingest

    def fake_clone(owner, name, dest, ref=None):
        if not dest.exists():
            shutil.copytree(fixture_repo, dest)
        sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return git.Checkout(path=dest, sha=sha, date="2026-01-01T00:00:00+00:00")

    monkeypatch.setattr(git, "clone", fake_clone)
    repo_id = ingest.register_repo("https://github.com/acme/widget")
    ingest.ingest(repo_id, embedder=HashEmbedder())
    return repo_id

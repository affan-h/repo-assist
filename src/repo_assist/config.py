"""Runtime settings, read from environment variables (and a local .env file)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    agent_model: str
    judge_model: str
    embedding_model: str
    github_token: str | None
    max_threads: int
    max_commits: int
    model_cache_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "repo_assist.db"

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"


@lru_cache
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("REPO_ASSIST_DATA_DIR", "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=data_dir,
        agent_model=os.environ.get("REPO_ASSIST_AGENT_MODEL", "google:gemini-3.5-flash-lite"),
        # Free-tier friendly default; set a stronger or different-family model to reduce judge bias.
        judge_model=os.environ.get("REPO_ASSIST_JUDGE_MODEL", "google:gemini-3.5-flash-lite"),
        embedding_model=os.environ.get("REPO_ASSIST_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        github_token=os.environ.get("GITHUB_TOKEN") or None,
        # Upper bounds on history ingestion. Threads = releases + issues + PRs + discussions.
        max_threads=int(os.environ.get("REPO_ASSIST_MAX_THREADS", "8000")),
        max_commits=int(os.environ.get("REPO_ASSIST_MAX_COMMITS", "5000")),
        model_cache_dir=Path(os.environ.get("REPO_ASSIST_MODEL_DIR", data_dir / "models")),
    )

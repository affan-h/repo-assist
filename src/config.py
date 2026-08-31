"""
Global configuration constants for repo-assist.

Single source of truth for filesystem paths and shared pipeline settings.
All paths are relative to `src/` (the assumed working directory for all
pipeline phase scripts and CLI invocations).
"""

from pathlib import Path

# Canonical path to the SQLite code graph database relative to src/
DB_PATH = "data/code_graph.db"

# Bounded history fetching policy (scope guards against unbounded API calls/mining)
MAX_PRS = 300
MAX_ISSUES = 300
MAX_DISCUSSIONS_PER_CATEGORY = 300
MAX_COMMITS_PER_FILE = 500

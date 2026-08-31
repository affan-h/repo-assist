import os
from pathlib import Path

# Base directory paths anchored to the location of this configuration file
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = SRC_DIR / "data"
REPOS_DIR = PROJECT_ROOT / "repos"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Canonical absolute path string to the SQLite code graph database
DB_PATH = str(DATA_DIR / "code_graph.db")

# Bounded history fetching policy (scope guards against unbounded API calls/mining)
MAX_PRS = 300
MAX_ISSUES = 300
MAX_DISCUSSIONS_PER_CATEGORY = 300
MAX_COMMITS_PER_FILE = 500

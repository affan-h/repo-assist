"""
Phase 2, Step 1 -- commit history mining.

Verified against real PyDriller output (a real cloned repo, not
documentation) before writing this:
  - Commit fields: hash, author.name, author.email, author_date, msg, merge
  - ModifiedFile fields: filename, change_type (enum), added_lines, deleted_lines
  - Merge commits show 0 modified_files by default -- confirmed real
    behavior, not a bug we need to work around, but a case we must
    handle explicitly (we record merge commits separately, without
    per-file attribution, rather than silently dropping them).
  - filepath= filtering on Repository() correctly scopes traversal to
    commits touching a specific file -- confirmed working.

This module mines commit history for every FILE already present in our
Phase 1 graph (data/code_graph.db), not the whole repo blindly -- we
only care about history for files we can actually connect to a symbol.

Storage: a new SQLite table, commits, keyed by (repo, file_path, commit_hash).
One row per (file, commit) pair, since a single commit can touch multiple
files we care about, and a single file has many commits.

Run with:
    python3 src/mine_history.py
"""

import sqlite3
from pathlib import Path
from pydriller import Repository


def init_commits_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS commits (
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            author_name TEXT,
            author_email TEXT,
            author_date TEXT,
            message TEXT,
            is_merge INTEGER NOT NULL,
            added_lines INTEGER,
            deleted_lines INTEGER,
            change_type TEXT,
            PRIMARY KEY (repo, file_path, commit_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_commits_file ON commits(repo, file_path);
        CREATE INDEX IF NOT EXISTS idx_commits_hash ON commits(commit_hash);
    """)
    conn.commit()
    return conn


def get_indexed_files(db_path: str, repo: str) -> list[str]:
    """Return every file path already known to our Phase 1 graph for this
    repo, so we only mine history for files we can actually connect to
    a symbol -- not the whole repo's history blindly."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT path FROM files WHERE repo = ?", (repo,))
    files = [row[0] for row in cur.fetchall()]
    conn.close()
    return files


def mine_file_history(repo_path: str, repo_name: str, file_path: str, conn: sqlite3.Connection):
    """Mine commit history for ONE file, insert rows into the commits table."""
    cur = conn.cursor()
    count = 0

    for commit in Repository(repo_path, filepath=file_path).traverse_commits():
        if commit.merge:
            # Merge commits show 0 modified_files by default (confirmed
            # via real PyDriller output) -- we still record that this
            # merge touched the file's history line, but without
            # per-file added/deleted line counts, since PyDriller can't
            # give us that without specifying which parent to diff against.
            cur.execute(
                """INSERT OR REPLACE INTO commits
                   (repo, file_path, commit_hash, author_name, author_email,
                    author_date, message, is_merge, added_lines, deleted_lines, change_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (repo_name, file_path, commit.hash, commit.author.name, commit.author.email,
                 str(commit.author_date), commit.msg, 1, None, None, "MERGE"),
            )
            count += 1
            continue

        for mf in commit.modified_files:
            if mf.filename != Path(file_path).name:
                continue  # Repository(filepath=...) can occasionally include
                          # renamed/related files; only keep the exact match
            cur.execute(
                """INSERT OR REPLACE INTO commits
                   (repo, file_path, commit_hash, author_name, author_email,
                    author_date, message, is_merge, added_lines, deleted_lines, change_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (repo_name, file_path, commit.hash, commit.author.name, commit.author.email,
                 str(commit.author_date), commit.msg, 0, mf.added_lines, mf.deleted_lines,
                 str(mf.change_type)),
            )
            count += 1

    return count


def main():
    db_path = "data/code_graph.db"
    init_commits_table(db_path)

    repos = {
        "httpx": "repos/httpx",
        "got": "repos/got",
    }

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    for repo_name, repo_path in repos.items():
        files = get_indexed_files(db_path, repo_name)
        print(f"Mining history for {repo_name}: {len(files)} indexed files...")

        total_commits_recorded = 0
        for i, file_path in enumerate(files):
            count = mine_file_history(repo_path, repo_name, file_path, conn)
            total_commits_recorded += count
            if (i + 1) % 5 == 0 or (i + 1) == len(files):
                print(f"  [{i+1}/{len(files)}] {file_path}: {count} commit-rows")

        conn.commit()
        print(f"  {repo_name}: {total_commits_recorded} total commit-file rows recorded\n")

    conn.close()
    print("Done. Mined history stored in data/code_graph.db's 'commits' table.")


if __name__ == "__main__":
    main()

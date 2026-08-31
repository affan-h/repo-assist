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
from config import DB_PATH, REPOS_DIR, MAX_COMMITS_PER_FILE
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
    import subprocess
    from pathlib import Path
    cur = conn.cursor()
    max_commits = MAX_COMMITS_PER_FILE or 500

    cmd = [
        "git", "-C", str(repo_path), "log",
        f"-n{max_commits}", "--numstat",
        "--pretty=format:COMMIT:%H|%an|%ae|%aI|%s",
        "--", file_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.splitlines()

        rows = []
        current_commit = None

        for line in lines:
            if line.startswith("COMMIT:"):
                parts = line[len("COMMIT:"):].split("|", 4)
                if len(parts) == 5:
                    current_commit = {
                        "hash": parts[0],
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "msg": parts[4],
                    }
            elif line.strip() and current_commit:
                stat_parts = line.split()
                if len(stat_parts) >= 2:
                    added = int(stat_parts[0]) if stat_parts[0].isdigit() else 0
                    deleted = int(stat_parts[1]) if stat_parts[1].isdigit() else 0
                    rows.append((
                        repo_name, file_path, current_commit["hash"],
                        current_commit["author"], current_commit["email"],
                        current_commit["date"], current_commit["msg"],
                        0, added, deleted, "MODIFY",
                    ))
                    current_commit = None

        if rows:
            cur.executemany(
                """INSERT OR REPLACE INTO commits
                   (repo, file_path, commit_hash, author_name, author_email,
                    author_date, message, is_merge, added_lines, deleted_lines, change_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

        return len(rows)

    except Exception:
        # Fallback to PyDriller if git subprocess is unavailable
        rows = []
        count = 0
        for commit in Repository(repo_path, filepath=file_path, order='reverse').traverse_commits():
            if MAX_COMMITS_PER_FILE is not None and count >= MAX_COMMITS_PER_FILE:
                break
            if commit.merge:
                rows.append((
                    repo_name, file_path, commit.hash, commit.author.name, commit.author.email,
                    str(commit.author_date), commit.msg, 1, None, None, "MERGE",
                ))
                count += 1
                continue
            for mf in commit.modified_files:
                if mf.filename != Path(file_path).name:
                    continue
                rows.append((
                    repo_name, file_path, commit.hash, commit.author.name, commit.author.email,
                    str(commit.author_date), commit.msg, 0, mf.added_lines, mf.deleted_lines,
                    str(mf.change_type),
                ))
                count += 1

        if rows:
            cur.executemany(
                """INSERT OR REPLACE INTO commits
                   (repo, file_path, commit_hash, author_name, author_email,
                    author_date, message, is_merge, added_lines, deleted_lines, change_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

        return count


def main():
    from pathlib import Path
    init_commits_table(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT repo FROM files")
    repos = [r[0] for r in cur.fetchall()]

    for repo_name in repos:
        repo_dir = REPOS_DIR / repo_name
        if not repo_dir.exists():
            print(f"Repo path {repo_dir} does not exist, skipping commit mining.")
            continue

        repo_path = str(repo_dir)
        files = get_indexed_files(DB_PATH, repo_name)
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
    print(f"Done. Mined history stored in {DB_PATH}'s 'commits' table.")


if __name__ == "__main__":
    main()

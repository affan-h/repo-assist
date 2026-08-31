"""
Phase 3, Step 1 -- churn scoring.

Before designing the full risk/blast-radius model, compute a real churn
score per file using data ALREADY in the commits table (built in Phase 2
via PyDriller) and inspect its actual distribution -- same discipline as
every prior phase: look at real numbers before building further logic
on assumptions about their shape.

Churn = number of distinct commits that touched a file. This is the
simplest, most directly available signal from what we've already built,
and a well-established real proxy for "this file is a hotspot" in the
empirical software engineering literature (files that change often are
statistically more likely to contain future bugs).

Deliberately NOT yet included: complexity, incident correlation --
those come after we've seen whether churn alone produces a sensible,
usable distribution.

Run with:
    python3 src/compute_churn.py
"""

import sqlite3

from config import DB_PATH


def compute_churn_per_file(conn: sqlite3.Connection, repo: str) -> list[tuple[str, int]]:
    """Returns [(file_path, distinct_commit_count), ...] sorted by
    churn descending."""
    cur = conn.cursor()
    cur.execute("""
        SELECT file_path, COUNT(DISTINCT commit_hash) as churn
        FROM commits
        WHERE repo = ?
        GROUP BY file_path
        ORDER BY churn DESC
    """, (repo,))
    return cur.fetchall()


def init_churn_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS churn_scores (
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            commit_count INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (repo, file_path)
        );
    """)
    conn.commit()
    return conn


def main():
    conn = init_churn_table(DB_PATH)

    import time
    now = str(time.time())

    for repo in ["httpx", "got"]:
        print("=" * 70)
        print(f"CHURN SCORES: {repo}")
        print("=" * 70)

        churn_data = compute_churn_per_file(conn, repo)
        if not churn_data:
            print(f"  No commit data found for {repo} -- check Phase 2's commits table.")
            continue

        cur = conn.cursor()
        for file_path, count in churn_data:
            cur.execute(
                "INSERT OR REPLACE INTO churn_scores (repo, file_path, commit_count, computed_at) VALUES (?, ?, ?, ?)",
                (repo, file_path, count, now),
            )
        conn.commit()

        print(f"  Total files with commit history: {len(churn_data)}")
        print(f"\n  Top 10 highest-churn files:")
        for file_path, count in churn_data[:10]:
            print(f"    {count:4} commits  {file_path}")

        print(f"\n  Bottom 5 lowest-churn files:")
        for file_path, count in churn_data[-5:]:
            print(f"    {count:4} commits  {file_path}")

        counts = [c for _, c in churn_data]
        print(f"\n  Distribution: min={min(counts)}, max={max(counts)}, "
              f"median={sorted(counts)[len(counts)//2]}, "
              f"mean={sum(counts)/len(counts):.1f}")
        print()

    conn.close()


if __name__ == "__main__":
    main()

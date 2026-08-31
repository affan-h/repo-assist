"""
Phase 3, Step 3 -- combined risk score.

File-level score (complexity is per-symbol, churn is per-file, so we
aggregate complexity UP to file level by summing branch counts of all
non-class symbols in that file -- deliberately excluding classes, same
reasoning as compute_complexity.py, since class-level branch counts
just re-count their methods).

risk_score = percentile_rank(churn) + percentile_rank(total_branches),
averaged (0-1 scale). Percentile rank, not raw normalization, so one
extreme outlier (e.g. got's 214-commit file) doesn't compress everyone
else's score toward zero -- avoids needing an arbitrary weight between
the two signals.

Run with:
    python3 src/compute_risk_scores.py
"""

import sqlite3

from config import DB_PATH


def init_risk_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS risk_scores (
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            churn INTEGER NOT NULL,
            total_branches INTEGER NOT NULL,
            churn_percentile REAL NOT NULL,
            complexity_percentile REAL NOT NULL,
            risk_score REAL NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (repo, file_path)
        );
    """)
    conn.commit()
    return conn


def percentile_ranks(values: list[float]) -> list[float]:
    """Returns each value's percentile rank (0-1) within the list,
    ties get the same rank. Avoids needing min/max normalization,
    which is sensitive to a single outlier."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return [sorted_vals.index(v) / max(1, n - 1) for v in values]


def compute_risk(conn: sqlite3.Connection, repo: str):
    cur = conn.cursor()

    cur.execute("SELECT file_path, commit_count FROM churn_scores WHERE repo = ?", (repo,))
    churn = dict(cur.fetchall())

    cur.execute("""
        SELECT file_path, SUM(branch_count) FROM complexity_scores
        WHERE repo = ? AND qualified_name NOT IN (
            SELECT qualified_name FROM symbols WHERE repo = ? AND kind = 'class'
        )
        GROUP BY file_path
    """, (repo, repo))
    complexity = dict(cur.fetchall())

    all_files = sorted(set(churn) | set(complexity))
    churn_vals = [churn.get(f, 0) for f in all_files]
    complexity_vals = [complexity.get(f, 0) for f in all_files]

    churn_pct = percentile_ranks(churn_vals)
    complexity_pct = percentile_ranks(complexity_vals)

    import time
    now = str(time.time())
    results = []
    for i, f in enumerate(all_files):
        risk = (churn_pct[i] + complexity_pct[i]) / 2
        conn.execute(
            """INSERT OR REPLACE INTO risk_scores
               (repo, file_path, churn, total_branches, churn_percentile,
                complexity_percentile, risk_score, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (repo, f, churn_vals[i], complexity_vals[i], churn_pct[i], complexity_pct[i], risk, now),
        )
        results.append((f, churn_vals[i], complexity_vals[i], risk))

    conn.commit()
    return sorted(results, key=lambda r: -r[3])


def main():
    conn = init_risk_table(DB_PATH)

    cur = conn.cursor()
    cur.execute("SELECT DISTINCT repo FROM churn_scores UNION SELECT DISTINCT repo FROM complexity_scores")
    repos = [row[0] for row in cur.fetchall()]

    for repo in repos:
        print("=" * 70)
        print(f"RISK SCORES: {repo}")
        print("=" * 70)
        results = compute_risk(conn, repo)
        if not results:
            print("  No data.")
            continue
        print(f"  Top 10 highest risk files:")
        for f, churn, branches, risk in results[:10]:
            print(f"    risk={risk:.2f}  churn={churn:4}  branches={branches:4}  {f}")
        print()

    conn.close()


if __name__ == "__main__":
    main()

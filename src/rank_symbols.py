"""
rank_symbols.py — PageRank over the symbol graph.

Runs a PageRank pass over CALLS/IMPORTS/EXTENDS edges using networkx,
and writes a pagerank_score field onto each symbol row in data/code_graph.db.

Run with:
    python3 src/rank_symbols.py
"""

import sqlite3
import networkx as nx
from config import DB_PATH


def compute_pagerank(db_path: str = DB_PATH, repo: str | None = None) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure pagerank_score column exists
    try:
        cur.execute("ALTER TABLE symbols ADD COLUMN pagerank_score REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Fetch distinct repos
    if repo:
        repos = [repo]
    else:
        cur.execute("SELECT DISTINCT repo FROM symbols")
        repos = [row[0] for row in cur.fetchall()]

    counts_by_repo = {}

    for repo in repos:
        G = nx.DiGraph()

        # Add all symbols in this repo as nodes
        cur.execute(
            "SELECT file_path, qualified_name, start_line FROM symbols WHERE repo = ?",
            (repo,),
        )
        symbols = cur.fetchall()
        for fp, qn, start_line in symbols:
            G.add_node((fp, qn, start_line))

        # Add edges from symbol_edges (CALLS, INSTANTIATES, EXTENDS)
        cur.execute(
            """SELECT from_file, from_qualified_name, from_start_line,
                      to_file, to_qualified_name, to_start_line
               FROM symbol_edges WHERE repo = ?""",
            (repo,),
        )
        for f_fp, f_qn, f_line, t_fp, t_qn, t_line in cur.fetchall():
            u = (f_fp, f_qn, f_line)
            v = (t_fp, t_qn, t_line)
            if u in G and v in G:
                G.add_edge(u, v)

        if len(G) == 0:
            continue

        # Compute PageRank
        scores = nx.pagerank(G, alpha=0.85)

        # Batch update database
        updates = [
            (score, repo, fp, qn, start_line)
            for (fp, qn, start_line), score in scores.items()
        ]
        cur.executemany(
            """UPDATE symbols
               SET pagerank_score = ?
               WHERE repo = ? AND file_path = ? AND qualified_name = ? AND start_line = ?""",
            updates,
        )
        conn.commit()
        counts_by_repo[repo] = len(updates)

    conn.close()
    return counts_by_repo


def main():
    print("Computing PageRank scores for symbols...")
    counts = compute_pagerank(DB_PATH)
    total = sum(counts.values())
    for repo, count in sorted(counts.items()):
        print(f"  {repo}: {count} symbols scored")
    print(f"\nTotal symbols with non-null pagerank_score: {total}")


if __name__ == "__main__":
    main()

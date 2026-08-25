"""
Graph centrality scoring (v2, §4.2).

Real fix vs. the previous draft:
  - Uses rustworkx.pagerank, not networkx -- the plan is explicit that
    rustworkx is already a v1 dependency and networkx would be a new,
    unnecessary one (§4.2, §11: "verify rustworkx.pagerank exists with
    standard semantics before writing any custom traversal"). Verified
    directly: rustworkx 0.18.1 exposes a module-level rx.pagerank(graph)
    function (not a PyDiGraph method) whose docstring states it "tries to
    match NetworkX's power-iteration implementation" -- confirms §11's
    assumption, not a custom reimplementation.
  - Writes the real §5.1 schema: (repo, file_path, pagerank_score,
    in_degree, computed_at) -- the previous draft wrote (repo,
    symbol_name, score), which doesn't match what §5.1 specifies and
    would silently break anything reading the documented schema.
  - Real edges (CALLS + INSTANTIATES + EXTENDS, not just CALLS) are used
    to build the graph -- imports edges are aggregated separately, since
    imports connect files directly while symbol_edges connect symbols;
    symbol-level PageRank is aggregated UP to file-level by taking the max
    score of any symbol defined in that file, matching how a file's real
    "importance" should read: a file containing one highly-central symbol
    is a high-risk file to touch, even if its other symbols are minor.
"""

import sqlite3
from datetime import datetime, timezone

import rustworkx as rx

DB_PATH = "../data/code_graph.db"


def _init_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS centrality_scores (
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            pagerank_score REAL NOT NULL,
            in_degree INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (repo, file_path)
        )
    """)
    conn.commit()


def compute_all_centrality():
    with sqlite3.connect(DB_PATH) as conn:
        _init_table(conn)
        cursor = conn.cursor()

        repos = [r[0] for r in cursor.execute("SELECT DISTINCT repo FROM symbol_edges").fetchall()]

        for repo in repos:
            print(f"\n--- Computing centrality for {repo} ---")

            # Real structural edges: CALLS + INSTANTIATES + EXTENDS, per
            # Phase 1's real graph (project_context.md) -- not CALLS alone,
            # since a heavily-instantiated or heavily-extended symbol is
            # also structurally important, not just a heavily-called one.
            cursor.execute("""
                SELECT from_qualified_name, from_file, to_qualified_name, to_file, edge_type
                FROM symbol_edges
                WHERE repo = ? AND edge_type IN ('CALLS', 'INSTANTIATES', 'EXTENDS')
            """, (repo,))
            edges = cursor.fetchall()

            if not edges:
                print(f"  No structural edges found for {repo}. Skipping.")
                continue

            # Build graph on qualified_name nodes; track which file each
            # qualified_name lives in so we can aggregate up afterward.
            graph = rx.PyDiGraph()
            node_index: dict[str, int] = {}
            symbol_file: dict[str, str] = {}

            def get_or_add(qname: str, file: str) -> int:
                if qname not in node_index:
                    node_index[qname] = graph.add_node(qname)
                    symbol_file[qname] = file
                return node_index[qname]

            in_degree: dict[str, int] = {}
            for from_q, from_f, to_q, to_f, edge_type in edges:
                src = get_or_add(from_q, from_f)
                tgt = get_or_add(to_q, to_f)
                graph.add_edge(src, tgt, edge_type)
                in_degree[to_q] = in_degree.get(to_q, 0) + 1

            print(f"  Graph built: {graph.num_nodes()} nodes, {graph.num_edges()} edges.")

            pagerank_scores = rx.pagerank(graph, alpha=0.85)  # dict-like: node index -> score

            # Aggregate symbol-level scores up to file-level (max per file,
            # see module docstring for rationale).
            file_pagerank: dict[str, float] = {}
            file_in_degree: dict[str, int] = {}
            for qname, idx in node_index.items():
                f = symbol_file.get(qname)
                if not f:
                    continue
                score = pagerank_scores[idx]
                deg = in_degree.get(qname, 0)
                if f not in file_pagerank or score > file_pagerank[f]:
                    file_pagerank[f] = score
                file_in_degree[f] = file_in_degree.get(f, 0) + deg

            now = datetime.now(timezone.utc).isoformat()
            for f, score in file_pagerank.items():
                conn.execute("""
                    INSERT OR REPLACE INTO centrality_scores
                    (repo, file_path, pagerank_score, in_degree, computed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (repo, f, score, file_in_degree.get(f, 0), now))
            conn.commit()

            top_5 = sorted(file_pagerank.items(), key=lambda x: -x[1])[:5]
            print("  Top 5 most central files:")
            for f, score in top_5:
                print(f"    {f}: pagerank={score:.5f}  in_degree={file_in_degree.get(f, 0)}")


if __name__ == "__main__":
    compute_all_centrality()

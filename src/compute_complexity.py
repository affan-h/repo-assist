"""
Phase 3, Step 2 -- complexity scoring.

Verified real tree-sitter node types before writing this (not assumed):
  Python:     if_statement, elif_clause, for_statement, while_statement,
              except_clause (statement/clause nodes -- NOT the bare
              keyword leaf tokens like "if"/"elif", which appear
              separately in the tree and would double-count if included)
  TypeScript: if_statement, for_statement, while_statement, catch_clause,
              switch_case (same distinction: statement/clause nodes only)

Complexity score = count of branching constructs in a symbol's body,
plus its line count as a secondary signal. This is a lightweight proxy
for cyclomatic complexity, not the real McCabe metric -- deliberately
simple and grounded in what we can directly count from the AST we
already have, rather than importing a heavier complexity-analysis
dependency for a project at this scope.

Run with:
    python3 src/compute_complexity.py
"""

import sqlite3
from pathlib import Path

from config import DB_PATH
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts


PY_LANGUAGE = Language(tspython.language())
TS_LANGUAGE = Language(tsts.language_typescript())
py_parser = Parser(PY_LANGUAGE)
ts_parser = Parser(TS_LANGUAGE)

PY_BRANCH_NODES = {"if_statement", "elif_clause", "for_statement", "while_statement", "except_clause"}
TS_BRANCH_NODES = {"if_statement", "for_statement", "while_statement", "catch_clause", "switch_case"}


def count_branches(node, branch_node_types: set[str]) -> int:
    count = 1 if node.type in branch_node_types else 0
    for child in node.children:
        count += count_branches(child, branch_node_types)
    return count


def init_complexity_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS complexity_scores (
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            line_count INTEGER NOT NULL,
            branch_count INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (repo, file_path, qualified_name, start_line)
        );
    """)
    conn.commit()
    return conn


def compute_for_repo(conn: sqlite3.Connection, repo: str, repo_root: str, language: str):
    cur = conn.cursor()
    cur.execute("""
        SELECT file_path, qualified_name, start_line, end_line, kind
        FROM symbols WHERE repo = ?
    """, (repo,))
    symbols = cur.fetchall()

    parser = py_parser if language == "python" else ts_parser
    branch_nodes = PY_BRANCH_NODES if language == "python" else TS_BRANCH_NODES

    import time
    now = str(time.time())
    results = []

    for file_path, qualified_name, start_line, end_line, kind in symbols:
        full_path = Path(repo_root) / file_path
        try:
            source_code = full_path.read_bytes()
        except FileNotFoundError:
            continue

        lines = source_code.split(b"\n")
        snippet = b"\n".join(lines[start_line - 1:end_line])
        line_count = end_line - start_line + 1

        try:
            tree = parser.parse(snippet)
            branch_count = count_branches(tree.root_node, branch_nodes)
        except Exception:
            branch_count = 0

        cur.execute(
            """INSERT OR REPLACE INTO complexity_scores
               (repo, file_path, qualified_name, start_line, line_count, branch_count, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (repo, file_path, qualified_name, start_line, line_count, branch_count, now),
        )
        results.append((qualified_name, file_path, line_count, branch_count, kind))

    conn.commit()
    return results


def main():
    conn = init_complexity_table(DB_PATH)

    for repo, repo_root, language in [("httpx", "repos/httpx", "python"), ("got", "repos/got", "typescript")]:
        print("=" * 70)
        print(f"COMPLEXITY SCORES: {repo}")
        print("=" * 70)

        results = compute_for_repo(conn, repo, repo_root, language)
        if not results:
            print(f"  No symbols found for {repo}.")
            continue

        funcs_and_methods = [r for r in results if r[4] != "class"]
        classes = [r for r in results if r[4] == "class"]

        by_branches = sorted(funcs_and_methods, key=lambda r: -r[3])
        print(f"  Total symbols scored: {len(results)} ({len(funcs_and_methods)} functions/methods, {len(classes)} classes)")
        print(f"\n  Top 10 most complex FUNCTIONS/METHODS (classes excluded -- they aggregate all their methods' branches):")
        for qname, fpath, lines, branches, kind in by_branches[:10]:
            print(f"    branches={branches:3} lines={lines:4}  {fpath}:{qname}")

        if classes:
            by_class_branches = sorted(classes, key=lambda r: -r[3])
            print(f"\n  Top 5 largest CLASSES (for reference only, not used in ranking):")
            for qname, fpath, lines, branches, kind in by_class_branches[:5]:
                print(f"    branches={branches:3} lines={lines:4}  {fpath}:{qname}")

        branch_counts = [r[3] for r in funcs_and_methods]
        line_counts = [r[2] for r in funcs_and_methods]
        print(f"\n  Branch count distribution (functions/methods only): min={min(branch_counts)}, max={max(branch_counts)}, "
              f"median={sorted(branch_counts)[len(branch_counts)//2]}, "
              f"mean={sum(branch_counts)/len(branch_counts):.2f}")
        print(f"  Line count distribution (functions/methods only): min={min(line_counts)}, max={max(line_counts)}, "
              f"median={sorted(line_counts)[len(line_counts)//2]}, "
              f"mean={sum(line_counts)/len(line_counts):.1f}")
        print()

    conn.close()


if __name__ == "__main__":
    main()

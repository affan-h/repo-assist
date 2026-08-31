"""
Graph schema for the codebase intelligence project.

Node types:
    FileNode   — a source file
    SymbolNode — a class, function, or method (from extract_symbols.py)

Edge types:
    CONTAINS — File -> Symbol   (this file defines this symbol)
    IMPORTS  — File -> File     (this file imports from that file)

CALLS edges (Symbol -> Symbol) are deliberately NOT built yet — they
require import resolution to be correct first, since a call like
`transport.handle_request(...)` can only be resolved to a specific
Symbol once we know what `transport` actually refers to. That is
Step 9+.

Storage: rustworkx PyDiGraph for in-memory traversal, SQLite for
persistence between runs. multigraph=False is a deliberate choice:
if we ever add the same IMPORTS edge twice (e.g. re-running ingestion),
we want it to update in place, not silently create a parallel edge.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import rustworkx as rx


# ── Node payload types ───────────────────────────────────────────────────────

@dataclass
class FileNode:
    repo: str
    path: str          # relative to repo root, e.g. "httpx/_config.py"
    language: str       # "python" | "typescript"


@dataclass
class SymbolNode:
    repo: str
    name: str
    kind: str           # "class" | "function" | "method"
    file_path: str
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    pagerank_score: Optional[float] = None


# ── The graph wrapper ────────────────────────────────────────────────────────

class CodeGraph:
    """
    Wraps a rustworkx PyDiGraph with lookup indices, since rustworkx only
    identifies nodes by integer index — we need to go from
    (repo, file_path) -> index and (repo, file_path, symbol_name) -> index
    constantly, so we maintain those maps ourselves.
    """

    def __init__(self):
        self.graph = rx.PyDiGraph(multigraph=False)
        # repo -> file_path -> node_index
        self._file_index: dict[tuple[str, str], int] = {}
        # repo -> file_path -> symbol_name -> node_index
        # (symbol_name here is the "qualified" name: "ClassName.method_name"
        #  for methods, or just "func_name" for top-level functions/classes,
        #  since two methods in different classes can share a name — e.g.
        #  __init__ appears in Timeout, Limits, and Proxy in httpx/_config.py)
        #
        # IMPORTANT: the key also includes start_line. Originally it did not,
        # and this silently collapsed every @property / @x.setter pair (and
        # TypeScript get/set accessor pair) into a single node, because both
        # halves share the same qualified_name. Verified on real data: 12
        # such collisions in httpx, 50 in got (source/core/options.ts alone
        # accounts for all 50, since it's a large options class built almost
        # entirely from property accessor pairs). Losing the setter half of
        # every property is a real correctness bug for a code-intelligence
        # graph, not an acceptable simplification -- so start_line is now
        # part of the identity, since two symbols can share every other
        # attribute but never occupy the same line.
        self._symbol_index: dict[tuple[str, str, str, int], int] = {}

    # ── Node creation ────────────────────────────────────────────────────

    def add_file(self, repo: str, path: str, language: str) -> int:
        key = (repo, path)
        if key in self._file_index:
            return self._file_index[key]
        idx = self.graph.add_node(FileNode(repo=repo, path=path, language=language))
        self._file_index[key] = idx
        return idx

    def add_symbol(
        self,
        repo: str,
        name: str,
        kind: str,
        file_path: str,
        start_line: int,
        end_line: int,
        parent_class: Optional[str] = None,
    ) -> int:
        qualified_name = f"{parent_class}.{name}" if parent_class else name
        key = (repo, file_path, qualified_name, start_line)
        if key in self._symbol_index:
            return self._symbol_index[key]

        idx = self.graph.add_node(SymbolNode(
            repo=repo, name=name, kind=kind, file_path=file_path,
            start_line=start_line, end_line=end_line, parent_class=parent_class,
        ))
        self._symbol_index[key] = idx

        # Auto-wire the CONTAINS edge from the file to this symbol.
        # The file node must already exist — add_symbol assumes add_file
        # was called first for this file_path. We don't call add_file
        # implicitly here because we don't know the language at this point.
        file_key = (repo, file_path)
        if file_key in self._file_index:
            file_idx = self._file_index[file_key]
            if not self.graph.has_edge(file_idx, idx):
                self.graph.add_edge(file_idx, idx, "CONTAINS")

        return idx

    # ── Edge creation ────────────────────────────────────────────────────

    def add_import(self, repo: str, from_file: str, to_file: str) -> Optional[int]:
        """
        Record that from_file imports something from to_file.
        Returns None (and does nothing) if to_file isn't a known file node —
        this happens for external/third-party imports (e.g. `import os`,
        `from typing import Optional`), which we deliberately don't graph.
        """
        from_key = (repo, from_file)
        to_key = (repo, to_file)
        if from_key not in self._file_index or to_key not in self._file_index:
            return None
        from_idx = self._file_index[from_key]
        to_idx = self._file_index[to_key]
        if self.graph.has_edge(from_idx, to_idx):
            return None  # already recorded, multigraph=False means no duplicate needed
        return self.graph.add_edge(from_idx, to_idx, "IMPORTS")

    # ── Queries ──────────────────────────────────────────────────────────

    def get_file_index(self, repo: str, path: str) -> Optional[int]:
        return self._file_index.get((repo, path))

    def get_symbol_index(self, repo: str, file_path: str, qualified_name: str, start_line: Optional[int] = None) -> Optional[int]:
        """
        Look up a symbol node index. If start_line is given, returns the
        exact match. If not, returns the FIRST match by insertion order --
        callers that need to disambiguate between multiple same-named
        symbols (e.g. a @property/@setter pair, both named "auth") should
        use find_all_symbol_matches() instead, which is what fetch_symbol
        will use to build a disambiguation list rather than silently
        picking one.
        """
        if start_line is not None:
            return self._symbol_index.get((repo, file_path, qualified_name, start_line))
        matches = self.find_all_symbol_matches(repo, file_path, qualified_name)
        return matches[0][1] if matches else None

    def find_all_symbol_matches(
        self, repo: str, file_path: str, qualified_name: str
    ) -> list[tuple[int, int]]:
        """
        Returns [(start_line, node_index), ...] for every symbol matching
        this (repo, file_path, qualified_name) -- there can legitimately be
        more than one (property getter/setter pairs, TS get/set accessors).
        This is the correct entry point for a future fetch_symbol tool:
        if len(result) > 1, present a disambiguation list to the caller
        instead of guessing which one they meant.
        """
        out = []
        for (r, fp, qn, line), idx in self._symbol_index.items():
            if r == repo and fp == file_path and qn == qualified_name:
                out.append((line, idx))
        return sorted(out)

    def symbols_in_file(self, repo: str, file_path: str) -> list[SymbolNode]:
        file_idx = self.get_file_index(repo, file_path)
        if file_idx is None:
            return []
        out = []
        for succ_idx in self.graph.successor_indices(file_idx):
            payload = self.graph[succ_idx]
            if isinstance(payload, SymbolNode):
                out.append(payload)
        return out

    def files_importing(self, repo: str, target_file: str) -> list[str]:
        """Reverse lookup: which files import from target_file. Answers
        questions like httpx-T2, got-T4, got-T6 directly."""
        target_idx = self.get_file_index(repo, target_file)
        if target_idx is None:
            return []
        out = []
        for pred_idx in self.graph.predecessor_indices(target_idx):
            payload = self.graph[pred_idx]
            if isinstance(payload, FileNode):
                out.append(payload.path)
        return out

    def stats(self) -> dict:
        file_count = sum(1 for n in self.graph.nodes() if isinstance(n, FileNode))
        symbol_count = sum(1 for n in self.graph.nodes() if isinstance(n, SymbolNode))
        contains_edges = sum(1 for e in self.graph.edges() if e == "CONTAINS")
        import_edges = sum(1 for e in self.graph.edges() if e == "IMPORTS")
        return {
            "files": file_count,
            "symbols": symbol_count,
            "contains_edges": contains_edges,
            "import_edges": import_edges,
            "total_nodes": self.graph.num_nodes(),
            "total_edges": self.graph.num_edges(),
        }


# ── SQLite persistence ───────────────────────────────────────────────────────
# We persist as plain rows (not a serialized graph blob) so the data is
# queryable directly with SQL too, and so a corrupted/interrupted write
# doesn't lose the whole graph — only the batch in progress.

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    PRIMARY KEY (repo, path)
);

CREATE TABLE IF NOT EXISTS symbols (
    repo TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    parent_class TEXT,
    pagerank_score REAL,
    PRIMARY KEY (repo, file_path, qualified_name, start_line),
    FOREIGN KEY (repo, file_path) REFERENCES files(repo, path)
);

CREATE TABLE IF NOT EXISTS imports (
    repo TEXT NOT NULL,
    from_file TEXT NOT NULL,
    to_file TEXT NOT NULL,
    PRIMARY KEY (repo, from_file, to_file)
);

CREATE TABLE IF NOT EXISTS symbol_edges (
    repo TEXT NOT NULL,
    edge_type TEXT NOT NULL,          -- 'CALLS' | 'INSTANTIATES' | 'EXTENDS'
    from_file TEXT NOT NULL,
    from_qualified_name TEXT NOT NULL,
    from_start_line INTEGER NOT NULL,
    to_file TEXT NOT NULL,
    to_qualified_name TEXT NOT NULL,
    to_start_line INTEGER NOT NULL,
    PRIMARY KEY (repo, edge_type, from_file, from_qualified_name, from_start_line,
                 to_file, to_qualified_name, to_start_line)
);

CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(repo, file_path);
CREATE INDEX IF NOT EXISTS idx_imports_to ON imports(repo, to_file);
CREATE INDEX IF NOT EXISTS idx_symbol_edges_from ON symbol_edges(repo, from_file, from_qualified_name, from_start_line);
CREATE INDEX IF NOT EXISTS idx_symbol_edges_to ON symbol_edges(repo, to_file, to_qualified_name, to_start_line);
CREATE INDEX IF NOT EXISTS idx_symbol_edges_type ON symbol_edges(repo, edge_type);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def save_graph(cg: CodeGraph, db_path: str):
    conn = init_db(db_path)
    cur = conn.cursor()

    for node in cg.graph.nodes():
        if isinstance(node, FileNode):
            cur.execute(
                "INSERT OR REPLACE INTO files (repo, path, language) VALUES (?, ?, ?)",
                (node.repo, node.path, node.language),
            )

    for node in cg.graph.nodes():
        if isinstance(node, SymbolNode):
            qualified_name = f"{node.parent_class}.{node.name}" if node.parent_class else node.name
            cur.execute(
                """INSERT OR REPLACE INTO symbols
                   (repo, qualified_name, name, kind, file_path, start_line, end_line, parent_class, pagerank_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (node.repo, qualified_name, node.name, node.kind, node.file_path,
                 node.start_line, node.end_line, node.parent_class, getattr(node, "pagerank_score", None)),
            )

    for u, v, payload in cg.graph.weighted_edge_list():
        if payload == "IMPORTS":
            u_node = cg.graph[u]
            v_node = cg.graph[v]
            if isinstance(u_node, FileNode) and isinstance(v_node, FileNode):
                cur.execute(
                    "INSERT OR REPLACE INTO imports (repo, from_file, to_file) VALUES (?, ?, ?)",
                    (u_node.repo, u_node.path, v_node.path),
                )

        elif payload in ("CALLS", "INSTANTIATES", "EXTENDS"):
            u_node = cg.graph[u]
            v_node = cg.graph[v]
            if isinstance(u_node, SymbolNode) and isinstance(v_node, SymbolNode):
                u_qname = f"{u_node.parent_class}.{u_node.name}" if u_node.parent_class else u_node.name
                v_qname = f"{v_node.parent_class}.{v_node.name}" if v_node.parent_class else v_node.name
                cur.execute(
                    """INSERT OR REPLACE INTO symbol_edges
                       (repo, edge_type, from_file, from_qualified_name, from_start_line,
                        to_file, to_qualified_name, to_start_line)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (u_node.repo, payload, u_node.file_path, u_qname, u_node.start_line,
                     v_node.file_path, v_qname, v_node.start_line),
                )

    conn.commit()
    conn.close()


# ── Self-test ─────────────────────────────────────────────────────────────

def _self_test():
    """
    Build a tiny synthetic graph mirroring the got source/index.ts case we
    found in Step 7 verification, and confirm the queries return correct
    answers. This runs with no dependency on the real repos, so it's a
    fast sanity check of the schema logic itself.
    """
    cg = CodeGraph()

    cg.add_file("got", "source/index.ts", "typescript")
    cg.add_file("got", "source/create.ts", "typescript")
    cg.add_symbol("got", "create", "function", "source/create.ts", 10, 25)

    cg.add_import("got", "source/index.ts", "source/create.ts")

    stats = cg.stats()
    assert stats["files"] == 2, f"expected 2 files, got {stats['files']}"
    assert stats["symbols"] == 1, f"expected 1 symbol, got {stats['symbols']}"
    assert stats["import_edges"] == 1, f"expected 1 import edge, got {stats['import_edges']}"

    importers = cg.files_importing("got", "source/create.ts")
    assert importers == ["source/index.ts"], f"unexpected importers: {importers}"

    symbols = cg.symbols_in_file("got", "source/create.ts")
    assert len(symbols) == 1 and symbols[0].name == "create"

    print("Self-test passed.")
    print(f"Stats: {stats}")
    print(f"Files importing source/create.ts: {importers}")


if __name__ == "__main__":
    _self_test()

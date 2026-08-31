"""
The complete Phase 1 pipeline, run in the correct order on ONE graph
object, saved ONCE at the end.

This is the fix for a real gap found via verification: resolve_calls.py,
resolve_calls_typed.py, resolve_calls_typed_ts.py, and resolve_inheritance.py
were each written and tested as standalone scripts -- each builds its OWN
graph via build_graph_with_imports() and saves its own (partial) result.
Running resolve_inheritance.py last meant its save_graph() call persisted
a graph that only ever had EXTENDS added to it, silently overwriting any
previous CALLS/INSTANTIATES-containing database. Confirmed via direct
SQL query against the real database: symbol_edges contained only EXTENDS
rows (65), zero CALLS or INSTANTIATES, despite both having been resolved
correctly and verified in their own standalone test runs.

Correct order (each phase depends on state from the previous one):
  1. Files + symbols + imports   (build_graph_with_imports)
  2. Base CALLS/INSTANTIATES     (resolve_calls.py)          -- needs imports resolved
  3. Typed CALLS (Python)        (resolve_calls_typed.py)    -- needs base CALLS as a foundation to add on top of
  4. Typed CALLS (TypeScript)    (resolve_calls_typed_ts.py) -- same, TS side
  5. EXTENDS                     (resolve_inheritance.py)    -- independent of CALLS, but run last for clarity

Run with:
    python3 src/build_full_graph.py
"""

import subprocess
from pathlib import Path
import sys

from config import DB_PATH, REPOS_DIR
from graph_schema import save_graph
from resolve_imports import build_graph_with_imports
from resolve_calls import resolve_calls_for_file, CallResolutionStats
from resolve_calls_typed import process_python_file, TypedCallStats
from resolve_calls_typed_ts import process_typescript_file, TypedCallStatsTS
from resolve_inheritance import resolve_inheritance_for_file, InheritanceStats
from extract_symbols import find_source_files


def ensure_repo_cloned(repo_spec: str, repos_dir: Path = REPOS_DIR) -> tuple[str, Path]:
    """
    Ensures a repository is cloned locally.
    Supports full URLs (https://github.com/psf/requests.git), 'owner/repo' (psf/requests), or local folder names.
    Performs full git clone (no shallow depth) so PyDriller history mining works completely.
    """
    repos_dir.mkdir(parents=True, exist_ok=True)
    if repo_spec.startswith("http://") or repo_spec.startswith("https://") or repo_spec.startswith("git@"):
        clone_url = repo_spec
        name = repo_spec.rstrip("/").split("/")[-1].removesuffix(".git")
    elif "/" in repo_spec:
        name = repo_spec.split("/")[-1]
        clone_url = f"https://github.com/{repo_spec}.git"
    else:
        name = repo_spec
        clone_url = f"https://github.com/{repo_spec}.git"

    dest = repos_dir / name
    if not dest.exists():
        print(f"Cloning {clone_url} into {dest} (full history)...")
        subprocess.run(["git", "clone", clone_url, str(dest)], check=True)
    return name, dest


def discover_repos(args: list[str] | None = None, repos_dir: Path = REPOS_DIR) -> list[tuple[str, Path]]:
    if args:
        return [ensure_repo_cloned(arg, repos_dir) for arg in args]

    # Deterministic default order for baseline repos, followed by any additional local repos
    known = [("httpx", repos_dir / "httpx"), ("got", repos_dir / "got")]
    repos = [(name, path) for name, path in known if path.exists()]
    seen = {name for name, _ in repos}

    if repos_dir.exists():
        for d in sorted(repos_dir.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name not in seen:
                repos.append((d.name, d))

    return repos


def main():
    repos = discover_repos(sys.argv[1:] if len(sys.argv) > 1 else None)

    print("=" * 70)
    print("PHASE 1 (complete): building the full graph in one pipeline")
    print("=" * 70)

    repos_dict = {name: root for name, root in repos}
    print("\n[1/5] Files, symbols, imports...")
    cg = build_graph_with_imports(repos=repos_dict)

    print("\n[2/5] Base CALLS/INSTANTIATES (self./this. direct calls)...")
    for repo_name, repo_root in repos:
        if not repo_root.exists():
            continue
        call_stats = CallResolutionStats()
        for f in find_source_files(repo_root):
            resolve_calls_for_file(f, repo_root, repo_name, cg, call_stats)
        call_stats.report(repo_name)

    print("\n[3/5] Typed CALLS -- Python (local type inference)...")
    for repo_name, repo_root in repos:
        if not repo_root.exists():
            continue
        py_typed_stats = TypedCallStats()
        py_files = find_source_files(repo_root, extensions=(".py",))
        if py_files:
            for py_file in py_files:
                process_python_file(py_file, repo_root, repo_name, cg, py_typed_stats)
            py_typed_stats.report(repo_name)

    print("\n[4/5] Typed CALLS -- TypeScript (local type inference)...")
    for repo_name, repo_root in repos:
        if not repo_root.exists():
            continue
        ts_typed_stats = TypedCallStatsTS()
        ts_files = find_source_files(repo_root, extensions=(".ts", ".tsx"))
        if ts_files:
            for ts_file in ts_files:
                process_typescript_file(ts_file, repo_root, repo_name, cg, ts_typed_stats)
            ts_typed_stats.report(repo_name)

    print("\n[5/5] EXTENDS (inheritance)...")
    for repo_name, repo_root in repos:
        if not repo_root.exists():
            continue
        inherit_stats = InheritanceStats()
        for f in find_source_files(repo_root):
            resolve_inheritance_for_file(f, repo_root, repo_name, cg, inherit_stats)
        inherit_stats.report(repo_name)

    print("\n" + "=" * 70)
    print("FINAL GRAPH STATS (single combined graph, all passes applied)")
    print("=" * 70)
    stats = cg.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    calls = sum(1 for e in cg.graph.edges() if e == "CALLS")
    instantiates = sum(1 for e in cg.graph.edges() if e == "INSTANTIATES")
    extends = sum(1 for e in cg.graph.edges() if e == "EXTENDS")
    print(f"  CALLS edges: {calls}")
    print(f"  INSTANTIATES edges: {instantiates}")
    print(f"  EXTENDS edges: {extends}")

    save_graph(cg, DB_PATH)
    print(f"\nGraph persisted to {DB_PATH} (ONE combined save, all edge types included)")


if __name__ == "__main__":
    main()

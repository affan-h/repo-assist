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

from pathlib import Path
import sys
sys.path.insert(0, "src")

from graph_schema import save_graph
from resolve_imports import build_graph_with_imports
from resolve_calls import resolve_python_calls, resolve_typescript_calls, CallResolutionStats
from resolve_calls_typed import process_python_file, TypedCallStats
from resolve_calls_typed_ts import process_typescript_file, TypedCallStatsTS
from resolve_inheritance import resolve_python_inheritance, resolve_typescript_inheritance, InheritanceStats
from extract_symbols import should_skip_dir


def main():
    httpx_root = Path("repos/httpx")
    got_root = Path("repos/got")

    print("=" * 70)
    print("PHASE 1 (complete): building the full graph in one pipeline")
    print("=" * 70)

    print("\n[1/5] Files, symbols, imports...")
    cg = build_graph_with_imports()

    print("\n[2/5] Base CALLS/INSTANTIATES (self./this. direct calls)...")
    httpx_call_stats = CallResolutionStats()
    for py_file in sorted(httpx_root.rglob("*.py")):
        if any(should_skip_dir(p) for p in py_file.relative_to(httpx_root).parts[:-1]):
            continue
        resolve_python_calls(py_file, httpx_root, cg, httpx_call_stats)
    httpx_call_stats.report("httpx")

    got_call_stats = CallResolutionStats()
    for ts_file in sorted(got_root.rglob("*.ts")):
        if any(should_skip_dir(p) for p in ts_file.relative_to(got_root).parts[:-1]):
            continue
        resolve_typescript_calls(ts_file, got_root, cg, got_call_stats)
    got_call_stats.report("got")

    print("\n[3/5] Typed CALLS -- Python (local type inference)...")
    py_typed_stats = TypedCallStats()
    for py_file in sorted(httpx_root.rglob("*.py")):
        if any(should_skip_dir(p) for p in py_file.relative_to(httpx_root).parts[:-1]):
            continue
        process_python_file(py_file, httpx_root, "httpx", cg, py_typed_stats)
    py_typed_stats.report("httpx")

    print("\n[4/5] Typed CALLS -- TypeScript (local type inference)...")
    ts_typed_stats = TypedCallStatsTS()
    for ts_file in sorted(got_root.rglob("*.ts")):
        if any(should_skip_dir(p) for p in ts_file.relative_to(got_root).parts[:-1]):
            continue
        process_typescript_file(ts_file, got_root, "got", cg, ts_typed_stats)
    ts_typed_stats.report("got")

    print("\n[5/5] EXTENDS (inheritance)...")
    httpx_inherit_stats = InheritanceStats()
    for py_file in sorted(httpx_root.rglob("*.py")):
        if any(should_skip_dir(p) for p in py_file.relative_to(httpx_root).parts[:-1]):
            continue
        resolve_python_inheritance(py_file, httpx_root, cg, httpx_inherit_stats)
    httpx_inherit_stats.report("httpx")

    got_inherit_stats = InheritanceStats()
    for ts_file in sorted(got_root.rglob("*.ts")):
        if any(should_skip_dir(p) for p in ts_file.relative_to(got_root).parts[:-1]):
            continue
        resolve_typescript_inheritance(ts_file, got_root, cg, got_inherit_stats)
    got_inherit_stats.report("got")

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

    save_graph(cg, "data/code_graph.db")
    print("\nGraph persisted to data/code_graph.db (ONE combined save, all edge types included)")


if __name__ == "__main__":
    main()

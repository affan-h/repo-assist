"""
Item 2 -- Inheritance / EXTENDS edges.

Confirmed tree-sitter shapes before writing this:
  Python:     class_definition -> argument_list -> identifier(s)
              (supports multiple inheritance: class Foo(Bar, Baz))
  TypeScript: class_declaration -> class_heritage -> extends_clause -> identifier
              (single inheritance only, per TS/JS language rules)

This adds:
  1. EXTENDS edges: Class -> ParentClass, for both languages.
  2. resolve_with_inheritance(): given a class and a method name not
     found directly on it, walk up the EXTENDS chain and check each
     ancestor. This directly addresses the attempted_but_class_not_found
     cases from Step 10b -- some of those 78 unresolved lookups are very
     likely inherited methods, not genuine gaps.

Deliberately NOT handling: multiple inheritance method resolution order
(MRO) -- if a class has two parents that both define the same method
name, we return the FIRST parent's version (first in the argument_list),
not Python's actual C3 linearization order. This is a known, accepted
simplification: computing real MRO is a solved but non-trivial algorithm,
and multiple inheritance with colliding method names is rare in practice
for the kind of code we're indexing (neither httpx nor got are expected
to lean heavily on this pattern, though this is an assumption, not yet
verified against the real repos).

Run with:
    python3 src/resolve_inheritance.py
"""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts

import sys
sys.path.insert(0, "src")
from graph_schema import CodeGraph, save_graph
from resolve_imports import build_graph_with_imports
from extract_symbols import should_skip_dir


PY_LANGUAGE = Language(tspython.language())
TS_LANGUAGE = Language(tsts.language_typescript())
py_parser = Parser(PY_LANGUAGE)
ts_parser = Parser(TS_LANGUAGE)


def _text(node, source_code) -> str:
    return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


class InheritanceStats:
    def __init__(self):
        self.extends_edges = 0
        self.classes_with_multiple_parents = 0

    def report(self, label: str):
        print(f"  [{label}] EXTENDS edges added={self.extends_edges} "
              f"classes with 2+ parents={self.classes_with_multiple_parents}")


# ── Python: extract base classes ─────────────────────────────────────────

def extract_python_base_classes(class_node, source_code) -> list[str]:
    """Returns base class names in declaration order. Only handles the
    simple identifier case (class Foo(Bar)) -- does NOT resolve dotted
    base classes (class Foo(module.Bar)) or dynamic bases
    (class Foo(get_base_class())), which stay unresolved by design,
    same "don't guess" discipline as everywhere else in this project."""
    bases = []
    for child in class_node.children:
        if child.type == "argument_list":
            for c in child.children:
                if c.type == "identifier":
                    bases.append(_text(c, source_code))
    return bases


def resolve_python_inheritance(file_path: Path, repo_root: Path, cg: CodeGraph, stats: InheritanceStats):
    rel_path = str(file_path.relative_to(repo_root))
    source_code = file_path.read_bytes()
    tree = py_parser.parse(source_code)
    root = tree.root_node

    def walk(node):
        if node.type == "class_definition":
            cls_name = _text(node.child_by_field_name("name"), source_code)
            bases = extract_python_base_classes(node, source_code)
            if len(bases) > 1:
                stats.classes_with_multiple_parents += 1
            for base_name in bases:
                child_idx = cg.find_all_symbol_matches("httpx", rel_path, cls_name)
                # Base class may be defined in ANY file (not just this one) --
                # search the whole repo, same approach as call resolution.
                parent_match = _find_class_in_repo(cg, "httpx", base_name)
                if child_idx and parent_match is not None:
                    child_node_idx = child_idx[0][1]
                    parent_node_idx = parent_match
                    if not cg.graph.has_edge(child_node_idx, parent_node_idx):
                        cg.graph.add_edge(child_node_idx, parent_node_idx, "EXTENDS")
                        stats.extends_edges += 1
        for child in node.children:
            walk(child)

    walk(root)


# ── TypeScript: extract base class ───────────────────────────────────────

def extract_typescript_base_class(class_node, source_code) -> str | None:
    """TS classes support only single inheritance -- one extends_clause,
    one identifier."""
    for child in class_node.children:
        if child.type == "class_heritage":
            for c in child.children:
                if c.type == "extends_clause":
                    for cc in c.children:
                        if cc.type == "identifier":
                            return _text(cc, source_code)
    return None


def resolve_typescript_inheritance(file_path: Path, repo_root: Path, cg: CodeGraph, stats: InheritanceStats):
    rel_path = str(file_path.relative_to(repo_root))
    source_code = file_path.read_bytes()
    tree = ts_parser.parse(source_code)
    root = tree.root_node

    def walk(node):
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                cls_name = _text(name_node, source_code)
                base_name = extract_typescript_base_class(node, source_code)
                if base_name is not None:
                    child_idx = cg.find_all_symbol_matches("got", rel_path, cls_name)
                    parent_match = _find_class_in_repo(cg, "got", base_name)
                    if child_idx and parent_match is not None:
                        child_node_idx = child_idx[0][1]
                        parent_node_idx = parent_match
                        if not cg.graph.has_edge(child_node_idx, parent_node_idx):
                            cg.graph.add_edge(child_node_idx, parent_node_idx, "EXTENDS")
                            stats.extends_edges += 1
        for child in node.children:
            walk(child)

    walk(root)


# ── Shared: find a class symbol anywhere in the repo ─────────────────────

def _find_class_in_repo(cg: CodeGraph, repo: str, class_name: str):
    """Search the whole repo for a top-level class symbol matching
    class_name (parent_class is None, kind == 'class'). Returns node
    index or None."""
    for (r, fp, qn, line), idx in cg._symbol_index.items():
        if r != repo:
            continue
        node = cg.graph[idx]
        if node.kind == "class" and node.parent_class is None and node.name == class_name:
            return idx
    return None


# ── Inherited method resolution -- the actual payoff ─────────────────────

def resolve_method_with_inheritance(
    cg: CodeGraph, repo: str, class_name: str, method_name: str, max_depth: int = 5
):
    """
    Given a class name and a method name NOT found directly on that class,
    walk up the EXTENDS chain (as far as it goes, up to max_depth to guard
    against any accidental cycle) and check each ancestor for the method.
    Returns (node_index, file_path, defining_class_name) or None.

    This is the direct fix for Step 10b's attempted_but_class_not_found
    cases -- e.g. self.foo() where foo is defined on a parent class.
    """
    class_idx = _find_class_in_repo(cg, repo, class_name)
    if class_idx is None:
        return None

    current_idx = class_idx
    current_name = class_name
    visited = set()

    for _ in range(max_depth):
        if current_idx in visited:
            break  # guard against a cycle, shouldn't happen but don't hang if it does
        visited.add(current_idx)

        direct = None
        for (r, fp, qn, line), idx in cg._symbol_index.items():
            if r == repo and qn == f"{current_name}.{method_name}":
                direct = (idx, fp, current_name)
                break
        if direct is not None:
            return direct

        parents = [
            succ for succ in cg.graph.successor_indices(current_idx)
            if cg.graph.get_edge_data(current_idx, succ) == "EXTENDS"
        ]
        if not parents:
            return None
        current_idx = parents[0]  # first parent only -- documented MRO simplification
        current_name = cg.graph[current_idx].name

    return None


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Rebuilding graph with files, symbols, imports (Steps 7-9)...\n")
    cg = build_graph_with_imports()

    httpx_root = Path("repos/httpx")
    got_root = Path("repos/got")

    print("Resolving inheritance (EXTENDS edges)...")

    py_stats = InheritanceStats()
    for py_file in sorted(httpx_root.rglob("*.py")):
        if any(should_skip_dir(part) for part in py_file.relative_to(httpx_root).parts[:-1]):
            continue
        resolve_python_inheritance(py_file, httpx_root, cg, py_stats)
    py_stats.report("httpx")

    ts_stats = InheritanceStats()
    for ts_file in sorted(got_root.rglob("*.ts")):
        if any(should_skip_dir(part) for part in ts_file.relative_to(got_root).parts[:-1]):
            continue
        resolve_typescript_inheritance(ts_file, got_root, cg, ts_stats)
    ts_stats.report("got")

    extends_edges = sum(1 for e in cg.graph.edges() if e == "EXTENDS")
    print(f"\nTotal EXTENDS edges in graph: {extends_edges}")

    save_graph(cg, "data/code_graph.db")
    print("Graph persisted to data/code_graph.db")
    print("(files, symbols, imports, and CALLS/INSTANTIATES/EXTENDS edges")
    print(" are all now included in the persisted database.)")


if __name__ == "__main__":
    main()

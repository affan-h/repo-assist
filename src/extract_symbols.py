"""
Symbol extraction across the full repo tree.

Goal: walk every relevant source file in httpx (Python) and got (TypeScript),
extract every class/function/method definition with its exact file + line
location, and print a summary. This is the direct precursor to building
the actual graph nodes in Step 8 — we want to see real numbers (how many
files, how many symbols) before we commit to a graph schema.

Scope, deliberately: Python and TypeScript ONLY. No C++. matching-engine-cpp
was dropped from this project's pinned repos in favor of httpx + got, and
there is no reason to introduce a third language/grammar for a repo that
isn't part of Phase 0's evaluation set.

Run with:
    python3 src/extract_symbols.py
"""

from pathlib import Path
from dataclasses import dataclass, field
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    name: str
    kind: str          # "class" | "function" | "method"
    file_path: str      # relative to repo root, e.g. "httpx/_config.py"
    start_line: int      # 1-indexed
    end_line: int
    parent_class: str | None = None   # set if this is a method


@dataclass
class FileResult:
    file_path: str
    symbols: list[Symbol] = field(default_factory=list)
    parse_error: bool = False
    skipped_reason: str | None = None


# ── Language setup ──────────────────────────────────────────────────────────

PY_LANGUAGE = Language(tspython.language())
# tree_sitter_typescript exposes two grammars: language_typescript() for .ts
# and language_tsx() for .tsx (JSX-in-TS). got is plain TypeScript, no JSX,
# so we only need language_typescript().
TS_LANGUAGE = Language(tsts.language_typescript())

py_parser = Parser(PY_LANGUAGE)
ts_parser = Parser(TS_LANGUAGE)


# ── Directories to skip in both repos ────────────────────────────────────────
# Test files and build artifacts are noise for a first pass — we want to see
# real source symbol counts, not test-suite bulk. We will decide later
# (Phase 2 planning) whether tests should be indexed separately.

SKIP_DIR_NAMES = {
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", "test", "tests",
}


def should_skip_dir(dir_name: str) -> bool:
    return dir_name in SKIP_DIR_NAMES or dir_name.startswith(".")


# ── Python extraction ────────────────────────────────────────────────────────

def extract_python_symbols(file_path: Path, repo_root: Path) -> FileResult:
    rel_path = str(file_path.relative_to(repo_root))
    source_code = file_path.read_bytes()
    tree = py_parser.parse(source_code)
    root = tree.root_node

    result = FileResult(file_path=rel_path, parse_error=root.has_error)

    def get_name(node) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return "?"
        return source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

    def walk(node, current_class: str | None):
        if node.type == "class_definition":
            name = get_name(node)
            result.symbols.append(Symbol(
                name=name, kind="class", file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))
            for child in node.children:
                walk(child, current_class=name)
            return  # already recursed with updated current_class

        if node.type == "function_definition":
            name = get_name(node)
            kind = "method" if current_class else "function"
            result.symbols.append(Symbol(
                name=name, kind=kind, file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_class=current_class,
            ))
            # Do not recurse into function bodies for nested defs in this
            # pass — nested/inner functions are rare and out of scope for
            # the symbol graph's first version.
            return

        for child in node.children:
            walk(child, current_class)

    walk(root, current_class=None)
    return result


# ── TypeScript extraction ────────────────────────────────────────────────────
# TypeScript's grammar has more definition shapes than Python: class
# declarations, function declarations, arrow functions assigned to const,
# and exported variants of each. We handle the common cases explicitly and
# print anything unrecognized so we can see what we're missing, rather than
# silently dropping symbols.

def extract_typescript_symbols(file_path: Path, repo_root: Path) -> FileResult:
    rel_path = str(file_path.relative_to(repo_root))
    source_code = file_path.read_bytes()
    tree = ts_parser.parse(source_code)
    root = tree.root_node

    result = FileResult(file_path=rel_path, parse_error=root.has_error)

    # Graceful degradation: collect the byte ranges of every ERROR node.
    # A single unsupported construct (e.g. "export type * from ...", which
    # tree-sitter-typescript 0.23.2 does not yet parse) should not discard
    # every valid symbol in the rest of the file. We only exclude symbols
    # whose own span overlaps an ERROR node's span.
    error_spans: list[tuple[int, int]] = []

    def collect_error_spans(node):
        if node.type == "ERROR":
            error_spans.append((node.start_byte, node.end_byte))
        for child in node.children:
            collect_error_spans(child)

    collect_error_spans(root)

    def overlaps_error(node) -> bool:
        return any(
            node.start_byte < err_end and node.end_byte > err_start
            for err_start, err_end in error_spans
        )

    def get_name(node) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return "?"
        return source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

    def walk(node, current_class: str | None):
        node_type = node.type

        # class Foo { ... }  — including "export class Foo"
        if node_type == "class_declaration":
            name = get_name(node)
            if not overlaps_error(node):
                result.symbols.append(Symbol(
                    name=name, kind="class", file_path=rel_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))
            for child in node.children:
                walk(child, current_class=name)
            return

        # function foo() { ... }  — including "export function foo"
        if node_type == "function_declaration":
            name = get_name(node)
            kind = "method" if current_class else "function"
            if not overlaps_error(node):
                result.symbols.append(Symbol(
                    name=name, kind=kind, file_path=rel_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_class=current_class,
                ))
            return

        # method_definition covers class methods: foo() { ... } inside a
        # class body. IMPORTANT: this node type also appears for method
        # shorthand inside plain object literals (e.g.
        # `suite.add('x', { async fn(x) { ... } })`), which is NOT a class
        # method. We must check the immediate parent is class_body before
        # recording it — otherwise callback-style object methods get
        # misfiled as class methods with a stale/wrong current_class.
        if node_type == "method_definition":
            if node.parent is not None and node.parent.type == "class_body":
                if not overlaps_error(node):
                    name = get_name(node)
                    result.symbols.append(Symbol(
                        name=name, kind="method", file_path=rel_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_class=current_class,
                    ))
            # No recursion needed — method_definition's own children are
            # just its params/body, which contain no further top-level
            # symbols we track in this pass.
            return

        # const foo = () => { ... }  or  const foo = function() { ... }
        # This is common in got's codebase — arrow functions assigned to consts.
        if node_type == "lexical_declaration":
            if not overlaps_error(node):
                for declarator in node.children:
                    if declarator.type != "variable_declarator":
                        continue
                    name_node = declarator.child_by_field_name("name")
                    value_node = declarator.child_by_field_name("value")
                    if name_node is None or value_node is None:
                        continue
                    if value_node.type in ("arrow_function", "function_expression"):
                        name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                        kind = "method" if current_class else "function"
                        result.symbols.append(Symbol(
                            name=name, kind=kind, file_path=rel_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            parent_class=current_class,
                        ))
            # DO recurse into the arrow_function/function_expression bodies
            # we just recorded above -- CHANGED from the original "don't
            # descend" rule after confirming a real cost via ground-truth
            # audit: got.extend() (a real, central public API method) is
            # defined INSIDE a factory function's body ("const create = (...)
            # => { ... got.extend = (...) => {...}; ... }"), and the old
            # rule silently dropped it and everything else nested this way.
            # The original concern (object-literal methods getting misfiled
            # with a stale current_class) is a SEPARATE, already-guarded
            # case -- the method_definition handler above only records a
            # method if node.parent.type == "class_body", so a
            # method-shorthand object literal inside this body still won't
            # be misfiled even with full descent enabled here.
            for child in node.children:
                walk(child, current_class)
            return

        # x.propertyName = (...) => { ... }  or  x.propertyName = function() {...}
        # CONFIRMED REAL GAP (ground-truth audit, W6 "got.extend()" question):
        # this is a DISTINCT tree-sitter shape from lexical_declaration
        # ("const x = ...") -- it's an assignment_expression whose left side
        # is a member_expression (object.property), not a plain identifier.
        # Real example: "got.extend = (...instancesOrOptions) => {...}" in
        # source/create.ts was silently never captured as a symbol at all
        # before this fix, confirmed via direct query (zero rows matching
        # "extend" in the whole symbols table despite the function being
        # real and central to got's public API). Confirmed exact real
        # tree-sitter shape via direct parser inspection before writing this:
        # expression_statement -> assignment_expression -> member_expression
        # (object "." property) "=" arrow_function/function_expression.
        if node_type == "expression_statement":
            for child in node.children:
                if child.type == "assignment_expression":
                    left = child.child_by_field_name("left")
                    right = child.child_by_field_name("right")
                    if (left is not None and right is not None
                            and left.type == "member_expression"
                            and right.type in ("arrow_function", "function_expression")):
                        prop_node = left.child_by_field_name("property")
                        if prop_node is not None and not overlaps_error(child):
                            name = source_code[prop_node.start_byte:prop_node.end_byte].decode("utf-8", errors="replace")
                            kind = "method" if current_class else "function"
                            result.symbols.append(Symbol(
                                name=name, kind=kind, file_path=rel_path,
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                parent_class=current_class,
                            ))
                            # DO descend into the arrow function's body too
                            # (same change as lexical_declaration above) --
                            # a real nested definition inside got.extend's
                            # own body should still be found, not silently
                            # dropped just because we already recorded the
                            # outer assignment as a symbol.
                            walk(right, current_class)
                            return
            # fell through: not the assignment-to-function pattern, walk normally
            for child in node.children:
                walk(child, current_class)
            return

        for child in node.children:
            walk(child, current_class)

    walk(root, current_class=None)
    return result


# ── Repo walkers ──────────────────────────────────────────────────────────

def walk_repo(repo_root: Path, extension: str, extractor) -> list[FileResult]:
    results = []
    for path in sorted(repo_root.rglob(f"*{extension}")):
        if any(should_skip_dir(part) for part in path.relative_to(repo_root).parts[:-1]):
            continue
        try:
            results.append(extractor(path, repo_root))
        except Exception as e:
            rel = str(path.relative_to(repo_root))
            results.append(FileResult(file_path=rel, skipped_reason=f"{type(e).__name__}: {e}"))
    return results


def print_summary(repo_name: str, results: list[FileResult]):
    total_files = len(results)
    error_files = [r for r in results if r.parse_error]
    skipped_files = [r for r in results if r.skipped_reason]
    all_symbols = [s for r in results for s in r.symbols]

    classes = [s for s in all_symbols if s.kind == "class"]
    functions = [s for s in all_symbols if s.kind == "function"]
    methods = [s for s in all_symbols if s.kind == "method"]

    print("=" * 60)
    print(f"{repo_name}")
    print("=" * 60)
    print(f"  Files parsed:      {total_files}")
    print(f"  Files with errors: {len(error_files)}")
    print(f"  Files skipped:     {len(skipped_files)}")
    print(f"  Total symbols:     {len(all_symbols)}")
    print(f"    Classes:         {len(classes)}")
    print(f"    Functions:       {len(functions)}")
    print(f"    Methods:         {len(methods)}")

    if error_files:
        print(f"\n  Files with parse errors (first 5):")
        for r in error_files[:5]:
            print(f"    - {r.file_path}")

    if skipped_files:
        print(f"\n  Files skipped due to exceptions (first 5):")
        for r in skipped_files[:5]:
            print(f"    - {r.file_path}: {r.skipped_reason}")

    print()


def main():
    httpx_root = Path("repos/httpx")
    got_root = Path("repos/got")

    if not httpx_root.exists() or not got_root.exists():
        print("ERROR: repos/httpx or repos/got not found. Check Step 5 (git clone) completed.")
        return

    print("Extracting Python symbols from httpx...\n")
    httpx_results = walk_repo(httpx_root, ".py", extract_python_symbols)
    print_summary("httpx (Python)", httpx_results)

    print("Extracting TypeScript symbols from got...\n")
    got_results = walk_repo(got_root, ".ts", extract_typescript_symbols)
    print_summary("got (TypeScript)", got_results)

    # Show a handful of real extracted symbols from each, for a sanity check
    print("=" * 60)
    print("SAMPLE: first 10 symbols from httpx")
    print("=" * 60)
    httpx_symbols = [s for r in httpx_results for s in r.symbols]
    for s in httpx_symbols[:10]:
        parent = f" (in {s.parent_class})" if s.parent_class else ""
        print(f"  {s.kind:10} {s.name:25}{parent:20} {s.file_path}:{s.start_line}")

    print()
    print("=" * 60)
    print("SAMPLE: first 10 symbols from got")
    print("=" * 60)
    got_symbols = [s for r in got_results for s in r.symbols]
    for s in got_symbols[:10]:
        parent = f" (in {s.parent_class})" if s.parent_class else ""
        print(f"  {s.kind:10} {s.name:25}{parent:20} {s.file_path}:{s.start_line}")


if __name__ == "__main__":
    main()

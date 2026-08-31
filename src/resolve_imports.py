"""
Step 9 — Import resolution.

Walks every Python file in httpx and every TypeScript file in got, extracts
import statements, resolves them to actual files already known to the
CodeGraph (built in Step 7/8), and adds IMPORTS edges.

Resolution rules, confirmed against real tree-sitter output (not assumed):

PYTHON (httpx):
    import_from_statement -> relative_import -> import_prefix (dots) + dotted_name (module)
    Single dot "." = same package as the importing file.
    Multiple dots ".." = go up one package level per extra dot.
    Absolute imports (e.g. "import os") have no relative_import child at all
    and are external -> always a no-op, per graph_schema.py's design.

TYPESCRIPT (got):
    import_statement -> string -> string_fragment holds the raw path, e.g.
    './create.js' or '../core/options.js'. Relative paths start with
    "./" or "../". Bare paths (e.g. 'node:https', 'benchmark', 'request')
    are external packages -> no-op. TypeScript source files are referenced
    with a .js extension in the import path even though the file on disk
    is .ts (standard Node ESM convention) -- we must rewrite .js -> .ts
    before checking if the file exists in our graph.

Run with:
    python3 src/resolve_imports.py
"""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts

from config import DB_PATH
from graph_schema import CodeGraph, save_graph
from extract_symbols import (
    walk_repo,
    extract_python_symbols,
    extract_typescript_symbols,
    extract_symbols_for_file,
    find_source_files,
    should_skip_dir,
    tsx_parser,
)


PY_LANGUAGE = Language(tspython.language())
TS_LANGUAGE = Language(tsts.language_typescript())
py_parser = Parser(PY_LANGUAGE)
ts_parser = Parser(TS_LANGUAGE)


# ── Python import resolution ─────────────────────────────────────────────────

def resolve_python_imports(file_path: Path, repo_root: Path) -> list[str]:
    """
    Returns a list of file paths (relative to repo_root) that this file
    imports from, for every RESOLVABLE relative import. External/absolute
    imports are silently excluded -- they cannot resolve to a file in our
    graph, per graph_schema.py's design.
    """
    rel_path = file_path.relative_to(repo_root)
    source_code = file_path.read_bytes()
    tree = py_parser.parse(source_code)
    root = tree.root_node

    resolved: list[str] = []

    def walk(node):
        if node.type == "import_from_statement":
            relative_import_node = None
            for child in node.children:
                if child.type == "relative_import":
                    relative_import_node = child
                    break

            if relative_import_node is not None:
                # Count leading dots (import_prefix children, each is one ".")
                dot_count = sum(
                    1 for c in relative_import_node.children if c.type == "import_prefix"
                )
                # Get the module name being imported FROM (not the names
                # being imported). Two distinct grammar shapes, confirmed
                # by direct tree-sitter inspection:
                #   "from ._config import X"  -> dotted_name ("_config") is
                #                                 INSIDE relative_import.
                #   "from . import _api"      -> relative_import has ONLY
                #                                 a dot, no dotted_name inside
                #                                 it. The imported name
                #                                 ("_api") is a SIBLING of
                #                                 relative_import on the
                #                                 outer import_from_statement,
                #                                 and in this form it IS the
                #                                 submodule being imported
                #                                 (from the current package),
                #                                 not a symbol inside a module.
                module_name = None
                for c in relative_import_node.children:
                    if c.type == "dotted_name":
                        module_name = source_code[c.start_byte:c.end_byte].decode("utf-8", errors="replace")
                        break

                if module_name is None:
                    # "from . import x" form: the first dotted_name that is
                    # a DIRECT CHILD of import_from_statement (i.e. NOT
                    # inside relative_import) is the submodule name.
                    for c in node.children:
                        if c.type == "dotted_name":
                            module_name = source_code[c.start_byte:c.end_byte].decode(
                                "utf-8", errors="replace"
                            )
                            break
                        if c.type == "wildcard_import":
                            # "from . import *" -- genuinely ambiguous which
                            # submodule(s) are involved without deeper
                            # resolution. Leave module_name as None; this
                            # will correctly fall through to the
                            # __init__.py case below, which is the closest
                            # reasonable interpretation for a wildcard.
                            break

                target = _resolve_python_relative(rel_path, dot_count, module_name, repo_root)
                if target is not None:
                    resolved.append(target)

            # Absolute imports (no relative_import child) are external.
            # Deliberately no-op, matching graph_schema.py's design.

        for child in node.children:
            walk(child)

    walk(root)
    return resolved


def _resolve_python_relative(
    importing_file: Path, dot_count: int, module_name: str | None, repo_root: Path
) -> str | None:
    """
    dot_count=1 means "same package as importing_file". Each extra dot
    means "go up one more package level" (dot_count=2 -> parent package).
    module_name is None only for "from . import *" (wildcard) -- in every
    other case (including "from . import x", where x is a submodule name)
    module_name is now populated by the caller before this function runs.
    """
    # Start at the importing file's own directory (its package).
    current_dir = importing_file.parent
    # dot_count=1 -> stay in current_dir. dot_count=2 -> go up one. etc.
    for _ in range(dot_count - 1):
        current_dir = current_dir.parent

    if module_name is None:
        # "from . import x" -- refers to current_dir/__init__.py
        candidate = current_dir / "__init__.py"
    else:
        # module_name might itself be dotted, e.g. "foo.bar" -> foo/bar.py
        candidate = current_dir / (module_name.replace(".", "/") + ".py")

    full_candidate = repo_root / candidate
    if full_candidate.exists():
        return str(candidate)
    return None


# ── TypeScript import resolution ─────────────────────────────────────────────

def resolve_typescript_imports(file_path: Path, repo_root: Path) -> list[str]:
    rel_path = file_path.relative_to(repo_root)
    source_code = file_path.read_bytes()
    parser = tsx_parser if file_path.suffix == ".tsx" else ts_parser
    tree = parser.parse(source_code)
    root = tree.root_node

    resolved: list[str] = []

    def walk(node):
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "string":
                    for grandchild in child.children:
                        if grandchild.type == "string_fragment":
                            raw_path = source_code[grandchild.start_byte:grandchild.end_byte].decode(
                                "utf-8", errors="replace"
                            )
                            target = _resolve_typescript_relative(rel_path, raw_path, repo_root)
                            if target is not None:
                                resolved.append(target)
        for child in node.children:
            walk(child)

    walk(root)
    return resolved


def _resolve_typescript_relative(importing_file: Path, raw_path: str, repo_root: Path) -> str | None:
    # Bare imports (no leading "./" or "../") are external packages -- no-op.
    # This matches confirmed examples: 'node:https', 'benchmark', 'node-fetch',
    # 'request' all lack a leading dot.
    if not (raw_path.startswith("./") or raw_path.startswith("../")):
        return None

    # Resolve the relative path from the importing file's directory.
    current_dir = importing_file.parent
    # raw_path is like './create.js' or '../core/options.js'
    combined = (current_dir / raw_path).as_posix()

    # Normalize away any ../ segments (e.g. "source/../core" -> "core")
    combined_path = Path(combined)
    parts = []
    for part in combined_path.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part == ".":
            continue
        else:
            parts.append(part)
    normalized = Path(*parts) if parts else Path(".")

    # Import paths reference the eventual .js output even though the
    # source file on disk is .ts/.tsx -- rewrite extension before checking.
    if normalized.suffix in (".js", ".jsx", ""):
        if (repo_root / normalized.with_suffix(".ts")).exists():
            normalized = normalized.with_suffix(".ts")
        elif (repo_root / normalized.with_suffix(".tsx")).exists():
            normalized = normalized.with_suffix(".tsx")

    full_candidate = repo_root / normalized
    if full_candidate.exists():
        return str(normalized)
    return None


def resolve_imports_for_file(file_path: Path, repo_root: Path) -> list[str]:
    """Dispatches to the appropriate language import resolver based on file extension."""
    if file_path.suffix == ".py":
        return resolve_python_imports(file_path, repo_root)
    elif file_path.suffix in (".ts", ".tsx"):
        return resolve_typescript_imports(file_path, repo_root)
    return []


# ── Main pipeline ─────────────────────────────────────────────────────────

def build_graph_with_imports(repos: dict[str, Path] | None = None) -> CodeGraph:
    cg = CodeGraph()
    if repos is None:
        repos = {
            "httpx": Path("repos/httpx"),
            "got": Path("repos/got"),
        }

    # ── Pass 1: add all File and Symbol nodes first. ──────────────────────
    # Import resolution needs every file already present as a node before
    # we can check "does this target file exist in our graph" -- so this
    # must be a separate, earlier pass, not interleaved with import edges.

    print("Pass 1: indexing files and symbols...")

    for repo_name, repo_root in repos.items():
        if not repo_root.exists():
            print(f"  {repo_name}: root directory {repo_root} not found, skipping.")
            continue
        files = find_source_files(repo_root)
        for f in files:
            rel_path = str(f.relative_to(repo_root))
            language = "python" if f.suffix == ".py" else "typescript"
            cg.add_file(repo_name, rel_path, language)
            file_result = extract_symbols_for_file(f, repo_root)
            for s in file_result.symbols:
                cg.add_symbol(repo_name, s.name, s.kind, s.file_path, s.start_line, s.end_line, s.parent_class)

        print(f"  {repo_name}: {len(files)} files indexed")

    print()

    # ── Pass 2: resolve imports now that every file node exists. ──────────

    print("Pass 2: resolving imports...")

    for repo_name, repo_root in repos.items():
        if not repo_root.exists():
            continue
        import_count = 0
        files = find_source_files(repo_root)
        for f in files:
            from_file = str(f.relative_to(repo_root))
            targets = resolve_imports_for_file(f, repo_root)
            for target in targets:
                edge = cg.add_import(repo_name, from_file, target)
                if edge is not None:
                    import_count += 1

        print(f"  {repo_name}: {import_count} IMPORTS edges resolved")

    print()
    return cg


def main():
    cg = build_graph_with_imports()
    stats = cg.stats()

    print("=" * 60)
    print("FINAL GRAPH STATS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── Verification: the specific case we found broken in Step 7 ─────────
    print()
    print("=" * 60)
    print("VERIFICATION: source/index.ts -> source/create.ts")
    print("(This is the exact gap we found when verify_index_ts.py showed")
    print(" 0 symbols for source/index.ts -- confirming import resolution")
    print(" now captures that relationship as a graph edge.)")
    print("=" * 60)
    importers = cg.files_importing("got", "source/create.ts")
    print(f"  Files importing source/create.ts: {importers}")
    if "source/index.ts" in importers:
        print("  CONFIRMED: source/index.ts -> source/create.ts edge exists.")
    else:
        print("  NOT FOUND -- resolution logic needs further debugging.")

    save_graph(cg, DB_PATH)
    print(f"\nGraph persisted to {DB_PATH}")


if __name__ == "__main__":
    main()

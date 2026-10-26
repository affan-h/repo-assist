"""Parse source files with tree-sitter into symbol-level chunks, a symbol table, and call sites.

Chunking follows the code's own structure rather than fixed-size windows:
every function/method is a chunk, a class contributes a header chunk (signature,
docstring, attributes), and module-level code between definitions is grouped
into small windows. Each chunk records its exact line range so citations point
at real lines.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath

import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from repo_assist.store import CallSite, Chunk, Symbol

MAX_CHUNK_LINES = 80     # longer definitions are split into windows
WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MAX_FILE_BYTES = 300_000

_PY = Language(tree_sitter_python.language())
_TS = Language(tree_sitter_typescript.language_typescript())
_TSX = Language(tree_sitter_typescript.language_tsx())
LANGUAGES = {".py": _PY, ".ts": _TS, ".mts": _TS, ".cts": _TS, ".tsx": _TSX,
             ".js": _TSX, ".mjs": _TSX, ".cjs": _TSX, ".jsx": _TSX}

# Node type -> symbol kind. "container" kinds are descended into for members.
_DEFINITIONS = {
    "function_definition": "function",
    "class_definition": "class",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}
_CONTAINERS = {"class"}
_FUNCTION_VALUES = {"arrow_function", "function_expression", "function"}
_CALLS = {"call", "call_expression", "new_expression"}

_SKIP_DIRS = {"node_modules", "vendor", "dist", "build", "site-packages", ".github"}
_TEST_DIRS = {"test", "tests", "__tests__", "testing"}


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    return (
        any(part.lower() in _TEST_DIRS for part in p.parts[:-1])
        or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
        or ".test." in name or ".spec." in name
    )


def is_indexable_source(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        p.suffix in LANGUAGES
        and not any(part in _SKIP_DIRS for part in p.parts)
        and not p.name.endswith((".min.js", ".d.ts"))
        and not is_test_path(path)
    )


@dataclass
class _Def:
    node: Node
    kind: str
    name: str
    qualified_name: str
    start: int  # 1-based inclusive
    end: int
    children: list["_Def"]


@dataclass
class ParsedFile:
    chunks: list[Chunk]
    symbols: list[Symbol]
    calls: list[CallSite]


def parse_file(path: str, source: bytes) -> ParsedFile:
    language = LANGUAGES[PurePosixPath(path).suffix]
    tree = Parser(language).parse(source)
    lines = source.decode("utf-8", errors="replace").splitlines()

    defs = _collect_definitions(tree.root_node, prefix="")
    symbols = [
        Symbol(d.name, d.qualified_name, d.kind, path, d.start, d.end)
        for d in _walk_defs(defs)
    ]
    chunks = _chunk_definitions(path, lines, defs) + _chunk_module_code(path, lines, defs)
    calls = _collect_calls(tree.root_node, path, defs)
    return ParsedFile(chunks=chunks, symbols=symbols, calls=calls)


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

def _definition_of(node: Node) -> tuple[str, str] | None:
    """Return (kind, name) if `node` defines a named symbol."""
    if node.type in _DEFINITIONS:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _DEFINITIONS[node.type], name_node.text.decode()
    if node.type == "variable_declarator":  # const handler = () => {...}
        value = node.child_by_field_name("value")
        name_node = node.child_by_field_name("name")
        if value is not None and value.type in _FUNCTION_VALUES and name_node is not None:
            return "function", name_node.text.decode()
    return None


_WRAPPERS = {"export_statement", "lexical_declaration", "variable_declaration", "class_body", "block",
             "module", "program", "statement_block", "expression_statement", "ambient_declaration"}


def _collect_definitions(node: Node, prefix: str) -> list[_Def]:
    found = []
    for child in node.named_children:
        span = child  # line range to cite; includes decorators for decorated definitions
        if child.type == "decorated_definition":
            child = child.child_by_field_name("definition")
            if child is None:
                continue
        definition = _definition_of(child)
        if definition is None:
            if child.type in _WRAPPERS:
                found.extend(_collect_definitions(child, prefix))
            continue
        kind, name = definition
        qualified = f"{prefix}{name}"
        members = _collect_definitions(child, f"{qualified}.") if kind in _CONTAINERS else []
        if kind == "function" and prefix:
            kind = "method"
        found.append(_Def(child, kind, name, qualified, span.start_point.row + 1, span.end_point.row + 1, members))
    return found


def _walk_defs(defs: list[_Def]):
    for d in defs:
        yield d
        yield from _walk_defs(d.children)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _make_chunk(path: str, lines: list[str], start: int, end: int, symbol: str | None,
                kind_label: str, extra_terms: list[str] | None = None) -> Chunk:
    text = "\n".join(lines[start - 1:end])
    context = f"{path}\n{kind_label} {symbol}" if symbol else path
    return Chunk(
        kind="code", ref=f"{path}:{start}-{end}", path=path, start_line=start, end_line=end,
        symbol=symbol, text=text, embed_text=f"{context}\n{text}"[:4000],
        extra_search_terms=extra_terms or [],
    )


def _windows(start: int, end: int) -> list[tuple[int, int]]:
    if end - start + 1 <= MAX_CHUNK_LINES:
        return [(start, end)]
    spans, s = [], start
    while s <= end:
        e = min(s + WINDOW_LINES - 1, end)
        spans.append((s, e))
        if e == end:
            break
        s = e - WINDOW_OVERLAP + 1
    return spans


def _chunk_definitions(path: str, lines: list[str], defs: list[_Def]) -> list[Chunk]:
    chunks = []
    for d in _walk_defs(defs):
        if d.children:
            # Class header: from the class line up to its first member.
            header_end = min(d.children[0].start - 1, d.start + WINDOW_LINES - 1)
            member_names = [c.name for c in d.children]
            chunk = _make_chunk(path, lines, d.start, max(header_end, d.start), d.qualified_name,
                                d.kind, extra_terms=member_names)
            chunk.embed_text += "\nmembers: " + ", ".join(member_names)
            chunks.append(chunk)
        else:
            for s, e in _windows(d.start, d.end):
                chunks.append(_make_chunk(path, lines, s, e, d.qualified_name, d.kind))
    return chunks


def _chunk_module_code(path: str, lines: list[str], defs: list[_Def]) -> list[Chunk]:
    """Group top-level lines not covered by any definition (imports, constants, scripts)."""
    covered = set()
    for d in defs:
        covered.update(range(d.start, d.end + 1))
    chunks, run = [], []
    for lineno in range(1, len(lines) + 1):
        if lineno in covered:
            if run:
                chunks.extend(_flush_run(path, lines, run))
                run = []
        else:
            run.append(lineno)
    if run:
        chunks.extend(_flush_run(path, lines, run))
    return chunks


def _flush_run(path: str, lines: list[str], run: list[int]) -> list[Chunk]:
    meaningful = [n for n in run if lines[n - 1].strip()]
    if len(meaningful) < 3:
        return []
    start, end = meaningful[0], meaningful[-1]
    return [_make_chunk(path, lines, s, e, None, "module") for s, e in _windows(start, end)]


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------

def _callee_name(node: Node) -> str | None:
    target = node.child_by_field_name("function") or node.child_by_field_name("constructor")
    if target is None:
        return None
    if target.type in {"identifier", "type_identifier"}:
        return target.text.decode()
    if target.type in {"attribute", "member_expression"}:
        prop = target.child_by_field_name("attribute") or target.child_by_field_name("property")
        return prop.text.decode() if prop is not None else None
    return None


def _collect_calls(root: Node, path: str, defs: list[_Def]) -> list[CallSite]:
    spans = sorted(_walk_defs(defs), key=lambda d: d.end - d.start)  # innermost first

    def enclosing(line: int) -> str:
        for d in spans:
            if d.start <= line <= d.end:
                return d.qualified_name
        return "<module>"

    calls, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.type in _CALLS:
            name = _callee_name(node)
            if name:
                line = node.start_point.row + 1
                calls.append(CallSite(callee=name, caller=enclosing(line), path=path, line=line))
        stack.extend(node.named_children)
    return calls

"""Parse source files with tree-sitter into symbol-level chunks, a symbol table, and call sites.

Chunking follows the code's own structure rather than fixed-size windows:
every function/method is a chunk, a class contributes a header chunk (signature,
docstring, attributes), and module-level code between definitions is grouped
into small windows. Each chunk records its exact line range so citations point
at real lines.

Languages listed in _SPECS get symbol extraction; any other language that has a
tree-sitter grammar is still indexed, as line windows without symbols.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath

from tree_sitter import Node, Parser
from tree_sitter_language_pack import detect_language_from_path, get_parser

from repo_assist.store import CallSite, Chunk, Symbol

MAX_CHUNK_LINES = 80     # longer definitions are split into windows
WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MAX_FILE_BYTES = 300_000
MAX_NESTING = 12         # how deep to look through non-definition nodes for definitions

_JS = {
    "function_declaration": "function", "generator_function_declaration": "function",
    "class_declaration": "class", "abstract_class_declaration": "class", "method_definition": "method",
    "interface_declaration": "interface", "type_alias_declaration": "type", "enum_declaration": "enum",
}
_C = {"function_definition": "function", "struct_specifier": "type", "enum_specifier": "enum",
      "union_specifier": "type"}

# Per language: definition node type -> symbol kind.
_SPECS: dict[str, dict[str, str]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": _JS, "typescript": _JS, "tsx": _JS,
    "go": {"function_declaration": "function", "method_declaration": "method", "type_spec": "type"},
    "java": {"class_declaration": "class", "interface_declaration": "interface", "enum_declaration": "enum",
             "record_declaration": "class", "method_declaration": "method", "constructor_declaration": "method"},
    "kotlin": {"class_declaration": "class", "object_declaration": "class", "function_declaration": "function"},
    "rust": {"function_item": "function", "struct_item": "type", "enum_item": "enum", "trait_item": "trait",
             "impl_item": "impl", "mod_item": "module"},
    "c": _C,
    "cpp": {**_C, "class_specifier": "class", "namespace_definition": "module"},
    "csharp": {"class_declaration": "class", "interface_declaration": "interface", "struct_declaration": "type",
               "enum_declaration": "enum", "record_declaration": "class", "method_declaration": "method",
               "constructor_declaration": "method", "namespace_declaration": "module"},
    "ruby": {"class": "class", "module": "module", "method": "method", "singleton_method": "method"},
    "php": {"class_declaration": "class", "interface_declaration": "interface", "trait_declaration": "trait",
            "function_definition": "function", "method_declaration": "method"},
    "swift": {"class_declaration": "class", "protocol_declaration": "interface", "function_declaration": "function"},
    "scala": {"class_definition": "class", "object_definition": "class", "trait_definition": "trait",
              "function_definition": "function"},
}
SYMBOL_LANGUAGES = sorted(_SPECS)
_CONTAINERS = {"class", "interface", "module", "impl", "trait", "enum"}
_NEEDS_BODY = {"struct_specifier", "enum_specifier", "union_specifier", "class_specifier"}  # C/C++ type uses
_FUNCTION_VALUES = {"arrow_function", "function_expression", "function"}
_NAME_TYPES = {"identifier", "type_identifier", "field_identifier", "property_identifier", "constant", "name",
               "simple_identifier", "qualified_identifier", "destructor_name", "operator_name"}
_CALLS = {"call", "call_expression", "new_expression", "method_invocation", "invocation_expression",
          "function_call_expression", "member_call_expression", "scoped_call_expression",
          "object_creation_expression"}

# Data, markup and config formats: covered by grep and the docs index, not chunked as code.
_NON_CODE = {"markdown", "markdown_inline", "rst", "json", "jsonc", "json5", "yaml", "toml", "xml", "csv",
             "tsv", "ini", "properties", "gitignore", "gitattributes", "diff", "html", "css", "scss", "less",
             "svg", "requirements", "latex", "bibtex", "po", "text"}

_SKIP_DIRS = {"node_modules", "vendor", "dist", "build", "site-packages", ".github", "third_party"}
_TEST_DIRS = {"test", "tests", "__tests__", "testing", "spec", "specs"}
_TEST_SUFFIXES = ("_test.py", "_test.go", "_test.rs", "_spec.rb", "test.java", "tests.java", "test.kt",
                  "tests.cs", "test.cs", "test.php")


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    return (
        any(part.lower() in _TEST_DIRS for part in p.parts[:-1])
        or name.startswith("test_") or name.endswith(_TEST_SUFFIXES) or name == "conftest.py"
        or ".test." in name or ".spec." in name
    )


def language_for(path: str) -> str | None:
    """The tree-sitter language for a source file, or None if it isn't code we index."""
    p = PurePosixPath(path)
    if p.name.endswith((".min.js", ".d.ts", ".lock")):
        return None
    language = detect_language_from_path(path)
    return None if language is None or language in _NON_CODE else language


def is_indexable_source(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        language_for(path) is not None
        and not any(part in _SKIP_DIRS for part in p.parts)
        and not is_test_path(path)
    )


@lru_cache
def _parser(language: str) -> Parser:
    return get_parser(language)


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
    language = language_for(path)
    if language is None:
        raise ValueError(f"No code grammar for {path}")
    tree = _parser(language).parse(source)
    lines = source.decode("utf-8", errors="replace").splitlines()

    spec = _SPECS.get(language)
    defs = _collect_definitions(tree.root_node, "", spec, 0) if spec else []
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

def _text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _name_of(node: Node) -> str | None:
    if node.type == "impl_item":  # Rust: `impl Foo` / `impl Trait for Foo` -> members are Foo.*
        target = node.child_by_field_name("type")
        return _text(target) if target is not None else None
    named = node.child_by_field_name("name")
    if named is not None:
        return _text(named)
    declarator = node.child_by_field_name("declarator")  # C/C++: int *f(int x) { ... }
    while declarator is not None and declarator.type not in _NAME_TYPES:
        declarator = declarator.child_by_field_name("declarator")
    if declarator is not None:
        return _text(declarator)
    return next((_text(c) for c in node.named_children if c.type in _NAME_TYPES), None)


def _definition_of(node: Node, spec: dict[str, str]) -> tuple[str, str, str] | None:
    """Return (kind, name, prefix_override) if `node` defines a named symbol."""
    if node.type in spec:
        if node.type in _NEEDS_BODY and node.child_by_field_name("body") is None:
            return None  # `struct S x;` mentions a type, it doesn't define one
        name = _name_of(node)
        if not name:
            return None
        prefix = ""
        receiver = node.child_by_field_name("receiver")  # Go: func (s *Server) Serve() -> Server.Serve
        if receiver is not None:
            types = re.findall(r"[A-Za-z_]\w*", _text(receiver))
            prefix = f"{types[-1]}." if types else ""
        return spec[node.type], name, prefix
    if node.type == "variable_declarator":  # JS/TS: const handler = () => {...}
        value = node.child_by_field_name("value")
        name_node = node.child_by_field_name("name")
        if value is not None and value.type in _FUNCTION_VALUES and name_node is not None:
            return "function", _text(name_node), ""
    return None


def _collect_definitions(node: Node, prefix: str, spec: dict[str, str], depth: int) -> list[_Def]:
    found = []
    for child in node.named_children:
        span = child  # line range to cite; includes decorators for decorated definitions
        if child.type == "decorated_definition":
            child = child.child_by_field_name("definition")
            if child is None:
                continue
        definition = _definition_of(child, spec)
        if definition is None:
            if depth < MAX_NESTING:  # look through wrappers (exports, blocks, namespaces...)
                found.extend(_collect_definitions(child, prefix, spec, depth + 1))
            continue
        kind, raw_name, receiver_prefix = definition
        qualified = f"{prefix or receiver_prefix}{raw_name}".replace("::", ".")
        name = qualified.rsplit(".", 1)[-1]
        members = _collect_definitions(child, f"{qualified}.", spec, 0) if kind in _CONTAINERS else []
        if kind == "function" and (prefix or receiver_prefix):
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

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GENERICS = re.compile(r"<[^<>]*>")


def _callee_name(node: Node) -> str | None:
    """Last identifier of the called expression: `a.b(c).run()` -> run, `Vec::<u8>::new()` -> new."""
    target = None
    if node.type in {"method_invocation", "member_call_expression", "scoped_call_expression"}:
        target = node.child_by_field_name("name")
    for field in ("method", "function", "constructor", "type"):
        target = target or node.child_by_field_name(field)
    if target is None:
        return None
    identifiers = _IDENTIFIER.findall(_GENERICS.sub("", _text(target)))
    return identifiers[-1] if identifiers else None


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

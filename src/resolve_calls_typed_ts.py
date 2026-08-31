"""
Item 1 -- TypeScript local type inference, extending call resolution.

Direct port of resolve_calls_typed.py's Python logic to TypeScript,
using the grammar shapes confirmed BEFORE Step 10b even started (see
diagnose_type_inference.py's TypeScript output):

  1. const x = new SomeClass(...)         -- constructor assignment
  2. this.x = new SomeClass(...)          -- constructor assignment to property
  3. this.x = param                       -- property assigned from a
                                              typed constructor parameter
  4. constructor(private x: Type)         -- parameter property (TS-specific:
                                              declares AND assigns in one step)
  5. function f(x: SomeType)              -- type-annotated parameter

Same scope discipline as the Python version: no return-type inference,
no reassignment tracking, no cross-file type resolution beyond IMPORTS
edges already built. Unresolvable receivers stay unresolved.

Run with:
    python3 src/resolve_calls_typed_ts.py
"""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_typescript as tsts

from graph_schema import CodeGraph, save_graph
from resolve_imports import build_graph_with_imports
from extract_symbols import find_source_files, should_skip_dir, tsx_parser


TS_LANGUAGE = Language(tsts.language_typescript())
ts_parser = Parser(TS_LANGUAGE)


# ── Stats ────────────────────────────────────────────────────────────────────

class TypedCallStatsTS:
    def __init__(self):
        self.type_map_entries = 0
        self.resolved_local_var = 0
        self.resolved_this_attr = 0
        self.attempted_but_class_not_found = 0

    def report(self, label: str):
        print(f"  [{label}] type_map_entries={self.type_map_entries} "
              f"resolved(local_var)={self.resolved_local_var} "
              f"resolved(this_attr)={self.resolved_this_attr} "
              f"attempted_but_class_not_found={self.attempted_but_class_not_found}")


# ── Shared helpers ────────────────────────────────────────────────────────

def _text(node, source_code) -> str:
    return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _member_parts(node, source_code):
    """For a member_expression (e.g. this.transport), return
    (receiver_text_or_None, property_name) if it's a simple two-level
    member access, else (None, None). Mirrors _attribute_parts from the
    Python module, confirmed against the same grammar shapes verified
    before Step 10b: member_expression -> [this|identifier, ., property_identifier]."""
    children = [c for c in node.children if c.type != "."]
    if len(children) == 2 and children[1].type in ("property_identifier", "private_property_identifier"):
        if children[0].type in ("this", "identifier"):
            receiver = "this" if children[0].type == "this" else _text(children[0], source_code)
            return receiver, _text(children[1], source_code)
    return None, None


# ── Phase A: build the type map for one class ────────────────────────────────

def build_type_map_ts(class_node, source_code) -> dict[str, str]:
    """
    Scans a class body for:
      - this.x = new SomeClass(...)      -> type_map["this.x"] = "SomeClass"
      - const x = new SomeClass(...)     -> type_map["x"] = "SomeClass"      (local var)
      - constructor(private x: Type)     -> type_map["x"] = "Type"          (param)
                                             ALSO: parameter properties auto-assign
                                             to `this.x`, confirmed via real
                                             tree-sitter evidence (required_parameter
                                             with accessibility_modifier) -- so this
                                             ALSO populates type_map["this.x"] directly,
                                             no separate "this.x = param" propagation
                                             step needed like Python required.
      - function f(x: SomeType)          -> type_map["x"] = "SomeType"      (param)
      - this.x = param                   -> type_map["this.x"] = type_map["param"]
                                             IF param's type is already known
                                             (mirrors Python's propagation step,
                                             for the non-parameter-property case
                                             where assignment happens in the
                                             constructor BODY rather than the
                                             parameter list).

    Same flat-per-class simplification as the Python version: local
    variables/params are NOT function-scoped in this map. Accepted for
    the same reason -- rare collision in practice, real complexity to fix.
    """
    type_map: dict[str, str] = {}

    def collect_params_and_parameter_properties(node):
        if node.type == "required_parameter" or node.type == "optional_parameter":
            name_node = None
            type_node = None
            has_accessibility_modifier = False
            for c in node.children:
                if c.type == "identifier" and name_node is None:
                    name_node = c
                if c.type == "type_annotation":
                    type_node = c
                if c.type == "accessibility_modifier":
                    has_accessibility_modifier = True
            if name_node is not None and type_node is not None:
                type_id = next(
                    (c for c in type_node.children if c.type == "type_identifier"), None
                )
                if type_id is not None:
                    param_name = _text(name_node, source_code)
                    type_name = _text(type_id, source_code)
                    type_map[param_name] = type_name
                    if has_accessibility_modifier:
                        # Parameter property: constructor(private x: Type)
                        # auto-assigns to this.x -- confirmed via real
                        # tree-sitter evidence before Step 10b.
                        type_map[f"this.{param_name}"] = type_name
        if node.type != "class_declaration" or node is class_node:
            for child in node.children:
                collect_params_and_parameter_properties(child)

    def collect_assignments(node):
        # this.x = new SomeClass(...)  OR  const x = new SomeClass(...)
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None and right is not None:
                if right.type == "new_expression":
                    fn_node = None
                    for c in right.children:
                        if c.type == "identifier":
                            fn_node = c
                            break
                    if fn_node is not None:
                        class_name = _text(fn_node, source_code)
                        if left.type == "member_expression":
                            receiver, attr_name = _member_parts(left, source_code)
                            if receiver == "this":
                                type_map[f"this.{attr_name}"] = class_name
                elif right.type == "identifier" and left.type == "member_expression":
                    # this.x = param -- propagate if param's type is known
                    receiver, attr_name = _member_parts(left, source_code)
                    if receiver == "this":
                        param_name = _text(right, source_code)
                        if param_name in type_map:
                            type_map[f"this.{attr_name}"] = type_map[param_name]

        if node.type == "variable_declarator":
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")
            if name_node is not None and value_node is not None and value_node.type == "new_expression":
                fn_node = None
                for c in value_node.children:
                    if c.type == "identifier":
                        fn_node = c
                        break
                if fn_node is not None and name_node.type == "identifier":
                    type_map[_text(name_node, source_code)] = _text(fn_node, source_code)

        if node.type != "class_declaration" or node is class_node:
            for child in node.children:
                collect_assignments(child)

    collect_params_and_parameter_properties(class_node)
    collect_assignments(class_node)

    return type_map


# ── Phase B: resolve calls using the type map ────────────────────────────────

def find_symbol_in_repo(cg: CodeGraph, repo: str, qualified_name: str):
    matches = []
    for (r, fp, qn, line), idx in cg._symbol_index.items():
        if r == repo and qn == qualified_name:
            matches.append((line, idx, fp))
    if not matches:
        return None
    matches.sort()
    _, idx, fp = matches[0]
    return idx, fp


def resolve_calls_using_type_map_ts(
    class_node, source_code, rel_path: str, repo: str, cg: CodeGraph,
    type_map: dict[str, str], stats: TypedCallStatsTS,
):
    cls_name_node = class_node.child_by_field_name("name")
    cls_name = _text(cls_name_node, source_code)

    def get_enclosing_method_name(node):
        current = node.parent
        while current is not None:
            if current.type == "method_definition" and current.parent is not None and current.parent.type == "class_body":
                name_node = current.child_by_field_name("name")
                return _text(name_node, source_code)
            current = current.parent
        return None

    def record(caller_method, target_idx: int):
        if caller_method is None:
            return
        caller_qname = f"{cls_name}.{caller_method}"
        caller_match = find_symbol_in_repo(cg, repo, caller_qname)
        if caller_match is None:
            return
        caller_idx, _ = caller_match
        if not cg.graph.has_edge(caller_idx, target_idx):
            cg.graph.add_edge(caller_idx, target_idx, "CALLS")

    def walk(node):
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node is not None and fn_node.type == "member_expression":
                # Method name is the LAST identifier-like child, same fix
                # as the Python bug found via verification -- extracting
                # it fresh here rather than relying on a helper that only
                # handles the simple 2-child case, since chained receivers
                # (this.x.y()) need the SAME correct extraction.
                fn_children_no_dots = [c for c in fn_node.children if c.type != "."]
                method_name = None
                if fn_children_no_dots and fn_children_no_dots[-1].type in (
                    "property_identifier", "private_property_identifier"
                ):
                    method_name = _text(fn_children_no_dots[-1], source_code)

                receiver, _unused = _member_parts(fn_node, source_code)

                var_key = None
                bucket = None
                if receiver is not None and receiver != "this":
                    var_key = receiver
                    bucket = "local_var"
                elif fn_node.children and fn_node.children[0].type == "member_expression":
                    # chained: this.transport.handleRequest()
                    inner_receiver, inner_attr = _member_parts(fn_node.children[0], source_code)
                    if inner_receiver == "this":
                        var_key = f"this.{inner_attr}"
                        bucket = "this_attr"

                if var_key is not None and method_name is not None and var_key in type_map:
                    target_class = type_map[var_key]
                    target = find_symbol_in_repo(cg, repo, f"{target_class}.{method_name}")
                    if target is not None:
                        target_idx, _ = target
                        caller_method = get_enclosing_method_name(node)
                        record(caller_method, target_idx)
                        if bucket == "local_var":
                            stats.resolved_local_var += 1
                        else:
                            stats.resolved_this_attr += 1
                    else:
                        stats.attempted_but_class_not_found += 1

        if node.type != "class_declaration" or node is class_node:
            for child in node.children:
                walk(child)

    walk(class_node)


# ── Per-file driver ──────────────────────────────────────────────────────

def process_typescript_file(file_path: Path, repo_root: Path, repo: str, cg: CodeGraph, stats: TypedCallStatsTS):
    rel_path = str(file_path.relative_to(repo_root))
    source_code = file_path.read_bytes()
    parser = tsx_parser if file_path.suffix == ".tsx" else ts_parser
    tree = parser.parse(source_code)
    root = tree.root_node

    def walk(node):
        if node.type == "class_declaration":
            type_map = build_type_map_ts(node, source_code)
            stats.type_map_entries += len(type_map)
            resolve_calls_using_type_map_ts(node, source_code, rel_path, repo, cg, type_map, stats)
        for child in node.children:
            walk(child)

    walk(root)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Rebuilding graph with files, symbols, imports, and base CALLS edges (Steps 7-10)...\n")
    cg = build_graph_with_imports()

    from resolve_calls import resolve_calls_for_file, CallResolutionStats

    httpx_root = Path("repos/httpx")
    got_root = Path("repos/got")

    repos = [
        ("httpx", httpx_root),
        ("got", got_root),
    ]

    for repo_name, repo_root in repos:
        if not repo_root.exists():
            continue
        base_stats = CallResolutionStats()
        for f in find_source_files(repo_root):
            resolve_calls_for_file(f, repo_root, repo_name, cg, base_stats)

    base_edges = sum(1 for e in cg.graph.edges() if e == "CALLS")
    print(f"Base (untyped, both languages) CALLS edges: {base_edges}\n")

    print("Resolving typed CALLS edges for TypeScript (got)...")
    stats = TypedCallStatsTS()
    if got_root.exists():
        for ts_file in find_source_files(got_root, extensions=(".ts", ".tsx")):
            process_typescript_file(ts_file, got_root, "got", cg, stats)
        stats.report("got")

    total_edges = sum(1 for e in cg.graph.edges() if e == "CALLS")
    print(f"\nTotal CALLS edges after TypeScript typed resolution: {total_edges}")
    print(f"  (base: {base_edges}, added by TS type inference: {total_edges - base_edges})")


if __name__ == "__main__":
    main()

"""
Step 10b -- Local type inference, extending call resolution.

Scope, confirmed against real tree-sitter evidence before writing this
(see diagnose_type_inference.py output):

  1. x = SomeClass(...)              -- constructor assignment (Python)
  2. self.x = SomeClass(...)         -- constructor assignment to attribute
  3. def f(self, x: SomeClass)       -- type-annotated parameter (Python)

TypeScript (new/const, parameter properties) is NOT yet implemented in
this pass -- Python only, to keep this step reviewable in isolation.
TypeScript follows the identical strategy once this is confirmed correct
on real data.

Deliberately OUT of scope: inference through function return types
(x = some_factory() -- unknown without analyzing that function's return),
reassignment tracking, cross-file type resolution beyond IMPORTS edges.
A variable whose type can't be determined this way stays unresolved --
same "don't guess" discipline as resolve_calls.py.

Design: two clearly separated phases per class.
  Phase A: build_type_map()      -- scan the class, produce
                                     {var_or_self_attr_name: ClassName}
  Phase B: resolve_calls_using_type_map() -- walk the class's calls,
                                     look up receivers in the type map,
                                     resolve to CALLS edges if found.

Run with:
    python3 src/resolve_calls_typed.py
"""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

import sys
sys.path.insert(0, "src")
from graph_schema import CodeGraph, save_graph
from resolve_imports import build_graph_with_imports
from extract_symbols import should_skip_dir


PY_LANGUAGE = Language(tspython.language())
py_parser = Parser(PY_LANGUAGE)


# ── Stats ────────────────────────────────────────────────────────────────────

class TypedCallStats:
    def __init__(self):
        self.type_map_entries = 0
        self.resolved_local_var = 0
        self.resolved_self_attr = 0
        self.attempted_but_class_not_found = 0

    def report(self, label: str):
        print(f"  [{label}] type_map_entries={self.type_map_entries} "
              f"resolved(local_var)={self.resolved_local_var} "
              f"resolved(self_attr)={self.resolved_self_attr} "
              f"attempted_but_class_not_found={self.attempted_but_class_not_found}")


# ── Small AST helpers, shared by both phases ─────────────────────────────────

def _text(node, source_code) -> str:
    return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _attribute_parts(node, source_code):
    """For an `attribute` node (e.g. self.transport), return
    (receiver_text_or_None, attr_name) if it's a simple two-level attribute
    (identifier.identifier), else (None, None)."""
    children = [c for c in node.children if c.type != "."]
    if len(children) == 2 and children[0].type == "identifier" and children[1].type == "identifier":
        return _text(children[0], source_code), _text(children[1], source_code)
    return None, None


# ── Phase A: build the type map for one class ────────────────────────────────

def build_type_map(class_node, source_code) -> dict[str, str]:
    """
    Scans a class body for:
      - self.x = SomeClass(...)   -> type_map["self.x"] = "SomeClass"
      - x = SomeClass(...)        -> type_map["x"] = "SomeClass"       (local var)
      - def f(self, x: SomeClass) -> type_map["x"] = "SomeClass"       (param)
      - self.x = param            -> type_map["self.x"] = type_map["param"]
                                      IF param's type is already known
                                      (from a typed parameter). This closes
                                      a real gap found via verification: the
                                      common __init__ pattern
                                      "def __init__(self, transport: Transport):
                                           self.transport = transport"
                                      was NOT populating self.transport's type,
                                      even though the type is fully knowable --
                                      it's just one level of indirection through
                                      the parameter. Two-pass: parameters must
                                      be scanned before self.x = param
                                      assignments, so we do params first,
                                      then a second pass for assignments.
      - with SomeClass(...) as x: -> type_map["x"] = "SomeClass"
                                      Found via direct verification: httpx's
                                      own request()/stream() functions use
                                      exactly this pattern ("with Client(...)
                                      as client: return client.request(...)"),
                                      which the plain `assignment`-node check
                                      above never catches, since a with-statement
                                      binds its variable via `as_pattern`, a
                                      structurally different node. This was a
                                      real, confirmed gap -- get_callees()
                                      returned nothing for request()/stream()
                                      even though the actual relationship
                                      (calls Client.request / Client.stream)
                                      is real and present in the source.
                                      Real tree-sitter shape confirmed:
                                      with_statement -> with_clause -> with_item
                                      -> as_pattern -> [call(identifier, ...),
                                      as, as_pattern_target -> identifier].

    Local-variable and parameter entries are function-scoped in reality,
    but we deliberately keep ONE flat map per class rather than per-function
    scoping. This is a known simplification: if two methods each have a
    local variable named `client` with DIFFERENT types, this map only
    keeps the last one seen. Accepted for now because it's rare in
    practice and the alternative (per-function maps) adds real complexity
    for a case we haven't confirmed actually occurs in our two repos.
    """
    type_map: dict[str, str] = {}

    def collect_params(node):
        if node.type == "typed_parameter":
            name_node = None
            type_node = None
            for c in node.children:
                if c.type == "identifier" and name_node is None:
                    name_node = c
                if c.type == "type":
                    type_node = c
            if name_node is not None and type_node is not None:
                type_id = next((c for c in type_node.children if c.type == "identifier"), None)
                if type_id is not None:
                    type_map[_text(name_node, source_code)] = _text(type_id, source_code)
        if node.type != "class_definition" or node is class_node:
            for child in node.children:
                collect_params(child)

    def collect_assignments(node):
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None and right is not None:
                if right.type == "call":
                    fn_node = right.children[0] if right.children else None
                    if fn_node is not None and fn_node.type == "identifier":
                        class_name = _text(fn_node, source_code)
                        if left.type == "identifier":
                            type_map[_text(left, source_code)] = class_name
                        elif left.type == "attribute":
                            receiver, attr_name = _attribute_parts(left, source_code)
                            if receiver == "self":
                                type_map[f"self.{attr_name}"] = class_name

                elif right.type == "identifier" and left.type == "attribute":
                    # self.x = param_name -- if param_name's type is already
                    # known (from Pass 1's typed-parameter scan), propagate
                    # it to self.x too.
                    receiver, attr_name = _attribute_parts(left, source_code)
                    if receiver == "self":
                        param_name = _text(right, source_code)
                        if param_name in type_map:
                            type_map[f"self.{attr_name}"] = type_map[param_name]

        if node.type != "class_definition" or node is class_node:
            for child in node.children:
                collect_assignments(child)

    def collect_with_statements(node):
        # "with SomeClass(...) as x:" -- confirmed via direct tree-sitter
        # inspection this is a SEPARATE node structure from `assignment`,
        # and the real root cause of request()/stream() showing no known
        # relationships despite the actual code using exactly this pattern.
        if node.type == "as_pattern":
            call_node = None
            target_node = None
            for c in node.children:
                if c.type == "call":
                    call_node = c
                if c.type == "as_pattern_target":
                    target_node = c
            if call_node is not None and target_node is not None:
                fn_node = call_node.children[0] if call_node.children else None
                target_id = next((c for c in target_node.children if c.type == "identifier"), None)
                if fn_node is not None and fn_node.type == "identifier" and target_id is not None:
                    class_name = _text(fn_node, source_code)
                    var_name = _text(target_id, source_code)
                    type_map[var_name] = class_name

        if node.type != "class_definition" or node is class_node:
            for child in node.children:
                collect_with_statements(child)

    collect_params(class_node)      # Pass 1: typed parameters, so their types are known first
    collect_assignments(class_node)  # Pass 2: assignments, including self.x = param
    collect_with_statements(class_node)  # Pass 3: with-statement variable bindings

    return type_map


# ── Phase B: resolve calls using the type map ────────────────────────────────

def find_symbol_in_repo(cg: CodeGraph, repo: str, qualified_name: str):
    """Search the WHOLE repo (not just one file) for a matching
    qualified_name, since the target class is typically defined in a
    different file than the caller. Returns (node_index, file_path)
    for the first match by line order, or None."""
    matches = []
    for (r, fp, qn, line), idx in cg._symbol_index.items():
        if r == repo and qn == qualified_name:
            matches.append((line, idx, fp))
    if not matches:
        return None
    matches.sort()
    _, idx, fp = matches[0]
    return idx, fp


def resolve_calls_using_type_map(
    scope_node, source_code, rel_path: str, repo: str, cg: CodeGraph,
    type_map: dict[str, str], stats: TypedCallStats,
):
    """
    scope_node can be EITHER a class_definition (methods resolve to
    "ClassName.method_name") OR a module-level function_definition
    (resolves to just "function_name", no class prefix).

    REAL BUG FOUND AND FIXED: this function originally assumed
    scope_node was always a class_definition, unconditionally computing
    cls_name = scope_node's own name field. When called (even
    experimentally) on a plain function_definition, this silently
    produced a WRONG qualified name (e.g. "request.request" instead of
    "request") -- record() then correctly found no such symbol and
    silently declined to add the edge, which looked like "the with-
    statement detection doesn't work" but was actually "this whole
    layer was never wired up for module-level functions at all."
    Confirmed via direct testing against httpx's real request()/stream()
    source, which use the "with Client(...) as client: client.request(...)"
    pattern specifically at module level, not inside a class.
    """
    is_class_scope = scope_node.type == "class_definition"
    cls_name = None
    if is_class_scope:
        cls_name = _text(scope_node.child_by_field_name("name"), source_code)

    def get_enclosing_method_name(node):
        current = node.parent
        while current is not None and current is not scope_node:
            if current.type == "function_definition":
                return _text(current.child_by_field_name("name"), source_code)
            current = current.parent
        return None

    def record(caller_method, target_idx: int):
        if is_class_scope:
            if caller_method is None:
                return
            caller_qname = f"{cls_name}.{caller_method}"
        else:
            # Module-level function: scope_node itself is the calling
            # function -- no enclosing method to look up separately,
            # and no class prefix.
            caller_qname = _text(scope_node.child_by_field_name("name"), source_code)

        caller_match = find_symbol_in_repo(cg, repo, caller_qname)
        if caller_match is None:
            return
        caller_idx, _ = caller_match
        if not cg.graph.has_edge(caller_idx, target_idx):
            cg.graph.add_edge(caller_idx, target_idx, "CALLS")

    def walk(node):
        if node.type == "call":
            fn_node = node.children[0] if node.children else None
            if fn_node is not None and fn_node.type == "attribute":
                # The method being called is ALWAYS the last identifier
                # child of fn_node, regardless of whether the receiver is
                # a simple identifier (client.request) or a chained
                # attribute (self.transport.handle_request). Extracting
                # this directly, rather than relying on _attribute_parts
                # (which only handles the simple 2-child case and returns
                # None for a 3-level chain), fixes a real bug found via
                # verification: method_name was silently staying None for
                # every chained self.x.y() call, so self_attr resolution
                # never fired even when the type map had the right entry.
                fn_children_no_dots = [c for c in fn_node.children if c.type != "."]
                method_name = None
                if fn_children_no_dots and fn_children_no_dots[-1].type == "identifier":
                    method_name = _text(fn_children_no_dots[-1], source_code)

                receiver, _unused = _attribute_parts(fn_node, source_code)

                var_key = None
                bucket = None
                if receiver is not None and receiver != "self":
                    var_key = receiver
                    bucket = "local_var"
                elif fn_node.children and fn_node.children[0].type == "attribute":
                    # chained: self.transport.handle_request()
                    inner_receiver, inner_attr = _attribute_parts(fn_node.children[0], source_code)
                    if inner_receiver == "self":
                        var_key = f"self.{inner_attr}"
                        bucket = "self_attr"

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
                            stats.resolved_self_attr += 1
                    else:
                        stats.attempted_but_class_not_found += 1

        if node.type != "class_definition" or node is scope_node:
            for child in node.children:
                walk(child)

    walk(scope_node)


# ── Per-file driver ──────────────────────────────────────────────────────

def process_python_file(file_path: Path, repo_root: Path, repo: str, cg: CodeGraph, stats: TypedCallStats):
    rel_path = str(file_path.relative_to(repo_root))
    source_code = file_path.read_bytes()
    tree = py_parser.parse(source_code)
    root = tree.root_node

    def is_inside_class(node) -> bool:
        current = node.parent
        while current is not None:
            if current.type == "class_definition":
                return True
            current = current.parent
        return False

    def walk(node):
        if node.type == "class_definition":
            type_map = build_type_map(node, source_code)
            stats.type_map_entries += len(type_map)
            resolve_calls_using_type_map(node, source_code, rel_path, repo, cg, type_map, stats)
            return  # class body already fully processed above; don't also
                     # treat its methods as separate module-level functions

        if node.type == "function_definition" and not is_inside_class(node):
            # REAL FIX: module-level functions (e.g. httpx's request()/
            # stream(), which use "with Client(...) as client:
            # client.request(...)") were never covered by this layer
            # before -- confirmed via direct testing that get_callees()
            # returned nothing for either, despite the real, correct
            # relationship being present in the source. resolve_calls_
            # using_type_map() was generalized to handle a plain function
            # scope (see its own docstring for the bug found and fixed),
            # so we now call it here too, not just for classes.
            type_map = build_type_map(node, source_code)
            stats.type_map_entries += len(type_map)
            resolve_calls_using_type_map(node, source_code, rel_path, repo, cg, type_map, stats)
            # Still descend into it in case of nested function definitions.

        for child in node.children:
            walk(child)

    walk(root)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Rebuilding graph with files, symbols, imports, and base CALLS edges (Steps 7-10)...\n")
    cg = build_graph_with_imports()

    # Also run the base (untyped) call resolver first, so this pass adds
    # ON TOP of it rather than replacing it.
    from resolve_calls import resolve_python_calls, resolve_typescript_calls, CallResolutionStats

    httpx_root = Path("repos/httpx")
    got_root = Path("repos/got")

    base_httpx_stats = CallResolutionStats()
    for py_file in sorted(httpx_root.rglob("*.py")):
        if any(should_skip_dir(part) for part in py_file.relative_to(httpx_root).parts[:-1]):
            continue
        resolve_python_calls(py_file, httpx_root, cg, base_httpx_stats)

    base_got_stats = CallResolutionStats()
    for ts_file in sorted(got_root.rglob("*.ts")):
        if any(should_skip_dir(part) for part in ts_file.relative_to(got_root).parts[:-1]):
            continue
        resolve_typescript_calls(ts_file, got_root, cg, base_got_stats)

    base_edges = sum(1 for e in cg.graph.edges() if e == "CALLS")
    print(f"Base (untyped) CALLS edges: {base_edges}\n")

    print("Resolving typed CALLS edges (Step 10b, Python only)...")
    stats = TypedCallStats()
    for py_file in sorted(httpx_root.rglob("*.py")):
        if any(should_skip_dir(part) for part in py_file.relative_to(httpx_root).parts[:-1]):
            continue
        process_python_file(py_file, httpx_root, "httpx", cg, stats)

    stats.report("httpx")

    total_edges = sum(1 for e in cg.graph.edges() if e == "CALLS")
    print(f"\nTotal CALLS edges after typed resolution: {total_edges}")
    print(f"  (base: {base_edges}, added by type inference: {total_edges - base_edges})")


if __name__ == "__main__":
    main()

"""
Step 10 -- CALLS edge resolution.

Deliberately scoped, per the reasoning laid out before writing this:
full call resolution requires type inference we're not building. We
resolve only the tractable subset:

  1. self.method(...) / this.method(...) where the receiver is EXACTLY
     "self"/"this" with ONE dot -- i.e. a direct method call on the
     enclosing class. Verified via real tree-sitter output that this
     shape is: attribute/member_expression with two DIRECT children,
     an identifier("self"/"this") and an identifier(method_name).

  2. self._stream.close() -- REJECTED. Confirmed via real evidence that
     this shape has attribute.children[0] be ANOTHER attribute node
     (self._stream), not a plain identifier. We don't know _stream's
     type, so we do not guess. Any chained attribute/member access is
     excluded, not resolved.

  3. Direct calls to a bare identifier (e.g. create_ssl_context(url),
     _port_or_default(url)) where that identifier matches a top-level
     function/class already known in the SAME FILE, or reachable via
     an already-resolved IMPORTS edge from that file.

Everything else is left unresolved and explicitly counted, not
silently dropped -- so we know our real coverage, not an inflated one.

Run with:
    python3 src/resolve_calls.py
"""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts

import sys
sys.path.insert(0, "src")
from graph_schema import CodeGraph, SymbolNode, save_graph
from resolve_imports import build_graph_with_imports


PY_LANGUAGE = Language(tspython.language())
TS_LANGUAGE = Language(tsts.language_typescript())
py_parser = Parser(PY_LANGUAGE)
ts_parser = Parser(TS_LANGUAGE)


def ensure_module_symbol(cg: CodeGraph, repo: str, file_path: str):
    """Adds a synthetic '<module>' symbol for this file if not already
    present, via the NORMAL add_symbol() path -- not a bypass -- so it
    persists through save_graph() and is queryable via query_tools.py
    exactly like any other symbol. start_line=0 is a deliberate sentinel:
    real symbols start at line 1+ (tree-sitter/1-indexed), so 0 can never
    collide with a real symbol's start_line for the same qualified_name.
    kind="module" is a new, distinct value -- callers filtering on
    kind == "class"/"function"/"method" (e.g. resolve_inheritance.py's
    complexity-exclusion logic) will not accidentally pick this up."""
    key = (repo, file_path, "<module>", 0)
    if key not in cg._symbol_index:
        cg.add_symbol(repo, "<module>", "module", file_path, 0, 0, parent_class=None)


# ── Result tracking ──────────────────────────────────────────────────────────

class CallResolutionStats:
    def __init__(self):
        self.resolved = 0
        self.resolved_instantiates = 0                # constructor calls, tracked separately from method CALLS
        self.unresolved_chained_receiver = 0        # self.x.y() style -- correctly excluded
        self.unresolved_non_self_receiver = 0        # obj.method() where obj != self/this
        self.unresolved_target_not_found = 0          # self.foo() but foo isn't a known method
        self.unresolved_other = 0
        self.other_samples: list[str] = []            # first N raw call texts, for inspection
        self.non_self_samples: list[str] = []

    def record_other(self, call_text: str):
        self.unresolved_other += 1
        if len(self.other_samples) < 15:
            self.other_samples.append(call_text)

    def report(self, label: str):
        print(f"  [{label}] resolved(CALLS)={self.resolved} "
              f"resolved(INSTANTIATES)={self.resolved_instantiates} "
              f"unresolved(chained_receiver)={self.unresolved_chained_receiver} "
              f"unresolved(non_self_receiver)={self.unresolved_non_self_receiver} "
              f"unresolved(target_not_found)={self.unresolved_target_not_found} "
              f"unresolved(other)={self.unresolved_other}")

    def report_samples(self, label: str):
        if self.non_self_samples:
            print(f"\n  [{label}] sample of non-self/this receiver calls (first 15):")
            for s in self.non_self_samples:
                print(f"    {s!r}")
        if self.other_samples:
            print(f"\n  [{label}] sample of genuinely unclassified 'other' calls (first 15):")
            for s in self.other_samples:
                print(f"    {s!r}")


# ── Python call resolution ───────────────────────────────────────────────────

def resolve_python_calls(
    file_path: Path, repo_root: Path, cg: CodeGraph, stats: CallResolutionStats
):
    rel_path = str(file_path.relative_to(repo_root))
    ensure_module_symbol(cg, "httpx", rel_path)
    source_code = file_path.read_bytes()
    tree = py_parser.parse(source_code)
    root = tree.root_node

    def get_enclosing_context(node):
        """Walk up from a call node to find which function/method it's inside,
        and which class (if any) that method belongs to. Returns
        (caller_qualified_name, caller_class_or_None).

        REAL GAP FOUND AND FIXED (via router-layer testing, not assumed):
        a call/instantiation at true module level -- e.g.
        "DEFAULT_LIMITS = Limits(...)" in httpx/_config.py, outside any
        function or class -- previously walked all the way up and returned
        (None, None). _record_call/_record_instantiates both bail out
        silently on caller_qname=None, so these edges were dropped
        entirely, with no error or count to signal it. Confirmed via
        direct testing: httpx's real Limits class had genuinely ZERO
        INSTANTIATES edges anywhere in the graph despite being
        instantiated directly in its own defining file.

        FIX: return a synthetic "<module>" pseudo-caller instead of None,
        so the edge gets recorded with a real, queryable source rather
        than silently vanishing. This pseudo-symbol must be added to the
        graph via the normal add_symbol() path (see ensure_module_symbol()
        below) so it persists through save_graph() and is queryable via
        query_tools.py exactly like any other symbol -- not a special-cased
        bypass of the normal storage path.
        """
        current = node.parent
        while current is not None:
            if current.type == "function_definition":
                name_node = current.child_by_field_name("name")
                fn_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                # Keep walking up to see if THIS function is itself inside a class
                outer = current.parent
                while outer is not None:
                    if outer.type == "class_definition":
                        cls_name_node = outer.child_by_field_name("name")
                        cls_name = source_code[cls_name_node.start_byte:cls_name_node.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                        return (f"{cls_name}.{fn_name}", cls_name)
                    outer = outer.parent
                return (fn_name, None)
            current = current.parent
        # No enclosing function found -- this is true module-level code.
        return ("<module>", None)

    def walk(node):
        if node.type == "call":
            fn_node = node.children[0] if node.children else None

            if fn_node is not None and fn_node.type == "attribute":
                attr_children = [c for c in fn_node.children if c.type not in (".",)]
                # Confirmed shape: identifier("self") then identifier(method_name)
                # for a resolvable direct call. Any other shape (e.g. first
                # child itself being "attribute") is a chained receiver --
                # excluded per the evidence gathered before writing this.
                if (
                    len(attr_children) == 2
                    and attr_children[0].type == "identifier"
                    and attr_children[1].type == "identifier"
                ):
                    receiver = source_code[attr_children[0].start_byte:attr_children[0].end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    method_name = source_code[attr_children[1].start_byte:attr_children[1].end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    if receiver == "self":
                        caller_qname, caller_class = get_enclosing_context(node)
                        if caller_class is not None:
                            target = cg.find_all_symbol_matches(
                                "httpx", rel_path, f"{caller_class}.{method_name}"
                            )
                            if target:
                                _record_call(cg, "httpx", rel_path, caller_qname, target[0][1])
                                stats.resolved += 1
                            else:
                                stats.unresolved_target_not_found += 1
                        else:
                            call_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                            stats.record_other(f"self.{method_name}() outside any class: {call_text[:50]}")
                    else:
                        # e.g. response.json(), url.copy_with(...) -- a call on
                        # some OTHER object entirely, not self. This is a large,
                        # expected bucket: we deliberately never intended to
                        # resolve these without type inference. Recorded as
                        # its own tracked category rather than folded into a
                        # generic "other", so the real composition is visible.
                        stats.unresolved_non_self_receiver += 1
                        call_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                        if len(stats.non_self_samples) < 15:
                            stats.non_self_samples.append(call_text[:60])
                elif len(attr_children) == 2 and attr_children[0].type == "attribute":
                    # self._stream.close() shape -- chained receiver, excluded by design
                    stats.unresolved_chained_receiver += 1
                else:
                    call_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                    stats.record_other(f"unrecognized attribute-call shape: {call_text[:50]}")

            elif fn_node is not None and fn_node.type == "identifier":
                # Plain function call: foo(...) -- check same-file top-level symbols
                fn_name = source_code[fn_node.start_byte:fn_node.end_byte].decode("utf-8", errors="replace")
                caller_qname, _ = get_enclosing_context(node)
                target = cg.find_all_symbol_matches("httpx", rel_path, fn_name)
                if target:
                    target_idx = target[0][1]
                    target_node = cg.graph[target_idx]
                    if target_node.kind == "class":
                        # SomeClass(...) is a CONSTRUCTOR call, not a method
                        # call. Recording this as CALLS would be wrong --
                        # the caller isn't calling a method named "Client",
                        # it's instantiating the Client class. Found via
                        # direct verification against a synthetic repo:
                        # self.client = Client() was being recorded as
                        # "__init__ CALLS Client", which is factually wrong
                        # and would corrupt any call-chain trace (e.g.
                        # httpx-T1) that crossed a constructor call.
                        # Recorded as its own honest edge type instead.
                        _record_instantiates(cg, "httpx", rel_path, caller_qname, target_idx)
                        stats.resolved_instantiates += 1
                    else:
                        _record_call(cg, "httpx", rel_path, caller_qname, target_idx)
                        stats.resolved += 1
                else:
                    stats.unresolved_target_not_found += 1
            else:
                fn_text = source_code[fn_node.start_byte:fn_node.end_byte].decode("utf-8", errors="replace") if fn_node else "?"
                stats.record_other(f"call with unhandled function-node type ({fn_node.type if fn_node else 'None'}): {fn_text[:50]}")

        for child in node.children:
            walk(child)

    walk(root)


# ── TypeScript call resolution ───────────────────────────────────────────────

def resolve_typescript_calls(
    file_path: Path, repo_root: Path, cg: CodeGraph, stats: CallResolutionStats
):
    rel_path = str(file_path.relative_to(repo_root))
    ensure_module_symbol(cg, "got", rel_path)
    source_code = file_path.read_bytes()
    tree = ts_parser.parse(source_code)
    root = tree.root_node

    def get_enclosing_context(node):
        """Same fix as the Python resolver's get_enclosing_context -- a
        module-level instantiation (e.g. a top-level `const x = new Foo();`
        or bare `Foo();` outside any class/function) previously returned
        (None, None), silently dropping the edge. Now returns a synthetic
        "<module>" pseudo-caller instead."""
        current = node.parent
        while current is not None:
            if current.type == "method_definition" and current.parent is not None and current.parent.type == "class_body":
                name_node = current.child_by_field_name("name")
                fn_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                class_node = current.parent.parent  # class_body -> class_declaration
                if class_node is not None and class_node.type == "class_declaration":
                    cls_name_node = class_node.child_by_field_name("name")
                    cls_name = source_code[cls_name_node.start_byte:cls_name_node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    return (f"{cls_name}.{fn_name}", cls_name)
                return (fn_name, None)
            if current.type == "function_declaration":
                name_node = current.child_by_field_name("name")
                fn_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                return (fn_name, None)
            current = current.parent
        # No enclosing method/function found -- true module-level code.
        return ("<module>", None)

    def walk(node):
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")

            if fn_node is not None and fn_node.type == "member_expression":
                mem_children = [c for c in fn_node.children if c.type != "."]
                if (
                    len(mem_children) == 2
                    and mem_children[0].type == "this"
                    and mem_children[1].type in ("property_identifier", "private_property_identifier")
                ):
                    method_name = source_code[mem_children[1].start_byte:mem_children[1].end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    caller_qname, caller_class = get_enclosing_context(node)
                    if caller_class is not None:
                        target = cg.find_all_symbol_matches(
                            "got", rel_path, f"{caller_class}.{method_name}"
                        )
                        if target:
                            _record_call(cg, "got", rel_path, caller_qname, target[0][1])
                            stats.resolved += 1
                        else:
                            stats.unresolved_target_not_found += 1
                    else:
                        call_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                        stats.record_other(f"this.{method_name}() outside any class: {call_text[:50]}")
                elif len(mem_children) == 2 and mem_children[0].type == "member_expression":
                    # this.foo.bar() shape -- chained receiver, excluded by design
                    stats.unresolved_chained_receiver += 1
                elif len(mem_children) == 2 and mem_children[0].type != "this":
                    # e.g. response.json(), got.extend(...) -- a call on some
                    # OTHER identifier entirely, not this. Large, expected
                    # bucket -- we never intended to resolve these without
                    # type inference. Tracked as its own category.
                    stats.unresolved_non_self_receiver += 1
                    call_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                    if len(stats.non_self_samples) < 15:
                        stats.non_self_samples.append(call_text[:60])
                else:
                    call_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                    stats.record_other(f"unrecognized member_expression shape: {call_text[:50]}")

            elif fn_node is not None and fn_node.type == "identifier":
                fn_name = source_code[fn_node.start_byte:fn_node.end_byte].decode("utf-8", errors="replace")
                caller_qname, _ = get_enclosing_context(node)
                target = cg.find_all_symbol_matches("got", rel_path, fn_name)
                if target:
                    target_idx = target[0][1]
                    target_node = cg.graph[target_idx]
                    if target_node.kind == "class":
                        # Same fix as the Python resolver: a bare identifier
                        # call resolving to a class is a construction, not a
                        # method call. Kept as its own honest edge type.
                        _record_instantiates(cg, "got", rel_path, caller_qname, target_idx)
                        stats.resolved_instantiates += 1
                    else:
                        _record_call(cg, "got", rel_path, caller_qname, target_idx)
                        stats.resolved += 1
                else:
                    stats.unresolved_target_not_found += 1
            else:
                fn_text = source_code[fn_node.start_byte:fn_node.end_byte].decode("utf-8", errors="replace") if fn_node else "?"
                stats.record_other(f"call with unhandled function-node type ({fn_node.type if fn_node else 'None'}): {fn_text[:50]}")

        for child in node.children:
            walk(child)

    walk(root)


# ── Shared edge recording ────────────────────────────────────────────────────

def _record_call(cg: CodeGraph, repo: str, caller_file: str, caller_qname: str | None, target_idx: int):
    """Add a CALLS edge. caller_qname is now always a real string --
    get_enclosing_context() returns "<module>" for calls at true module
    level (outside any def), not None. The None check/type hint remain
    only as defensive guards, not the expected path."""
    if caller_qname is None:
        return
    caller_matches = cg.find_all_symbol_matches(repo, caller_file, caller_qname)
    if not caller_matches:
        return
    caller_idx = caller_matches[0][1]
    if not cg.graph.has_edge(caller_idx, target_idx):
        cg.graph.add_edge(caller_idx, target_idx, "CALLS")


def _record_instantiates(cg: CodeGraph, repo: str, caller_file: str, caller_qname: str | None, target_idx: int):
    """Add an INSTANTIATES edge -- distinct from CALLS. Represents 'this
    method/function creates an instance of this class', e.g.
    self.client = Client(). This is deliberately NOT a CALLS edge: found
    via direct verification (a synthetic repo test) that treating
    constructor calls as method calls produces factually wrong edges
    that would corrupt any call-chain trace crossing a constructor."""
    if caller_qname is None:
        return
    caller_matches = cg.find_all_symbol_matches(repo, caller_file, caller_qname)
    if not caller_matches:
        return
    caller_idx = caller_matches[0][1]
    if not cg.graph.has_edge(caller_idx, target_idx):
        cg.graph.add_edge(caller_idx, target_idx, "INSTANTIATES")


# ── Main pipeline ─────────────────────────────────────────────────────────

def main():
    print("Rebuilding graph with files, symbols, and imports (Steps 7-9)...\n")
    cg = build_graph_with_imports()

    httpx_root = Path("repos/httpx")
    got_root = Path("repos/got")

    from extract_symbols import should_skip_dir

    print("\nResolving CALLS edges (Step 10)...")

    httpx_stats = CallResolutionStats()
    for py_file in sorted(httpx_root.rglob("*.py")):
        if any(should_skip_dir(part) for part in py_file.relative_to(httpx_root).parts[:-1]):
            continue
        resolve_python_calls(py_file, httpx_root, cg, httpx_stats)
    httpx_stats.report("httpx")
    httpx_stats.report_samples("httpx")

    got_stats = CallResolutionStats()
    for ts_file in sorted(got_root.rglob("*.ts")):
        if any(should_skip_dir(part) for part in ts_file.relative_to(got_root).parts[:-1]):
            continue
        resolve_typescript_calls(ts_file, got_root, cg, got_stats)
    got_stats.report("got")
    got_stats.report_samples("got")

    call_edges = sum(1 for e in cg.graph.edges() if e == "CALLS")
    instantiates_edges = sum(1 for e in cg.graph.edges() if e == "INSTANTIATES")
    module_symbols = sum(
        1 for n in cg.graph.nodes()
        if hasattr(n, "kind") and n.kind == "module"
    )
    print(f"\nTotal CALLS edges added: {call_edges}")
    print(f"Total INSTANTIATES edges added: {instantiates_edges}")
    print(f"'<module>' pseudo-symbols created (one per file with module-level "
          f"calls/instantiations): {module_symbols}")
    print("(Compare instantiates_edges against the pre-fix count to confirm "
          "the module-level gap -- e.g. httpx's Limits class -- is now captured.)")

    save_graph(cg, "data/code_graph.db")
    print("Graph persisted to data/code_graph.db")
    print("(CALLS and INSTANTIATES edges are included in the persisted database.)")


if __name__ == "__main__":
    main()

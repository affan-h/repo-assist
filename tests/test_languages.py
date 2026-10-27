"""Symbol and call-site extraction across languages (grammars from tree-sitter-language-pack)."""

import pytest

from repo_assist.ingest import code

CASES = {
    "server.go": (
        "package web\n\ntype Server struct {\n\taddr string\n}\n\n"
        "func (s *Server) Serve(port int) error {\n\treturn s.listen(port)\n}\n\n"
        "func NewServer() *Server {\n\treturn &Server{}\n}\n",
        {"Server": "type", "Server.Serve": "method", "NewServer": "function"},
        {("listen", "Server.Serve")},
    ),
    "Client.java": (
        "package a;\n\npublic class Client {\n  public Client() { init(); }\n"
        "  public Response send(Request r) {\n    return transport.handle(r);\n  }\n}\n",
        {"Client": "class", "Client.Client": "method", "Client.send": "method"},
        {("init", "Client.Client"), ("handle", "Client.send")},
    ),
    "lib.rs": (
        "pub struct Pool { size: usize }\n\nimpl Pool {\n    pub fn new(size: usize) -> Self {\n"
        "        Self::validate(size);\n        Pool { size }\n    }\n}\n\nfn helper() {}\n",
        {"Pool": "impl", "Pool.new": "method", "helper": "function"},
        {("validate", "Pool.new")},
    ),
    "pool.cpp": (
        "class Pool {\n public:\n  void drain();\n};\n\nvoid Pool::drain() {\n  release(42);\n}\n",
        {"Pool": "class", "Pool.drain": "function"},
        {("release", "Pool.drain")},
    ),
    "client.rb": (
        "module Http\n  class Client\n    def get(url)\n      request(:get, url)\n    end\n  end\nend\n",
        {"Http": "module", "Http.Client": "class", "Http.Client.get": "method"},
        {("request", "Http.Client.get")},
    ),
}


@pytest.mark.parametrize("path", sorted(CASES))
def test_symbols_and_calls(path):
    source, expected_symbols, expected_calls = CASES[path]
    parsed = code.parse_file(path, source.encode())
    symbols = {s.qualified_name: s.kind for s in parsed.symbols}
    for name, kind in expected_symbols.items():
        assert symbols.get(name) == kind, (name, symbols)
    calls = {(c.callee, c.caller) for c in parsed.calls}
    assert expected_calls <= calls, calls
    for chunk in parsed.chunks:  # every chunk cites exact lines that exist in the file
        assert chunk.ref == f"{path}:{chunk.start_line}-{chunk.end_line}"
        assert 1 <= chunk.start_line <= chunk.end_line <= source.count("\n") + 1


def test_languages_without_a_spec_are_indexed_as_windows():
    source = "\n".join(f"local x{i} = {i}" for i in range(100)) + "\n"
    parsed = code.parse_file("init.lua", source.encode())
    assert parsed.symbols == []
    assert parsed.chunks and parsed.chunks[0].start_line == 1 and parsed.chunks[-1].end_line == 100


@pytest.mark.parametrize("path, indexable", [
    ("cmd/server/main.go", True),
    ("src/lib.rs", True),
    ("app/models/user.rb", True),
    ("scripts/build.lua", True),
    ("README.md", False),            # docs have their own chunker
    ("package.json", False),         # data/config: grep covers it
    (".github/workflows/ci.yml", False),
    ("server_test.go", False),       # tests: reachable via grep/read_file
    ("src/test/java/AppTest.java", False),
    ("vendor/github.com/x/y.go", False),
    ("assets/app.min.js", False),
])
def test_indexable_source(path, indexable):
    assert code.is_indexable_source(path) is indexable

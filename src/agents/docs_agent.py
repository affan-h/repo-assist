"""
Docs specialist agent (v2, §4.3).

Real, tested v1 keyword/IDF search (query_tools.search_docs) plus the new
v2 semantic layer (embeddings.search_semantic). Both result sets are kept
separate and returned together, per §4.3: "real, useful signal for later
analysis of which retrieval strategy actually works better per question
type" -- not merged/deduped here, so that signal isn't thrown away.

Real bug fixed here, found via a live grader run: PR bodies and doc_chunk
content were truncated to 600 chars each. Traced via a direct example --
the "got.extend()" question's top semantic hit (documentation/10-instances.md,
score 0.637, genuinely the correct source) contains the real answer, but it
runs well past 600 chars; the cut landed mid-explanation. The synthesizer,
starved of the back half of its own best evidence, filled the gap with
plausible-sounding trained knowledge instead -- which the verifier then
correctly flagged as unsupported, triggered a synthesis_error retry, and
the retry regenerated a DIFFERENT unsupported version of the same guess,
since the retry never fixed the actual problem (evidence volume). This
wasn't a synthesizer prompting problem or a verifier miscalibration --
it was evidence being discarded before synthesis ever saw it. Raised to
a real, generous cap; see the constant below for the reasoning on the
specific number chosen.
"""

import sqlite3
from config import DB_PATH
import query_tools as tools
from embeddings import search_semantic

# Real cap, not a guess: real doc chunks run up to ~400 words per
# build_embeddings_index.py's own CHUNK_SIZE, which is roughly 2000-2800
# characters of English/code-mixed text. 600 was cutting most real chunks
# off mid-explanation (confirmed directly against a real chunk that was
# ~1800 chars and needed all of it). 3000 comfortably covers a full real
# chunk without needing to guess further; orchestrator.py's own evidence
# formatting still applies a final safety cap before the LLM call, so this
# isn't the only backstop against an unbounded prompt.
MAX_EVIDENCE_CHARS = 3000


def execute(repo: str, question: str, focus_notes: str = "") -> dict:
    print(f"  [Agent: Docs] Gathering semantic and keyword documentation...")
    evidence: dict = {"focus_notes": focus_notes}

    # 1. v1 keyword/IDF search -- real, tested, structurally can't find
    # conceptually-related content with no literal term overlap (§4.1).
    try:
        evidence["keyword_docs"] = tools.search_docs(repo, question)
    except Exception as e:
        evidence["keyword_docs"] = []
        evidence["keyword_docs_error"] = str(e)

    # 2. v2 semantic search -- real vectors, brute-force cosine (§3.1/§4.1).
    try:
        semantic_hits = search_semantic(DB_PATH, repo, question, top_k=5)
    except Exception as e:
        semantic_hits = []
        evidence["semantic_error"] = str(e)

    semantic_results = []
    if semantic_hits:
        with sqlite3.connect(DB_PATH) as conn:
            for hit in semantic_hits:
                src_type = hit["source_type"]
                src_id = hit["source_id"]
                score = hit["score"]

                if src_type == "summary":
                    # source_id format from build_embeddings_index.py:
                    # "<file_path>::<qualified_name>::<start_line>" -- real
                    # fix vs. qualified_name alone, which collided 66 real
                    # summaries in got (e.g. TypeScript interface property
                    # declarations sharing a name across multiple real
                    # locations -- same collision class as Phase 1's own
                    # getter/setter dedup bug in project_context.md).
                    parts = src_id.split("::")
                    row = None
                    if len(parts) == 3:
                        file_path, qname, start_line = parts
                        if start_line.isdigit():
                            row = conn.execute(
                                "SELECT summary FROM summaries WHERE repo=? AND file_path=? "
                                "AND qualified_name=? AND start_line=?",
                                (repo, file_path, qname, int(start_line)),
                            ).fetchone()
                    if row:
                        semantic_results.append({
                            "source_type": "summary", "source_id": src_id,
                            "score": score, "text": row[0],
                        })

                elif src_type == "pr":
                    # source_id format from build_embeddings_index.py: "PR#<num>_chunk<idx>"
                    pr_num = src_id.split("_")[0].replace("PR#", "")
                    row = conn.execute(
                        "SELECT title, body FROM pr_cache WHERE repo=? AND pr_number=?",
                        (repo, pr_num),
                    ).fetchone()
                    if row:
                        body = (row[1] or "")[:MAX_EVIDENCE_CHARS]
                        semantic_results.append({
                            "source_type": "pr", "source_id": pr_num,
                            "score": score, "text": f"{row[0]} - {body}",
                        })
                elif src_type == "doc_chunk":
                    # source_id format from build_embeddings_index.py:
                    # "<file_path>#chunk<idx>" for a doc row embedded whole, or
                    # "<file_path>#chunk<idx>.<sub_idx>" when a single docs-table
                    # row was itself too long and got split further (real fix,
                    # found via a direct 4,710-token crash on real data -- see
                    # build_embeddings_index.py's docstring). The DB lookup is
                    # keyed on the original docs.chunk_index only -- sub_idx
                    # exists purely to keep source_id unique in the embeddings
                    # table's PRIMARY KEY, not as a separate DB column.
                    file_path, _, chunk_part = src_id.rpartition("#chunk")
                    base_chunk_part = chunk_part.split(".")[0]
                    row = None
                    if base_chunk_part.isdigit():
                        row = conn.execute(
                            "SELECT heading, content FROM docs WHERE repo=? AND file_path=? AND chunk_index=?",
                            (repo, file_path, int(base_chunk_part)),
                        ).fetchone()
                    if row:
                        semantic_results.append({
                            "source_type": "doc_chunk", "source_id": src_id,
                            "score": score, "text": f"[{row[0]}] {row[1][:MAX_EVIDENCE_CHARS]}",
                        })
                # other source_types (issue, discussion) intentionally not yet
                # wired here -- build_embeddings_index.py doesn't index them
                # currently; extend both together rather than reading a
                # source_type the indexer never writes.

    evidence["semantic_results"] = semantic_results
    return evidence

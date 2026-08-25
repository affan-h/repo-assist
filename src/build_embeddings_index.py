"""
One-time embeddings indexing (v2, §4.1/§7.1).

Real fixes vs. the previous draft:
  - Uses embeddings.py's compute_embedding(text, is_query=False) rather
    than calling a model object's .encode() directly -- this applies the
    real document-prompt prefix EmbeddingGemma expects (see embeddings.py's
    docstring for why, including the real optimum/transformers version
    conflict that forced a switch to raw ONNX Runtime instead of
    sentence-transformers). This file doesn't need to know HOW embeddings
    get computed, only that indexed content should use is_query=False.
  - §4.1 says "every doc chunk (`docs` table)" -- the previous draft only
    indexed summaries and PRs, silently dropping the docs table entirely.
    Added below.
  - Kept issues/discussions un-indexed for now (matching what docs_agent.py
    currently reads back), but this is now stated explicitly rather than
    silently omitted -- extend both together if you add issue/discussion
    indexing.
"""

import sqlite3
from datetime import datetime, timezone

from embeddings import compute_embedding, get_embedding_model, MODEL_NAME

DB_PATH = "../data/code_graph.db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def init_embeddings_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            repo TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            vector BLOB NOT NULL,
            model_name TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (repo, source_type, source_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_repo_type ON embeddings(repo, source_type);")
    conn.commit()


def chunk_text(text: str) -> list[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP)]


def _insert_vector(conn, repo, source_type, source_id, vector):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO embeddings (repo, source_type, source_id, vector, model_name, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (repo, source_type, source_id, vector.tobytes(), MODEL_NAME, now))


def build_index():
    get_embedding_model()  # load once up front, real load-time cost surfaced immediately not on first use
    with sqlite3.connect(DB_PATH) as conn:
        init_embeddings_table(conn)
        cursor = conn.cursor()

        # 1. Symbol summaries (Phase 2's `summaries` table)
        #
        # Real fix, found via a direct row-count discrepancy on real data:
        # summaries' REAL primary key is (repo, file_path, qualified_name,
        # start_line) -- four columns, confirmed via `.schema summaries`.
        # Using qualified_name alone as embeddings' source_id silently
        # collided 66 real summaries in got (e.g. "Options.agent",
        # "Options.body" -- TypeScript interface properties/declaration
        # merging sharing a name across multiple real declarations) --
        # INSERT OR REPLACE overwrote earlier ones with later ones, same
        # collision class Phase 1's own getter/setter dedup bug already
        # hit once (see project_context.md). Fixed: source_id now encodes
        # the full real key so two distinct summaries can never collide.
        print("\n--- Indexing symbol summaries ---")
        cursor.execute("SELECT repo, file_path, qualified_name, start_line, summary FROM summaries")
        rows = cursor.fetchall()
        indexed = 0
        for repo, file_path, name, start_line, summary in rows:
            if not summary:
                continue
            vector = compute_embedding(summary, is_query=False)
            source_id = f"{file_path}::{name}::{start_line}"
            _insert_vector(conn, repo, "summary", source_id, vector)
            indexed += 1
        conn.commit()
        print(f"Indexed {indexed}/{len(rows)} summaries (skipped rows with empty summary).")

        # 2. PRs (pr_cache) -- chunked, since bodies can run long
        print("\n--- Indexing PRs ---")
        try:
            cursor.execute("SELECT repo, pr_number, title, body FROM pr_cache")
            pr_rows = cursor.fetchall()
            chunk_count = 0
            for repo, pr_num, title, body in pr_rows:
                text = f"Title: {title}\n\n{body or ''}"
                for idx, chunk in enumerate(chunk_text(text)):
                    source_id = f"PR#{pr_num}_chunk{idx}"
                    vector = compute_embedding(chunk, is_query=False)
                    _insert_vector(conn, repo, "pr", source_id, vector)
                    chunk_count += 1
            conn.commit()
            print(f"Indexed {len(pr_rows)} PRs into {chunk_count} chunks.")
        except sqlite3.OperationalError as e:
            print(f"Could not index PRs (pr_cache missing/empty?): {e}")

        # 3. Docs (README/CHANGELOG/documentation chunks) -- required by
        # §4.1 ("every doc chunk (`docs` table)"), missing from the
        # previous draft entirely.
        #
        # Real fix, found via a direct crash on real data: one doc row's
        # content tokenized to 4,710 tokens -- over 2x the model's real
        # 2048-token ceiling -- and crashed the model's RotaryEmbedding op.
        # compute_embedding() now truncates defensively so this can never
        # crash again, but silent truncation alone would throw away real
        # content for long doc chunks. Rechunk by word count here too
        # (same chunk_text() used for PRs) so long docs become multiple
        # real, fully-embedded chunks instead of one truncated one.
        print("\n--- Indexing docs ---")
        try:
            cursor.execute("SELECT repo, file_path, chunk_index, heading, content FROM docs")
            doc_rows = cursor.fetchall()
            doc_indexed = 0
            oversized_rows = 0
            for repo, file_path, chunk_index, heading, content in doc_rows:
                if not content:
                    continue
                text = f"{heading}\n\n{content}" if heading else content
                sub_chunks = chunk_text(text)
                if len(sub_chunks) > 1:
                    oversized_rows += 1
                for sub_idx, sub_text in enumerate(sub_chunks):
                    source_id = f"{file_path}#chunk{chunk_index}"
                    if len(sub_chunks) > 1:
                        source_id += f".{sub_idx}"
                    vector = compute_embedding(sub_text, is_query=False)
                    _insert_vector(conn, repo, "doc_chunk", source_id, vector)
                    doc_indexed += 1
            conn.commit()
            print(f"Indexed {len(doc_rows)} doc rows into {doc_indexed} chunks "
                  f"({oversized_rows} row(s) were oversized and split further).")
        except sqlite3.OperationalError as e:
            print(f"Could not index docs (docs table missing/empty?): {e}")

        print("\nPhase A indexing complete.")
        cursor.execute("SELECT source_type, COUNT(*) FROM embeddings GROUP BY source_type")
        for source_type, count in cursor.fetchall():
            print(f"  {source_type}: {count} vectors")


if __name__ == "__main__":
    build_index()

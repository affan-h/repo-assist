"""
Phase 2, Step 5 -- bulk Discussion indexing (new strategy).

Real finding that motivated this pivot: Discussion #3007 in httpx
("Make working with SSL easier") contains the actual settled rationale
for the SSLContext API design, confirmed by direct inspection of the
page. But it is marked "Unanswered" by GitHub -- no comment was ever
selected as the official answer, despite a clear real conclusion in the
reply thread. Our keyword-search approach (fetch_discussion.py) failed
to surface it even with in:title removed and first: raised to 15,
because GitHub's relevance ranking for a common term ("SSLContext")
across 32 matches doesn't reliably put the right OLDER discussion near
the top.

New strategy: stop searching per-PR, per-guess. Instead, bulk-index
all discussions in a known design-relevant category (confirmed via the
live page: #3007 sits in category "💡 Ideas") upfront, the same way we
already bulk-mine commit history. Store everything, regardless of
whether GitHub marked an answer -- we already proved that signal is
unreliable. Linking discussions to specific PRs/commits is a SEPARATE,
later step (fuzzy match by shared vocabulary/timing), not done here.

Verified schema before writing this:
  - repository(owner, name) { discussionCategories(first: N) { nodes
    { id, name } } } -- confirmed real, gets category IDs.
  - repository(owner, name) { discussions(categoryId: ID, first: N,
    after: String) { pageInfo { hasNextPage, endCursor }, nodes {...} } }
    -- confirmed real, standard cursor-based pagination.
  - A repo has at most 25 categories (documented limit) -- small,
    boundable list, safe to fetch in one call.

Run with:
    python3 src/index_discussions.py <owner> <repo> [category_name]
    e.g. python3 src/index_discussions.py encode httpx Ideas
    (omit category_name to first list all available categories)
"""

import os
import sys
import sqlite3
import json
import time
import requests

from config import DB_PATH


GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


CATEGORIES_QUERY = """
query($owner: String!, $repo: String!) {
  repository(owner: $owner, name: $repo) {
    discussionCategories(first: 25) {
      nodes {
        id
        name
        description
      }
    }
  }
  rateLimit { limit remaining resetAt }
}
"""


DISCUSSIONS_IN_CATEGORY_QUERY = """
query($owner: String!, $repo: String!, $categoryId: ID!, $after: String) {
  repository(owner: $owner, name: $repo) {
    discussions(categoryId: $categoryId, first: 25, after: $after,
                orderBy: {field: CREATED_AT, direction: ASC}) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
        url
        createdAt
        author { login }
        isAnswered
        answer {
          body
          author { login }
        }
        comments(first: 30) {
          totalCount
          nodes {
            author { login }
            body
            createdAt
            replies(first: 10) {
              nodes {
                author { login }
                body
                createdAt
              }
            }
          }
        }
      }
    }
  }
  rateLimit { limit remaining resetAt }
}
"""


def init_discussions_index_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS discussions_index (
            repo TEXT NOT NULL,
            discussion_number INTEGER NOT NULL,
            category TEXT,
            title TEXT,
            body TEXT,
            url TEXT,
            author TEXT,
            created_at TEXT,
            is_answered INTEGER,
            answer_body TEXT,
            answer_author TEXT,
            comments_json TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (repo, discussion_number)
        );
        CREATE INDEX IF NOT EXISTS idx_discussions_repo ON discussions_index(repo);
    """)
    conn.commit()
    return conn


def get_categories(owner: str, repo: str, token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"query": CATEGORIES_QUERY, "variables": {"owner": owner, "repo": repo}}
    response = requests.post(GITHUB_GRAPHQL_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]["repository"]["discussionCategories"]["nodes"]


def index_all_discussions_in_category(
    owner: str, repo: str, category_id: str, category_name: str,
    token: str, db_path: str,
):
    """
    Paginates through EVERY discussion in the given category, storing
    each one fully -- including comments and their replies -- regardless
    of whether GitHub marked an official answer. This is a bulk,
    upfront index, not a per-query lazy fetch, since we already learned
    that per-query keyword search misses real results.
    """
    headers = {"Authorization": f"Bearer {token}"}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    after_cursor = None
    total_indexed = 0

    while True:
        payload = {
            "query": DISCUSSIONS_IN_CATEGORY_QUERY,
            "variables": {"owner": owner, "repo": repo, "categoryId": category_id, "after": after_cursor},
        }
        response = requests.post(GITHUB_GRAPHQL_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        discussions_conn = data["data"]["repository"]["discussions"]
        rate_limit = data["data"]["rateLimit"]

        for disc in discussions_conn["nodes"]:
            answer = disc.get("answer")
            cur.execute(
                """INSERT OR REPLACE INTO discussions_index
                   (repo, discussion_number, category, title, body, url, author,
                    created_at, is_answered, answer_body, answer_author,
                    comments_json, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    repo, disc["number"], category_name, disc["title"], disc["body"],
                    disc["url"], disc["author"]["login"] if disc["author"] else None,
                    disc["createdAt"], 1 if disc["isAnswered"] else 0,
                    answer["body"] if answer else None,
                    answer["author"]["login"] if answer and answer["author"] else None,
                    json.dumps(disc["comments"]["nodes"]),
                    str(time.time()),
                ),
            )
            total_indexed += 1

        conn.commit()

        print(f"  [{total_indexed}/{discussions_conn['totalCount']}] indexed so far "
              f"(rate limit: {rate_limit['remaining']}/{rate_limit['limit']})")

        if not discussions_conn["pageInfo"]["hasNextPage"]:
            break
        after_cursor = discussions_conn["pageInfo"]["endCursor"]

    conn.close()
    return total_indexed


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 src/index_discussions.py <owner> <repo> [category_name]")
        print("Example: python3 src/index_discussions.py encode httpx Ideas")
        print("(omit category_name to list all available categories first)")
        sys.exit(1)

    owner, repo = sys.argv[1], sys.argv[2]
    category_name_filter = sys.argv[3] if len(sys.argv) > 3 else None

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable not set.")

    init_discussions_index_table(DB_PATH)

    print(f"Fetching discussion categories for {owner}/{repo}...")
    categories = get_categories(owner, repo, token)

    print(f"\nAvailable categories:")
    for cat in categories:
        print(f"  {cat['name']!r} (id={cat['id']})  -- {cat.get('description', '')}")

    if category_name_filter is None:
        print("\nNo category specified -- listed available categories above.")
        print("Re-run with a category name to index it, e.g.:")
        print(f"  python3 src/index_discussions.py {owner} {repo} \"Ideas\"")
        return

    matching = [c for c in categories if c["name"].lower() == category_name_filter.lower()]
    if not matching:
        print(f"\nNo category matching '{category_name_filter}' found. "
              f"Available: {[c['name'] for c in categories]}")
        return

    category = matching[0]
    print(f"\nIndexing all discussions in category '{category['name']}'...")
    total = index_all_discussions_in_category(
        owner, repo, category["id"], category["name"], token, DB_PATH
    )
    print(f"\nDone. Indexed {total} discussions from '{category['name']}' into {DB_PATH}.")


if __name__ == "__main__":
    main()

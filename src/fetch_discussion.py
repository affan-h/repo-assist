"""
Phase 2, Step 4 -- GitHub Discussions search, extending fetch_pr.py.

Motivated by a real finding: PR #3319 in httpx ("Introduce new SSLContext
API...") turned out to be a large release-staging PR whose own body
doesn't explain the SSLContext design rationale -- that rationale lives
in Discussion #3007 ("Make working with SSL easier."), which we would
have completely missed with PR-only mining.

Verified before writing this (see conversation research):
  - Discussion has NO direct number-based lookup shortcut like
    repository(owner, name) { pullRequest(number: N) } does for PRs.
    We don't have discussion numbers from our commit data anyway --
    only PR numbers -- so keyword SEARCH is the correct mechanism,
    not a lookup we were missing.
  - search(query: "repo:owner/name in:title keyword", type: DISCUSSION,
    first: N) is the real, confirmed query shape. type: DISCUSSION is
    its own distinct search type (separate from ISSUE, which covers
    both issues and PRs together).
  - Discussion.answer / answerChosenBy fields (confirmed in schema)
    flag GitHub's own "marked as answer" signal -- a strong, native
    indicator of settled rationale, worth surfacing distinctly from
    ordinary comments.

Design: given a PR's title (already cached by fetch_pr.py), extract
plausible search keywords and search for a matching Discussion. This is
a heuristic, not a guaranteed link -- multiple discussions could match,
or none could. We surface candidates and let the caller (eventually,
the verifier agent) judge relevance, rather than silently picking one.

Run with:
    python3 src/fetch_discussion.py <owner> <repo> "<search keywords>"
    e.g. python3 src/fetch_discussion.py encode httpx "SSLContext SSL"
"""

import os
import sys
import sqlite3
import json
import time
import requests


GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


DISCUSSION_SEARCH_QUERY = """
query($searchQuery: String!, $maxResults: Int!) {
  search(query: $searchQuery, type: DISCUSSION, first: $maxResults) {
    discussionCount
    nodes {
      ... on Discussion {
        number
        title
        body
        url
        createdAt
        author { login }
        answer {
          body
          author { login }
        }
        comments(first: 20) {
          totalCount
          nodes {
            author { login }
            body
            createdAt
          }
        }
      }
    }
  }
  rateLimit {
    limit
    remaining
    resetAt
  }
}
"""


def init_discussion_cache_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS discussion_cache (
            repo TEXT NOT NULL,
            search_query TEXT NOT NULL,
            discussion_number INTEGER,
            title TEXT,
            body TEXT,
            url TEXT,
            author TEXT,
            answer_body TEXT,
            answer_author TEXT,
            comments_json TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (repo, search_query, discussion_number)
        );
    """)
    conn.commit()
    return conn


def get_cached_discussion_search(db_path: str, repo: str, search_query: str) -> list[dict] | None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM discussion_cache WHERE repo = ? AND search_query = ?",
        (repo, search_query),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None
    columns = ["repo", "search_query", "discussion_number", "title", "body", "url",
               "author", "answer_body", "answer_author", "comments_json", "fetched_at"]
    return [dict(zip(columns, row)) for row in rows]


def search_discussions_live(owner: str, repo: str, keywords: str, token: str, max_results: int = 15) -> list[dict]:
    """
    Live GraphQL search call. Only invoked on a cache miss.

    IMPORTANT FIX: no longer restricts to `in:title`. Confirmed directly
    from GitHub's own search docs: "When you omit the in qualifier,
    GitHub searches the title, body, and comments." Our original
    in:title-only search was why Discussion #3007 ("Make working with
    SSL easier") never surfaced for a query like "SSLContext" -- that
    term appears in the discussion's BODY, not its title, and in:title
    was silently excluding it. Dropping the restriction searches all
    three fields, which is what we actually want for rationale-mining.

    max_results raised from a hardcoded 5 to a configurable value
    (default 15) -- verified this matters directly: a real search for
    "SSL" matched 27 discussions, and our target result did not appear
    in the first 5.
    """
    headers = {"Authorization": f"Bearer {token}"}
    search_string = f"repo:{owner}/{repo} {keywords}"

    payload = {
        "query": DISCUSSION_SEARCH_QUERY,
        "variables": {"searchQuery": search_string, "maxResults": max_results},
    }

    response = requests.post(GITHUB_GRAPHQL_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    results = data["data"]["search"]["nodes"]
    rate_limit = data["data"]["rateLimit"]
    print(f"  Rate limit: {rate_limit['remaining']}/{rate_limit['limit']} remaining")
    print(f"  Search matched {data['data']['search']['discussionCount']} discussion(s) total "
          f"(showing up to {max_results})")

    return results


def save_discussion_results(db_path: str, repo: str, search_query: str, results: list[dict]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not results:
        # Record that this search was attempted and found NOTHING --
        # important so we don't re-search the same empty query later,
        # and so a future verifier can distinguish "searched, found
        # nothing" from "never searched."
        cur.execute(
            """INSERT OR REPLACE INTO discussion_cache
               (repo, search_query, discussion_number, title, body, url,
                author, answer_body, answer_author, comments_json, fetched_at)
               VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)""",
            (repo, search_query, str(time.time())),
        )
    else:
        for disc in results:
            answer = disc.get("answer")
            cur.execute(
                """INSERT OR REPLACE INTO discussion_cache
                   (repo, search_query, discussion_number, title, body, url,
                    author, answer_body, answer_author, comments_json, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    repo, search_query, disc["number"], disc["title"], disc["body"], disc["url"],
                    disc["author"]["login"] if disc["author"] else None,
                    answer["body"] if answer else None,
                    answer["author"]["login"] if answer and answer["author"] else None,
                    json.dumps(disc["comments"]["nodes"]),
                    str(time.time()),
                ),
            )

    conn.commit()
    conn.close()


def _translate_cache_rows(cached: list[dict] | None) -> list[dict]:
    """
    Single source of truth for turning raw cache rows into the public
    return shape. A cache row with discussion_number IS NULL is our
    sentinel for "searched, found nothing" -- it must become an empty
    list, not be returned as if it were a real discussion.

    This was previously duplicated (checked on the cache-hit path only)
    and NOT applied on the fresh-search path, which caused a real bug:
    a genuinely empty search result printed a fake "#None: None" result
    to the user instead of correctly reporting zero matches. Factoring
    this into one function used by both paths closes that gap for good.
    """
    if cached is None:
        return []
    if len(cached) == 1 and cached[0]["discussion_number"] is None:
        return []
    return cached


def search_discussions(
    owner: str, repo: str, keywords: str, db_path: str = "data/code_graph.db", max_results: int = 15
) -> list[dict]:
    """
    Main entry point. Checks cache first (keyed by exact search string),
    only hits the live API on a miss.
    """
    init_discussion_cache_table(db_path)
    # Cache key matches what we actually send to GitHub now that in:title
    # is no longer prepended -- keeping key and query in sync matters,
    # since a stale key format would silently create duplicate cache
    # entries for the same real search.
    search_query_key = keywords

    cached = get_cached_discussion_search(db_path, repo, search_query_key)
    if cached is not None:
        translated = _translate_cache_rows(cached)
        if not translated:
            print(f"  Cache hit: previously searched '{keywords}', found nothing.")
        else:
            print(f"  Cache hit: {len(translated)} cached discussion(s) for '{keywords}'.")
        return translated

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable not set.")

    print(f"  Cache miss: searching Discussions for '{keywords}' live...")
    results = search_discussions_live(owner, repo, keywords, token, max_results=max_results)
    save_discussion_results(db_path, repo, search_query_key, results)

    return _translate_cache_rows(get_cached_discussion_search(db_path, repo, search_query_key))


def main():
    if len(sys.argv) != 4:
        print('Usage: python3 src/fetch_discussion.py <owner> <repo> "<keywords>"')
        print('Example: python3 src/fetch_discussion.py encode httpx "SSLContext SSL"')
        sys.exit(1)

    owner, repo, keywords = sys.argv[1], sys.argv[2], sys.argv[3]

    results = search_discussions(owner, repo, keywords)

    if not results:
        print(f"\nNo discussions found matching '{keywords}' in {owner}/{repo}.")
        return

    print(f"\n{'='*70}")
    print(f"Found {len(results)} discussion(s):")
    print(f"{'='*70}")
    for disc in results:
        print(f"\n#{disc['discussion_number']}: {disc['title']}")
        print(f"  URL: {disc['url']}")
        print(f"  Author: {disc['author']}")
        if disc['answer_body']:
            print(f"  MARKED ANSWER (by {disc['answer_author']}): {disc['answer_body'][:200]}")
        else:
            print(f"  (no marked answer)")
        comments = json.loads(disc['comments_json']) if disc['comments_json'] else []
        print(f"  {len(comments)} comment(s) in thread")


if __name__ == "__main__":
    main()

"""
Phase 2 extension -- GitHub GraphQL Issue fetcher, mirroring fetch_pr.py's
proven pattern exactly (same auth, same rate-limit handling, same lazy
fetch-on-miss + cache design).

WHY THIS EXISTS: full 56-question eval run confirmed several real "why"
ground truths cite bare GitHub ISSUE numbers, not PRs -- Y1 (#572), Y3
(#1274, #1173), Y7 (#572). fetch_pr.py only fetches PRs; there was no
mechanism at all to fetch a standalone Issue. Real query shape
(repository(owner,name){ issue(number){...} }) confirmed via GitHub's own
community discussions before writing this.

Design: same "lazy" on-demand fetch as fetch_pr.py -- no bulk pre-fetch,
cache immediately on first fetch, never re-hit the API for a cached issue.

Setup: same GITHUB_TOKEN env var fetch_pr.py already uses -- no new auth
needed.

Run with:
    python3 src/fetch_issue.py <owner> <repo> <issue_number>
    e.g. python3 src/fetch_issue.py encode httpx 572

CONFIRMED REAL, EXTERNAL CONSTRAINT (as of testing on 2026-07-15, NOT a bug
in this code): httpx's maintainer closed off Issues access on the encode/httpx
repository (see github.com/encode/httpx/discussions/3784, "Closing off
access.", posted Feb 27 2026). Direct GraphQL testing confirmed:
  - repository(...).issue(number: N) on encode/httpx -> NOT_FOUND for every
    real, previously-existing issue number tested (#572, #1274), even
    though the issue content is still publicly visible via github.com's
    historical page cache / search engines.
  - repository(...).pullRequest(...) and repository(...).discussion(...)
    on encode/httpx BOTH still resolve fine (confirmed live) -- only the
    Issues surface is closed for this specific repo, not Discussions/PRs.
  - sindresorhus/got's Issues are CONFIRMED STILL OPEN (tested, got issue
    #1 resolves fine) -- this closure is httpx-specific, not universal.

PRACTICAL IMPLICATION: httpx "why" ground truths citing bare Issue numbers
(e.g. #572, #1274, #1173) are permanently unanswerable via live fetch for
as long as this closure remains in effect -- structurally identical to
unanswerable_why in practice, even though categorized as "why" in the eval
set. This is logged here as a known, real, external limitation -- not
something get_issue() can route around, since the API itself returns
NOT_FOUND, not a rate-limit or auth error that a retry could fix.
"""

import os
import sys
import sqlite3
import json
import time

from config import DB_PATH
import requests


GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


ISSUE_QUERY = """
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      number
      title
      body
      state
      author { login }
      createdAt
      closedAt
      comments(first: 50) {
        totalCount
        nodes {
          author { login }
          body
          createdAt
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


def init_issue_cache_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS issue_cache (
            repo TEXT NOT NULL,
            issue_number INTEGER NOT NULL,
            title TEXT,
            body TEXT,
            state TEXT,
            author TEXT,
            created_at TEXT,
            closed_at TEXT,
            comments_json TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (repo, issue_number)
        );
    """)
    conn.commit()
    return conn


def get_cached_issue(db_path: str, repo: str, issue_number: int) -> dict | None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM issue_cache WHERE repo = ? AND issue_number = ?", (repo, issue_number))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    columns = ["repo", "issue_number", "title", "body", "state", "author",
               "created_at", "closed_at", "comments_json", "fetched_at"]
    return dict(zip(columns, row))


def fetch_issue_from_github(owner: str, repo: str, issue_number: int, token: str) -> dict:
    """Live GraphQL call. Only invoked on a cache miss."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "query": ISSUE_QUERY,
        "variables": {"owner": owner, "repo": repo, "number": issue_number},
    }

    response = requests.post(GITHUB_GRAPHQL_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    issue_data = data["data"]["repository"]["issue"]
    rate_limit = data["data"]["rateLimit"]

    if issue_data is None:
        raise ValueError(f"Issue #{issue_number} not found in {owner}/{repo} "
                          f"(wrong number, or it's a PR not a standalone issue)")

    print(f"  Rate limit: {rate_limit['remaining']}/{rate_limit['limit']} "
          f"remaining, resets at {rate_limit['resetAt']}")

    return issue_data


def save_issue_to_cache(db_path: str, repo: str, issue_data: dict):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT OR REPLACE INTO issue_cache
           (repo, issue_number, title, body, state, author, created_at,
            closed_at, comments_json, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            repo,
            issue_data["number"],
            issue_data["title"],
            issue_data["body"],
            issue_data["state"],
            issue_data["author"]["login"] if issue_data["author"] else None,
            issue_data["createdAt"],
            issue_data["closedAt"],
            json.dumps(issue_data["comments"]["nodes"]),
            str(time.time()),
        ),
    )
    conn.commit()
    conn.close()


def get_issue(owner: str, repo: str, issue_number: int, db_path: str = DB_PATH) -> dict:
    """Main entry point -- same lazy fetch-on-miss pattern as fetch_pr.py's get_pr()."""
    init_issue_cache_table(db_path)

    cached = get_cached_issue(db_path, repo, issue_number)
    if cached is not None:
        print(f"  Cache hit: Issue #{issue_number} already fetched previously.")
        return cached

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable not set. "
            "Create a token at https://github.com/settings/tokens and "
            "run: export GITHUB_TOKEN=ghp_your_token_here"
        )

    print(f"  Cache miss: fetching Issue #{issue_number} from GitHub live...")
    issue_data = fetch_issue_from_github(owner, repo, issue_number, token)
    save_issue_to_cache(db_path, repo, issue_data)

    return get_cached_issue(db_path, repo, issue_number)


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 src/fetch_issue.py <owner> <repo> <issue_number>")
        print("Example: python3 src/fetch_issue.py encode httpx 572")
        sys.exit(1)

    owner, repo, issue_number = sys.argv[1], sys.argv[2], int(sys.argv[3])
    issue = get_issue(owner, repo, issue_number)

    print("\n" + "=" * 70)
    print(f"Issue #{issue['issue_number']}: {issue['title']}")
    print("=" * 70)
    print(f"Author: {issue['author']}  |  State: {issue['state']}")
    print(f"\nBody:\n{(issue['body'] or '(no description)')[:500]}")

    comments = json.loads(issue["comments_json"])
    print(f"\n{len(comments)} comment(s)")
    for c in comments[:3]:
        author = c["author"]["login"] if c["author"] else "?"
        print(f"  [comment by {author}] {c['body'][:100]}")


if __name__ == "__main__":
    main()

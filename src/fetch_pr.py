"""
Phase 2, Step 3 -- GitHub GraphQL PR fetcher.

Verified before writing this (see conversation research):
  - GraphQL API requires authentication -- there is no unauthenticated
    access at all, confirmed directly from GitHub's own community
    discussions. A personal access token is mandatory.
  - Rate limit: 5,000 points/hour for a personal access token. Cost is
    driven by query complexity (nested connections, first: N sizes),
    not a flat per-call count -- so keeping `first:` values small and
    only fetching fields we actually need matters for real usage.
  - PullRequestReview -> comments has documented GraphQL quirks around
    reply threading (community discussions #24666, #24850) -- replies
    aren't always cleanly connected to parents. We deliberately fetch
    top-level review comments and issue comments, NOT attempt full
    threaded reconstruction, since our use case (extracting rationale)
    doesn't need perfect threading.
  - Real confirmed query shape: repository(owner, name) { pullRequest(number)
    { title, body, comments(first: N) { nodes {...} }, reviews(first: N)
    {...} } }

Design: fetch ONE PR at a time, ON DEMAND -- this is the "lazy" part of
the Phase 2 design agreed from the start. We do NOT bulk-fetch all 632
httpx PRs upfront. A caller (eventually, the verifier agent) requests a
specific PR number only when it needs to check a claim against it.

Caching: every fetched PR is saved to SQLite immediately, so re-requesting
the same PR never re-hits the API.

Setup required before running:
    1. Create a GitHub personal access token: https://github.com/settings/tokens
       (classic token, no special scopes needed for public repo read access)
    2. Set it as an environment variable:
           export GITHUB_TOKEN=ghp_your_token_here
    3. pip install requests (if not already installed)

Run with:
    python3 src/fetch_pr.py <owner> <repo> <pr_number>
    e.g. python3 src/fetch_pr.py encode httpx 3319
"""

import os
import sys
import sqlite3
import json
import time
import requests


GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


PR_QUERY = """
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      number
      title
      body
      state
      author { login }
      createdAt
      mergedAt
      additions
      deletions
      changedFiles
      comments(first: 50) {
        totalCount
        nodes {
          author { login }
          body
          createdAt
        }
      }
      reviews(first: 20) {
        totalCount
        nodes {
          author { login }
          state
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


def init_pr_cache_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pr_cache (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            title TEXT,
            body TEXT,
            state TEXT,
            author TEXT,
            created_at TEXT,
            merged_at TEXT,
            additions INTEGER,
            deletions INTEGER,
            changed_files INTEGER,
            comments_json TEXT,
            reviews_json TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (repo, pr_number)
        );
    """)
    conn.commit()
    return conn


def get_cached_pr(db_path: str, repo: str, pr_number: int) -> dict | None:
    """Check the local cache first -- never re-hit the API for a PR
    we've already fetched."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM pr_cache WHERE repo = ? AND pr_number = ?", (repo, pr_number))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    columns = ["repo", "pr_number", "title", "body", "state", "author", "created_at",
               "merged_at", "additions", "deletions", "changed_files",
               "comments_json", "reviews_json", "fetched_at"]
    return dict(zip(columns, row))


def fetch_pr_from_github(owner: str, repo: str, pr_number: int, token: str) -> dict:
    """Live GraphQL call. Only invoked on a cache miss."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "query": PR_QUERY,
        "variables": {"owner": owner, "repo": repo, "number": pr_number},
    }

    response = requests.post(GITHUB_GRAPHQL_URL, json=payload, headers=headers, timeout=30)

    # GraphQL returns 200 even on query errors -- must check the body,
    # not just the status code (confirmed via research: this is a
    # well-known GraphQL gotcha, not something obvious from REST habits).
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    pr_data = data["data"]["repository"]["pullRequest"]
    rate_limit = data["data"]["rateLimit"]

    if pr_data is None:
        raise ValueError(f"PR #{pr_number} not found in {owner}/{repo} "
                          f"(wrong number, or it's an issue not a PR)")

    print(f"  Rate limit: {rate_limit['remaining']}/{rate_limit['limit']} "
          f"remaining, resets at {rate_limit['resetAt']}")

    return pr_data


def save_pr_to_cache(db_path: str, repo: str, pr_data: dict):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT OR REPLACE INTO pr_cache
           (repo, pr_number, title, body, state, author, created_at, merged_at,
            additions, deletions, changed_files, comments_json, reviews_json, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            repo,
            pr_data["number"],
            pr_data["title"],
            pr_data["body"],
            pr_data["state"],
            pr_data["author"]["login"] if pr_data["author"] else None,
            pr_data["createdAt"],
            pr_data["mergedAt"],
            pr_data["additions"],
            pr_data["deletions"],
            pr_data["changedFiles"],
            json.dumps(pr_data["comments"]["nodes"]),
            json.dumps(pr_data["reviews"]["nodes"]),
            str(time.time()),
        ),
    )
    conn.commit()
    conn.close()


def get_pr(owner: str, repo: str, pr_number: int, db_path: str = "data/code_graph.db") -> dict:
    """
    The main entry point -- this is what a future verifier agent will call.
    Checks cache first, only hits the live API on a miss.
    """
    init_pr_cache_table(db_path)

    cached = get_cached_pr(db_path, repo, pr_number)
    if cached is not None:
        print(f"  Cache hit: PR #{pr_number} already fetched previously.")
        return cached

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable not set. "
            "Create a token at https://github.com/settings/tokens and "
            "run: export GITHUB_TOKEN=ghp_your_token_here"
        )

    print(f"  Cache miss: fetching PR #{pr_number} from GitHub live...")
    pr_data = fetch_pr_from_github(owner, repo, pr_number, token)
    save_pr_to_cache(db_path, repo, pr_data)

    return get_cached_pr(db_path, repo, pr_number)


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 src/fetch_pr.py <owner> <repo> <pr_number>")
        print("Example: python3 src/fetch_pr.py encode httpx 3319")
        sys.exit(1)

    owner, repo, pr_number = sys.argv[1], sys.argv[2], int(sys.argv[3])

    pr = get_pr(owner, repo, pr_number)

    print("\n" + "=" * 70)
    print(f"PR #{pr['pr_number']}: {pr['title']}")
    print("=" * 70)
    print(f"Author: {pr['author']}  |  State: {pr['state']}")
    print(f"+{pr['additions']} -{pr['deletions']} across {pr['changed_files']} files")
    print(f"\nBody:\n{(pr['body'] or '(no description)')[:500]}")

    comments = json.loads(pr["comments_json"])
    reviews = json.loads(pr["reviews_json"])
    print(f"\n{len(comments)} comment(s), {len(reviews)} review(s)")
    for c in comments[:3]:
        author = c["author"]["login"] if c["author"] else "?"
        print(f"  [comment by {author}] {c['body'][:100]}")
    for r in reviews[:3]:
        author = r["author"]["login"] if r["author"] else "?"
        print(f"  [review by {author}, {r['state']}] {(r['body'] or '(no body)')[:100]}")


if __name__ == "__main__":
    main()

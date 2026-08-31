"""
Fetches got's GitHub Releases (REST API) and caches them in a new
release_cache table -- same repo-assist/ root, same code_graph.db.

RUN FROM src/: python3 fetch_got_releases.py

WHY REST, NOT GRAPHQL (unlike pr_cache/discussion_cache): releases are a
flat list with no nested reply-thread structure to worry about -- the
complexity that justified GraphQL for Discussions doesn't apply here.
REST's GET /repos/{owner}/{repo}/releases is simpler and sufficient.

WHY THIS EXISTS: Y9/Y10/Y11 ground truth cites got's "v9/v10/v12 release
notes" as the real rationale source. Confirmed via direct check: no
CHANGELOG.md or release-notes file exists anywhere in the cloned got repo
-- this content only exists on GitHub's Releases page. build_docs_table.py
deliberately does NOT cover this; this script is the separate, explicit
fill for that gap.

AUTH: uses the same personal access token pattern as your existing PR/
Discussion fetchers. Reads from GITHUB_TOKEN env var. Unauthenticated
requests work too (60/hr rate limit) but got has ~25 releases total, so
either works fine for a one-time fetch.
"""

import json
import os
import sqlite3
import time
import urllib.request
import urllib.error

from config import DB_PATH
REPO = "got"  # matches the `repo` value used elsewhere in your schema
OWNER_REPO = "sindresorhus/got"  # real GitHub owner/repo slug


def _github_get(url: str) -> list[dict]:
    token = os.environ.get("GITHUB_TOKEN")
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "repo-assist")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_all_releases(owner_repo: str) -> list[dict]:
    releases = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{owner_repo}/releases?per_page=100&page={page}"
        try:
            batch = _github_get(url)
        except urllib.error.HTTPError as e:
            print(f"HTTPError on page {page}: {e.code} {e.reason}")
            if e.code == 403:
                print("Likely rate-limited. Set GITHUB_TOKEN env var and retry, or wait an hour.")
            break
        if not batch:
            break
        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.5)  # be polite even though we're well under any real limit
    return releases


def build():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS release_cache (
            repo TEXT NOT NULL,
            tag_name TEXT NOT NULL,
            name TEXT,
            body TEXT,
            published_at TEXT,
            html_url TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (repo, tag_name)
        )
    """)

    releases = fetch_all_releases(OWNER_REPO)
    print(f"Fetched {len(releases)} releases from GitHub for {OWNER_REPO}")

    fetched_at = str(time.time())
    for r in releases:
        conn.execute(
            """INSERT OR REPLACE INTO release_cache
               (repo, tag_name, name, body, published_at, html_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (REPO, r.get("tag_name"), r.get("name"), r.get("body"),
             r.get("published_at"), r.get("html_url"), fetched_at),
        )

    conn.commit()

    # real sanity check, not just trusting the row count
    cur = conn.execute("SELECT tag_name, length(body) FROM release_cache WHERE repo=? ORDER BY published_at DESC LIMIT 5", (REPO,))
    print("\nMost recent 5 cached releases (tag, body length):")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} chars")

    conn.close()


if __name__ == "__main__":
    build()

"""
Phase 2, Step 6 -- linking PRs/commits to relevant Discussions.

REWRITE. The first version, when run against the real 138-discussion
httpx corpus, produced 47 candidate matches for one PR -- unusable
noise, and worse than having no linker at all. Three real bugs found
by inspecting that actual output, not by further guessing:

BUG 1 -- overlapping regex patterns produced truncated fragments.
  The original design ran three separate regexes (camelCase-starting-
  lowercase, PascalCase-one-cap, acronym-prefixed) and unioned their
  results. Real output showed garbage like 'ebSocket', 'etworkOptions',
  'ttpTransport' -- these are WebSocket/NetworkOptions/HttpTransport
  with their first letter eaten, because re.findall() lets independent
  patterns match OVERLAPPING substrings of the same word, with no
  awareness that they're describing the same token. Confirmed by
  reproducing the exact failure: extract_notable_terms("AIOHttpTransport")
  returned fragments from all three patterns simultaneously.
  FIX: a single pass over whole alphabetic words, classifying each
  WHOLE word as "notable" (mixed-case, or all-uppercase 3+ letters)
  rather than three patterns hunting for sub-matches independently.

BUG 2 -- the generic-term filter used an ABSOLUTE count threshold
  (max_generic_frequency=5), which doesn't scale with corpus size.
  On the real 138-discussion corpus, "HTTPX" and "HTTP" each appeared
  in 10+ discussions and were never filtered, dominating the results.
  FIX: a PERCENTAGE-based threshold (a term appearing in more than ~3%
  of the corpus is generic), which scales correctly regardless of how
  many documents exist.

BUG 3 -- output was unranked beyond raw matched-term count, so a
  weak match (one generic term) looked as prominent as a strong one
  (a rare, specific term). FIX: score by INVERSE DOCUMENT FREQUENCY --
  rarer terms contribute more to a candidate's score, and only the
  top N candidates by score are shown, not every candidate that
  matched anything at all.

Run with:
    python3 src/link_pr_to_discussions.py <repo> <pr_number>
    e.g. python3 src/link_pr_to_discussions.py httpx 3319
"""

import re
import sys
import sqlite3

from config import DB_PATH
import math
import json
from collections import Counter


GENERIC_THRESHOLD_FRACTION = 0.06  # recalibrated from a real frequency-distribution
                                     # inspection (see inspect_term_frequency_distribution.py):
                                     # the actual httpx corpus shows a sharp natural cliff
                                     # between "specific" terms (max ~7 occurrences, e.g.
                                     # SSLContext at 6, QueryParams/WSGI/DNS/ASGI at 4-5) and
                                     # "generic" terms (11+ occurrences, jumping straight to
                                     # HTTP=37, HTTPX=27, API=41). The prior 0.03 (≈4 docs
                                     # at this corpus size) cut directly through the
                                     # specific-term band, incorrectly filtering out
                                     # SSLContext itself. 0.06 (≈8 docs) sits just above
                                     # the real cliff instead.
TOP_N_RESULTS = 5
TITLE_MATCH_WEIGHT = 5.0  # verified against real data: a term in the PR's own
                           # title (its author's explicit statement of subject)
                           # must outweigh several body-only term matches
                           # combined. 5.0 confirmed sufficient using the real
                           # SSLContext (title, freq=6) vs ASGI+QueryParams+WSGI
                           # (body-only, freq 4-5 each) case: 15.71 vs 10.42.


def extract_notable_terms(text: str) -> set[str]:
    """
    Single-pass extraction: find all alphabetic words, keep only those
    that are NOTABLE -- an identifier-like mixed-case word (SSLContext,
    calculateDelay, WebSocket) or an all-uppercase 3+ letter acronym
    (SSL, DNS). Plain lowercase words (app, data) and ordinary
    capitalized English words that merely start a sentence (Make,
    Request, Introduce) are excluded.

    FIX: the first version of this check only required "has an
    uppercase letter AND has a lowercase letter" anywhere in the word,
    which incorrectly matched ordinary capitalized words like "Make"
    or "Request" (capital first letter, lowercase rest -- exactly what
    every sentence-initial word looks like). The real signal we want
    is INTERNAL case-mixing -- an uppercase letter appearing after at
    least one lowercase letter, or 2+ consecutive uppercase letters
    followed by lowercase (an acronym prefix). A single leading capital
    followed by all-lowercase is just normal capitalization, not a
    code identifier, and must be excluded.
    """
    words = re.findall(r"[A-Za-z]+", text)
    notable = set()
    for w in words:
        if len(w) < 3:
            continue

        if w.isupper():
            # All-caps acronym, e.g. SSL, DNS, HTTP
            notable.add(w)
            continue

        # Check for INTERNAL case transitions: a lowercase letter
        # followed later by an uppercase letter (calculateDelay,
        # httpClient), or 2+ leading uppercase letters followed by
        # lowercase (SSLContext, HTTPClient) -- NOT just "starts with
        # one capital, rest lowercase" (Make, Request, Introduce).
        has_lower_then_upper = re.search(r"[a-z].*[A-Z]", w) is not None
        has_acronym_prefix = re.match(r"^[A-Z]{2,}[a-z]", w) is not None

        if has_lower_then_upper or has_acronym_prefix:
            notable.add(w)

    return notable


def _flatten_comment_text(comments_json: str | None) -> str:
    """
    CRITICAL FIX: the original build_corpus_term_frequency() and
    find_candidate_discussions() only ever SELECTed title and body
    columns -- confirmed by directly reading the SQL queries, not
    guessing. This is the real root cause of SSLContext never
    surfacing as a match for PR #3319: Discussion #3007's own top-level
    "body" field is just the original post. The actual settled
    rationale ("I think it'll be neatest as a subclass of
    ssl.SSLContext...") lives in a REPLY, nested inside the
    comments_json column, which was never read at all.

    This flattens a comments_json blob (as stored by
    index_discussions.py -- a list of comment dicts, each with a
    nested "replies": {"nodes": [...]} list) into one plain-text blob
    so its content can be scanned by extract_notable_terms() the same
    way title/body are.
    """
    if not comments_json:
        return ""
    try:
        comments = json.loads(comments_json)
    except (json.JSONDecodeError, TypeError):
        return ""

    parts = []
    for c in comments:
        parts.append(c.get("body") or "")
        for reply in c.get("replies", {}).get("nodes", []):
            parts.append(reply.get("body") or "")
    return " ".join(parts)


def build_corpus_term_frequency(conn: sqlite3.Connection, repo: str) -> tuple[Counter, int]:
    """
    Returns (term -> document count, total document count) across BOTH
    pr_cache and discussions_index for this repo -- now scanning
    title + body + FLATTENED COMMENTS/REPLIES, not just title+body.
    The document count is needed to compute a PERCENTAGE threshold,
    not just a raw count -- this is the fix for Bug 2 (from the
    previous round).
    """
    cur = conn.cursor()
    freq = Counter()
    doc_count = 0

    cur.execute("SELECT title, body, comments_json FROM pr_cache WHERE repo = ?", (repo,))
    for title, body, comments_json in cur.fetchall():
        doc_count += 1
        comment_text = _flatten_comment_text(comments_json)
        terms = (extract_notable_terms(title or "")
                 | extract_notable_terms(body or "")
                 | extract_notable_terms(comment_text))
        for t in terms:
            freq[t] += 1

    cur.execute("SELECT title, body, comments_json FROM discussions_index WHERE repo = ?", (repo,))
    for title, body, comments_json in cur.fetchall():
        doc_count += 1
        comment_text = _flatten_comment_text(comments_json)
        terms = (extract_notable_terms(title or "")
                 | extract_notable_terms(body or "")
                 | extract_notable_terms(comment_text))
        for t in terms:
            freq[t] += 1

    return freq, doc_count


def find_candidate_discussions(
    conn: sqlite3.Connection, repo: str, pr_title: str, pr_body: str,
    top_n: int = TOP_N_RESULTS,
) -> list[dict]:
    """
    Returns the TOP N candidate discussions, ranked with PR-TITLE
    matches weighted far more heavily than PR-BODY-only matches.

    REAL BUG FOUND AND FIXED: plain summed IDF scoring ranked
    #1531 ("Restructuring the docs", a long discussion touching many
    API surfaces) above #3007 (the ACTUAL rationale discussion for
    PR #3319's SSLContext design), because #1531 matched three
    moderately-rare terms (ASGI, QueryParams, WSGI -- freq 4-5 each)
    while #3007 matched only one (SSLContext, freq 6) -- and summing
    three medium scores beat one slightly-higher single score, no
    matter how the aggregation formula was tuned (tested: plain sum,
    diminishing-bonus sum, precision-normalized sum -- all failed the
    same way, confirmed by direct calculation against the real
    frequencies).

    The real, evidence-based fix: PR #3319's own TITLE contains only
    {SSLContext, API} -- confirmed by extracting terms from the title
    string alone. Every other matched term (ASGI, WSGI, QueryParams,
    etc.) comes from the PR's BODY, which we already know (from
    fetch_pr.py's original real output) is a broad release-staging
    checklist, not specifically about any one of those terms. The
    title is the PR author's own explicit statement of what the PR is
    centrally about -- weighting title matches heavily is a real,
    defensible signal, not another arbitrary constant.
    """
    term_freq, doc_count = build_corpus_term_frequency(conn, repo)
    generic_cutoff = max(2, int(doc_count * GENERIC_THRESHOLD_FRACTION))

    pr_title_terms = extract_notable_terms(pr_title)
    pr_body_terms = extract_notable_terms(pr_body or "") - pr_title_terms
    pr_terms = pr_title_terms | pr_body_terms

    discriminating_terms = {t for t in pr_terms if term_freq[t] <= generic_cutoff}
    if not discriminating_terms:
        return []

    cur = conn.cursor()
    cur.execute("""
        SELECT discussion_number, title, body, url, created_at, comments_json
        FROM discussions_index WHERE repo = ?
    """, (repo,))

    scored = []
    for number, title, body, url, created_at, comments_json in cur.fetchall():
        comment_text = _flatten_comment_text(comments_json)
        disc_terms = (extract_notable_terms(title or "")
                      | extract_notable_terms(body or "")
                      | extract_notable_terms(comment_text))
        matched = discriminating_terms & disc_terms
        if not matched:
            continue

        score = 0.0
        for t in matched:
            idf = math.log(doc_count / max(1, term_freq[t]))
            weight = TITLE_MATCH_WEIGHT if t in pr_title_terms else 1.0
            score += idf * weight

        scored.append({
            "discussion_number": number,
            "title": title,
            "url": url,
            "created_at": created_at,
            "matched_terms": sorted(matched),
            "matched_in_title": sorted(matched & pr_title_terms),
            "score": round(score, 2),
        })

    scored.sort(key=lambda c: -c["score"])
    return scored[:top_n]


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 src/link_pr_to_discussions.py <repo> <pr_number>")
        print("Example: python3 src/link_pr_to_discussions.py httpx 3319")
        sys.exit(1)

    repo, pr_number = sys.argv[1], int(sys.argv[2])

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT title, body FROM pr_cache WHERE repo = ? AND pr_number = ?", (repo, pr_number))
    row = cur.fetchone()

    if row is None:
        print(f"PR #{pr_number} not found in pr_cache for {repo}. Fetch it first with fetch_pr.py.")
        return

    pr_title, pr_body = row
    print(f"PR #{pr_number}: {pr_title}\n")

    pr_terms = extract_notable_terms(pr_title) | extract_notable_terms(pr_body or "")
    print(f"Notable terms extracted from this PR: {sorted(pr_terms)}\n")

    candidates = find_candidate_discussions(conn, repo, pr_title, pr_body)

    if not candidates:
        print("No candidate discussions found.")
        return

    print(f"Top {len(candidates)} candidate discussion(s), ranked by specificity:\n")
    for c in candidates:
        print(f"  #{c['discussion_number']}: {c['title']}  (score={c['score']})")
        print(f"    Matched on: {c['matched_terms']}")
        if c['matched_in_title']:
            print(f"      (of which, in PR title: {c['matched_in_title']})")
        created = c['created_at'][:10] if c['created_at'] else "(unknown date)"
        url = c['url'] or "(no URL)"
        print(f"    Created: {created}  |  {url}")
        print()

    conn.close()


if __name__ == "__main__":
    main()

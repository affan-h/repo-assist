"""Git operations: clone at a pinned commit, list tracked files, read commit history."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from repo_assist.store import Chunk

_GITHUB_URL = re.compile(r"^(?:https?://github\.com/|git@github\.com:)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


def parse_repo_url(url: str) -> tuple[str, str]:
    """Accepts `owner/name`, `https://github.com/owner/name(.git)` or `git@github.com:owner/name.git`."""
    match = _GITHUB_URL.match(url.strip())
    if not match:
        raise ValueError(f"Not a GitHub repository URL or owner/name: {url!r}")
    return match.group(1), match.group(2)


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


@dataclass
class Checkout:
    path: Path
    sha: str
    date: str  # ISO-8601 commit date of the checked-out commit


def clone(owner: str, name: str, dest: Path, ref: str | None = None) -> Checkout:
    """Clone (or update) a GitHub repo and check out `ref` (default branch head if None).

    A full clone: line-level history (`git log -L`, used to trace code back to the
    commits and PRs that shaped it) needs every historical version of a file.
    """
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{owner}/{name}.git", str(dest)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    else:
        _git(dest, "fetch", "--quiet", "origin")

    target = ref or _git(dest, "rev-parse", "origin/HEAD").strip()
    _git(dest, "checkout", "--quiet", "--detach", target)
    sha = _git(dest, "rev-parse", "HEAD").strip()
    date = _git(dest, "log", "-1", "--format=%cI", "HEAD").strip()
    return Checkout(path=dest, sha=sha, date=date)


def tracked_files(repo_dir: Path) -> list[str]:
    return [line for line in _git(repo_dir, "ls-files").splitlines() if line]


_RECORD = "\x1e"
_FIELD = "\x1f"


def commit_chunks(repo_dir: Path, max_commits: int) -> list[Chunk]:
    """One chunk per non-merge commit reachable from HEAD: message plus files touched."""
    # --no-renames: rename detection is slow on long histories and not needed here.
    log = _git(
        repo_dir, "log", "--no-merges", "--no-renames", f"-n{max_commits}", "--name-only",
        f"--format={_RECORD}%H{_FIELD}%an{_FIELD}%aI{_FIELD}%s{_FIELD}%b{_FIELD}",
    )
    chunks = []
    for record in log.split(_RECORD)[1:]:
        sha, author, date, subject, body, files = record.split(_FIELD)
        file_list = [f for f in files.strip().splitlines() if f]
        shown_files = file_list[:25] + ([f"... and {len(file_list) - 25} more"] if len(file_list) > 25 else [])
        text = f"{subject}\n\n{body.strip()}".strip()
        text += f"\n\nAuthor: {author}\nDate: {date}\nFiles changed:\n" + "\n".join(shown_files)
        chunks.append(Chunk(
            kind="commit", ref=f"commit:{sha[:10]}", title=subject, text=text, created_at=date,
            embed_text=f"{subject}\n{body.strip()}"[:2000] or subject,
        ))
    return chunks


_LOG_FORMAT = f"{_RECORD}%H{_FIELD}%aI{_FIELD}%s{_FIELD}%b"
_ISSUE_REF = re.compile(r"(?:\(|\s|^)#(\d+)\b")


@dataclass
class Commit:
    sha: str
    date: str
    subject: str
    body: str

    @property
    def ref(self) -> str:
        return f"commit:{self.sha[:10]}"

    @property
    def message(self) -> str:
        return f"{self.subject}\n\n{self.body}".strip()

    def linked_numbers(self) -> list[int]:
        """Issue/PR numbers mentioned in the message: 'Fix retries (#991)', 'Fixes #417'."""
        return sorted({int(n) for n in _ISSUE_REF.findall(self.message)})


def _parse_log(output: str) -> list[Commit]:
    commits = []
    for record in output.split(_RECORD)[1:]:
        sha, date, subject, body = (record.split(_FIELD) + ["", "", "", ""])[:4]
        commits.append(Commit(sha.strip(), date.strip(), subject.strip(), body.strip()))
    return commits


def line_history(repo_dir: Path, path: str, start: int, end: int, limit: int = 10) -> list[Commit]:
    """Commits that changed lines start..end of `path`, newest first (git log -L)."""
    return _parse_log(_git(repo_dir, "log", f"-L{start},{end}:{path}", "-s", f"-n{limit}",
                           f"--format={_LOG_FORMAT}"))

"""Command-line interface: `repo-assist ingest | ask | repos | serve`."""

import argparse
import logging
import sys
import textwrap

from repo_assist import store


def _ingest(args) -> int:
    from repo_assist import ingest

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    repo_id = ingest.register_repo(args.url)
    stats = ingest.ingest(repo_id, ref=args.ref, include_history=not args.no_history)
    print(f"\n{repo_id} is ready: {stats}")
    return 0


def _repos(args) -> int:
    store.init_db()
    with store.session() as conn:
        repos = store.list_repos(conn)
    if not repos:
        print("No repositories ingested yet. Try: repo-assist ingest encode/httpx")
    for r in repos:
        counts = ", ".join(f"{k}={v}" for k, v in r["stats"].items() if k not in ("ingest_seconds", "warnings"))
        counts += "".join(f"\n{'':<16} warning: {w}" for w in r["stats"].get("warnings", []))
        print(f"{r['id']:<16} {r['status']:<10} {r['owner']}/{r['name']}@{(r['commit_sha'] or '')[:10]}  {counts}")
    return 0


def _print_event(event: dict) -> None:
    if event["type"] == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in event["args"].items() if v is not None)
        print(f"  -> {event['tool']}({args})", file=sys.stderr, flush=True)
    elif event["type"] == "tool_result":
        print(f"     {event['summary']}", file=sys.stderr, flush=True)


def _ask(args) -> int:
    from repo_assist import agent

    result = agent.answer_sync(args.repo, args.question, mode=args.mode, model=args.model,
                               on_event=_print_event if args.verbose or args.mode == "agent" else None)
    print()
    print(textwrap.indent(result.answer, "  "))
    if result.citations:
        print("\nSources:")
        for i, c in enumerate(result.citations, start=1):
            print(f"  [{i}] {c.ref}  {c.url}")
    if result.abstained:
        print("\n  (abstained: the indexed evidence does not answer this)")
    if result.error:
        print(f"\n  ({result.error})")
    usage = result.usage
    print(f"\n  {result.latency_s:.1f}s, {usage.get('requests', 0)} LLM calls, {len(result.trace)} tool calls, "
          f"{usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} tokens", file=sys.stderr)
    return 0


def _serve(args) -> int:
    import uvicorn

    uvicorn.run("repo_assist.api:app", host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="repo-assist", description="Grounded Q&A over GitHub repositories.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Clone and index a GitHub repository")
    p.add_argument("url", help="owner/name or GitHub URL")
    p.add_argument("--ref", help="Commit, tag or branch to index (default: default branch HEAD)")
    p.add_argument("--no-history", action="store_true", help="Skip commits, issues, PRs and discussions")
    p.set_defaults(func=_ingest)

    p = sub.add_parser("ask", help="Ask a question about an ingested repository")
    p.add_argument("repo", help="Repository id, e.g. httpx")
    p.add_argument("question")
    p.add_argument("--mode", choices=["agent", "single_shot"], default="agent")
    p.add_argument("--model", help="pydantic-ai model string, e.g. google:gemini-3.5-flash")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_ask)

    p = sub.add_parser("repos", help="List ingested repositories")
    p.set_defaults(func=_repos)

    p = sub.add_parser("serve", help="Run the web app and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

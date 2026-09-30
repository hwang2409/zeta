"""Terminal commands for reviewing drafts and operating the daemon."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..core.session import env_home
from .authoring import import_jobs, listing, show
from .commands import review_job
from .daemon import serve
from .store import SQLiteStore
from .trigger import Webhook
from .webhook import DEFAULT_WEBHOOK_PORT


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "automation", help="draft, approve, and run automations"
    )
    verbs = parser.add_subparsers(dest="automation_verb", required=True)
    verbs.add_parser("list", help="list drafts and approved automations")
    for verb in ("show", "approve", "disable"):
        subparser = verbs.add_parser(verb)
        subparser.add_argument("name")
    importer = verbs.add_parser("import", help="import JSON entries as inert drafts")
    importer.add_argument("path", type=Path)
    daemon = verbs.add_parser("daemon", help="run the single-worker foreground daemon")
    daemon.add_argument("--webhook-host", default="127.0.0.1")
    daemon.add_argument("--webhook-port", type=int, default=DEFAULT_WEBHOOK_PORT)
    daemon.add_argument("--allow-non-loopback", action="store_true")
    webhook = verbs.add_parser("webhook", help="inspect or rotate webhook credentials")
    webhook_verbs = webhook.add_subparsers(dest="webhook_verb", required=True)
    for verb in ("url", "show-secret", "rotate-secret", "rotate-url"):
        operation = webhook_verbs.add_parser(verb)
        operation.add_argument("name")


def run(args: argparse.Namespace) -> int:
    home = env_home()
    try:
        if args.automation_verb == "daemon":
            asyncio.run(
                serve(
                    home,
                    webhook_host=args.webhook_host,
                    webhook_port=args.webhook_port,
                    allow_non_loopback=args.allow_non_loopback,
                    on_ready=lambda host, port: print(
                        f"webhook receiver listening on {host}:{port}",
                        file=sys.stderr,
                        flush=True,
                    ),
                )
            )
            return 0
        with SQLiteStore(home) as store:
            if args.automation_verb == "list":
                print(listing(store))
            elif args.automation_verb == "webhook":
                state = store.get(args.name)
                if not isinstance(state.job.trigger, Webhook):
                    raise ValueError(f"automation is not a webhook: {args.name}")
                if not state.enabled:
                    raise ValueError(f"webhook automation is disabled: {args.name}")
                credentials = store.webhook_credentials(args.name)
                if args.webhook_verb == "url":
                    print(
                        f"http://127.0.0.1:{DEFAULT_WEBHOOK_PORT}/hooks/{credentials.token}\n"
                        "Loopback URL only; public exposure is planned for phase 2."
                    )
                elif args.webhook_verb == "show-secret":
                    print(credentials.secret.hex())
                elif args.webhook_verb == "rotate-secret":
                    store.rotate_webhook_secret(args.name)
                    print(f"{args.name} webhook secret rotated.")
                elif args.webhook_verb == "rotate-url":
                    store.rotate_webhook_url(args.name)
                    print(f"{args.name} webhook URL rotated.")
            elif args.automation_verb == "show":
                print(show(store, args.name))
            elif args.automation_verb == "import":
                print(import_jobs(store, args.path, cwd=str(Path.cwd()), home=home))
            elif args.automation_verb == "disable":
                store.disable(args.name)
                print(f"{args.name} disabled.")
            elif args.automation_verb == "approve":
                review = asyncio.run(review_job(store, args.name, home))
                print(review.text)
                if not sys.stdin.isatty():
                    raise ValueError(
                        "approval requires an interactive terminal; use /automations approve in the TUI"
                    )
                if (
                    input(
                        "Type the approval token to arm this exact revision: "
                    ).strip()
                    != review.token
                ):
                    raise ValueError("approval token did not match; job remains inert")
                store.approve(
                    review.name, review.revision, review.recipient, datetime.now(UTC)
                )
                print(f"{review.name} r{review.revision} armed for {review.recipient}.")
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError) as exc:
        print(f"zeta automation: {exc}", file=sys.stderr)
        return 1

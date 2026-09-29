"""Human-only approval flow, shared by CLI and slash commands."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..skills import discover_session_skills
from ..tools import ToolRegistry
from .authoring import import_jobs, listing, show
from .delivery import SlackDelivery
from .services import mount_services
from .store import SQLiteStore
from .trigger import Webhook
from .webhook import DEFAULT_WEBHOOK_PORT


@dataclass(frozen=True)
class Review:
    name: str
    revision: int
    recipient: str
    token: str
    text: str


async def review_job(store: SQLiteStore, name: str, home: Path) -> Review:
    state = store.get(name)
    policy = ApprovalPolicy(default=ApprovalDecision.DENY, always_allow=state.job.allow)
    registry = ToolRegistry(
        state.job.cwd,
        approval_policy=policy,
        enforce_approvals=True,
        skill_catalog=discover_session_skills(home=home),
    )
    mount = None
    try:
        mount = await mount_services(state.job, registry, home)
        recipient = await SlackDelivery(mount).resolve(state.job.deliver)
        if store.get(name).revision != state.revision:
            raise ValueError("draft changed during review; inspect the new revision")
        payload = json.dumps(
            [name, state.job.document(), state.revision, recipient], sort_keys=True
        )
        token = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return Review(
            name,
            state.revision,
            recipient,
            token,
            show(store, name)
            + f"\nResolved recipient: {recipient}\nApproval token: {token}",
        )
    finally:
        if mount is not None:
            await mount.close()
        await registry.close()


async def slash(args: str, *, home: Path, cwd: str) -> str:
    parts = shlex.split(args)
    with SQLiteStore(home) as store:
        if not parts:
            return listing(store)
        if parts[0] == "approve" and len(parts) in {2, 3}:
            review = await review_job(store, parts[1], home)
            if len(parts) == 2:
                return (
                    review.text
                    + f"\nTo arm this exact job: /automations approve {review.name} {review.token}"
                )
            if parts[2] != review.token:
                raise ValueError(
                    "review changed or token is invalid; inspect the job again"
                )
            store.approve(
                review.name, review.revision, review.recipient, datetime.now(UTC)
            )
            return f"{review.name} r{review.revision} armed for {review.recipient}."
        if parts[0] == "disable" and len(parts) == 2:
            store.disable(parts[1])
            return f"{parts[1]} disabled."
        if parts[0] == "import" and len(parts) == 2:
            return import_jobs(store, Path(parts[1]), cwd=cwd, home=home)
        if len(parts) == 3 and parts[:2] == ["webhook", "url"]:
            state = store.get(parts[2])
            if not isinstance(state.job.trigger, Webhook):
                raise ValueError(f"automation is not a webhook: {parts[2]}")
            token = store.webhook_credentials(parts[2]).token
            return (
                f"http://127.0.0.1:{DEFAULT_WEBHOOK_PORT}/hooks/{token}\n"
                "Loopback URL only; public exposure is planned for phase 2."
            )
        if len(parts) == 3 and parts[:2] == ["webhook", "show-secret"]:
            return store.webhook_credentials(parts[2]).secret.hex()
        if len(parts) == 3 and parts[:2] == ["webhook", "rotate-secret"]:
            store.rotate_webhook_secret(parts[2])
            return f"{parts[2]} webhook secret rotated."
        if len(parts) == 3 and parts[:2] == ["webhook", "rotate-url"]:
            store.rotate_webhook_url(parts[2])
            return f"{parts[2]} webhook URL rotated."
        if parts[0] == "show" and len(parts) == 2:
            return show(store, parts[1])
        if len(parts) == 1:
            return show(store, parts[0])
        raise ValueError(
            "usage: /automations [<name> | approve <name> [token] | disable <name> | "
            "import <file> | webhook <url|show-secret|rotate-secret|rotate-url> <name>]"
        )

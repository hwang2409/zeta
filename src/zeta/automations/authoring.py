"""Draft-only authoring, import, and human-readable inspection."""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfoNotFoundError

from ..core.session import env_home
from ..providers.factory import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL
from ..settings import load_settings
from .models import Job, parse_job
from .store import SQLiteStore


def resolve_job(
    name: str,
    document: object,
    *,
    cwd: str,
    provider: str | None = None,
    model: str | None = None,
    home: Path | None = None,
) -> Job:
    if not isinstance(document, dict):
        raise TypeError("job must be an object")
    settings = load_settings(home=home or env_home()).settings
    value = dict(document)
    selected_provider = value.setdefault(
        "provider", provider or settings.provider or "claude"
    )
    selected_model = model if provider == selected_provider else None
    value.setdefault(
        "model",
        selected_model
        or settings.model
        or {
            "claude": DEFAULT_CLAUDE_MODEL,
            "codex": DEFAULT_CODEX_MODEL,
            "fake": "fake",
        }.get(selected_provider, ""),
    )
    value.setdefault("cwd", str(Path(cwd).expanduser().resolve()))
    return parse_job(name, value)


def import_jobs(store: SQLiteStore, path: Path, *, cwd: str, home: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("automation file must map job names to job objects")
    results = []
    for name, value in document.items():
        try:
            job = resolve_job(name, value, cwd=cwd, home=home)
            state = store.draft(job, source=str(path.resolve()))
            results.append(f"{name} r{state.revision}: draft (not armed)")
        except (ValueError, TypeError, KeyError, ZoneInfoNotFoundError) as exc:
            store.record_error(name, str(exc))
            results.append(f"{name}: invalid: {exc}")
    return "\n".join(results) or "No automations found."


def show(store: SQLiteStore, name: str) -> str:
    state = store.get(name)
    lines = [
        f"{name} r{state.revision}: {'armed' if state.enabled else 'draft/disabled'}",
        json.dumps(state.job.document(), indent=2, ensure_ascii=False),
        f"Approved recipient: {state.recipient or 'none'}",
        f"Last consumed window: {state.last_run or 'none'}",
    ]
    for run in store.runs(name):
        lines.append(f"{run.status}: {run.due_at} {run.detail}")
        if run.session_id:
            lines.append(f"  zeta --resume {run.session_id}")
        if run.delivery:
            lines.append(f"  delivery: {run.delivery}")
    return "\n".join(lines)


def listing(store: SQLiteStore) -> str:
    lines = []
    for state in store.jobs():
        runs = store.runs(state.job.name)
        last = runs[0] if runs else None
        lines.append(
            f"{state.job.name} r{state.revision} | {'armed' if state.enabled else 'draft/disabled'}"
            f" | {state.job.trigger.kind} | {state.job.deliver} | {last.status if last else 'never run'}"
        )
        if last and last.session_id:
            lines.append(f"  zeta --resume {last.session_id}")
    lines.extend(f"ERROR: {error}" for error in store.errors())
    return (
        "\n".join(lines)
        or "No automations. Ask the agent to draft one, or import an automations.json file."
    )

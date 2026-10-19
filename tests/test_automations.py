from __future__ import annotations

from pathlib import Path

from zeta.core.abort import AbortSignal
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent import ChildApprovalPolicy


async def test_unattended_allow_list_gates_even_exempt_and_internal_calls(
    tmp_path: Path,
) -> None:
    policy = ApprovalPolicy(
        default=ApprovalDecision.DENY, always_allow=["permitted(ok*)"]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        approval_policy=policy,
        approval_store=store,
        enforce_approvals=True,
        skill_catalog=SkillCatalog.empty(),
    )
    calls = []
    for name in ("permitted", "forbidden"):
        registry.register(
            name,
            lambda args: calls.append(args) or "executed",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
            approval_subject="value",
            requires_approval=False,
        )
    allowed = await registry.execute(ToolCall("1", "permitted", {"value": "okay"}))
    denied = await registry.execute(
        ToolCall("2", "forbidden", {"value": "okay"}), _skip_approval=True
    )
    scoped = await registry.execute(ToolCall("3", "permitted", {"value": "bad"}))
    assert not allowed["isError"]
    assert denied["isError"] and scoped["isError"]
    assert calls == [{"value": "okay"}]
    child_store = ConversationStore(tmp_path / "child")
    child = registry.clone_for_session(child_store)
    child.set_approval_policy(
        ChildApprovalPolicy(policy, child_store, "child", "child-1")
    )
    denied_child = await child.execute(ToolCall("4", "forbidden", {"value": "okay"}))
    assert denied_child["isError"]
    assert registry.denied_tools == ["forbidden", "permitted", "forbidden"]
    assert calls == [{"value": "okay"}]
    await child.close()
    await registry.close()


import asyncio
import json
import os
import selectors
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.automations import commands
from zeta.automations import runner as automation_runner
from zeta.automations.authoring import import_jobs, listing, resolve_job
from zeta.automations.commands import review_job
from zeta.automations.daemon import daemon_lock, serve
from zeta.automations.delivery import SlackDelivery
from zeta.automations.models import instant, parse_job
from zeta.automations.runner import _receipt, poll_events, run_claimed
from zeta.automations.services import (
    configure_inbox,
    mount_services,
    resolve_inbox_grants,
    validate_permissions,
)
from zeta.automations.store import SQLiteStore
from zeta.automations.tick import tick
from zeta.automations.trigger import Schedule, cron_matches, parse_trigger
from zeta.core.session import SessionManager
from zeta.core.slash import create_slash_registry
from zeta.mcp.config import load_mcp_config, server_to_json
from zeta.mcp.mount import MCPMount
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.prompts import load_identity
from zeta.protocol.types import (
    ASSISTANT_RESPONSE_SYNTHETIC,
    MessageOrigin,
    MessageRole,
    TextContent,
)
from zeta.runtime.unattended import build_unattended_loop
from zeta.skills import discover_session_skills

START = datetime(2026, 9, 9, 11, 0, tzinfo=UTC)
DUE = START + timedelta(hours=1)


async def test_no_delivery_job_approves_and_runs_without_slack_config(
    tmp_path: Path,
) -> None:
    document = _job(tmp_path).document()
    document.pop("deliver")
    document["servers"] = []
    job = parse_job("inbox-only", document)
    with SQLiteStore(tmp_path) as store:
        state = store.draft(job)
        review = await review_job(store, job.name, tmp_path)
        assert review.recipient is None
        store.approve(job.name, state.revision, review.recipient, START)
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)
        await run_claimed(
            store,
            occurrence,
            run_id,
            home=tmp_path,
            backend=FakeBackend([ScriptedTurn(content=[TextContent("sent")])]),
            mount_factory=mount_services,
        )
        run = store.runs(job.name)[0]
        assert run.status == "completed"
        assert run.delivery == ""


async def test_no_delivery_job_with_inbox_grant_approves_and_sends_without_slack_config(
    tmp_path: Path,
) -> None:
    projects = ProjectRegistry(tmp_path / "projects")
    source = projects.create_project("source", "source", tmp_path)
    target = projects.create_project("target", "target")
    document = _job(tmp_path).document()
    document.pop("deliver")
    document["servers"] = []
    document["allow"] = ["inbox(send target)"]
    job = parse_job("inbox-only", document)

    with SQLiteStore(tmp_path) as store:
        state = store.draft(job)
        review = await review_job(store, job.name, tmp_path)
        assert review.recipient is None
        assert review.inbox_target_ids == (target.project_id,)
        store.approve(
            job.name,
            state.revision,
            review.recipient,
            START,
            review.inbox_target_ids,
        )
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)
        assert run_id
        await run_claimed(
            store,
            occurrence,
            run_id,
            home=tmp_path,
            backend=FakeBackend(
                [
                    ScriptedTurn(
                        tool_calls=[
                            ToolCall(
                                "send",
                                "inbox",
                                {
                                    "action": "send",
                                    "project": "target",
                                    "kind": "question",
                                    "title": "title",
                                    "body": "body",
                                },
                            )
                        ]
                    ),
                    ScriptedTurn(content=[TextContent("sent")]),
                ]
            ),
            mount_factory=mount_services,
        )
        run = store.runs(job.name)[0]
        assert run.status == "completed", run.detail
        assert run.delivery == ""

    messages = ProjectInbox(
        projects, sessions_root=tmp_path / "sessions"
    ).list(target.project_id)
    assert len(messages["new"]) == 1
    message = messages["new"][0]
    assert message["from"] == {
        "project": source.project_id,
        "session": run.session_id,
    }


def test_approve_rejects_recipient_delivery_mismatch(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path) as store:
        delivery = _job(tmp_path)
        state = store.draft(delivery)
        with pytest.raises(ValueError, match="recipient must match"):
            store.approve(delivery.name, state.revision, None, START)
        no_delivery = parse_job(
            "inbox-only",
            {**delivery.document(), "servers": [], "deliver": "none"},
        )
        state = store.draft(no_delivery)
        with pytest.raises(ValueError, match="recipient must match"):
            store.approve(no_delivery.name, state.revision, "U123", START)


def test_automation_schema_does_not_require_delivery() -> None:
    registry = ToolRegistry(
        Path.cwd(), register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    from zeta.tools.automation import register

    register(registry)
    schema = registry.definitions_by_name["automation"].parameters
    assert "deliver" not in schema["properties"]["job"]["required"]


async def test_runtime_missing_delivery_tool_has_actionable_error(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.test_mcp import _FakeClient
    from zeta.mcp import mount as mount_module

    class Client(_FakeClient):
        async def list_tools(self):
            return []

    monkeypatch.setattr(mount_module, "_build_client", Client)
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "slack": {
                        "transport": "streamable-http",
                        "url": "https://mcp.slack.com/mcp",
                        "auth": {"type": "oauth"},
                    }
                }
            }
        )
    )
    job = _job(tmp_path)
    with SQLiteStore(tmp_path) as store:
        _arm(store, job)
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)

        await run_claimed(
            store,
            occurrence,
            run_id,
            home=tmp_path,
            backend=FakeBackend([ScriptedTurn(content=[TextContent("sent")])]),
            mount_factory=mount_services,
        )
        run = store.runs(job.name)[0]
        assert run.status == "failed"
        assert "delivery to slack needs slack__slack_send_message" in run.detail


def _job(tmp_path: Path, name: str = "brief", *, poll: bool = False):
    return parse_job(
        name,
        {
            "prompt": "Summarize new activity.",
            "trigger": {"kind": "poll", "condition": "New urgent activity"}
            if poll
            else {"kind": "schedule", "cron": "0 8 * * 1-5"},
            "servers": ["slack"],
            "allow": [],
            "deliver": "slack:U123",
            "provider": "codex",
            "model": "fake",
            "cwd": str(tmp_path),
        },
    )


def _arm(store: SQLiteStore, job, now: datetime = START) -> None:
    state = store.draft(job)
    store.approve(job.name, state.revision, "U123", now)


@pytest.mark.parametrize(
    ("allow", "message"),
    [
        (("inbox",), "inbox\\(send <project>\\)"),
        (("inbox(*)",), "wildcard"),
        (("inbox(send missing)",), "unknown inbox target project"),
    ],
)
def test_automation_inbox_rules_require_a_named_project(
    tmp_path: Path, allow: tuple[str, ...], message: str
) -> None:
    home = tmp_path / ".zeta"
    projects = ProjectRegistry(home / "projects")
    projects.create_project("source", "source", tmp_path)
    job = replace(_job(tmp_path), allow=allow)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=discover_session_skills(home=home),
        approval_policy=ApprovalPolicy(default=ApprovalDecision.DENY, always_allow=allow),
        project_registry=projects,
    )
    try:
        with pytest.raises(ValueError, match=message):
            configure_inbox(job, registry, home)
    finally:
        asyncio.run(registry.close())


def test_automation_without_inbox_rule_has_no_inbox_tool(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    projects = ProjectRegistry(home / "projects")
    projects.create_project("source", "source", tmp_path)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=discover_session_skills(home=home),
        project_registry=projects,
    )
    try:
        configure_inbox(_job(tmp_path), registry, home)
        assert "inbox" not in registry.definitions_by_name
    finally:
        asyncio.run(registry.close())


@pytest.mark.parametrize(
    "allow",
    [("inbox",), ("inbox(*)",), ("inbox(send missing)",), ("inbox(send target)",)],
)
def test_automation_inbox_draft_validation_applies_to_both_draft_paths(
    tmp_path: Path, allow: tuple[str, ...]
) -> None:
    projects = ProjectRegistry(tmp_path / "projects")
    if allow != ("inbox(send target)",):
        projects.create_project("source", "source", tmp_path)
    projects.create_project("target", "target")
    bad = replace(_job(tmp_path), allow=allow)
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"bad": bad.document()}))
    with SQLiteStore(tmp_path) as store:
        assert "invalid" in import_jobs(store, path, cwd=str(tmp_path), home=tmp_path)
        assert "bad" not in {state.job.name for state in store.jobs()}
    session = SessionManager(tmp_path).create(provider="codex", model="fake", cwd=tmp_path)
    registry = ToolRegistry(tmp_path, session_store=session.store, skill_catalog=SkillCatalog.empty())
    try:
        result = asyncio.run(registry.execute(ToolCall(
            "draft", "automation", {"action": "draft", "name": "bad", "job": bad.document()}
        )))
        assert result["isError"]
        with SQLiteStore(tmp_path) as store, pytest.raises(ValueError):
            store.get("bad")
    finally:
        asyncio.run(registry.close())


def test_automation_inbox_registers_send_only_for_an_approved_target(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    projects = ProjectRegistry(home / "projects")
    projects.create_project("source", "source", tmp_path)
    projects.create_project("target", "target")
    allow = ("inbox(send target)",)
    job = replace(_job(tmp_path), allow=allow)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=discover_session_skills(home=home),
        approval_policy=ApprovalPolicy(default=ApprovalDecision.DENY, always_allow=allow),
        project_registry=projects,
    )
    try:
        configure_inbox(job, registry, home)
        assert registry.definitions_by_name["inbox"].actions.keys() == {"send"}
        validate_permissions(job, registry)
    finally:
        asyncio.run(registry.close())


async def _run_inbox_send(
    tmp_path: Path, target: str, allow: tuple[str, ...], action: str = "send"
):
    home = tmp_path
    projects = ProjectRegistry(home / "projects")
    source = projects.create_project("source", "source", tmp_path)
    target_project = projects.create_project("target", "target")
    job = replace(_job(tmp_path), allow=allow)
    grants = resolve_inbox_grants(job, projects)
    assert grants is not None
    with SQLiteStore(tmp_path) as store:
        state = store.draft(job)
        store.approve(job.name, state.revision, "U123", START, grants.target_ids)
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)
        assert run_id
        async def mount_factory(job, registry, home, runtime_grants):
            configure_inbox(job, registry, home, runtime_grants)
            return MCPMount(registry, {})
        arguments = {"action": action}
        if action == "send":
            arguments.update(project=target, kind="question", title="title", body="body")
        elif action == "claim":
            arguments["id"] = "missing"
        backend = FakeBackend([
            ScriptedTurn(tool_calls=[ToolCall("call", "inbox", arguments)]),
            ScriptedTurn(content=[TextContent("sent")]),
        ])
        await run_claimed(store, occurrence, run_id, home=tmp_path, backend=backend,
                          mount_factory=mount_factory, delivery=RecordingDelivery())
        messages = ProjectInbox(projects, sessions_root=home / "sessions").list(target_project.project_id)
        return store.runs(job.name)[0], messages, source.project_id


async def test_automation_inbox_send_reaches_target_with_job_identity(tmp_path: Path) -> None:
    run, messages, source_id = await _run_inbox_send(tmp_path, "target", ("inbox(send target)",))
    assert run.status == "completed", run.detail
    assert len(messages["new"]) == 1
    message = messages["new"][0]
    assert message["from"]["project"] == source_id
    assert message["from"]["session"] == run.session_id


async def test_automation_inbox_send_to_unlisted_project_is_denied(tmp_path: Path) -> None:
    run, messages, _ = await _run_inbox_send(tmp_path, "source", ("inbox(send target)",))
    assert run.status == "failed"
    assert "approved" in run.detail or "allow-list" in run.detail
    assert not messages["new"]


async def test_automation_inbox_list_and_claim_are_rejected(tmp_path: Path) -> None:
    for action in ("list", "claim"):
        run, messages, _ = await _run_inbox_send(tmp_path / action, "target", ("inbox(send target)",), action)
        assert run.status == "failed"
        assert action in run.detail or "allow-list" in run.detail
        assert not messages["new"]


async def test_inbox_grant_pinned_to_approved_project_id(tmp_path: Path) -> None:
    home = tmp_path
    projects = ProjectRegistry(home / "projects")
    projects.create_project("source", "source", tmp_path)
    target = projects.create_project("target", "target")
    job = replace(_job(tmp_path), allow=("inbox(send target)",))
    grants = resolve_inbox_grants(job, projects)
    assert grants is not None
    with SQLiteStore(tmp_path) as store:
        state = store.draft(job)
        store.approve(job.name, state.revision, "U123", START, grants.target_ids)
        shutil.rmtree(projects.root / target.project_id)
        replacement = projects.create_project("target", "replacement")
        assert replacement.project_id != target.project_id
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)
        await run_claimed(store, occurrence, run_id, home=tmp_path, backend=FakeBackend([]),
                          mount_factory=_empty_mount, delivery=RecordingDelivery())
        assert store.runs(job.name)[0].status == "failed"
        assert "changed" in store.runs(job.name)[0].detail


@pytest.mark.parametrize(
    ("when", "matches"),
    [
        ("2026-09-09T11:59:00+00:00", False),
        ("2026-09-09T12:00:00+00:00", True),
        ("2026-09-09T12:01:00+00:00", False),
        ("2026-09-12T12:00:00+00:00", False),
    ],
)
def test_cron_matches_local_minute_and_weekday_boundaries(
    when: str, matches: bool
) -> None:
    assert cron_matches(Schedule("0 8 * * 1-5"), instant(when)) is matches


def test_cron_supports_ranges_lists_steps_and_posix_day_matching() -> None:
    assert cron_matches(Schedule("0,30 8-10/2 1 * 3"), DUE)
    assert not cron_matches(Schedule("*/20 9 * * *"), DUE)
    assert cron_matches(Schedule("0 8 9 * 1"), DUE)
    assert not cron_matches(Schedule("0 8 * * 1"), DUE)
    with pytest.raises(ValueError):
        parse_trigger({"kind": "schedule", "cron": "* * * *"})
    with pytest.raises(ValueError):
        parse_trigger({"kind": "schedule", "cron": "*/0 * * * *"})
    with pytest.raises(ValueError):
        parse_trigger({"kind": "webhook"})


def test_cron_skips_nonexistent_times_and_second_fold() -> None:
    trigger = Schedule("30 1 * * *")
    assert cron_matches(trigger, instant("2026-11-01T05:30:00+00:00"))
    assert not cron_matches(trigger, instant("2026-11-01T06:30:00+00:00"))
    spring = Schedule("30 2 * * *")
    assert not any(
        cron_matches(
            spring, instant("2026-03-08T06:00:00+00:00") + timedelta(minutes=i)
        )
        for i in range(180)
    )


def test_tick_is_read_only_and_fires_only_due_approved_jobs(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        store.draft(_job(tmp_path, "inert"))
        _arm(store, replace(_job(tmp_path, "later"), trigger=Schedule("0 9 * * *")))
        before = store.db.total_changes
        assert tick(store, DUE - timedelta(seconds=1)) == ()
        first = tick(store, DUE)
        assert [item.name for item in first] == ["brief"]
        assert first == tick(store, DUE)
        assert store.db.total_changes == before
        assert store.claim(first[0])
        assert not tick(store, DUE)
        assert store.claim(first[0]) is None


def test_catch_up_is_latest_only_and_bounded_to_two_hours(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        assert len(tick(store, DUE + timedelta(hours=2))) == 1
        assert not tick(store, DUE + timedelta(hours=2, seconds=1))
        _arm(
            store, replace(_job(tmp_path, "frequent"), trigger=Schedule("*/5 * * * *"))
        )
        occurrence = next(
            item
            for item in tick(store, DUE + timedelta(minutes=32))
            if item.name == "frequent"
        )
        assert occurrence.due_at == DUE + timedelta(minutes=30)


def test_arming_never_replays_occurrences_before_approval(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path), now=DUE + timedelta(minutes=1))
        assert not tick(store, DUE + timedelta(minutes=30))


def test_claim_rechecks_approval_and_revision_across_connections(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path) as first, SQLiteStore(tmp_path) as second:
        _arm(first, _job(tmp_path))
        occurrence = tick(first, DUE)[0]
        second.disable("brief")
        assert first.claim(occurrence) is None
        first.approve("brief", 1, "U123", START)
        assert second.claim(occurrence)
        assert first.claim(occurrence) is None
        second.draft(replace(_job(tmp_path), prompt="Changed"))
        with pytest.raises(ValueError, match="changed"):
            first.approve("brief", 1, "U123", START)
        assert not first.get("brief").enabled


def test_poll_interval_and_last_check_prevent_duplicate_checks(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path, poll=True))
        assert not tick(store, START + timedelta(seconds=299))
        due = tick(store, START + timedelta(seconds=300))[0]
        assert store.claim(due)
        assert not tick(store, due.checked_at)
        assert due.last_run == START


def test_poll_response_filters_old_future_and_duplicate_events() -> None:
    rows = [
        {"id": key, "timestamp": timestamp, "context": "source evidence"}
        for key, timestamp in [
            ("old", START.isoformat()),
            ("new", (START + timedelta(seconds=1)).isoformat()),
            ("new", (START + timedelta(seconds=1)).isoformat()),
            ("future", (DUE + timedelta(seconds=1)).isoformat()),
        ]
    ]
    events = poll_events(json.dumps({"events": rows}), START, DUE)
    assert [event.id for event in events] == ["new"]
    for bad in (
        "plain prose",
        "{}",
        '{"events":[{}]}',
        '{"events":[{"id":"x","timestamp":"2026-01-01","context":"x"}]}',
    ):
        with pytest.raises(ValueError):
            poll_events(bad, START, DUE)


class RecordingDelivery:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, str]] = []

    async def resolve(self, target: str) -> str:
        return "U123"

    async def send(self, recipient: str, name: str, session_id: str, text: str) -> str:
        self.sent.append((recipient, name, session_id, text))
        return '{"ts":"123.456"}'


async def _empty_mount(job, registry: ToolRegistry, home: Path) -> MCPMount:
    return MCPMount(registry, {})


async def test_draft_approve_fire_deliver_inspect_and_resume_round_trip(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    origin = SessionManager(tmp_path).create(
        provider="codex", model="fake", cwd=tmp_path
    )
    registry = ToolRegistry(tmp_path, session_store=origin.store, skill_catalog=SkillCatalog.empty())
    job = _job(tmp_path)
    response = await registry.execute(
        ToolCall(
            "draft",
            "automation",
            {"action": "draft", "name": job.name, "job": job.document()},
        )
    )
    assert not response["isError"], response
    sender = RecordingDelivery()
    with SQLiteStore(tmp_path) as store:
        assert not tick(store, DUE)
        store.approve("brief", 1, "U123", START)
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)
        assert run_id
        backend = FakeBackend(
            [ScriptedTurn(content=[TextContent("Your morning brief.")])]
        )
        await run_claimed(
            store,
            occurrence,
            run_id,
            home=tmp_path,
            backend=backend,
            mount_factory=_empty_mount,
            delivery=sender,
        )
        run = store.runs("brief")[0]
        assert run.status == "completed"
        assert json.loads(run.delivery)["recipient"] == "U123"
        assert sender.sent == [("U123", "brief", run.session_id, "Your morning brief.")]
        assert "completed" in listing(store)
        resumed = SessionManager(tmp_path).open(run.session_id)
        assert not hasattr(resumed.metadata, "compaction")
        assert "recall_history" in {
            schema["name"] for schema in backend.calls[0][1]
        }
        assert any(
            "Your morning brief." in str(message.content)
            for message in resumed.store.messages()
        )
        assert any(
            "123.456" in str(message.content) for message in resumed.store.messages()
        )
        assert not tick(store, DUE)
    assert "brief" in await commands.slash("", home=tmp_path, cwd=str(tmp_path))
    assert (
        "automations"
        in create_slash_registry(zeta_home=tmp_path, project_dir=tmp_path, skill_catalog=SkillCatalog.empty()).help_text()
    )
    await registry.close()


async def test_poll_matches_are_consumed_before_execution_and_not_replayed(
    tmp_path: Path,
) -> None:
    sender = RecordingDelivery()
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path, poll=True))
        due = tick(store, START + timedelta(minutes=5))[0]
        run_id = store.claim(due)
        event = {
            "events": [
                {
                    "id": "slack:123",
                    "timestamp": (START + timedelta(minutes=1)).isoformat(),
                    "context": "urgent",
                }
            ]
        }
        backend = FakeBackend(
            [
                ScriptedTurn(content=[TextContent(json.dumps(event))]),
                ScriptedTurn(content=[TextContent("Acted.")]),
            ]
        )
        await run_claimed(
            store,
            due,
            run_id,
            home=tmp_path,
            backend=backend,
            mount_factory=_empty_mount,
            delivery=sender,
        )
        assert store.get("brief").last_run == due.checked_at
        assert len(sender.sent) == 1
        second = tick(store, START + timedelta(minutes=10))[0]
        # Even an incorrectly retimestamped repeated source ID is deduplicated.
        event["events"][0]["timestamp"] = (START + timedelta(minutes=6)).isoformat()
        second_id = store.claim(second)
        second_backend = FakeBackend(
            [ScriptedTurn(content=[TextContent(json.dumps(event))])]
        )
        await run_claimed(
            store,
            second,
            second_id,
            home=tmp_path,
            backend=second_backend,
            mount_factory=_empty_mount,
            delivery=sender,
        )
        assert store.runs("brief")[0].status == "no_match"
        assert len(second_backend.calls) == 1
        assert len(sender.sent) == 1


async def test_negative_polls_advance_cursor_and_invalid_checks_preserve_it(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path, poll=True))
        for minute, text, status, expected in [
            (5, '{"events":[]}', "no_match", 5),
            (10, "invalid", "failed", 5),
        ]:
            occurrence = tick(store, START + timedelta(minutes=minute))[0]
            run_id = store.claim(occurrence)
            backend = FakeBackend([ScriptedTurn(content=[TextContent(text)])])
            await run_claimed(
                store,
                occurrence,
                run_id,
                home=tmp_path,
                backend=backend,
                mount_factory=_empty_mount,
                delivery=RecordingDelivery(),
            )
            assert store.runs("brief")[0].status == status
            assert store.get("brief").last_run == START + timedelta(minutes=expected)


async def test_unlisted_tool_denial_is_durable_and_prevents_delivery(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)
        sender = RecordingDelivery()
        backend = FakeBackend(
            [
                ScriptedTurn(tool_calls=[ToolCall("todo", "todo", {})]),
                ScriptedTurn(content=[TextContent("Pretending success")]),
            ]
        )
        await run_claimed(
            store,
            occurrence,
            run_id,
            home=tmp_path,
            backend=backend,
            mount_factory=_empty_mount,
            delivery=sender,
        )
        run = store.runs("brief")[0]
        assert run.status == "failed"
        assert "allow-list" in run.detail
        assert "--yolo" not in run.detail
        assert not sender.sent
        transcript = SessionManager(tmp_path).open(run.session_id).store
        assert any(
            message.tool_result and message.tool_result.is_error
            for message in transcript.messages()
        )


def test_unattended_runtime_rejects_ollama_sessions_clearly(tmp_path: Path) -> None:
    session = SessionManager(tmp_path).create(
        provider="ollama",
        model="qwen3:4b",
        cwd=tmp_path,
        compaction_budget=40_960,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        with pytest.raises(
            ValueError, match="Ollama automations are not yet supported"
        ):
            build_unattended_loop(session, home=tmp_path, allow=())
    finally:
        session.store.close()


async def test_unattended_runtime_preserves_persisted_allow_layers(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-exist"
    session = SessionManager(tmp_path).create(
        provider="codex",
        model="fake",
        cwd=tmp_path,
        skill_catalog=discover_session_skills(home=tmp_path),
        tool_allow=("*",),
        tool_allow_layers=(("remote__*",), ("*",)),
    )
    loop = build_unattended_loop(
        session, home=tmp_path, allow=("bash",), backend=FakeBackend([])
    )
    try:
        result = await loop.tool_registry.execute(
            ToolCall("blocked-bash", "bash", {"command": f"touch {marker}"})
        )

        assert result["isError"] is True
        assert not marker.exists()
    finally:
        await loop.close()


async def test_unattended_runtime_ignores_global_yolo_hooks_and_project_tools(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    (tmp_path / "settings.toml").write_text(
        'yolo = true\n[approval]\nallow = ["bash"]\n'
    )
    catalog = discover_session_skills(home=tmp_path)
    session = SessionManager(tmp_path).create(
        provider="codex",
        model="fake",
        cwd=tmp_path,
        system_prompt=load_identity(catalog=catalog),
        skill_catalog=catalog,
    )
    loop = build_unattended_loop(
        session, home=tmp_path, allow=(), backend=FakeBackend([])
    )
    assert (
        loop.tool_registry.approval_policy.decide(loop.tool_registry.resolve_call("bash", {"command": "echo x"}))
        == ApprovalDecision.DENY
    )
    assert loop.hooks is None
    assert loop._mcp_mount_attempted
    await loop.close()


async def test_automation_run_keeps_approved_skill_snapshot(
    tmp_path: Path,
) -> None:
    catalog = discover_session_skills(home=tmp_path)
    session = SessionManager(tmp_path).create(
        provider="codex",
        model="fake",
        cwd=tmp_path,
        system_prompt=load_identity(catalog=catalog),
        skill_catalog=catalog,
    )
    skill = tmp_path / "skills" / "late.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: late\ndescription: late description\n---\n\nlate body\n",
        encoding="utf-8",
    )
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    loop = build_unattended_loop(
        session, home=tmp_path, allow=("skill",), backend=backend
    )
    try:
        _ = [
            event
            async for event in loop.run_turn("run", origin=MessageOrigin.AUTOMATION_PROMPT)
        ]
        prompt = "\n".join(
            block.text
            for message in backend.calls[0][0]
            if message.role is MessageRole.SYSTEM
            for block in message.content
            if isinstance(block, TextContent)
        )
        loaded = await loop.tool_registry.execute(
            ToolCall("late", "skill", {"name": "late"})
        )
        assert "late description" not in prompt
        assert loaded["isError"] is True
        assert "available skills: none" in loaded["content"][0]["text"]
    finally:
        await loop.close()


async def test_automation_catalog_excludes_project_skills_from_prompt_and_tool(
    tmp_path: Path,
) -> None:
    project_skill = tmp_path / ".zeta" / "skills" / "project.md"
    project_skill.parent.mkdir(parents=True)
    project_skill.write_text(
        "---\nname: project\ndescription: project skill\n---\n\nproject body\n",
        encoding="utf-8",
    )
    catalog = discover_session_skills(home=tmp_path)
    session = SessionManager(tmp_path).create(
        provider="codex",
        model="fake",
        cwd=tmp_path,
        system_prompt=load_identity(catalog=catalog),
        skill_catalog=catalog,
    )
    loop = build_unattended_loop(
        session,
        home=tmp_path,
        allow=("skill",),
        backend=FakeBackend([]),
    )
    try:
        loaded = await loop.tool_registry.execute(
            ToolCall("project", "skill", {"name": "project"})
        )
        assert "project skill" not in session.metadata.system_prompt
        assert [item["name"] for item in session.metadata.skill_catalog or []] == []
        assert loaded["isError"] is True
        assert "available skills: none" in loaded["content"][0]["text"]
    finally:
        await loop.close()


def test_import_keeps_project_grants_inert_and_isolates_malformed_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "automations.json"
    path.write_text(
        json.dumps({"good": _job(tmp_path).document(), "broken": {"trigger": "bad"}})
    )
    with SQLiteStore(tmp_path) as store:
        report = import_jobs(store, path, cwd=str(tmp_path), home=tmp_path)
        assert "good r1: draft" in report and "broken: invalid" in report
        assert not tick(store, DUE)
        store.approve("good", 1, "U123", START)
        path.write_text('{"good":{"allow":["bash"]}}')
        assert store.get("good").job.allow == ()
        assert store.get("good").enabled
        store.db.execute("UPDATE revisions SET document='invalid' WHERE name='good'")
        store.db.commit()
        assert not tick(store, DUE)
        assert "malformed stored job" in listing(store)


def test_recovery_records_interrupted_and_uncertain_without_replaying(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        _arm(store, _job(tmp_path, "second"))
        due = tick(store, DUE)
        first_id = store.claim(due[0])
        second_id = store.claim(due[1])
        store.finish(second_id, "sending")
        store.recover()
        assert store.runs(due[0].name)[0].status == "interrupted"
        assert store.runs(due[1].name)[0].status == "uncertain"
        assert not tick(store, DUE)
        assert store.claim(due[0]) is None
        assert first_id


def test_daemon_lock_refuses_second_owner(tmp_path: Path) -> None:
    with (
        daemon_lock(tmp_path),
        pytest.raises(RuntimeError, match="already running"),
        daemon_lock(tmp_path),
    ):
        pass


async def test_daemon_remains_responsive_and_aborts_worker_on_shutdown(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
    stopped = asyncio.Event()
    ready = asyncio.Event()
    started = asyncio.Event()
    canceled = asyncio.Event()

    async def runner(store, occurrence, run_id, *, home):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            store.finish(run_id, "interrupted")
            canceled.set()

    task = asyncio.create_task(
        serve(
            tmp_path,
            stop=stopped,
            clock=lambda: DUE,
            interval=0.01,
            runner=runner,
            webhook_port=0,
            on_ready=lambda _host, _port: ready.set(),
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=2)
    await asyncio.wait_for(started.wait(), timeout=2)
    stopped.set()
    await asyncio.wait_for(task, timeout=2)
    assert canceled.is_set()


def test_subjectless_mcp_scoped_allow_is_rejected_without_widening(
    tmp_path: Path,
) -> None:
    job = replace(_job(tmp_path), allow=("slack__history(channel*)",))
    policy = ApprovalPolicy(default=ApprovalDecision.DENY, always_allow=job.allow)
    registry = ToolRegistry(tmp_path, register_builtin=False, approval_policy=policy, skill_catalog=SkillCatalog.empty())
    registry.register("slack__history", lambda args: "x")
    with pytest.raises(ValueError, match="no approval subject"):
        validate_permissions(job, registry)
    assert policy.decide(registry.resolve_call("slack__history", {})) == ApprovalDecision.DENY


def test_fixed_client_oauth_config_interpolates_and_round_trips(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SLACK_CLIENT_SECRET", "secret-value")
    path = tmp_path / "mcp.json"
    raw = {
        "transport": "streamable-http",
        "url": "https://mcp.slack.com/mcp",
        "auth": {
            "type": "oauth",
            "client_id": "client-123",
            "client_secret": "${SLACK_CLIENT_SECRET}",
            "callback_port": 3118,
            "scopes": ["chat:write"],
        },
        "approval_subjects": {"slack_send_message": "channel_id"},
    }
    path.write_text(json.dumps({"servers": {"slack": raw}}))
    config = load_mcp_config(path).servers["slack"]
    assert config.client_secret == "secret-value"
    assert config.client_id == "client-123"
    assert config.scopes == ("chat:write",)
    assert server_to_json(config)["approval_subjects"] == {
        "slack_send_message": "channel_id"
    }
    assert server_to_json(config)["auth"]["callback_port"] == 3118


class SlackClient:
    def __init__(self, search_text: str = "User U123") -> None:
        self.calls = []
        self.search_text = search_text

    async def call_tool(self, name, arguments, abort_signal):
        self.calls.append((name, arguments))
        return {
            "content": [
                {
                    "type": "text",
                    "text": self.search_text if "search" in name else '{"ts":"123"}',
                }
            ],
            "isError": False,
        }


class SlackMount:
    def __init__(self, client) -> None:
        self.client = client

    async def call_tool(self, server, name, arguments):
        return await self.client.call_tool(name, arguments, AbortSignal())


async def test_slack_delivery_pins_recipient_and_bounds_output() -> None:
    client = SlackClient()
    sender = SlackDelivery(SlackMount(client))
    assert await sender.resolve("slack:@austin") == "U123"
    await sender.send("U123", "brief", "session-123", "x" * 6000)
    tool, args = client.calls[-1]
    assert tool == "slack_send_message"
    assert args["channel_id"] == "U123"
    assert len(args["message"]) == 4500
    assert "Truncated" in args["message"]
    client.search_text = "User U123 and User U456"
    with pytest.raises(ValueError, match="ambiguous"):
        await sender.resolve("slack:@austin")


async def test_human_review_token_arms_only_the_exact_revision(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.test_mcp import _FakeClient
    from zeta.mcp import mount as mount_module
    from zeta.mcp.client import MCPTool

    class ReviewClient(_FakeClient):
        async def list_tools(self):
            return [
                MCPTool(
                    "slack_send_message",
                    "",
                    {
                        "type": "object",
                        "properties": {
                            "channel_id": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                )
            ]

    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setattr(mount_module, "_build_client", ReviewClient)
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "slack": {
                        "transport": "streamable-http",
                        "url": "https://mcp.slack.com/mcp",
                        "auth": {"type": "oauth"},
                    }
                }
            }
        )
    )
    with SQLiteStore(tmp_path) as store:
        store.draft(_job(tmp_path))
    preview = await commands.slash("approve brief", home=tmp_path, cwd=str(tmp_path))
    token = preview.split("Approval token: ")[1].splitlines()[0]
    assert "Resolved recipient: U123" in preview
    with SQLiteStore(tmp_path) as store:
        store.draft(_job(tmp_path, "other"))
    other = await commands.slash("approve other", home=tmp_path, cwd=str(tmp_path))
    assert other.split("Approval token: ")[1].splitlines()[0] != token
    with SQLiteStore(tmp_path) as store:
        assert not store.get("brief").enabled
        store.draft(replace(_job(tmp_path), prompt="Changed after review"))
    with pytest.raises(ValueError, match="review changed"):
        await commands.slash(f"approve brief {token}", home=tmp_path, cwd=str(tmp_path))
    preview = await commands.slash("approve brief", home=tmp_path, cwd=str(tmp_path))
    token = preview.split("Approval token: ")[1].splitlines()[0]
    assert "armed" in await commands.slash(
        f"approve brief {token}", home=tmp_path, cwd=str(tmp_path)
    )
    with SQLiteStore(tmp_path) as store:
        assert store.get("brief").revision == 2
        assert store.get("brief").enabled


async def test_fixed_client_oauth_skips_registration_and_refreshes_with_same_credentials(
    tmp_path: Path,
) -> None:
    from urllib.parse import parse_qs, urlparse

    from tests.test_mcp_oauth import _FakeAuthServer, _fire_redirect
    from zeta.mcp.oauth import authorize, refresh_access_token
    from zeta.mcp.oauth_store import load_token

    auth = _FakeAuthServer()

    async def handler(request):
        response = await auth.handle(request)
        if request.url.path.endswith("oauth-authorization-server"):
            data = response.json()
            data.pop("registration_endpoint")
            return httpx.Response(200, json=data, request=request)
        return response

    tasks = []

    async def redirect(url):
        params = parse_qs(urlparse(url).query)
        assert params["client_id"] == ["registered-client"]
        assert params["scope"] == ["chat:write"]
        assert params["code_challenge_method"] == ["S256"]
        await _fire_redirect(
            params["redirect_uri"][0] + "?code=code&state=" + params["state"][0]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        token = await authorize(
            server_name="slack",
            server_url=auth.server_url,
            home=str(tmp_path),
            client_id="registered-client",
            client_secret="registered-secret",
            scopes=("chat:write",),
            http_client=client,
            browser_opener=lambda url: tasks.append(asyncio.create_task(redirect(url))),
        )
        await asyncio.gather(*tasks)
        assert not auth.registrations
        assert auth.token_requests[0]["client_secret"] == "registered-secret"
        stored = load_token("slack", home=str(tmp_path))
        assert stored.client_id == "registered-client"
        await refresh_access_token(
            token_endpoint=token.token_endpoint,
            refresh_token=token.refresh_token,
            client_id=token.client_id,
            client_secret=token.client_secret,
            resource=token.resource,
            http_client=client,
        )
        assert auth.token_requests[-1]["grant_type"] == "refresh_token"
        assert auth.token_requests[-1]["client_secret"] == "registered-secret"


async def test_only_selected_home_servers_mount_and_scoped_mcp_rules_work(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.test_mcp import _FakeClient
    from zeta.automations.services import mount_services
    from zeta.mcp import mount as mount_module
    from zeta.mcp.client import MCPTool

    selected = []

    class Client(_FakeClient):
        def __init__(self, config):
            super().__init__(config)
            selected.append(config.name)

        async def list_tools(self):
            return [
                MCPTool(
                    "history",
                    "",
                    {"type": "object", "properties": {"channel": {"type": "string"}}},
                ),
                MCPTool(
                    "slack_send_message",
                    "",
                    {
                        "type": "object",
                        "properties": {
                            "channel_id": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                ),
            ]

    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setattr(mount_module, "_build_client", Client)
    config = {
        "transport": "streamable-http",
        "url": "https://mcp.slack.com/mcp",
        "auth": {"type": "oauth"},
        "approval_subjects": {"history": "channel"},
    }
    (tmp_path / "mcp.json").write_text(
        json.dumps({"servers": {"slack": config, "unselected": config}})
    )
    project = tmp_path / ".zeta"
    project.mkdir()
    (project / "mcp.json").write_text(json.dumps({"servers": {"hostile": config}}))
    job = replace(_job(tmp_path), allow=("slack__history(C123)",))
    policy = ApprovalPolicy(default=ApprovalDecision.DENY, always_allow=job.allow)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        enforce_approvals=True,
        approval_store=ConversationStore(tmp_path / "session"),
skill_catalog=SkillCatalog.empty(),
    )
    mount = await mount_services(job, registry, tmp_path)
    try:
        assert selected == ["slack"]
        assert (
            policy.decide(registry.resolve_call("slack__history", {"channel": "C123"}))
            == ApprovalDecision.ALLOW
        )
        assert (
            policy.decide(registry.resolve_call("slack__history", {"channel": "C456"})) == ApprovalDecision.DENY
        )
        allowed = await registry.execute(
            ToolCall("read-1", "slack__history", {"channel": "C123"})
        )
        denied = await registry.execute(
            ToolCall("read-2", "slack__history", {"channel": "C456"})
        )
        assert not allowed["isError"]
        assert denied["isError"]
    finally:
        await mount.close()
        await registry.close()


async def test_cancel_during_delivery_is_uncertain_and_never_replayed(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    class SlowDelivery(RecordingDelivery):
        async def send(self, recipient, name, session_id, text):
            started.set()
            self.sent.append((recipient, name, session_id, text))
            await asyncio.Event().wait()

    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        due = tick(store, DUE)[0]
        run_id = store.claim(due)
        sender = SlowDelivery()
        run = asyncio.create_task(
            run_claimed(
                store,
                due,
                run_id,
                home=tmp_path,
                backend=FakeBackend([ScriptedTurn(content=[TextContent("message")])]),
                mount_factory=_empty_mount,
                delivery=sender,
                timeout_seconds=30,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=10)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        assert store.runs("brief")[0].status == "uncertain"
        assert store.runs("brief")[0].delivery == "U123"
        store.recover()
        assert not tick(store, DUE)
        assert len(sender.sent) == 1


async def test_timeout_during_delivery_is_uncertain_and_never_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real asyncio.timeout expiry, fired only after send() started: capture the
    # runner's Timeout object and reschedule its deadline to "now".
    started = asyncio.Event()
    timeouts: list[asyncio.Timeout] = []
    real_timeout = asyncio.timeout

    def controlled_timeout(_delay: float | None) -> asyncio.Timeout:
        timeout = real_timeout(None)
        timeouts.append(timeout)
        return timeout

    monkeypatch.setattr(automation_runner.asyncio, "timeout", controlled_timeout)

    class SlowDelivery(RecordingDelivery):
        async def send(self, recipient, name, session_id, text):
            started.set()
            self.sent.append((recipient, name, session_id, text))
            await asyncio.Event().wait()

    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        due = tick(store, DUE)[0]
        run_id = store.claim(due)
        sender = SlowDelivery()
        run = asyncio.create_task(
            run_claimed(
                store,
                due,
                run_id,
                home=tmp_path,
                backend=FakeBackend([ScriptedTurn(content=[TextContent("message")])]),
                mount_factory=_empty_mount,
                delivery=sender,
                timeout_seconds=30,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=10)
        timeouts[-1].reschedule(asyncio.get_running_loop().time())
        await asyncio.wait_for(run, timeout=10)
        assert store.runs("brief")[0].status == "uncertain"
        assert store.runs("brief")[0].delivery == "U123"
        store.recover()
        assert not tick(store, DUE)
        assert len(sender.sent) == 1


async def test_approval_change_after_claim_cancels_before_creating_a_session(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        due = tick(store, DUE)[0]
        run_id = store.claim(due)
        store.disable("brief")
        await run_claimed(store, due, run_id, home=tmp_path)
        assert store.runs("brief")[0].status == "canceled"
        assert not (tmp_path / "sessions").exists()


async def test_missing_mcp_configuration_fails_visibly_and_keeps_other_jobs_eligible(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    with SQLiteStore(tmp_path) as store:
        _arm(store, _job(tmp_path))
        _arm(store, _job(tmp_path, "other"))
        due = tick(store, DUE)[0]
        run_id = store.claim(due)
        await run_claimed(store, due, run_id, home=tmp_path, backend=FakeBackend([]))
        assert store.runs("brief")[0].status == "failed"
        assert "unavailable MCP configuration" in store.runs("brief")[0].detail
        assert [item.name for item in tick(store, DUE)] == ["other"]


def test_automation_receipt_is_non_consuming(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _receipt(store, "Automation failed: test")

    assert store.messages()[-1].metadata["response_state"] == ASSISTANT_RESPONSE_SYNTHETIC


def test_draft_defaults_are_explicit_and_invalid_types_are_rejected(
    tmp_path: Path,
) -> None:
    raw = _job(tmp_path).document()
    for field in ("cwd", "provider", "model"):
        raw.pop(field)
    job = resolve_job("new", raw, cwd=str(tmp_path), home=tmp_path)
    assert job.provider == "claude" and job.model == "claude-sonnet-4-6"
    assert job.cwd == str(tmp_path.resolve())
    assert job.trigger.timezone == "America/Toronto"
    with pytest.raises(ValueError):
        parse_trigger({"kind": "poll", "condition": "x", "interval_seconds": True})
    with pytest.raises(ValueError):
        parse_job("bad", {**job.document(), "allow": ["unselected__tool"]})


async def test_automation_tool_has_no_arming_operation(tmp_path: Path) -> None:
    registry = ToolRegistry(
        tmp_path, session_store=ConversationStore(tmp_path / "session"),
        skill_catalog=SkillCatalog.empty(),
    )
    result = await registry.execute(
        ToolCall("arm", "automation", {"action": "approve", "name": "job"})
    )
    assert result["isError"]
    await registry.close()


def test_project_mcp_cannot_declare_the_subject_of_a_trusted_grant(
    tmp_path: Path,
) -> None:
    from zeta.mcp.config import load_mcp_config_overlay

    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    (project / ".zeta").mkdir(parents=True)
    raw = {
        "transport": "streamable-http",
        "url": "https://mcp.example",
        "approval_subjects": {"tool": "trusted_field"},
    }
    (home / "mcp.json").write_text(json.dumps({"servers": {"service": raw}}))
    raw["approval_subjects"] = {"tool": "attacker_field"}
    (project / ".zeta" / "mcp.json").write_text(
        json.dumps({"servers": {"service": raw}})
    )
    assert load_mcp_config(home / "mcp.json").servers["service"].approval_subjects == {
        "tool": "trusted_field"
    }
    loaded = load_mcp_config_overlay(home=home, project_dir=project)
    assert loaded.servers["service"].approval_subjects == {}


def test_cli_import_list_show_and_disable_preserve_draft_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from zeta.cli.main import main

    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"brief": _job(tmp_path).document()}))
    assert main(["automation", "import", str(path)]) == 0
    assert "draft (not armed)" in capsys.readouterr().out
    assert main(["automation", "list"]) == 0
    assert "brief" in capsys.readouterr().out
    assert main(["automation", "show", "brief"]) == 0
    assert "Approved recipient: none" in capsys.readouterr().out
    assert main(["automation", "disable", "brief"]) == 0
    with SQLiteStore(tmp_path) as store:
        assert not tick(store, DUE)


def test_cli_does_not_accept_piped_approval(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import io

    from zeta.automations import cli
    from zeta.cli.main import main

    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    with SQLiteStore(tmp_path) as store:
        store.draft(_job(tmp_path))

    async def review(*args):
        return commands.Review("brief", 1, "U123", "token", "reviewed draft")

    monkeypatch.setattr(cli, "review_job", review)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("token\n"))
    assert main(["--yolo", "automation", "approve", "brief"]) == 1
    assert "interactive terminal" in capsys.readouterr().err
    with SQLiteStore(tmp_path) as store:
        assert not store.get("brief").enabled


def test_cli_approval_persists_resolved_inbox_targets(
    tmp_path: Path, monkeypatch
) -> None:
    import io

    from zeta.automations import cli
    from zeta.cli.main import main

    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    with SQLiteStore(tmp_path) as store:
        store.draft(_job(tmp_path))

    async def review(*args):
        return commands.Review(
            "brief", 1, "U123", "token", "reviewed draft", ("target-id",)
        )

    monkeypatch.setattr(cli, "review_job", review)
    monkeypatch.setattr(cli.sys, "stdin", TTY("token\n"))
    assert main(["automation", "approve", "brief"]) == 0
    with SQLiteStore(tmp_path) as store:
        assert store.get("brief").inbox_target_ids == ("target-id",)


def _start_cli_daemon(
    tmp_path: Path, *, port: int = 0
) -> tuple[subprocess.Popen[bytes], int, bytes, bytes]:
    code = (
        "from zeta.cli.main import main; "
        "raise SystemExit(main([\"automation\", \"daemon\", "
        f"\"--webhook-port\", \"{port}\"]))"
    )
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env={**os.environ, "ZETA_HOME": str(tmp_path)},
    )
    assert process.stdout is not None
    assert process.stderr is not None
    streams = {process.stdout: bytearray(), process.stderr: bytearray()}
    selector = selectors.DefaultSelector()
    for stream in streams:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    deadline = started + 60
    prefix = b"webhook receiver listening on "

    def collect_remaining() -> None:
        output, errors = process.communicate(timeout=10)
        streams[process.stdout].extend(output or b"")
        streams[process.stderr].extend(errors or b"")

    def diagnostics() -> str:
        elapsed = time.monotonic() - started
        return (
            f"daemon did not become ready after {elapsed:.3f}s; "
            f"stdout={bytes(streams[process.stdout])!r}; "
            f"stderr={bytes(streams[process.stderr])!r}"
        )

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if process.poll() is not None:
                collect_remaining()
                raise AssertionError(
                    f"daemon exited with code {process.returncode} after "
                    f"{elapsed:.3f}s; stdout={bytes(streams[process.stdout])!r}; "
                    f"stderr={bytes(streams[process.stderr])!r}"
                )
            if now >= deadline:
                if process.poll() is None:
                    process.kill()
                collect_remaining()
                raise AssertionError(diagnostics())
            for key, _ in selector.select(timeout=min(0.1, deadline - now)):
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if chunk:
                    streams[stream].extend(chunk)
                    if stream is process.stderr:
                        for line in bytes(streams[stream]).splitlines():
                            if line.startswith(prefix):
                                selector.close()
                                return (
                                    process,
                                    int(line.rsplit(b":", 1)[1]),
                                    bytes(streams[process.stdout]),
                                    bytes(streams[process.stderr]),
                                )
                elif process.poll() is not None:
                    collect_remaining()
                    raise AssertionError(diagnostics())
    except BaseException:
        if process.poll() is None:
            process.kill()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)
        raise
    finally:
        selector.close()


def test_cli_daemon_exits_cleanly_on_sigterm(tmp_path: Path) -> None:
    process, _port, startup_output, startup_errors = _start_cli_daemon(tmp_path)
    try:
        process.terminate()
        output, errors = process.communicate(timeout=10)
        output = startup_output + (output or b"")
        errors = startup_errors + (errors or b"")
        assert process.returncode == 0, (output, errors)
        with daemon_lock(tmp_path):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_daemon_exits_cleanly_on_sigterm_with_stuck_webhook_client(
    tmp_path: Path,
) -> None:
    webhook_job = replace(
        _job(tmp_path), trigger=parse_trigger({"kind": "webhook", "verify": "github"})
    )
    with SQLiteStore(tmp_path) as store:
        _arm(store, webhook_job)
        token = store.webhook_credentials("brief").token
    process, port, startup_output, startup_errors = _start_cli_daemon(tmp_path)
    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    try:
        client.sendall(
            f"POST /hooks/{token} HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Content-Length: 100\r\n\r\n"
            "x".encode()
        )
        time.sleep(0.1)
        started = time.monotonic()
        process.terminate()
        output, errors = process.communicate(timeout=5)
        output = startup_output + (output or b"")
        errors = startup_errors + (errors or b"")
        assert time.monotonic() - started < 3
        assert process.returncode == 0, (output, errors)
    finally:
        client.close()
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_daemon_reports_bind_failure_and_exits(tmp_path: Path) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        code = (
            "from zeta.cli.main import main; "
            "raise SystemExit(main([\"automation\", \"daemon\", "
            f"\"--webhook-port\", \"{port}\"]))"
        )
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "ZETA_HOME": str(tmp_path)},
            # Bound a hung daemon, allowing cold CLI imports on busy CI workers.
            timeout=30,
            check=False,
        )
    assert process.returncode == 1, (process.stdout, process.stderr)
    assert (
        f"failed to bind webhook receiver on 127.0.0.1:{port}: "
        in process.stderr
    )


def _slack_deferred_config() -> dict[str, object]:
    return {
        "transport": "streamable-http",
        "url": "https://mcp.slack.com/mcp",
        "auth": {"type": "oauth"},
        "approval_subjects": {"history": "channel"},
    }


def _slack_deferred_client_factory(
    monkeypatch, *, extra_tools: int = 40, include_send: bool = True
):
    """Patch the MCP client builder so 'slack' exposes >32 tools (deferred path)."""
    from tests.test_mcp import _FakeClient
    from zeta.mcp import mount as mount_module
    from zeta.mcp.client import MCPTool

    class Client(_FakeClient):
        async def list_tools(self):
            tools = [
                MCPTool(
                    "history",
                    "read channel history",
                    {"type": "object", "properties": {"channel": {"type": "string"}}},
                )
            ]
            if include_send:
                tools.append(
                    MCPTool(
                        "slack_send_message",
                        "send a message",
                        {
                            "type": "object",
                            "properties": {
                                "channel_id": {"type": "string"},
                                "message": {"type": "string"},
                            },
                        },
                    )
                )
            tools += [
                MCPTool(f"tool_{i}", f"filler {i}", {"type": "object"})
                for i in range(extra_tools)
            ]
            return tools

    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setattr(mount_module, "_build_client", Client)
    return Client


async def test_approval_rejects_missing_delivery_tool_with_actionable_error(
    tmp_path: Path, monkeypatch
) -> None:
    from zeta.automations.services import mount_services

    _slack_deferred_client_factory(monkeypatch, include_send=False)
    (tmp_path / "mcp.json").write_text(
        json.dumps({"servers": {"slack": _slack_deferred_config()}})
    )
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    with pytest.raises(ValueError, match="delivery to slack needs slack__slack_send_message"):
        await mount_services(_job(tmp_path), registry, tmp_path)
    await registry.close()


async def test_deferred_catalog_allows_scoped_rule_and_activates_delivery_tool(
    tmp_path: Path, monkeypatch
) -> None:
    from zeta.automations.services import mount_services
    from zeta.mcp.server_actor import MCP_EAGER_TOOL_LIMIT

    _slack_deferred_client_factory(monkeypatch)
    (tmp_path / "mcp.json").write_text(
        json.dumps({"servers": {"slack": _slack_deferred_config()}})
    )
    job = replace(_job(tmp_path), allow=("slack__history(C123)",))
    policy = ApprovalPolicy(default=ApprovalDecision.DENY, always_allow=job.allow)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        enforce_approvals=True,
        approval_store=ConversationStore(tmp_path / "session"),
        skill_catalog=SkillCatalog.empty(),
    )
    mount = await mount_services(job, registry, tmp_path)
    try:
        # Confirm the deferred path is exercised: more tools than the eager cap.
        actor = mount._actors["slack"]
        assert len(actor._tools) > MCP_EAGER_TOOL_LIMIT
        # The mandatory delivery tool is activated (before validation succeeds).
        assert "slack__slack_send_message" in registry.registered_names
        # The valid scoped allow rule activates its exact tool.
        assert "slack__history" in registry.registered_names
        # Deferred: unrequested tools are NOT auto-registered.
        assert "slack__tool_5" not in registry.registered_names
        assert (
            policy.decide(registry.resolve_call("slack__history", {"channel": "C123"}))
            == ApprovalDecision.ALLOW
        )
        # Re-activating the already-owned delivery tool remains accepted (idempotent).
        activated, rejected = mount.activate_tools(
            registry, ["slack__slack_send_message"]
        )
        assert activated == ["slack__slack_send_message"]
        assert rejected == []
    finally:
        await mount.close()
        await registry.close()


async def test_deferred_catalog_rejects_missing_tool(
    tmp_path: Path, monkeypatch
) -> None:
    from zeta.automations.services import mount_services

    _slack_deferred_client_factory(monkeypatch)
    (tmp_path / "mcp.json").write_text(
        json.dumps({"servers": {"slack": _slack_deferred_config()}})
    )
    job = replace(_job(tmp_path), allow=("slack__does_not_exist",))
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    with pytest.raises(ValueError, match="activation failed"):
        await mount_services(job, registry, tmp_path)
    await registry.close()


async def test_deferred_catalog_rejects_foreign_collision(
    tmp_path: Path, monkeypatch
) -> None:
    from zeta.automations.services import mount_services

    _slack_deferred_client_factory(monkeypatch)
    (tmp_path / "mcp.json").write_text(
        json.dumps({"servers": {"slack": _slack_deferred_config()}})
    )
    # A non-MCP tool already occupies the allowlisted name; the deferred catalog
    # must refuse to treat that foreign definition as an activated MCP tool.
    job = replace(_job(tmp_path), allow=("slack__history",))
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    registry.register("slack__history", lambda args: "foreign")
    with pytest.raises(ValueError, match="activation failed"):
        await mount_services(job, registry, tmp_path)
    await registry.close()


async def test_deferred_catalog_with_builtin_allow_rule_mounts(
    tmp_path: Path, monkeypatch
) -> None:
    from zeta.automations.services import mount_services

    _slack_deferred_client_factory(monkeypatch)
    (tmp_path / "mcp.json").write_text(
        json.dumps({"servers": {"slack": _slack_deferred_config()}})
    )
    # A built-in tool name in the allowlist must not be routed through MCP
    # activation (which only knows catalog names); it is validated normally.
    job = replace(_job(tmp_path), allow=("read",))
    registry = ToolRegistry(
        tmp_path,
        approval_store=ConversationStore(tmp_path / "session"),
        skill_catalog=SkillCatalog.empty(),
    )
    assert "read" in registry.registered_names
    mount = await mount_services(job, registry, tmp_path)
    try:
        # The delivery tool activated, and the built-in survived the mount.
        assert "slack__slack_send_message" in registry.registered_names
        assert "read" in registry.registered_names
    finally:
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True])
async def test_automation_uses_only_existing_project_without_git_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    from zeta.core import project_context

    cwd = tmp_path / "repo"
    cwd.mkdir()
    await asyncio.to_thread(
        subprocess.run, ["git", "init"], cwd=cwd, check=True, capture_output=True
    )
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    existing = (
        manager.project_registry.find_or_create_for_directory(cwd)
        if registered
        else None
    )

    def fail_new_discovery(*args: object, **kwargs: object) -> object:
        raise AssertionError("automation must not run automatic project discovery")

    monkeypatch.setattr(project_context, "_run_discovery_git", fail_new_discovery)
    with SQLiteStore(home) as store:
        _arm(store, replace(_job(cwd), name="project-scope"))
        occurrence = tick(store, DUE)[0]
        run_id = store.claim(occurrence)
        await run_claimed(
            store,
            occurrence,
            run_id,
            home=home,
            backend=FakeBackend(
                [ScriptedTurn(content=[TextContent("automation complete")])]
            ),
            mount_factory=_empty_mount,
            delivery=RecordingDelivery(),
        )
        run = store.runs("project-scope")[0]
        metadata = manager.read_metadata(run.session_id)
        assert metadata.project_id == (
            existing.project_id if existing is not None else None
        )
        assert len(manager.project_registry.list_projects()) == int(registered)


def test_action_allow_permissions_validate_and_match(tmp_path: Path) -> None:
    from zeta.tools.registry import ApprovalBinding, ToolAction

    allow = ("task(output)", "task(start pytest*)")
    job = replace(_job(tmp_path), allow=allow)
    policy = ApprovalPolicy(default=ApprovalDecision.DENY, always_allow=allow)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        approval_policy=policy,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.register(
        "task",
        lambda arguments, execution_context=None: "x",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "command": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        actions={
            "start": ToolAction(
                required_fields=frozenset({"command"}),
                allowed_fields=frozenset({"action", "command"}),
                requires_approval=True,
                capability_class="exec",
                approval_subject="command",
                binding=ApprovalBinding.CWD,
            ),
            "output": ToolAction(
                required_fields=frozenset({"task_id"}),
                allowed_fields=frozenset({"action", "task_id"}),
                requires_approval=False,
                capability_class="read",
            ),
        },
    )

    validate_permissions(job, registry)

    assert policy.decide(registry.resolve_call("task", {"action": "output", "task_id": "1"})) is ApprovalDecision.ALLOW
    assert policy.decide(registry.resolve_call("task", {"action": "start", "command": "pytest -q"})) is ApprovalDecision.ALLOW
    assert policy.decide(registry.resolve_call("task", {"action": "start", "command": "ruff"})) is ApprovalDecision.DENY

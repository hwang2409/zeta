from __future__ import annotations

import asyncio
import gc
import json
import multiprocessing
import threading
from multiprocessing.connection import Connection
from pathlib import Path
from typing import ClassVar

import pytest

from tests.support.owned import await_or_fail
from zeta.core.checkpoints import mutation as mutation_module
from zeta.core.store import (
    AppendConversationRows,
    ConversationAddress,
    ConversationDedupeConflict,
    ConversationDedupeKey,
    ConversationGenerationConflict,
    ConversationIntegrityError,
    ConversationRow,
    ConversationStore,
    ConversationTargetOwned,
    _async_writes,
    commit_conversation_async,
)
from zeta.core.store._step_delivery import PromptInput
from zeta.protocol.types import Message, MessageOrigin, MessageRole, TextContent


def _hold_conversation_owner(
    sessions: str,
    session_id: str,
    control: Connection,
) -> None:
    store = ConversationStore(
        Path(sessions), session_id=session_id, _owner_surface="tui"
    )
    control.send("ready")
    try:
        assert control.recv() == "release"
    finally:
        store.close()


def _warning(text: str, *, parent_id: str | None = None) -> ConversationRow:
    return ConversationRow("warning", {"message": text}, parent_id)


class _FinalizeAppendRows(AppendConversationRows[object]):
    cancellation_policy: ClassVar[mutation_module.MutationCancellationPolicy] = (
        mutation_module.MutationCancellationPolicy.FINALIZE_ONCE_SUBMITTED
    )


def test_commit_installs_one_multirow_delta_and_rejects_stale_generation(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    installed: list[object] = []
    store.on_entries_installed = installed.append
    generation = store.conversation_generation

    delta = store.commit(
        AppendConversationRows((_warning("one"), _warning("two")), result="done"),
        generation,
    )

    assert delta.result == "done"
    assert delta.generation_after != generation
    assert [entry.data["message"] for entry in delta.committed_entries] == [
        "one",
        "two",
    ]
    assert len(installed) == 1
    before = (
        store.path.read_bytes(),
        tuple(store.entries),
        store.conversation_generation,
    )
    with pytest.raises(ConversationGenerationConflict) as raised:
        store.commit(AppendConversationRows((_warning("stale"),)), generation)
    assert raised.value.actual_generation == delta.generation_after
    assert (
        store.path.read_bytes(),
        tuple(store.entries),
        store.conversation_generation,
    ) == before
    store.close()


def test_invalid_second_row_changes_neither_durable_nor_resident_state(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    before = (
        store.path.read_bytes(),
        tuple(store.entries),
        store.conversation_generation,
    )

    with pytest.raises(ConversationIntegrityError, match="invalid payload"):
        store.commit(
            AppendConversationRows(
                (_warning("valid"), ConversationRow("warning", {"message": 3}))
            ),
            store.conversation_generation,
        )

    assert (
        store.path.read_bytes(),
        tuple(store.entries),
        store.conversation_generation,
    ) == before
    store.close()


def test_dedupe_survives_restart_and_wins_before_stale_generation(
    tmp_path: Path,
) -> None:
    key = ConversationDedupeKey("run-publication", "run-1:route-1")
    command = AppendConversationRows((_warning("published"),), key, result="current")
    store = ConversationStore(tmp_path)
    first = store.commit(command, store.conversation_generation)
    first_id = first.committed_entries[0].id
    session_id = store.session_id
    store.close()

    reopened = ConversationStore(tmp_path, session_id=session_id)
    stale = reopened.conversation_generation
    reopened.append_compaction_marker("summary", 1, 1)
    installed: list[object] = []
    reopened.on_entries_installed = installed.append
    duplicate = reopened.commit(command, stale)

    assert duplicate.duplicate is True
    assert duplicate.committed_entries[0].id == first_id
    assert duplicate.entries_to_install == ()
    assert installed == []
    assert sum(entry.id == first_id for entry in reopened.entries) == 1
    with pytest.raises(ConversationDedupeConflict):
        reopened.commit(
            AppendConversationRows((_warning("changed"),), key),
            reopened.conversation_generation,
        )
    reopened.close()


def test_dedupe_survives_compaction_marker(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="dedupe-compaction")
    key = ConversationDedupeKey("message", "compacted")
    command = AppendConversationRows(
        (ConversationRow("warning", {"message": "once"}),),
        dedupe_key=key,
    )
    first = store.commit(command, store.conversation_generation)
    marker = store.append_compaction_marker("summary", 1, 1)

    duplicate = store.commit(command, 0)

    assert duplicate.duplicate
    assert duplicate.committed_entries == first.committed_entries
    assert store.replay() == [*first.committed_entries, marker]


def test_dedupe_returns_inactive_branch_publication_without_copying_it(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    anchor = store.commit(
        AppendConversationRows((_warning("anchor"),)), store.conversation_generation
    ).committed_entries[0]
    key = ConversationDedupeKey("run-publication", "run-1:route-1")
    command = AppendConversationRows((_warning("publication"),), key)
    publication = store.commit(command, store.conversation_generation)
    store.commit(
        AppendConversationRows((_warning("new branch", parent_id=anchor.id),)),
        store.conversation_generation,
    )
    count = len(store.entries)

    duplicate = store.commit(command, store.conversation_generation)

    assert duplicate.duplicate is True
    assert duplicate.committed_entries == publication.committed_entries
    assert len(store.entries) == count
    assert store.entries[-1].data["message"] == "new branch"
    store.close()


@pytest.mark.asyncio
async def test_commit_async_defers_cancellation_until_delta_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await asyncio.to_thread(ConversationStore, tmp_path)
    store.bind_owner_thread()
    committed = threading.Event()
    release = threading.Event()
    original = store._run_committed_mutation

    def pause_after_commit(
        command: object,
        expected_generation: int,
        cancelled_before_write: threading.Event,
    ) -> object:
        result = original(command, expected_generation, cancelled_before_write)
        committed.set()
        assert release.wait(10)
        return result

    # test-contract: seam pauses after the durable commit but before resident install
    # test-contract: seam pauses after durable commit
    monkeypatch.setattr(store, "_run_committed_mutation", pause_after_commit)
    task = asyncio.create_task(
        store.commit_async(
            AppendConversationRows((_warning("durable"),)),
            store.conversation_generation,
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(committed.wait, 10), 11)
        task.cancel()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert [entry.data["message"] for entry in store.entries] == ["durable"]
        generation = store.conversation_generation
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await asyncio.to_thread(store.close)

    reopened = await asyncio.to_thread(
        ConversationStore, tmp_path, session_id=store.session_id
    )
    try:
        assert reopened.conversation_generation == generation
        assert [entry.data["message"] for entry in reopened.entries] == ["durable"]
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.asyncio
async def test_commit_async_cancelled_before_write_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await asyncio.to_thread(ConversationStore, tmp_path)
    store.bind_owner_thread()
    reached = threading.Event()
    release = threading.Event()
    cancellation_seen = asyncio.Event()

    def observe_cancellation(_policy: object) -> None:
        cancellation_seen.set()

    def pause_before_write() -> None:
        reached.set()
        assert release.wait(10)

    # test-contract: seam observes the shared cancellation-policy boundary
    monkeypatch.setattr(store, "_submitted_mutation_cancelled", observe_cancellation)
    # test-contract: seam pauses the worker before its first durable write
    monkeypatch.setattr(store, "_before_committed_mutation", pause_before_write)
    before = (
        store.path.read_bytes(),
        tuple(store.entries),
        store.conversation_generation,
    )
    task = asyncio.create_task(
        store.commit_async(
            AppendConversationRows((_warning("not written"),)),
            store.conversation_generation,
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(reached.wait, 10), 11)
        task.cancel()
        await await_or_fail(
            cancellation_seen.wait(), what="pre-write cancellation policy"
        )
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert (
            store.path.read_bytes(),
            tuple(store.entries),
            store.conversation_generation,
        ) == before
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("policy", ["abort_before_write", "finalize_once_submitted"])
@pytest.mark.parametrize(
    "phase",
    [
        "success",
        "queued",
        "submitted",
        "during_write",
        "committed",
        "publish",
        "publish_failure",
        "final_callback_failure",
        "duplicate",
    ],
)
@pytest.mark.asyncio
async def test_mutation_cancellation_policy_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    phase: str,
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    store = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (store.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    store.bind_owner_thread()
    address = ConversationAddress("project", "root", "root")
    publications: list[object] = []
    final_callbacks = 0

    def publish(delta: object) -> None:
        publications.append(delta)
        if phase in {"publish_failure", "final_callback_failure"}:
            raise RuntimeError("policy-table publication failure")

    def fail_final_callback() -> None:
        nonlocal final_callbacks
        final_callbacks += 1
        raise RuntimeError("policy-table final callback failure")

    store.on_entries_installed = publish
    reached = asyncio.Event()
    cancellation_seen = asyncio.Event()
    worker_reached = threading.Event()
    worker_release = threading.Event()
    lock_held = False

    def observe_cancellation(_policy: object) -> None:
        cancellation_seen.set()

    # test-contract: seam observes the shared cancellation-policy boundary
    monkeypatch.setattr(store, "_submitted_mutation_cancelled", observe_cancellation)

    async def submit(index: int = 1) -> object:
        if policy == "abort_before_write":
            return await commit_conversation_async(
                address,
                AppendConversationRows(
                    (_warning(f"public-{index}"),),
                    dedupe_key=ConversationDedupeKey("policy-table", str(index)),
                ),
            )
        if phase == "final_callback_failure":
            return await store.append_message_async(
                Message(MessageRole.ASSISTANT, [TextContent(f"legacy-{index}")]),
                on_persisted=fail_final_callback,
            )
        return await store.commit_delivery_input(
            PromptInput(f"legacy-{index}", MessageOrigin.AGENT_SEND, None)
        )

    if phase == "queued":
        await store._async_write_lock.acquire()
        lock_held = True

        def observe_queue() -> None:
            reached.set()

        # test-contract: seam observes real resident lock admission
        monkeypatch.setattr(store, "_async_write_queued", observe_queue)
    elif phase == "submitted":

        def pause_before_write() -> None:
            worker_reached.set()
            assert worker_release.wait(10)

        # test-contract: seam pauses after executor submission, before write ownership
        monkeypatch.setattr(store, "_before_committed_mutation", pause_before_write)
    elif phase == "during_write":

        def pause_during_write() -> None:
            worker_reached.set()
            assert worker_release.wait(10)

        # test-contract: seam pauses after write ownership becomes irrevocable
        monkeypatch.setattr(
            store, "_after_committed_mutation_started", pause_during_write
        )
    elif phase in {"committed", "publish"}:

        async def pause_owner_phase() -> None:
            reached.set()
            assert await await_or_fail(
                asyncio.to_thread(worker_release.wait, 10),
                what=f"{phase} cancellation release",
            )

        seam = (
            "_before_committed_install"
            if phase == "committed"
            else "_before_committed_publish"
        )
        # test-contract: seam pauses at the owner-side durable/install boundary
        monkeypatch.setattr(store, seam, pause_owner_phase)

    try:
        if phase in {
            "success",
            "publish_failure",
            "final_callback_failure",
            "duplicate",
        }:
            if (
                phase == "final_callback_failure"
                and policy == "finalize_once_submitted"
            ):
                with pytest.raises(
                    RuntimeError, match="policy-table final callback failure"
                ):
                    await submit()
            else:
                await submit()
            if phase == "duplicate":
                await submit()
        else:
            task = asyncio.create_task(submit())
            if phase in {"queued", "committed", "publish"}:
                await await_or_fail(
                    reached.wait(), what=f"{policy} {phase} cancellation point"
                )
            else:
                assert await await_or_fail(
                    asyncio.to_thread(worker_reached.wait, 10),
                    what=f"{policy} {phase} worker cancellation point",
                )
            task.cancel()
            if phase != "queued":
                await await_or_fail(
                    cancellation_seen.wait(),
                    what=f"{policy} {phase} policy cancellation",
                )
            worker_release.set()
            if lock_held:
                store._async_write_lock.release()
                lock_held = False
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(task, what=f"{policy} {phase} cancellation outcome")

        expected = 1
        if phase == "queued" or (
            policy == "abort_before_write" and phase == "submitted"
        ):
            expected = 0
        elif phase == "duplicate" and policy == "finalize_once_submitted":
            expected = 2
        resident = tuple(entry.to_dict() for entry in store.entries)
        assert len(resident) == expected
        assert len(publications) == expected
        assert final_callbacks == (
            1
            if phase == "final_callback_failure" and policy == "finalize_once_submitted"
            else 0
        )
    finally:
        worker_release.set()
        if lock_held:
            store._async_write_lock.release()
        await asyncio.to_thread(store.close)

    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert tuple(entry.to_dict() for entry in reopened.entries) == resident
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.parametrize("target_kind", ["resident", "transient"])
@pytest.mark.parametrize("policy", ["abort_before_write", "finalize_once_submitted"])
@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "cancel_before_gate",
        "cancel_during_repair",
        "cancel_during_command",
        "command_failure_after_recovery",
        "duplicate_after_recovery",
    ],
)
@pytest.mark.asyncio
async def test_mutation_target_cancellation_policy_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    target_kind: str,
    scenario: str,
) -> None:
    from zeta.core.store import _log

    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    store = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (store.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    store.bind_owner_thread()
    address = ConversationAddress("project", "root", "root")
    publications: list[object] = []
    store.on_entries_installed = publications.append
    worker_reached = threading.Event()
    worker_release = threading.Event()
    cancellation_seen = asyncio.Event()
    key = ConversationDedupeKey("policy-table", f"{policy}-{target_kind}")
    command_type = (
        _FinalizeAppendRows
        if policy == "finalize_once_submitted"
        else AppendConversationRows
    )

    def command(*, invalid: bool = False) -> AppendConversationRows[object]:
        row = (
            ConversationRow("warning", {"message": 3})
            if invalid
            else _warning("policy target")
        )
        return command_type((row,), key)

    if scenario == "duplicate_after_recovery":
        first = await commit_conversation_async(address, command())
        assert first.duplicate is False
        publications.clear()

    torn = scenario != "success"
    torn_bytes = b'{"seq":99,"id":"torn"'
    if torn:
        with store.path.open("ab") as handle:
            handle.write(torn_bytes)
        before = store.path.read_bytes()

    if target_kind == "transient":
        await asyncio.to_thread(store.close)

    def pause() -> None:
        worker_reached.set()
        assert worker_release.wait(10)

    if scenario == "cancel_before_gate":
        # test-contract: seam pauses both targets before the one shared write gate
        monkeypatch.setattr(_async_writes, "_before_mutation_write_gate", pause)
    elif scenario == "cancel_during_repair":
        original_repair = _log.repair_transcript

        def paused_repair(*args: object, **kwargs: object) -> object:
            pause()
            return original_repair(*args, **kwargs)

        # test-contract: seam pauses #884 recovery before its first repair step
        monkeypatch.setattr(_log, "repair_transcript", paused_repair)
    elif scenario == "cancel_during_command":
        original_execute = _async_writes.execute_mutation

        def paused_execute(*args: object, **kwargs: object) -> object:
            pause()
            return original_execute(*args, **kwargs)

        # test-contract: seam pauses the real command after materialization
        monkeypatch.setattr(_async_writes, "execute_mutation", paused_execute)

    # test-contract: seam observes the one shared cancellation-policy owner
    monkeypatch.setattr(
        _async_writes,
        "_mutation_cancellation_observed",
        lambda _policy: cancellation_seen.set(),
    )

    submitted = command(invalid=scenario == "command_failure_after_recovery")
    task = asyncio.create_task(commit_conversation_async(address, submitted))
    should_cancel = scenario.startswith("cancel_")
    result = None
    try:
        if should_cancel:
            assert await await_or_fail(
                asyncio.to_thread(worker_reached.wait, 10),
                what=f"{target_kind} {policy} {scenario} worker entry",
            )
            task.cancel()
            await await_or_fail(
                cancellation_seen.wait(),
                what=f"{target_kind} {policy} {scenario} cancellation policy",
            )
            worker_release.set()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(
                    task, what=f"{target_kind} {policy} {scenario} outcome"
                )
        elif scenario == "command_failure_after_recovery":
            with pytest.raises(ConversationIntegrityError, match="invalid payload"):
                await await_or_fail(task, what=f"{target_kind} {policy} command error")
        else:
            result = await await_or_fail(
                task, what=f"{target_kind} {policy} {scenario} outcome"
            )
    finally:
        worker_release.set()
        if not task.done():
            task.cancel()
            await await_or_fail(
                asyncio.gather(task, return_exceptions=True),
                what=f"{target_kind} {policy} {scenario} task cleanup",
            )

    aborted_at_gate = scenario == "cancel_before_gate" and policy == "abort_before_write"
    if aborted_at_gate:
        assert store.path.read_bytes() == before
    if scenario == "duplicate_after_recovery":
        assert result is not None and result.duplicate is True

    resident = tuple(entry.to_dict() for entry in store.entries)
    if target_kind == "resident" and not aborted_at_gate:
        assert len(publications) == 1
        if scenario != "success":
            assert publications == ["reset"]
    if target_kind == "resident":
        await asyncio.to_thread(store.close)

    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        durable = tuple(entry.to_dict() for entry in reopened.entries)
        if target_kind == "resident" and not aborted_at_gate:
            assert durable == resident
        if aborted_at_gate:
            assert not any(
                entry.data.get("message") == "policy target"
                for entry in reopened.entries
            )
        elif scenario == "command_failure_after_recovery":
            assert [entry.type for entry in reopened.entries] == ["warning"]
            assert reopened.entries[0].data["kind"] == "transcript_recovery"
        elif scenario == "duplicate_after_recovery":
            assert [entry.type for entry in reopened.entries] == ["warning", "warning"]
            assert sum(
                entry.data.get("message") == "policy target"
                for entry in reopened.entries
            ) == 1
        elif torn:
            assert [entry.type for entry in reopened.entries] == ["warning", "warning"]
            assert reopened.entries[0].data["kind"] == "transcript_recovery"
        else:
            assert [entry.data["message"] for entry in reopened.entries] == [
                "policy target"
            ]
    finally:
        await asyncio.to_thread(reopened.close)

    second = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert tuple(entry.to_dict() for entry in second.entries) == durable
    finally:
        await asyncio.to_thread(second.close)


@pytest.mark.asyncio
async def test_address_commit_uses_registered_resident_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    store = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (store.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    store.bind_owner_thread()
    address = ConversationAddress("project", "root", "root")
    try:
        delta = await commit_conversation_async(
            address, AppendConversationRows((_warning("resident"),))
        )
        assert delta.committed_entries[0] is not None
        assert [entry.data["message"] for entry in store.entries] == ["resident"]
    finally:
        await asyncio.to_thread(store.close)


@pytest.mark.asyncio
async def test_address_commit_owner_refusal_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    created = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (created.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    await asyncio.to_thread(created.close)
    before = (created.path.read_bytes(), created.path.stat().st_mtime_ns)
    context = multiprocessing.get_context("spawn")
    parent_control, child_control = context.Pipe()
    process = context.Process(
        target=_hold_conversation_owner,
        args=(str(sessions), "root", child_control),
    )
    process.start()
    try:
        assert (
            await asyncio.wait_for(asyncio.to_thread(parent_control.recv), 11)
            == "ready"
        )
        with pytest.raises(ConversationTargetOwned) as raised:
            await commit_conversation_async(
                ConversationAddress("project", "root", "root"),
                AppendConversationRows((_warning("refused"),)),
            )
        assert raised.value.owner_surface == "tui"
        assert (created.path.read_bytes(), created.path.stat().st_mtime_ns) == before
    finally:
        parent_control.send("release")
        await asyncio.wait_for(asyncio.to_thread(process.join, 10), 11)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 10)
        child_control.close()
        parent_control.close()
        process.close()
        del process, child_control, parent_control
        gc.collect()
    recovered = await commit_conversation_async(
        ConversationAddress("project", "root", "root"),
        AppendConversationRows((_warning("recovered"),)),
    )
    assert recovered.committed_entries[0].data["message"] == "recovered"


@pytest.mark.asyncio
async def test_address_commit_without_resident_replays_on_later_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    created = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (created.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    await asyncio.to_thread(created.close)

    delta = await commit_conversation_async(
        ConversationAddress("project", "root", "root"),
        AppendConversationRows((_warning("transient"),)),
    )

    assert delta.duplicate is False
    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert [entry.data["message"] for entry in reopened.entries] == ["transient"]
    finally:
        await asyncio.to_thread(reopened.close)


def test_sync_and_async_adapters_have_the_same_typed_delta(tmp_path: Path) -> None:
    sync = ConversationStore(tmp_path / "sync")
    async_store = ConversationStore(tmp_path / "async")
    try:
        command = AppendConversationRows((_warning("same"),), result="ok")
        sync_delta = sync.commit(command, sync.conversation_generation)
        async_store.bind_owner_thread()
        async_delta = asyncio.run(
            async_store.commit_async(command, async_store.conversation_generation)
        )
        assert async_delta.result == sync_delta.result == "ok"
        assert (
            async_delta.committed_entries[0].data
            == sync_delta.committed_entries[0].data
        )
        assert (
            async_delta.entries_to_install[0].data
            == sync_delta.entries_to_install[0].data
        )
        assert async_delta.generation_after == sync_delta.generation_after
    finally:
        sync.close()
        async_store.close()


def test_dedupe_scope_includes_namespace_and_conversation(tmp_path: Path) -> None:
    key = ConversationDedupeKey("scope", "same-id")
    first = ConversationStore(tmp_path / "first")
    second = ConversationStore(tmp_path / "second")
    try:
        first.commit(
            AppendConversationRows((_warning("first"),), key),
            first.conversation_generation,
        )
        other = second.commit(
            AppendConversationRows((_warning("second"),), key),
            second.conversation_generation,
        )
        assert other.duplicate is False
        different = first.commit(
            AppendConversationRows(
                (_warning("different"),), ConversationDedupeKey("other", "same-id")
            ),
            first.conversation_generation,
        )
        assert different.duplicate is False
        assert [entry.data["message"] for entry in first.entries] == [
            "first",
            "different",
        ]
    finally:
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_address_commit_cancellation_after_resident_durable_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    store = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (store.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    store.bind_owner_thread()
    committed = threading.Event()
    release = threading.Event()
    original = store._run_committed_mutation

    def pause_after_commit(
        command: object, generation: int, cancelled: threading.Event
    ) -> object:
        result = original(command, generation, cancelled)
        committed.set()
        assert release.wait(10)
        return result

    # test-contract: seam pauses after durable commit
    monkeypatch.setattr(store, "_run_committed_mutation", pause_after_commit)
    task = asyncio.create_task(
        commit_conversation_async(
            ConversationAddress("project", "root", "root"),
            AppendConversationRows((_warning("address durable"),)),
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(committed.wait, 10), 11)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert [entry.data["message"] for entry in store.entries] == ["address durable"]
        generation = store.conversation_generation
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await asyncio.to_thread(store.close)
    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert reopened.conversation_generation == generation
        assert [entry.data["message"] for entry in reopened.entries] == [
            "address durable"
        ]
    finally:
        await asyncio.to_thread(reopened.close)


def test_typed_mutation_deletion_gate(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    try:
        assert not hasattr(store, "_run_durable_write")
        delta = store.commit(
            AppendConversationRows((_warning("public"),), result="public"),
            store.conversation_generation,
        )
        assert delta.result == "public"
    finally:
        store.close()


def test_append_validation_matches_replay_validation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    try:
        before = store.path.read_bytes()
        with pytest.raises(ConversationIntegrityError, match="invalid payload"):
            store.commit(
                AppendConversationRows((ConversationRow("warning", {"message": 3}),)),
                store.conversation_generation,
            )
        assert store.path.read_bytes() == before
        with pytest.raises(ValueError, match="invalid payload"):
            store._append_row("warning", {"message": 3})
        assert store.path.read_bytes() == before
    finally:
        store.close()


@pytest.mark.asyncio
async def test_commit_async_cancel_before_lock_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await asyncio.to_thread(ConversationStore, tmp_path)
    store.bind_owner_thread()
    await store._async_write_lock.acquire()
    before = store.path.read_bytes()
    queued = asyncio.Event()

    def observe_queue() -> None:
        queued.set()

    # New-interface mutation: removing the signal times out at queue admission.
    # test-contract: seam observes real resident lock admission
    monkeypatch.setattr(store, "_async_write_queued", observe_queue)
    task = asyncio.create_task(
        store.commit_async(
            AppendConversationRows((_warning("not locked"),)),
            store.conversation_generation,
        )
    )
    try:
        await await_or_fail(queued.wait(), what="public commit lock queue admission")
        task.cancel()
        store._async_write_lock.release()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert store.path.read_bytes() == before
        assert store.entries == []
    finally:
        if store._async_write_lock.locked():
            store._async_write_lock.release()
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await asyncio.to_thread(store.close)


@pytest.mark.asyncio
async def test_commit_async_close_race_keeps_durable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await asyncio.to_thread(ConversationStore, tmp_path)
    store.bind_owner_thread()
    committed = threading.Event()
    release = threading.Event()
    original = store._run_committed_mutation

    def pause_after_commit(
        command: object, generation: int, cancelled: threading.Event
    ) -> object:
        result = original(command, generation, cancelled)
        committed.set()
        assert release.wait(10)
        return result

    # test-contract: seam pauses after durable commit
    monkeypatch.setattr(store, "_run_committed_mutation", pause_after_commit)
    task = asyncio.create_task(
        store.commit_async(
            AppendConversationRows((_warning("close race"),)),
            store.conversation_generation,
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(committed.wait, 10), 11)
        await asyncio.to_thread(store.close)
        release.set()
        await asyncio.wait_for(task, 10)
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
    reopened = await asyncio.to_thread(
        ConversationStore, tmp_path, session_id=store.session_id
    )
    try:
        assert [entry.data["message"] for entry in reopened.entries] == ["close race"]
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.asyncio
async def test_legacy_delta_patch_contains_only_declared_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await asyncio.to_thread(ConversationStore, tmp_path)
    store.bind_owner_thread()
    observed: list[object] = []
    original = store._run_committed_mutation

    def capture(command: object, generation: int, cancelled: threading.Event) -> object:
        result = original(command, generation, cancelled)
        observed.append(result.delta)
        return result

    # test-contract: seam captures the typed committed delta
    monkeypatch.setattr(store, "_run_committed_mutation", capture)
    try:
        await store.append_pending_prompt_async("declared")
        patch = observed[0]._resident_patch
        assert set(patch.values) <= set(patch.expected)
        assert "cwd" not in patch.values
    finally:
        await asyncio.to_thread(store.close)


@pytest.mark.asyncio
async def test_transient_address_retries_conflict_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    created = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (created.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    await asyncio.to_thread(created.close)
    calls = 0
    original = ConversationStore.commit
    loop_thread = threading.get_ident()

    def conflict_once(
        self: ConversationStore,
        command: object,
        generation: int,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        assert threading.get_ident() != loop_thread
        calls += 1
        if calls == 1:
            raise ConversationGenerationConflict(generation, generation + 1)
        return original(self, command, generation, **kwargs)

    monkeypatch.setattr(ConversationStore, "commit", conflict_once)
    delta = await commit_conversation_async(
        ConversationAddress("project", "root", "root"),
        AppendConversationRows((_warning("retried"),)),
    )
    assert calls == 2
    assert delta.committed_entries[0].data["message"] == "retried"


@pytest.mark.asyncio
async def test_address_concurrent_resident_commits_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    store = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (store.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    store.bind_owner_thread()
    started = threading.Event()
    release = threading.Event()
    original = store._run_committed_mutation
    first = True
    queued = asyncio.Event()

    def observe_queue() -> None:
        if store._address_commit_waiters == 8:
            queued.set()

    def hold_first(
        command: object, generation: int, cancelled: threading.Event
    ) -> object:
        nonlocal first
        result = original(command, generation, cancelled)
        if first:
            first = False
            started.set()
            # test-contract: seam holds the worker at an explicit event
            assert release.wait(10)
        return result

    # test-contract: seam holds the first resident commit
    monkeypatch.setattr(store, "_run_committed_mutation", hold_first)
    # test-contract: seam observes resident lock queue admission
    monkeypatch.setattr(store, "_address_commit_queued", observe_queue)
    address = ConversationAddress("project", "root", "root")
    tasks = [
        asyncio.create_task(
            commit_conversation_async(
                address, AppendConversationRows((_warning(f"resident-{i}"),))
            )
        )
        for i in range(8)
    ]
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 10), 11)
        await await_or_fail(queued.wait(), what="resident lock queue admission")
        assert store._address_commit_waiters == 8
        release.set()
        results = await await_or_fail(
            asyncio.gather(*tasks, return_exceptions=True),
            what="resident address commits",
        )
        assert not [result for result in results if isinstance(result, BaseException)]
        messages = [entry.data["message"] for entry in store.entries]
        assert sorted(messages) == [f"resident-{i}" for i in range(8)]
        generation = store.conversation_generation
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        await asyncio.to_thread(store.close)
    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert reopened.conversation_generation == generation
        assert sorted(entry.data["message"] for entry in reopened.entries) == [
            f"resident-{i}" for i in range(8)
        ]
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.asyncio
async def test_address_transient_waiter_retries_after_owner_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    created = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (created.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    await asyncio.to_thread(created.close)
    entered = threading.Event()
    release = threading.Event()
    waiting = asyncio.Event()
    original = ConversationStore.commit
    calls = 0

    def hold_owner(
        self: ConversationStore,
        command: object,
        generation: int,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            # test-contract: seam holds the worker at an explicit event
            assert release.wait(10)
        return original(self, command, generation, **kwargs)

    monkeypatch.setattr(ConversationStore, "commit", hold_owner)
    # test-contract: seam observes transient reservation admission
    monkeypatch.setattr(
        mutation_module, "_reservation_waiting", lambda _path: waiting.set()
    )
    address = ConversationAddress("project", "root", "root")
    owner = asyncio.create_task(
        commit_conversation_async(address, AppendConversationRows((_warning("owner"),)))
    )
    waiter = asyncio.create_task(
        commit_conversation_async(
            address, AppendConversationRows((_warning("waiter"),))
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 11)
        await await_or_fail(waiting.wait(), what="transient reservation admission")
        assert not waiter.done()
        release.set()
        outcomes = await await_or_fail(
            asyncio.gather(owner, waiter, return_exceptions=True),
            what="transient owner and waiter",
        )
        assert not [
            outcome for outcome in outcomes if isinstance(outcome, BaseException)
        ]
    finally:
        release.set()
        for task in (owner, waiter):
            if not task.done():
                task.cancel()
        await asyncio.wait_for(
            asyncio.gather(owner, waiter, return_exceptions=True), 10
        )
    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert sorted(entry.data["message"] for entry in reopened.entries) == [
            "owner",
            "waiter",
        ]
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.asyncio
async def test_transient_waiter_cancellation_does_not_release_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    created = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (created.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    await asyncio.to_thread(created.close)
    entered = threading.Event()
    release = threading.Event()
    waiting = asyncio.Event()
    original = ConversationStore.commit
    calls = 0

    def hold_owner(
        self: ConversationStore,
        command: object,
        generation: int,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(10)
        return original(self, command, generation, **kwargs)

    monkeypatch.setattr(ConversationStore, "commit", hold_owner)
    # test-contract: seam observes transient reservation admission
    monkeypatch.setattr(
        mutation_module, "_reservation_waiting", lambda _path: waiting.set()
    )
    address = ConversationAddress("project", "root", "root")
    owner = asyncio.create_task(
        commit_conversation_async(address, AppendConversationRows((_warning("owner"),)))
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 11)
        waiter = asyncio.create_task(
            commit_conversation_async(
                address, AppendConversationRows((_warning("cancelled"),))
            )
        )
        await await_or_fail(waiting.wait(), what="transient reservation admission")
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 10)
        waiting.clear()
        third = asyncio.create_task(
            commit_conversation_async(
                address, AppendConversationRows((_warning("third"),))
            )
        )
        await await_or_fail(waiting.wait(), what="transient reservation admission")
        assert not third.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(owner, third), 20)
    finally:
        release.set()
        for task in (owner, waiter):
            if not task.done():
                task.cancel()
        await asyncio.wait_for(
            asyncio.gather(owner, waiter, return_exceptions=True), 10
        )
    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert sorted(entry.data["message"] for entry in reopened.entries) == [
            "owner",
            "third",
        ]
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.asyncio
async def test_transient_owner_failure_wakes_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    created = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (created.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    await asyncio.to_thread(created.close)
    entered = threading.Event()
    release = threading.Event()
    waiting = asyncio.Event()
    original = ConversationStore.commit
    calls = 0

    def fail_owner(
        self: ConversationStore,
        command: object,
        generation: int,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(10)
            raise ConversationIntegrityError("forced owner failure")
        return original(self, command, generation, **kwargs)

    monkeypatch.setattr(ConversationStore, "commit", fail_owner)
    # test-contract: seam observes transient reservation admission
    monkeypatch.setattr(
        mutation_module, "_reservation_waiting", lambda _path: waiting.set()
    )
    address = ConversationAddress("project", "root", "root")
    owner = asyncio.create_task(
        commit_conversation_async(
            address, AppendConversationRows((_warning("failed"),))
        )
    )
    tasks = [owner]
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 11)
        waiter = asyncio.create_task(
            commit_conversation_async(
                address, AppendConversationRows((_warning("survivor"),))
            )
        )
        tasks.append(waiter)
        await await_or_fail(waiting.wait(), what="transient reservation admission")
        assert not waiter.done()
        release.set()
        with pytest.raises(ConversationIntegrityError, match="forced owner failure"):
            await await_or_fail(owner, what="failing transient reservation owner")
        await await_or_fail(waiter, what="waiter after transient owner failure")
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert [entry.data["message"] for entry in reopened.entries] == ["survivor"]
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.asyncio
async def test_transient_owner_cancellation_after_worker_start_defers_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    sessions = home / "sessions"
    monkeypatch.setenv("ZETA_HOME", str(home))
    created = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    (created.session_dir / "meta.json").write_text(
        json.dumps({"project_id": "project"}), encoding="utf-8"
    )
    await asyncio.to_thread(created.close)
    entered = threading.Event()
    release = threading.Event()
    waiting = asyncio.Event()
    cancellation_seen = asyncio.Event()
    calls = 0

    def hold_owner_before_gate() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(10)

    # test-contract: seam pauses the reservation owner before shared write ownership
    monkeypatch.setattr(
        _async_writes, "_before_mutation_write_gate", hold_owner_before_gate
    )
    # test-contract: seam observes transient reservation admission
    monkeypatch.setattr(
        mutation_module, "_reservation_waiting", lambda _path: waiting.set()
    )
    # test-contract: seam observes cancellation before releasing the paused worker
    monkeypatch.setattr(
        _async_writes,
        "_mutation_cancellation_observed",
        lambda _policy: cancellation_seen.set(),
    )
    address = ConversationAddress("project", "root", "root")
    owner = asyncio.create_task(
        commit_conversation_async(address, AppendConversationRows((_warning("owner"),)))
    )
    waiter: asyncio.Task[object] | None = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 11)
        waiter = asyncio.create_task(
            commit_conversation_async(
                address, AppendConversationRows((_warning("waiter"),))
            )
        )
        await await_or_fail(waiting.wait(), what="transient reservation admission")
        owner.cancel()
        await await_or_fail(
            cancellation_seen.wait(), what="transient owner cancellation policy"
        )
        assert not owner.done()
        assert waiter is not None and not waiter.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(owner, what="cancelled transient reservation owner")
        await await_or_fail(waiter, what="waiter after transient owner cancellation")
    finally:
        release.set()
        tasks = (owner,) if waiter is None else (owner, waiter)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
    reopened = await asyncio.to_thread(ConversationStore, sessions, session_id="root")
    try:
        assert [entry.data["message"] for entry in reopened.entries] == ["waiter"]
    finally:
        await asyncio.to_thread(reopened.close)

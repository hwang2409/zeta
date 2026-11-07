from __future__ import annotations

import asyncio
import gc
import json
import multiprocessing
import threading
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

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
    commit_conversation_async,
)


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
            await asyncio.gather(task, return_exceptions=True)
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
    cancellation_seen: threading.Event | None = None
    original_start = store._start_committed_mutation

    def capture_cancellation(
        command: object, generation: int, cancelled: threading.Event
    ):
        nonlocal cancellation_seen
        cancellation_seen = cancelled
        return original_start(command, generation, cancelled)

    def pause_before_write() -> None:
        reached.set()
        assert release.wait(10)

    # test-contract: seam records the worker's cancellation gate
    monkeypatch.setattr(store, "_start_committed_mutation", capture_cancellation)
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
        assert cancellation_seen is not None
        assert await asyncio.wait_for(asyncio.to_thread(cancellation_seen.wait, 10), 11)
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
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(store.close)


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
        process.close()
        child_control.close()
        parent_control.close()
        del process, child_control, parent_control
        gc.collect()


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
            await asyncio.gather(task, return_exceptions=True)
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
        assert not hasattr(store, "_to_thread_durable")
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

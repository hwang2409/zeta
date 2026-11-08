"""Composer history finder and in-memory session history behavior."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from zeta.tui.session_prompt_history import SessionPromptHistory
from zeta.tui.transcript import FinderControl
from zeta.tui.transcript.finder_runtime import FinderRuntimeTarget
from zeta.tui.transcript.history_finder import HistoryFinder, build_history_candidates
from zeta.tui.transcript.transcript_finder import TranscriptFinderMixin


def test_finders_implement_the_runtime_interface() -> None:
    required = {
        name
        for name in vars(FinderRuntimeTarget)
        if not name.startswith("_")
    }
    for finder in (HistoryFinder, TranscriptFinderMixin):
        assert required <= set(dir(finder)), (
            finder.__name__,
            required - set(dir(finder)),
        )


def _assert_runtime_finder_contract(source: str) -> None:
    tree = ast.parse(source)
    runtime_attributes: set[str] = set()
    forbidden_probe: str | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            is_probe = isinstance(node.func, ast.Name) and node.func.id in {
                "getattr",
                "hasattr",
            }
            if is_probe and node.args:
                target = node.args[0]
                is_finder_holder = (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr.endswith("finder")
                )
                if (
                    isinstance(target, ast.Name)
                    and target.id == "finder"
                ) or (
                    isinstance(target, ast.Call)
                    and isinstance(target.func, ast.Attribute)
                    and isinstance(target.func.value, ast.Name)
                    and target.func.value.id == "self"
                    and target.func.attr == "_finder_target"
                ) or is_finder_holder:
                    forbidden_probe = node.func.id
        if not isinstance(node, ast.Attribute):
            continue
        value = node.value
        is_finder = isinstance(value, ast.Name) and value.id == "finder"
        is_finder_target = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "self"
            and value.func.attr == "_finder_target"
        )
        is_active_finder = (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
            and value.attr in {"_active_finder", "_history_finder"}
        )
        if is_finder or is_finder_target or is_active_finder:
            runtime_attributes.add(node.attr)

    assert forbidden_probe is None, (
        f"call the FinderRuntimeTarget method directly, not {forbidden_probe}"
    )
    protocol_attributes = {
        name for name in vars(FinderRuntimeTarget) if not name.startswith("_")
    }
    missing = runtime_attributes - protocol_attributes
    assert not missing, (
        f"missing finder method(s) {sorted(missing)}; "
        "add them to FinderRuntimeTarget"
    )


def test_runtime_finder_attributes_are_in_the_protocol() -> None:
    runtime_path = (
        Path(__file__).parents[1]
        / "src/zeta/tui/transcript/finder_runtime.py"
    )
    _assert_runtime_finder_contract(runtime_path.read_text())


def test_runtime_guard_rejects_finder_holder_probes() -> None:
    with pytest.raises(AssertionError, match="call the FinderRuntimeTarget method directly, not getattr"):
        _assert_runtime_finder_contract(
            'def probe(self):\n    return getattr(self._active_finder, "missing", None)\n'
        )


def test_runtime_guard_rejects_finder_attribute_probes() -> None:
    runtime_path = (
        Path(__file__).parents[1]
        / "src/zeta/tui/transcript/finder_runtime.py"
    )
    source = runtime_path.read_text()
    mutated = source.replace(
        "finder.finder_needs_ranking()",
        'getattr(finder, "not_in_protocol", lambda: False)()',
        1,
    )
    with pytest.raises(AssertionError, match="call the FinderRuntimeTarget method directly, not getattr"):
        _assert_runtime_finder_contract(mutated)


def _finder(history: SessionPromptHistory, path: str, *, on_accept=None) -> HistoryFinder:
    return HistoryFinder(lambda: None, lambda: history.snapshot(path), on_accept=on_accept)


async def _open(finder: HistoryFinder, buffer: Buffer) -> None:
    request = finder.open_finder(buffer)
    assert request is not None
    batches = [batch async for batch, _complete in finder.build_finder_candidate_batches(request)]
    assert len(batches) == 1
    assert finder.finder_publish_candidate_batch(
        request, batches[0], complete=True
    )
    result = finder.finder_rank()
    assert result is not None
    assert finder.finder_publish(result)


def test_candidates_are_newest_first_deduplicated_and_multiline() -> None:
    snapshot = build_history_candidates(("older match", "two\nlines", "older match   ", "newest match"))
    assert [candidate.text for candidate in snapshot.candidates] == [
        "newest match", "older match", "two ⏎ lines"
    ]
    assert snapshot.candidates[2].preview == ("two", "lines")


def test_session_history_is_scoped_and_navigation_restores_draft() -> None:
    history = SessionPromptHistory()
    history.replace("a", ("a one", "a two"))
    history.replace("b", ("b one",))
    assert history.older("a", Document("draft")).text == "a two"
    assert history.older("a", Document("a two")).text == "a one"
    assert history.older("b", Document("draft")).text == "b one"
    assert history.newer("b").text == "draft"


@pytest.mark.asyncio
async def test_finder_uses_viewed_session_snapshot() -> None:
    history = SessionPromptHistory()
    history.replace("a", ("a prompt",))
    history.replace("b", ("b prompt",))
    finder = _finder(history, "b")
    await _open(finder, Buffer())
    state = finder.finder_state()
    assert state is not None
    assert [row.candidate.text for row in state.rows] == ["b prompt"]


@pytest.mark.asyncio
async def test_accept_and_cancel_preserve_draft() -> None:
    history = SessionPromptHistory()
    history.replace("a", ("selected prompt",))
    buffer = Buffer()
    buffer.text = ""
    finder = _finder(history, "a")
    await _open(finder, buffer)
    assert finder.finder_accept()
    assert buffer.text == "selected prompt"


@pytest.mark.asyncio
async def test_empty_history_renders_empty_state() -> None:
    finder = _finder(SessionPromptHistory(), "missing")
    await _open(finder, Buffer())
    content = FinderControl(finder.finder_state).create_content(80, 20)
    rendered = "".join(
        fragment[1]
        for line in range(content.line_count)
        for fragment in content.get_line(line)
    )
    assert "no history" in rendered

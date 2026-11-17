from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from zeta.server.handlers.registry import HANDLERS, RESOLVE
from zeta.server.protocol import MethodDescriptor, ProtocolError
from zeta.server.session_controllers import SessionControllers

ROOT = Path(__file__).parents[1] / "src" / "zeta" / "server"
SCANNED = [*sorted(ROOT.rglob("*.py")), *sorted((ROOT.parent / "remote").rglob("*.py"))]
# the only request entries that resolve a record, with their one resolver;
# everything below them takes the record as an argument
RESOLVER_ENTRIES = {
    "handlers/registry.py:dispatch": "owned",
    "server.py:_Client._route_request": "owned",
    "session_replay/__init__.py:join_history_and_subscribe": "selected",
    "handlers/registry.py:child_run_events": "selected",
    "handlers/registry.py:path_complete": "selected",
    "handlers/registry.py:child_run_list": "selected",
    "connections.py:ConnectionRegistry.read_tree": "selected",
    "connections.py:ConnectionRegistry.read_history": "selected",
    "connections.py:ConnectionRegistry.read_child_run_history": "selected",
}
# `.owned`/`.selected` attributes that are not the resolvers
OTHER_ATTRIBUTES = {("accounts.py", "snapshot.selected")}
# the only callers of `retire`, each through the owner
RETIRE_CALLERS = {
    "parking.py:ParkedPairs.leave",
    "parking.py:ParkedPairs.close",
    "handlers/lifecycle.py:_session_swap",
}
# close state that only `SessionControllers` holds
CLOSE_STATE = {"closing", "close_task", "closing_session_id", "_closing", "_close_pair", "closing_tasks"}
RESOLVERS = {"owned", "selected"}
# the only places a request use is handed to an existing completion: the worker
# read and the approval settlement; delivery and setup tasks use `SessionUse.run`
TRANSFER_ENTRIES = {
    "connections.py:ConnectionRegistry.run_read",
    "control.py:ServeControl.resolve_approval",
}
# use and completion state, each written only by its owner module
USE_OWNER = "session_controllers.py"
COMPLETION_OWNER = "thread_call.py"


def _functions() -> list[tuple[str, ast.AST, Path]]:
    """``(path:qualified name, function, file)`` for every function, nested ones
    named under their parent."""
    found: list[tuple[str, ast.AST, Path]] = []

    def visit(node: ast.AST, rel: str, scope: str, file: Path) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, rel, f"{scope}{child.name}.", file)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append((f"{rel}:{scope}{child.name}", child, file))
                visit(child, rel, f"{scope}{child.name}.", file)
            else:
                visit(child, rel, scope, file)

    for file in SCANNED:
        if file.name != "session_controllers.py":
            visit(ast.parse(file.read_text()), _rel(file), "", file)
    return found


def _rel(file: Path) -> str:
    return (
        file.relative_to(ROOT).as_posix()
        if ROOT in file.parents
        else "../" + file.relative_to(ROOT.parent).as_posix()
    )


def _own_nodes(function: ast.AST) -> list[ast.AST]:
    """Nodes of ``function`` outside its nested functions and classes."""
    nodes: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def _resolver_violations() -> list[str]:
    """Every use of a resolver that is not one direct ``<x>.controllers.<kind>(...)``
    call in its entry, plus every entry whose count differs from one."""
    problems: list[str] = []
    counts = dict.fromkeys(RESOLVER_ENTRIES, 0)
    module_level: list[tuple[Path, ast.AST]] = [
        (file, ast.parse(file.read_text())) for file in SCANNED if file.name != "session_controllers.py"
    ]
    for file, tree in module_level:
        rel = _rel(file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in RESOLVERS:
                problems.append(f"{rel}:{node.lineno}: resolver name as a string")
    for name, function, file in _functions():
        rel = name.split(":", 1)[0]
        nodes = _own_nodes(function)
        callees = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
        scoped = {
            id(item.context_expr.func)
            for node in nodes
            if isinstance(node, (ast.With, ast.AsyncWith))
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
        }
        lambdas = [node for node in nodes if isinstance(node, ast.Lambda)]
        in_lambda = {id(child) for item in lambdas for child in ast.walk(item)}
        for node in nodes:
            if not (isinstance(node, ast.Attribute) and node.attr in RESOLVERS):
                continue
            where = f"{name}:{node.lineno}: {ast.unparse(node)}"
            if (rel, ast.unparse(node)) in OTHER_ATTRIBUTES:
                continue
            if not (
                id(node) in callees
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "controllers"
                and isinstance(node.ctx, ast.Load)
            ):
                problems.append(f"{where}: not a direct controllers.{node.attr}(...) call")
            elif id(node) in in_lambda:
                problems.append(f"{where}: resolver inside a lambda")
            elif RESOLVER_ENTRIES.get(name) != node.attr:
                problems.append(f"{where}: resolver outside its request entry")
            elif id(node) not in scoped:
                problems.append(f"{where}: resolved use not held by a with block")
            else:
                counts[name] += 1
    problems += [
        f"{name}: {count} resolver calls, expected 1" for name, count in counts.items() if count != 1
    ]
    return problems


def test_record_indexes_have_one_owner() -> None:
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                attrs = [child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)]
                text = ast.unparse(node)
                if (
                    any(
                        name in attrs
                        for name in (
                            "_parked",
                            "_attached",
                            "_records",
                            "_controller_mode",
                            "_controller_members",
                            "_hello_generations",
                            "_grace_session_id",
                        )
                    )
                    and (
                        "SessionController" in text
                        or "ServeControl" in text
                        or "_parked" in attrs
                        or "_controller_mode" in attrs
                        or "_controller_members" in attrs
                        or "_hello_generations" in attrs
                        or "_grace_session_id" in attrs
                    )
                ):
                    assert path.name == "session_controllers.py", f"{path}:{node.lineno}: {text}"


def test_server_has_only_private_test_aliases() -> None:
    tree = ast.parse((ROOT / "server.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ZetaServer")
    names = {node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "control" not in names
    assert "runtime" not in names
    assert "__getattr__" not in names
    assert "_control" + "_for_test" not in names
    assert "_runtime" + "_for_test" not in names


def test_every_method_has_a_resolve_policy() -> None:
    assert RESOLVE.keys() == HANDLERS.keys() == set(MethodDescriptor)


def test_records_are_resolved_once_at_request_entries() -> None:
    assert _resolver_violations() == []


def test_use_and_completion_state_have_one_owner() -> None:
    """Use sets, generations, and use construction live only in
    ``session_controllers.py``; worker completion tracking only in
    ``thread_call.py``; a use is handed to a completion only at a transfer entry."""
    problems: list[str] = []
    for file in sorted(ROOT.parent.rglob("*.py")):
        rel = file.relative_to(ROOT.parent).as_posix()
        serve = file in SCANNED
        for node in ast.walk(ast.parse(file.read_text())):
            text = ast.unparse(node) if isinstance(node, ast.expr | ast.stmt) else ""
            if file.name != USE_OWNER and (
                (isinstance(node, ast.Call) and text.startswith(("SessionUse(", "SessionController(")))
                or (
                    serve
                    and isinstance(node, ast.Attribute)
                    and node.attr in {"uses", "_generations", "_drains", "_settle", "generation"}
                    and not (node.attr == "generation" and isinstance(node.ctx, ast.Load))
                )
            ):
                problems.append(f"{rel}:{node.lineno}: {text}")
            if file.name != COMPLETION_OWNER and (
                (isinstance(node, ast.Call) and text.startswith("ThreadCallHandle("))
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"set_result", "set_exception", "cancel"}
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "completed"
                )
            ):
                problems.append(f"{rel}:{node.lineno}: {text}")
    transfers = [
        name
        for name, function, _file in _functions()
        for node in _own_nodes(function)
        if isinstance(node, ast.Attribute) and node.attr == "transfer_to"
    ]
    assert sorted(transfers) == sorted(TRANSFER_ENTRIES)
    assert problems == []


def test_every_resident_close_goes_through_retire() -> None:
    """No control or runtime closes outside ``SessionControllers``, the one
    close coordinator; every ``retire`` call goes
    through the owner from a named caller; and no module outside the owner
    holds close state (closing flags, close tasks, closing maps)."""
    problems: list[str] = []
    for name, function, _file in _functions():
        for node in _own_nodes(function):
            if isinstance(node, ast.Attribute) and node.attr in CLOSE_STATE:
                problems.append(f"{name}:{node.lineno}: close state {ast.unparse(node)}")
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            text = ast.unparse(node.func)
            if text.endswith(("control.close", "runtime.close")):
                problems.append(f"{name}:{node.lineno}: {text} outside the close coordinator")
            if node.func.attr == "retire" and (
                name not in RETIRE_CALLERS or not text.endswith("controllers.retire")
            ):
                problems.append(f"{name}:{node.lineno}: {text}")
    for file in SCANNED:
        if file.name in {"parking.py", "registry.py"} or file.name == "server.py":
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Attribute) and target.attr == "closing"
                    for target in node.targets
                ):
                    problems.append(f"{_rel(file)}:{node.lineno}: writes a closing flag")
    assert problems == []


def test_a_closing_record_stays_resident_until_its_close_ends() -> None:
    """Identity and generation survive transitions; ``retire`` stops resolution
    at once, keeps the record resident while a transferred use delays the close, closes
    control then runtime once, and only then drops the record."""
    events: list[str] = []

    class Runtime:
        session_id = "s"
        opened = object()

        async def close(self) -> None:
            events.append("runtime")

    class Control:
        def __init__(self) -> None:
            self.runtime = Runtime()

        async def close(self) -> None:
            events.append("control")

        async def wait_for_approval_settlement(self, timeout: float | None) -> None:
            events.append("settled")

    async def main() -> None:
        control = Control()
        owner = SessionControllers(control)  # type: ignore[arg-type]
        record = owner.attached()
        generation = record.generation
        assert owner.attach(record) is record
        assert owner.park(record) is record
        assert owner.unpark(record) is record
        assert owner.attach(record) is record
        assert owner.register(Control()).generation > generation  # type: ignore[arg-type]
        with owner.owned(None) as use:
            assert (use.record, use.generation, record.uses) == (record, generation, {use})
        assert record.uses == set()
        owner.park(record)
        held = owner.selected(None, "s")
        completion = asyncio.get_running_loop().create_future()
        held.transfer_to(completion)
        close = owner.retire(record)
        assert owner.retire(record) is close
        assert (record.closing, record.generation) == (True, generation)
        with pytest.raises(ProtocolError):
            owner.selected(None, "s")
        for _ in range(5):
            await asyncio.sleep(0)
        assert (events, record in owner.records()) == ([], True)
        completion.set_result(None)
        await await_or_fail(owner.wait_closed("s"), what="close after the use settles")
        assert events == ["control", "runtime", "settled"]
        assert record not in owner.records()

    asyncio.run(main())


def test_a_transferred_use_settles_only_with_its_completion() -> None:
    """Release after transfer is a no-op; the completion's settle (result, error,
    or cancel) releases the use once, and only then does the record's close
    start."""

    closed: list[str] = []

    class Runtime:
        opened = None

        async def close(self) -> None:
            closed.append("runtime")

    class Control:
        def __init__(self) -> None:
            self.runtime = Runtime()

        async def close(self) -> None:
            closed.append("control")

        async def wait_for_approval_settlement(self, timeout: float | None) -> None:
            pass

    async def main() -> None:
        owner = SessionControllers(Control())  # type: ignore[arg-type]
        for settle in ("result", "exception", "cancel"):
            closed.clear()
            use = owner.register(Control())  # type: ignore[arg-type]
            record = use.record
            completion = asyncio.get_running_loop().create_future()
            use.transfer_to(completion)
            with pytest.raises(RuntimeError):
                use.transfer_to(completion)
            use.release()
            close = owner.retire(record)
            for _ in range(5):
                await asyncio.sleep(0)
            assert (record.uses, closed, close.done()) == ({use}, [], False), settle
            if settle == "result":
                completion.set_result(None)
            elif settle == "exception":
                completion.set_exception(ValueError())
            else:
                completion.cancel()
            await await_or_fail(close, what=f"close after {settle}")
            assert (record.uses, closed) == (set(), ["control", "runtime"])
            if settle == "exception":
                completion.exception()

    asyncio.run(main())

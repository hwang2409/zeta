from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract
from zeta.core.slash import create_slash_registry
from zeta.server.protocol import (
    ErrorDescriptor,
    EventDescriptor,
    FrameCodec,
    MethodDescriptor,
    SlashErrorCodeDescriptor,
    SlashResultKindDescriptor,
)
from zeta.skills import SkillCatalog

ROOT = Path(__file__).parents[1]
PROTOCOL_DOC = ROOT / "docs" / "serve-protocol.md"
PROTOCOL_MODULE = ROOT / "src/zeta/server/protocol.py"
PROTOCOL_SOURCES = (ROOT / "src/zeta/server", ROOT / "src/zeta/protocol")
RAW_FRAME_BUILDERS = {
    "_bounded_json",
    "_error_notification",
    "_json_bytes",
    "_minimal_error",
}


def _section(document: str, heading: str, next_heading: str) -> str:
    return document.split(heading, 1)[1].split(next_heading, 1)[0]


def _documented_errors(document: str) -> set[tuple[str, int, str | None, str]]:
    section = _section(document, "Error descriptors are:", "## normative wire schema")
    pattern = re.compile(
        r"^\| `([a-z0-9_]+)` \| `(-\d+)` \| (?:`([a-z0-9_]+)`|none) \| ([^|]+?) \|$",
        re.MULTILINE,
    )
    return {
        (name, int(code), data_code or None, meaning)
        for name, code, data_code, meaning in pattern.findall(section)
    }


def _documented_events(document: str) -> set[str]:
    section = _section(document, "Event fields are:", "The `retry` data")
    names: set[str] = set()
    for row in section.splitlines():
        if row.startswith("| `"):
            event_cell = row.split("|", 2)[1].split(" (feature", 1)[0]
            names.update(re.findall(r"`([a-z][a-z0-9_]*)`", event_cell))
    return names


def _documented_slash_result_kinds(document: str) -> set[str]:
    section = _section(
        document,
        "Returns a `kind` discriminator:",
        "The shared slash registry",
    )
    return set(re.findall(r"^  - `([a-z_]+)`:", section, re.MULTILINE))


def _documented_slash_error_codes(document: str) -> set[str]:
    section = _section(
        document,
        "Structured slash error codes are:",
        "The shared slash registry",
    )
    return set(re.findall(r"^\| `([a-z_]+)` \|", section, re.MULTILINE))


def test_method_registry_matches_documented_table() -> None:
    document = PROTOCOL_DOC.read_text(encoding="utf-8")
    section = _section(document, "## method table", "## common JSON-RPC shapes")
    rows = re.findall(
        r"^\| `([^`]+)` \| `([^`]+)` \| `([^`]+)` \| `([^`]+)` \|$",
        section,
        re.MULTILINE,
    )
    assert rows == [
        (item.wire, item.since or "1.0", item.feature or "none", item.access.value)
        for item in MethodDescriptor
    ], "document every method in descriptor order"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_real_method_conformance_matrix(transport: str, tmp_path: Path) -> None:
    document = PROTOCOL_DOC.read_text(encoding="utf-8")
    section = _section(document, "## method table", "## common JSON-RPC shapes")
    table = [
        (name, since, feature, access)
        for name, since, feature, access in re.findall(
            r"^\| `([^`]+)` \| `([^`]+)` \| `([^`]+)` \| `([^`]+)` \|$",
            section,
            re.MULTILINE,
        )
    ]
    for version, features in (
        ("1.0", []),
        ("1.1", []),
        ("1.1", ["abort_scope", "delivery_id", "ping"]),
        ("1.2", ["abort_scope", "delivery_id", "ping", "serve_observers"]),
    ):
        contract = await ServeContract.start(
            tmp_path / f"{version}-{len(features)}", transport
        )  # type: ignore[arg-type]
        controller = observer = None
        try:
            controller = await contract.connect("controller")
            hello = await controller.request(
                "controller-hello",
                "hello",
                {"protocol_version": version, "features": features},
            )
            advertised = hello["result"]["capabilities"]["requests"]
            expected = [
                name
                for name, since, feature, _access in table
                if (since == "1.0" or version != "1.0")
                and (feature == "none" or feature in features)
            ]
            assert advertised == expected
            observer = await contract.connect("observer")
            observer_hello = await observer.request(
                "observer-hello",
                "hello",
                {"protocol_version": version, "role": "observer", "features": features},
            )
            if "serve_observers" not in features:
                assert observer_hello["error"]["code"] == -32602
                continue
            observer_expected = [
                name
                for name, since, feature, access in table
                if access == "read"
                and (since == "1.0" or version != "1.0")
                and (feature == "none" or feature in features)
            ]
            assert observer_hello["result"]["capabilities"]["requests"] == sorted(
                observer_expected
            )
            for method in ("send", "clear_steering"):
                reply = await observer.request(f"observer-{method}", method)
                assert reply["error"]["code"] == -32008
            for method in ("clear_steering", "delivery_status"):
                reply = await controller.request(f"w4-{method}", method)
                if method == "clear_steering" and "abort_scope" in features:
                    continue
                if method == "delivery_status" and "delivery_id" in features:
                    continue
                assert reply["error"]["code"] == -32601
        finally:
            await contract.close()




def test_builtin_busy_policy_is_explicit_and_read_only() -> None:
    registry = create_slash_registry(skill_catalog=SkillCatalog(()), surface="serve")
    safe = {
        command.name
        for command in registry.builtin_commands
        if command.serve is not None and command.serve.can_run_while_busy("")
    }
    assert safe == {
        "accounts",
        "help",
        "inbox",
        "memory",
        "model",
        "project",
        "status",
        "vim",
    }
    assert all(
        command.serve is not None and command.serve.mutates is not None
        for command in registry.builtin_commands
        if command.name in safe
    )
    assert registry.command_for("memory").serve.can_run_while_busy("")
    assert not registry.command_for("memory").serve.can_run_while_busy("undo")
    assert not registry.command_for("plan").serve.can_run_while_busy("")
    assert not registry.command_for("plan").serve.can_run_while_busy("on")
    assert not registry.command_for("plan").serve.can_run_while_busy("write a plan")
    assert not registry.command_for("implement").serve.can_run_while_busy("")





def test_protocol_registry_matches_documented_errors_and_events() -> None:
    document = PROTOCOL_DOC.read_text(encoding="utf-8")
    documented_errors = _documented_errors(document)
    registered_errors = {
        (
            descriptor.name.lower(),
            descriptor.code,
            descriptor.data_code,
            descriptor.meaning,
        )
        for descriptor in ErrorDescriptor
    }
    assert documented_errors == registered_errors, (
        "add exactly one descriptor in src/zeta/server/protocol.py and one matching "
        "error row in docs/serve-protocol.md"
    )
    assert _documented_events(document) == {event.value for event in EventDescriptor}, (
        "add exactly one event in src/zeta/server/protocol.py and one matching event row "
        "in docs/serve-protocol.md"
    )
    assert _documented_slash_result_kinds(document) == {
        kind.value for kind in SlashResultKindDescriptor
    }, "register and document each slash_run result kind"
    assert _documented_slash_error_codes(document) == {
        code.value for code in SlashErrorCodeDescriptor
    }, "register and document each structured slash_run error code"


def _annotation_name(argument: ast.arg) -> str | None:
    return argument.annotation.id if isinstance(argument.annotation, ast.Name) else None


def _class_methods(
    class_name: str,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(PROTOCOL_MODULE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_frame_codec_is_the_typed_error_and_notification_chokepoint() -> None:
    methods = _class_methods("FrameCodec")
    protocol_error_descriptor = _class_methods("ProtocolError")["__init__"].args.args[1]
    error_descriptor = methods["error_response"].args.args[2]
    event_descriptor = methods["notification"].args.args[1]
    assert _annotation_name(protocol_error_descriptor) == "ErrorDescriptor"
    assert _annotation_name(error_descriptor) == "ErrorDescriptor"
    assert _annotation_name(event_descriptor) == "EventDescriptor"
    assert not (
        {name for name in methods if not name.startswith("_")}
        - {
            "parse_request",
            "response",
            "serialized_size",
            "response_size",
            "response_fits",
            "error_response",
            "notification",
            "write",
        }
    ), "FrameCodec has an unreviewed public raw frame builder"


def _wire_keys(mapping: ast.Dict) -> set[str]:
    return {
        key.value
        for key in mapping.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _is_wire_error_or_notification(keys: set[str]) -> bool:
    return "jsonrpc" in keys and bool({"error", "method"} & keys)


def _enum_member(node: ast.AST | None, enum_name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == enum_name
    )


def _event_descriptor_expression(node: ast.AST | None) -> bool:
    return _enum_member(node, "EventDescriptor") or (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr in {"AUXILIARY_WIRE_EVENTS", "STREAM_EVENT_WIRE_EVENTS"}
    )


def _error_descriptor_expression(node: ast.AST | None) -> bool:
    if _enum_member(node, "ErrorDescriptor"):
        return True
    return (
        isinstance(node, ast.IfExp)
        and _error_descriptor_expression(node.body)
        and _error_descriptor_expression(node.orelse)
    )


def _call_argument(call: ast.Call, position: int, keyword: str) -> ast.AST | None:
    if len(call.args) > position:
        return call.args[position]
    return next((item.value for item in call.keywords if item.arg == keyword), None)


class _RawFrameBypassVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []
        self.typed_events: set[str] = set()
        self.derived_events: set[str] = set()
        self.derived_errors: set[str] = set()
        self.protocol_error_aliases = {"ProtocolError"}
        self.mapping_keys: dict[str, set[str]] = {}

    def _add(self, node: ast.AST, reason: str) -> None:
        try:
            label = self.path.relative_to(ROOT)
        except ValueError:
            label = self.path
        self.violations.append(f"{label}:{node.lineno}: {reason}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = (
            self.typed_events,
            self.derived_events,
            self.derived_errors,
            self.protocol_error_aliases,
            self.mapping_keys,
        )
        self.typed_events = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            if _annotation_name(argument) == "EventDescriptor"
        }
        self.derived_events = set()
        self.derived_errors = set()
        self.protocol_error_aliases = {"ProtocolError"}
        self.mapping_keys = {}
        for statement in node.body:
            self.visit(statement)
        (
            self.typed_events,
            self.derived_events,
            self.derived_errors,
            self.protocol_error_aliases,
            self.mapping_keys,
        ) = previous

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_assignment(target, node.value)
        self.generic_visit(node.value)

    def _record_assignment(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._record_name_assignment(target, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            self._record_sequence_assignment(target, value)
        elif isinstance(target, ast.Subscript):
            self._record_subscript_assignment(target)

    def _record_name_assignment(self, target: ast.Name, value: ast.expr) -> None:
        if isinstance(value, ast.Dict):
            self.mapping_keys[target.id] = _wire_keys(value)
        if isinstance(value, ast.Name) and value.id == "ProtocolError":
            self.protocol_error_aliases.add(target.id)
        if (
            isinstance(value, ast.Call)
            and _call_name(value.func) == "notification_wire_event"
        ) or _event_descriptor_expression(value):
            self.derived_events.add(target.id)
        if _error_descriptor_expression(value):
            self.derived_errors.add(target.id)

    def _record_sequence_assignment(
        self, target: ast.Tuple | ast.List, value: ast.expr
    ) -> None:
        if (
            isinstance(value, ast.Call)
            and _call_name(value.func) == "project_request_error"
            and target.elts
            and isinstance(target.elts[0], ast.Name)
        ):
            self.derived_errors.add(target.elts[0].id)

    def _record_subscript_assignment(self, target: ast.Subscript) -> None:
        if not (
            isinstance(target.value, ast.Name)
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        ):
            return
        keys = self.mapping_keys.setdefault(target.value.id, set())
        keys.add(target.slice.value)
        if _is_wire_error_or_notification(keys):
            self._add(target, "build error and notification frames through FrameCodec")

    def visit_Compare(self, node: ast.Compare) -> None:
        if (
            self.path.name in {"server.py", "mutation.py"}
            and isinstance(node.left, ast.Name)
            and node.left.id == "method"
        ):
            literals = []
            for item in node.comparators:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    literals.append(item.value)
                elif isinstance(item, ast.Set):
                    literals.extend(
                        value.value
                        for value in item.elts
                        if isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    )
            if any(value != "hello" for value in literals):
                self._add(node, "dispatch method names must use MethodDescriptor")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in RAW_FRAME_BUILDERS:
            self._add(node, f"raw frame builder {node.attr} is private to protocol.py")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if _is_wire_error_or_notification(_wire_keys(node)):
            self._add(node, "build error and notification frames through FrameCodec")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name == "_Frame":
            self._add(node, "only FrameCodec may construct transport frames")
        self._check_error_call(node, name)
        self._check_event_call(node, name)
        self._check_raw_write(node, name)
        self.generic_visit(node)

    def _check_error_call(self, node: ast.Call, name: str | None) -> None:
        if name not in self.protocol_error_aliases | {"error_response"}:
            return
        position = 0 if name in self.protocol_error_aliases else 1
        descriptor = _call_argument(node, position, "descriptor")
        valid = (
            _error_descriptor_expression(descriptor)
            or (
                isinstance(descriptor, ast.Attribute)
                and descriptor.attr == "descriptor"
            )
            or (
                isinstance(descriptor, ast.Name)
                and descriptor.id in self.derived_errors
            )
        )
        if not valid:
            self._add(node, "error emissions require an ErrorDescriptor member")
        data = _call_argument(node, position + 2, "data")
        if isinstance(data, ast.Name):
            keys = self.mapping_keys.get(data.id, set())
        elif isinstance(data, ast.Dict):
            keys = _wire_keys(data)
        else:
            keys = set()
        if "code" in keys:
            self._add(node, "descriptor data owns the structured error code")

    def _check_event_call(self, node: ast.Call, name: str | None) -> None:
        if name not in {"notification", "publish_controller", "_notify"}:
            return
        event = _call_argument(node, 0, "event")
        valid = (
            _event_descriptor_expression(event)
            or isinstance(event, ast.Starred)
            or (
                isinstance(event, ast.Name)
                and event.id in self.typed_events | self.derived_events
            )
        )
        if not valid:
            self._add(node, "event emissions require an EventDescriptor member")

    def _check_raw_write(self, node: ast.Call, name: str | None) -> None:
        if not node.args or name not in {"write", "_write", "_write_physical"}:
            return
        argument = node.args[-1]
        raw_bytes = (
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, bytes)
            and b'"jsonrpc"' in argument.value
        )
        raw_mapping = isinstance(argument, ast.Name) and _is_wire_error_or_notification(
            self.mapping_keys.get(argument.id, set())
        )
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        codec_write = name == "write" and (
            isinstance(receiver, ast.Name)
            and receiver.id == "codec"
            or isinstance(receiver, ast.Call)
            and _call_name(receiver.func) == "FrameCodec"
        )
        if raw_mapping or (codec_write and raw_bytes):
            self._add(node, "transport writes require a FrameCodec frame")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _source_bypasses(source: str) -> list[str]:
    visitor = _RawFrameBypassVisitor(Path("mutation.py"))
    visitor.visit(ast.parse(source))
    return visitor.violations


def _raw_frame_bypasses() -> list[str]:
    violations: list[str] = []
    for source_root in PROTOCOL_SOURCES:
        for path in source_root.rglob("*.py"):
            if path == PROTOCOL_MODULE:
                continue
            visitor = _RawFrameBypassVisitor(path)
            visitor.visit(
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            )
            violations.extend(visitor.violations)
    return violations


class _SlashRegistryBypassVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and "EXECUTABLE" in target.id
                and isinstance(node.value, (ast.Set, ast.Call))
            ):
                self.violations.append(
                    f"line {node.lineno}: slash availability belongs in ServeSlash"
                )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == "name"
            and any(
                isinstance(comparator, ast.Constant)
                and isinstance(comparator.value, str)
                for comparator in node.comparators
            )
        ):
            self.violations.append(
                f"line {node.lineno}: slash dispatch belongs in SlashCommand"
            )
        self.generic_visit(node)


def _slash_registry_bypasses(source: str) -> list[str]:
    visitor = _SlashRegistryBypassVisitor()
    visitor.visit(ast.parse(source))
    return visitor.violations


def test_no_serve_slash_allowlist_or_named_dispatch_bypasses_registry() -> None:
    source = (ROOT / "src/zeta/server/slash_commands.py").read_text()
    assert not (violations := _slash_registry_bypasses(source)), "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    (
        'EXECUTABLE_BUILTINS = frozenset({"status", "mcp"})',
        'def run(name):\n    if name == "mcp":\n        return "special"',
    ),
)
def test_slash_registry_gate_rejects_committed_bypass_mutations(source: str) -> None:
    assert _slash_registry_bypasses(source)


@pytest.mark.parametrize(
    "source",
    (
        'def dispatch(method):\n    if method == "ping":\n        return True\n',
        'def dispatch(method):\n    if method in {"approve", "deny"}:\n        return True\n',
    ),
)
def test_static_gate_rejects_raw_method_dispatch_mutations(source: str) -> None:
    assert _source_bypasses(source)


def test_no_code_bypasses_the_typed_frame_codec() -> None:
    assert not (violations := _raw_frame_bypasses()), (
        "error and notification wire frames have one typed chokepoint in "
        "src/zeta/server/protocol.py:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "source",
    (
        "async def emit(codec, event):\n    await codec.notification(event, 's')\n",
        "async def emit(client):\n    frame = {}\n    frame['jsonrpc'] = '2.0'\n    frame['error'] = {}\n    await client._write(frame)\n",
        'async def emit(codec, writer):\n    await codec.write(writer, b\'{"jsonrpc":"2.0"}\')\n',
        "def emit(code):\n    return ProtocolError(descriptor=code, message='bad')\n",
        "def emit(codec, code):\n    return codec.error_response(request_id=1, descriptor=code, message='bad')\n",
        "def emit(code):\n    error = ProtocolError\n    return error(code, 'bad')\n",
        "def emit(value):\n    data = {'code': value}\n    return ProtocolError(ErrorDescriptor.INVALID_PARAMS, 'bad', data)\n",
    ),
)
def test_static_gate_rejects_committed_bypass_mutations(source: str) -> None:
    assert _source_bypasses(source)


STRUCTURED_ERROR_FIXTURES = (
    (
        ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
        "session history changed",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32009,"message":"session history changed","data":{"code":"stale_history_snapshot"}}}\n',
    ),
    (
        ErrorDescriptor.CONTENT_NOT_FOUND,
        "content not found",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32012,"message":"content not found","data":{"code":"content_not_found"}}}\n',
    ),
    (
        ErrorDescriptor.CORRUPT_TRANSCRIPT,
        "transcript terminated invalid conversation row at row 2 (session s1)",
        {
            "kind": "invalid_row",
            "row": 2,
            "repairable": True,
            "session_id": "s1",
            "actions": [],
        },
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32000,"message":"transcript terminated invalid conversation row at row 2 (session s1)","data":{"code":"corrupt_transcript","kind":"invalid_row","row":2,"repairable":true,"session_id":"s1","actions":[]}}}\n',
    ),
    (
        ErrorDescriptor.ACTIVE_SESSION,
        "select another session",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32005,"message":"select another session","data":{"code":"active_session"}}}\n',
    ),
    (
        ErrorDescriptor.ACCOUNT_LOCK_TIMEOUT,
        "account lock timeout",
        {"retryable": True},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32008,"message":"account lock timeout","data":{"code":"account_lock_timeout","retryable":true}}}\n',
    ),
    (
        ErrorDescriptor.APPROVAL_PERSISTENCE_FAILED,
        "approval persistence failed",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32000,"message":"approval persistence failed","data":{"code":"approval_persistence_failed"}}}\n',
    ),
    (
        ErrorDescriptor.HOST_BINDING_REQUIRED,
        "session is not attached",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32003,"message":"session is not attached","data":{"code":"host_binding_required"}}}\n',
    ),
    (
        ErrorDescriptor.INVALID_CONTENT_READ,
        "invalid content read",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"message":"invalid content read","data":{"code":"invalid_content_read"}}}\n',
    ),
    (
        ErrorDescriptor.INVALID_SESSION_GRANTS_PARAMS,
        "session_grants takes no params",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"message":"session_grants takes no params","data":{"code":"invalid_session_grants_params"}}}\n',
    ),
    (
        ErrorDescriptor.INVALID_SESSION_GRANTS_REVOKE,
        "invalid revoke_session_grants params",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"message":"invalid revoke_session_grants params","data":{"code":"invalid_session_grants_revoke"}}}\n',
    ),
    (
        ErrorDescriptor.LOGIN_IN_PROGRESS,
        "login is already pending",
        None,
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32005,"message":"login is already pending","data":{"code":"login_in_progress"}}}\n',
    ),
    (
        ErrorDescriptor.OBSERVER_READ_ONLY,
        "method is unavailable",
        {"method": "send"},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32008,"message":"method is unavailable","data":{"code":"observer_read_only","method":"send"}}}\n',
    ),
    (
        ErrorDescriptor.PROJECT_NOT_FOUND,
        "project missing",
        {"project_id": "p1"},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"message":"project missing","data":{"code":"project_not_found","project_id":"p1"}}}\n',
    ),
    (
        ErrorDescriptor.PROJECT_REGISTRY_BUSY,
        "project registry busy",
        {"retryable": False},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32000,"message":"project registry busy","data":{"code":"project_registry_busy","retryable":false}}}\n',
    ),
    (
        ErrorDescriptor.SESSION_GRANT_NOT_FOUND,
        "session grant not found",
        {"grant": "bash"},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"message":"session grant not found","data":{"code":"session_grant_not_found","grant":"bash"}}}\n',
    ),
    (
        ErrorDescriptor.SESSION_MISMATCH,
        "session is not active",
        {"attached_session_id": "s1", "session_id": "s2"},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32003,"message":"session is not active","data":{"code":"session_mismatch","attached_session_id":"s1","session_id":"s2"}}}\n',
    ),
    (
        ErrorDescriptor.SESSION_NOT_FOUND,
        "session missing",
        {"session_id": "s1"},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"message":"session missing","data":{"code":"session_not_found","session_id":"s1"}}}\n',
    ),
    (
        ErrorDescriptor.TRANSCRIPT_REPAIR_REQUIRES_UPGRADE,
        "transcript repair needs the stored-format upgrade (home epoch 2, needs 4)",
        {"home_epoch": 2, "required_epoch": 4},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32013,"message":"transcript repair needs the stored-format upgrade (home epoch 2, needs 4)","data":{"code":"transcript_repair_requires_upgrade","home_epoch":2,"required_epoch":4}}}\n',
    ),
)


def test_documented_slash_table_matches_shared_registry() -> None:
    document = PROTOCOL_DOC.read_text()
    table = document.split(
        "The shared slash registry has this order and surface availability:", 1
    )[1].split("\n\n", 2)[1]
    documented = []
    for line in table.splitlines()[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        documented.append(
            (
                cells[0].strip("`"),
                cells[1] == "yes",
                cells[2].replace("`", ""),
            )
        )

    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())
    actual = [
        (command.name, command.serve is not None, command.description)
        for command in registry.builtin_commands
    ]
    assert documented == actual


def test_structured_error_frames_match_pinned_bytes() -> None:
    codec = FrameCodec()
    assert {descriptor for descriptor, _, _, _ in STRUCTURED_ERROR_FIXTURES} == {
        descriptor for descriptor in ErrorDescriptor if descriptor.data_code is not None
    }
    for descriptor, message, fields, origin_main_frame in STRUCTURED_ERROR_FIXTURES:
        assert codec.error_response(7, descriptor, message, fields) == origin_main_frame


def test_registered_protocol_values_match_pinned_bytes() -> None:
    codec = FrameCodec()
    assert (
        codec.notification(
            EventDescriptor.SESSION_GRANTS_CHANGED,
            "s1",
            data={"change": "clear", "changed": ["bash"], "grants": {}},
        )
        == b'{"jsonrpc":"2.0","method":"event","params":{"event":"session_grants_changed","session_id":"s1","data":{"change":"clear","changed":["bash"],"grants":{}}}}\n'
    )
    assert (
        codec.response(
            7,
            {
                "history": [
                    {
                        "id": "row1",
                        "role": "system",
                        "event": EventDescriptor.SERVED_TURN_END,
                        "data": {"turn_id": "turn1", "reason": "completed"},
                    }
                ]
            },
        )
        == b'{"jsonrpc":"2.0","id":7,"result":{"history":[{"id":"row1","role":"system","event":"served_turn_end","data":{"turn_id":"turn1","reason":"completed"}}]}}\n'
    )
    assert (
        codec.response(
            7,
            {
                "kind": SlashResultKindDescriptor.OUTPUT,
                "text": "credential error",
                "credential_store_error": {
                    "code": SlashErrorCodeDescriptor.MCP_CREDENTIAL_STORE,
                    "reason": "invalid_json",
                    "server": "live",
                    "credential_file": "/home/.zeta/mcp-oauth.json",
                    "operation": "load",
                    "recovery": "move_aside_then_auth",
                    "text": "credential error",
                },
            },
        )
        == b'{"jsonrpc":"2.0","id":7,"result":{"kind":"output","text":"credential error","credential_store_error":{"code":"mcp_credential_store","reason":"invalid_json","server":"live","credential_file":"/home/.zeta/mcp-oauth.json","operation":"load","recovery":"move_aside_then_auth","text":"credential error"}}}\n'
    )

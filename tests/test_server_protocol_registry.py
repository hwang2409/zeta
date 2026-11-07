from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from zeta.server.protocol import (
    ErrorDescriptor,
    EventDescriptor,
    FrameCodec,
    SlashResultKindDescriptor,
)

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
    section = _section(document, "Returns a `kind` discriminator:", "The built-ins")
    return set(re.findall(r"^  - `([a-z_]+)`:", section, re.MULTILINE))


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
            if isinstance(target, ast.Name):
                if isinstance(node.value, ast.Dict):
                    self.mapping_keys[target.id] = _wire_keys(node.value)
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "ProtocolError"
                ):
                    self.protocol_error_aliases.add(target.id)
                if (
                    isinstance(node.value, ast.Call)
                    and _call_name(node.value.func) == "notification_wire_event"
                ) or _event_descriptor_expression(node.value):
                    self.derived_events.add(target.id)
                if _error_descriptor_expression(node.value):
                    self.derived_errors.add(target.id)
            elif (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(node.value, ast.Call)
                and _call_name(node.value.func) == "project_request_error"
                and target.elts
                and isinstance(target.elts[0], ast.Name)
            ):
                self.derived_errors.add(target.elts[0].id)
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys = self.mapping_keys.setdefault(target.value.id, set())
                keys.add(target.slice.value)
                if _is_wire_error_or_notification(keys):
                    self._add(
                        target, "build error and notification frames through FrameCodec"
                    )
        self.generic_visit(node.value)

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
        ErrorDescriptor.SESSION_NOT_FOUND,
        "session missing",
        {"session_id": "s1"},
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"message":"session missing","data":{"code":"session_not_found","session_id":"s1"}}}\n',
    ),
)


def test_structured_error_frames_match_origin_main_bytes() -> None:
    codec = FrameCodec()
    assert {descriptor for descriptor, _, _, _ in STRUCTURED_ERROR_FIXTURES} == {
        descriptor for descriptor in ErrorDescriptor if descriptor.data_code is not None
    }
    for descriptor, message, fields, origin_main_frame in STRUCTURED_ERROR_FIXTURES:
        assert codec.error_response(7, descriptor, message, fields) == origin_main_frame

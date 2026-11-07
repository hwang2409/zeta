from __future__ import annotations

import ast
import re
from pathlib import Path

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


def _frame_codec_methods() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(PROTOCOL_MODULE.read_text(encoding="utf-8"))
    codec = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FrameCodec"
    )
    return {
        node.name: node
        for node in codec.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_frame_codec_is_the_typed_error_and_notification_chokepoint() -> None:
    methods = _frame_codec_methods()
    error_descriptor = methods["error_response"].args.args[2]
    event_descriptor = methods["notification"].args.args[1]
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


def _wire_keys(mapping: ast.Dict) -> set[object]:
    return {
        key.value
        for key in mapping.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _is_wire_error_or_notification(mapping: ast.Dict) -> bool:
    keys = _wire_keys(mapping)
    return "jsonrpc" in keys and ("error" in keys or "method" in keys)


class _RawFrameBypassVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []

    def _add(self, node: ast.AST, reason: str) -> None:
        self.violations.append(f"{self.path.relative_to(ROOT)}:{node.lineno}: {reason}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in RAW_FRAME_BUILDERS:
            self._add(node, f"raw frame builder {node.attr} is private to protocol.py")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if _is_wire_error_or_notification(node):
            self._add(node, "build error and notification frames through FrameCodec")
        self.generic_visit(node)


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

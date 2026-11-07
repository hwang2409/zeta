from __future__ import annotations

import ast
import re
from pathlib import Path

from zeta.server.protocol import ErrorDescriptor, EventDescriptor, FrameCodec

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
    (ErrorDescriptor.HISTORY_SNAPSHOT_STALE, "session history changed", None),
    (ErrorDescriptor.CONTENT_NOT_FOUND, "content not found", None),
    (ErrorDescriptor.ACTIVE_SESSION, "select another session", None),
    (ErrorDescriptor.ACCOUNT_LOCK_TIMEOUT, "account lock timeout", {"retryable": True}),
    (ErrorDescriptor.APPROVAL_PERSISTENCE_FAILED, "approval persistence failed", None),
    (ErrorDescriptor.HOST_BINDING_REQUIRED, "session is not attached", None),
    (ErrorDescriptor.INVALID_CONTENT_READ, "invalid content read", None),
    (ErrorDescriptor.LOGIN_IN_PROGRESS, "login is already pending", None),
    (ErrorDescriptor.OBSERVER_READ_ONLY, "method is unavailable", {"method": "send"}),
    (ErrorDescriptor.PROJECT_NOT_FOUND, "project missing", {"project_id": "p1"}),
    (
        ErrorDescriptor.PROJECT_REGISTRY_BUSY,
        "project registry busy",
        {"retryable": False},
    ),
    (ErrorDescriptor.SESSION_NOT_FOUND, "session missing", {"session_id": "s1"}),
)


def _origin_main_structured_error_fixture(
    descriptor: ErrorDescriptor, message: str, fields: dict[str, object] | None
) -> bytes:
    data = {"code": descriptor.data_code, **(fields or {})}
    import json

    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "error": {"code": descriptor.code, "message": message, "data": data},
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def test_structured_error_frames_match_origin_main_bytes() -> None:
    codec = FrameCodec()
    assert {descriptor for descriptor, _, _ in STRUCTURED_ERROR_FIXTURES} == {
        descriptor for descriptor in ErrorDescriptor if descriptor.data_code is not None
    }
    for descriptor, message, fields in STRUCTURED_ERROR_FIXTURES:
        assert codec.error_response(7, descriptor, message, fields) == (
            _origin_main_structured_error_fixture(descriptor, message, fields)
        )

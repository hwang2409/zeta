from __future__ import annotations

import ast
import re
from pathlib import Path

from zeta.server.protocol_registry import ErrorDescriptor, EventDescriptor

ROOT = Path(__file__).parents[1]
PROTOCOL_DOC = ROOT / "docs" / "serve-protocol.md"
PROTOCOL_SOURCES = (ROOT / "src/zeta/server", ROOT / "src/zeta/protocol")


def _section(document: str, heading: str, next_heading: str) -> str:
    return document.split(heading, 1)[1].split(next_heading, 1)[0]


def _documented_errors(document: str) -> set[tuple[str, int, str | None, str]]:
    section = _section(document, "Error descriptors are:", "## normative wire schema")
    rows = set()
    pattern = re.compile(
        r"^\| `([a-z0-9_]+)` \| `(-\d+)` \| (?:`([a-z0-9_]+)`|none) \| ([^|]+?) \|$",
        re.MULTILINE,
    )
    for name, code, data_code, meaning in pattern.findall(section):
        rows.add((name, int(code), data_code or None, meaning))
    return rows


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
        "add exactly one descriptor in src/zeta/server/protocol_registry.py and one "
        "matching error row in docs/serve-protocol.md"
    )
    assert _documented_events(document) == {event.value for event in EventDescriptor}, (
        "add exactly one event in src/zeta/server/protocol_registry.py and one matching "
        "event row in docs/serve-protocol.md"
    )


def _is_descriptor(node: ast.expr, descriptor: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == descriptor
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _dict_has_code(node: ast.expr) -> bool:
    return isinstance(node, ast.Dict) and any(
        isinstance(key, ast.Constant) and key.value == "code" for key in node.keys
    )


def test_protocol_emission_sites_use_registered_descriptors() -> None:
    violations: list[str] = []
    for source_root in PROTOCOL_SOURCES:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                descriptor_index = {
                    "ProtocolError": 0,
                    "error_response": 1,
                    "_minimal_error": 1,
                }.get(name)
                if descriptor_index is not None and len(node.args) > descriptor_index:
                    descriptor = node.args[descriptor_index]
                    allowed_forward = name == "error_response" and (
                        isinstance(descriptor, ast.Name)
                        and descriptor.id == "descriptor"
                        or isinstance(descriptor, ast.Attribute)
                        and descriptor.attr == "descriptor"
                    )
                    if (
                        not _is_descriptor(descriptor, "ErrorDescriptor")
                        and not allowed_forward
                    ):
                        violations.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: use ErrorDescriptor"
                        )
                    for argument in node.args[descriptor_index + 2 :]:
                        if _dict_has_code(argument):
                            violations.append(
                                f"{path.relative_to(ROOT)}:{node.lineno}: select the data.code "
                                "variant from ErrorDescriptor"
                            )
                if name == "notification" and path.name != "protocol.py":
                    is_central_encoder = (
                        path.name == "connections.py"
                        and node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "event"
                    )
                    if not is_central_encoder:
                        violations.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: route events through the "
                            "EventDescriptor mapping"
                        )
                if (
                    name in {"publish_controller", "_notify"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: use EventDescriptor"
                    )
    assert not violations, (
        "every serve emission must select a descriptor from "
        "src/zeta/server/protocol_registry.py; document it in docs/serve-protocol.md:\n"
        + "\n".join(violations)
    )

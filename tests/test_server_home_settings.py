"""``home_settings_v1`` through a real ``zeta serve`` websocket: negotiation, roles, and the update result."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract

FEATURES = ["home_settings_v1"]
BUILTIN = "http://127.0.0.1:4323/v1/traces"
ON = [{"key": "telemetry.enabled", "value": True}]
INVALID = {"code": -32602, "message": "invalid params", "data": {"code": "invalid_params"}}


# the names ``zeta.telemetry.runtime.preflight`` imports, so a stub tree on PYTHONPATH decides
# whether the serve subprocess can start telemetry, with or without zeta[otel] installed
_SDK_STUBS = {
    "requests/__init__.py": "",
    "opentelemetry/__init__.py": "",
    "opentelemetry/exporter/__init__.py": "",
    "opentelemetry/exporter/otlp/__init__.py": "",
    "opentelemetry/exporter/otlp/proto/__init__.py": "",
    "opentelemetry/exporter/otlp/proto/http/__init__.py": "",
    "opentelemetry/exporter/otlp/proto/http/trace_exporter.py": "OTLPSpanExporter = None\n",
    "opentelemetry/sdk/__init__.py": "",
    "opentelemetry/sdk/resources.py": "Resource = None\n",
    "opentelemetry/sdk/trace/__init__.py": "SpanLimits = TracerProvider = None\n",
    "opentelemetry/sdk/trace/export.py": "BatchSpanProcessor = None\n",
    "opentelemetry/sdk/trace/sampling.py": "ALWAYS_ON = ParentBased = None\n",
    "opentelemetry/trace/__init__.py": "",
    "opentelemetry/trace/propagation/__init__.py": "",
    "opentelemetry/trace/propagation/tracecontext.py": "TraceContextTextMapPropagator = None\n",
}
_SDK_MISSING = {"opentelemetry/__init__.py": "raise ModuleNotFoundError(\"No module named 'opentelemetry'\")\n"}


def _otel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> None:
    """Shadow the OpenTelemetry SDK for the serve subprocess and clear the OTEL environment."""
    root = tmp_path / "otel-stub"
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])))
    for name in list(os.environ):
        if name.startswith(("OTEL_EXPORTER_OTLP", "OTEL_PYTHON_EXPORTER_OTLP")) or name == "OTEL_SDK_DISABLED":
            monkeypatch.delenv(name)


def _revision(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


async def _client(contract: ServeContract, role: str = "controller", features: list[str] = FEATURES):
    client = await contract.connect(role)  # type: ignore[arg-type]
    hello = await client.request("hello", "hello", {
        "protocol_version": "1.2", "features": features, **({"role": "observer"} if role == "observer" else {}),
    })
    return client, hello["result"]


@pytest.mark.asyncio
async def test_observers_read_but_never_update_home_settings(tmp_path: Path) -> None:
    settings = tmp_path / "home" / "settings.toml"
    settings.parent.mkdir(parents=True)
    settings.write_bytes(b'model = "gpt-5.6-luna"\n')
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        plain, plain_hello = await _client(contract, features=[])
        observer, observer_hello = await _client(contract, "observer", ["serve_observers", *FEATURES])

        unnegotiated = await plain.request("get", "home_settings_get")
        got = await observer.request("get", "home_settings_get")
        refused = await observer.request(
            "update", "home_settings_update", {"changes": "not a change"}
        )
    finally:
        await contract.close()

    assert "home_settings_get" not in plain_hello["capabilities"]["requests"]
    assert unnegotiated["error"]["code"] == -32601
    assert "home_settings_get" in observer_hello["capabilities"]["requests"]
    assert "home_settings_update" not in observer_hello["capabilities"]["requests"]
    assert got["result"] == {
        "revision": _revision(b'model = "gpt-5.6-luna"\n'),
        "settings": {"telemetry": {"enabled": False, "source": "off", "editable": True}},
        "settings_malformed": False,
    }
    assert refused["error"] == {
        "code": -32008,
        "message": "method is unavailable to read-only observers",
        "data": {"code": "observer_read_only", "method": "home_settings_update", "reason": "observer_read_only"},
    }
    assert settings.read_bytes() == b'model = "gpt-5.6-luna"\n'


@pytest.mark.asyncio
async def test_controller_update_checks_revision_and_lists_restarts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _otel(tmp_path, monkeypatch, _SDK_STUBS)
    settings = tmp_path / "home" / "settings.toml"
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        controller, hello = await _client(contract)
        created = await controller.request("new", "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        got = await controller.request("get", "home_settings_get")
        invalid = [
            await controller.request("get-params", "home_settings_get", {"key": "otel.endpoint"}),
            await controller.request("extra", "home_settings_update", {
                "expected_revision": got["result"]["revision"], "changes": ON, "endpoint": BUILTIN,
            }),
            await controller.request("unknown", "home_settings_update", {
                "expected_revision": got["result"]["revision"],
                "changes": [{"key": "yolo", "value": True}],
            }),
            await controller.request("depth-bool", "home_settings_update", {
                "expected_revision": got["result"]["revision"],
                "changes": [{"key": "agents.max_depth", "value": True}],
            }),
        ]
        stale = await controller.request(
            "stale", "home_settings_update", {"expected_revision": _revision(b"other"), "changes": ON}
        )
        unchanged = settings.exists()
        updated = await controller.request(
            "update", "home_settings_update", {"expected_revision": got["result"]["revision"], "changes": ON}
        )
    finally:
        await contract.close()

    assert {"home_settings_get", "home_settings_update"} <= set(hello["capabilities"]["requests"])
    assert [reply["error"] for reply in invalid] == [INVALID] * 4
    assert stale["error"] == {
        "code": -32009,
        "message": "resource revision is stale",
        "data": {"code": "stale_settings_revision", "reason": "stale_settings_revision"},
    }
    assert unchanged is False
    written = settings.read_bytes()
    assert written == f'otel.endpoint = "{BUILTIN}"\n'.encode()
    assert updated["result"] == {
        "revision": _revision(written),
        "settings": {"telemetry": {"enabled": True, "source": "builtin", "editable": True}},
        "restart": {"serve": True, "sessions": [session_id], "complete": True},
    }


UNAVAILABLE = {
    "code": -32602,
    "message": "telemetry cannot start in this zeta install or environment",
    "data": {"code": "telemetry_unavailable", "detail": "OpenTelemetry support is not installed; install zeta[otel]"},
}
EXPORTER_ENV = {
    **UNAVAILABLE,
    "data": {
        "code": "telemetry_unavailable",
        "detail": "OTEL exporter environment is not allowed; configure zeta telemetry explicitly",
    },
}
OFF = [{"key": "telemetry.enabled", "value": False}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk", "environment", "body", "changes", "refusal"),
    [
        pytest.param(_SDK_MISSING, {}, b"", ON, UNAVAILABLE, id="sdk-missing-on-refused"),
        pytest.param(
            _SDK_STUBS, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:9/"}, b"", ON, EXPORTER_ENV,
            id="exporter-env-on-refused",
        ),
        pytest.param(_SDK_STUBS, {}, b"", ON, None, id="clean-on-accepted"),
        pytest.param(_SDK_MISSING, {}, f'otel.endpoint = "{BUILTIN}"\n'.encode(), OFF, None, id="sdk-missing-off-accepted"),
    ],
)
async def test_turning_telemetry_on_runs_the_start_preflight_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sdk, environment, body, changes, refusal
) -> None:
    """An on that would stop zeta's next start is refused with nothing written; off always applies."""
    _otel(tmp_path, monkeypatch, sdk)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)  # serve starts with telemetry off, so only the update checks it
    settings = tmp_path / "home" / "settings.toml"
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        settings.write_bytes(body)  # after start: serve would refuse to start with this telemetry on
        controller, _ = await _client(contract)
        reply = await controller.request(
            "update", "home_settings_update", {"expected_revision": _revision(body), "changes": changes}
        )
    finally:
        await contract.close()

    if refusal is not None:
        assert reply.get("error") == refusal
        assert settings.read_bytes() == body
    else:
        assert "error" not in reply, reply
        enabled = changes[0]["value"]
        assert reply["result"]["settings"]["telemetry"]["enabled"] is enabled
        assert settings.read_bytes() == (f'otel.endpoint = "{BUILTIN}"\n'.encode() if enabled else b"")


@pytest.mark.asyncio
async def test_a_filesystem_failure_names_no_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _otel(tmp_path, monkeypatch, _SDK_STUBS)
    home = tmp_path / "home"
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        controller, _ = await _client(contract)
        got = await controller.request("get", "home_settings_get")
        home.rename(tmp_path / "moved")
        try:
            replies = [
                await controller.request("get", "home_settings_get"),
                await controller.request(
                    "update", "home_settings_update", {"expected_revision": got["result"]["revision"], "changes": ON}
                ),
            ]
        finally:
            (tmp_path / "moved").rename(home)
    finally:
        await contract.close()

    for reply in replies:
        assert tmp_path.name not in json.dumps(reply)
        assert reply["error"] == {"code": -32000, "message": "unexpected server or storage failure"}

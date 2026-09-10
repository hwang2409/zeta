"""Consumer checks for the real app; build it with make gui-app first."""

import json
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bundle() -> Path:
    app = REPO / "dist/Zeta.app"
    if sys.platform != "darwin" or not app.exists():
        pytest.skip("requires macOS and make gui-app")
    return app


def test_bundle_layout_and_launchers(bundle: Path) -> None:
    contents = bundle / "Contents"
    plist = plistlib.loads((contents / "Info.plist").read_bytes())
    launcher = contents / "MacOS" / plist["CFBundleExecutable"]
    server = contents / "Resources/zeta-server"
    for path in (launcher, server, contents / "MacOS/zeta-gui"):
        assert path.is_file() and os.access(path, os.X_OK)
    for path in (launcher, server):
        script = path.read_text()
        assert str(REPO) not in script
        assert "/Users/" not in script
        assert "uv " not in script
        assert "--directory" not in script
    icon = contents / "Resources" / plist["CFBundleIconFile"]
    data = icon.read_bytes()
    assert data[:4] == b"icns"
    assert int.from_bytes(data[4:8], "big") == len(data)
    runtime = contents / "Resources/python"
    assert list(runtime.glob("bin/python3.*"))
    assert list(runtime.glob("lib/python*/site-packages/zeta/cli.py"))
    for link in bundle.rglob("*"):
        if link.is_symlink():
            assert link.resolve().is_relative_to(bundle.resolve()), link


def test_native_libraries_use_bundle_or_system_paths(bundle: Path) -> None:
    native = [bundle / "Contents/MacOS/zeta-gui"]
    native.extend((bundle / "Contents/Resources/python/bin").glob("python3.*"))
    native.extend(bundle.rglob("*.dylib"))
    native.extend(bundle.rglob("*.so"))
    for path in native:
        linked = subprocess.check_output(["otool", "-L", str(path)], text=True)
        for line in linked.splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            assert dependency.startswith(("@", "/usr/lib/", "/System/Library/")), (
                path, dependency
            )
        commands = subprocess.check_output(["otool", "-l", str(path)], text=True)
        lines = commands.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == "cmd LC_RPATH":
                rpath = lines[index + 2].strip().removeprefix("path ").split(" (", 1)[0]
                assert rpath.startswith(("@", "/usr/lib/", "/System/Library/")), (
                    path, rpath
                )


def test_relocated_server_without_checkout_or_uv(bundle: Path) -> None:
    # /tmp keeps the Unix socket below macOS's 104-byte path limit.
    with TemporaryDirectory(prefix="zeta-package-", dir="/tmp") as temporary:
        root = Path(temporary)
        moved = root / "Moved app/Zeta.app"
        shutil.copytree(bundle, moved, symlinks=True)
        env = dict(os.environ, ZETA_HOME=str(root / "home"), PATH="/usr/bin:/bin")
        # Isolated Python must also ignore hostile inherited Python settings.
        env.update(PYTHONHOME="/missing-python", PYTHONPATH=str(REPO / "src"))
        server = moved / "Contents/Resources/zeta-server"
        sock_path = root / "server.sock"
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen(
                [str(server), "serve", "--socket", str(sock_path)],
                cwd=root, env=env, stdout=log, stderr=log,
            )
            try:
                deadline = time.monotonic() + 10
                with socket.socket(socket.AF_UNIX) as client:
                    while True:
                        try:
                            client.connect(str(sock_path))
                            break
                        except (FileNotFoundError, ConnectionRefusedError):
                            if process.poll() is not None or time.monotonic() > deadline:
                                log.seek(0)
                                pytest.fail(f"bundled server did not start: {log.read()}")
                            time.sleep(0.02)
                    client.settimeout(3)
                    with client.makefile("rwb") as stream:
                        for request_id, method, params in (
                            (1, "hello", {"protocol_version": "1.1"}),
                            (2, "list_sessions", {}),
                        ):
                            stream.write(json.dumps({
                                "jsonrpc": "2.0", "id": request_id,
                                "method": method, "params": params,
                            }).encode() + b"\n")
                            stream.flush()
                            response = json.loads(stream.readline())
                            assert response["id"] == request_id
                            assert "result" in response, response
            finally:
                process.terminate()
                process.wait(timeout=10)

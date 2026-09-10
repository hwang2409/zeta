"""Build a self-contained, unsigned macOS app from the compiled GUI."""

import os
import plistlib
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

# Pin the standalone interpreter as well as the dependencies in uv.lock.
PYTHON_VERSION = "3.12.13"


def relocate_sysconfig(runtime: Path, source_root: Path) -> None:
    """Make the standalone interpreter report its current bundle prefix."""
    sysconfig_data, = (runtime / "lib").glob("python*/_sysconfigdata_*.py")
    source_prefix = str(source_root)
    data = sysconfig_data.read_text()
    if source_prefix not in data:
        raise RuntimeError(f"missing Python install prefix in {sysconfig_data}")
    data = data.replace(source_prefix, "__ZETA_RUNTIME_PREFIX__")
    sysconfig_data.write_text(
        "import sys\n\n"
        + data
        + "\nfor key, value in build_time_vars.items():\n"
        + "    if isinstance(value, str):\n"
        + "        build_time_vars[key] = value.replace(\n"
        + "            '__ZETA_RUNTIME_PREFIX__', sys.base_prefix\n"
        + "        )\n"
    )


def bundle_server(repo: Path, resources: Path) -> None:
    # A venv's interpreter/stdlib can point outside it. Copy the complete uv
    # standalone distribution instead, then install only locked runtime wheels.
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(repo / "dist/build-python"))
    subprocess.run(
        ["uv", "python", "install", "--no-bin", PYTHON_VERSION], env=env, check=True
    )
    source_python = Path(subprocess.check_output(
        ["uv", "python", "find", "--managed-python", PYTHON_VERSION], env=env, text=True
    ).strip()).resolve()
    source_root = source_python.parent.parent
    runtime = resources / "python"
    shutil.copytree(source_root, runtime, symlinks=True)
    python = runtime / "bin" / source_python.name
    # PEP 668 marker blocks uv (and pip) from installing here. We built this
    # copy specifically to install into, so drop the marker for this build.
    marker = next(runtime.glob("lib/python*/EXTERNALLY-MANAGED"), None)
    if marker is not None:
        marker.unlink()
    with TemporaryDirectory() as temporary:
        requirements = Path(temporary) / "requirements.txt"
        subprocess.run(
            ["uv", "export", "--frozen", "--no-dev", "--no-emit-project",
             "--no-header", "--output-file", str(requirements)],
            cwd=repo, check=True, stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python),
             "--require-hashes", "--requirements", str(requirements)], check=True,
        )
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", temporary], cwd=repo, check=True,
        )
        wheel, = Path(temporary).glob("*.whl")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            check=True,
        )
    # Console scripts contain build-time shebangs. The app needs only Python;
    # its relative shell launcher below calls the installed entry point directly.
    for executable in (runtime / "bin").iterdir():
        if executable != python:
            executable.unlink()
    relocate_sysconfig(runtime, source_root)
    # python-build-standalone leaves libpython3.12.dylib's install_name as an
    # absolute build path, and Tcl's optional dylibs use bare filenames. Rewrite
    # each to @rpath so a moved bundle stays relocatable.
    for dylib in runtime.rglob("*.dylib"):
        subprocess.run(
            ["install_name_tool", "-id", f"@rpath/{dylib.name}", str(dylib)],
            check=True, stderr=subprocess.DEVNULL,
        )
    for bytecode in runtime.rglob("*.pyc"):
        bytecode.unlink()
    subprocess.run([
        str(python), "-I", "-m", "compileall", "-q", "-f",
        "-s", str(runtime), "-p", ".", str(runtime / "lib"),
    ], check=True)
    server = resources / "zeta-server"
    server.write_text(f'''#!/bin/sh
APP_RESOURCES=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$APP_RESOURCES/python/bin/{python.name}" -I -c 'from zeta.cli import main; main()' "$@"
''')
    server.chmod(0o755)


def bundle_icon(repo: Path, resources: Path) -> None:
    with TemporaryDirectory() as temporary:
        iconset = Path(temporary) / "Zeta.iconset"
        iconset.mkdir()
        original = iconset / "icon_512x512@2x.png"
        subprocess.run(["swift", str(repo / "gui/app_icon.swift"), str(original)], check=True)
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                target = iconset / f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                if target != original:
                    pixels = str(size * scale)
                    subprocess.run(
                        ["sips", "-z", pixels, pixels, str(original), "--out", str(target)],
                        check=True, stdout=subprocess.DEVNULL,
                    )
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(resources / "Zeta.icns")],
            check=True,
        )


def package_app(repo: Path) -> Path:
    bundle = repo / "dist/Zeta.app"
    # Rebuild from scratch so removed dependencies cannot survive a warm build.
    if bundle.exists():
        shutil.rmtree(bundle)
    macos = bundle / "Contents/MacOS"
    resources = bundle / "Contents/Resources"
    macos.mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(repo / "gui/target/release/zeta-gui", macos / "zeta-gui")
    bundle_server(repo, resources)
    bundle_icon(repo, resources)
    launcher = macos / "zeta"
    launcher.write_text('''#!/bin/sh
APP_MACOS=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export ZETA_BIN="$APP_MACOS/../Resources/zeta-server"
exec "$APP_MACOS/zeta-gui" "$@"
''')
    launcher.chmod(0o755)
    with (bundle / "Contents/Info.plist").open("wb") as output:
        plistlib.dump(
            {
                "CFBundleName": "Zeta",
                "CFBundleDisplayName": "Zeta",
                "CFBundleIdentifier": "dev.zeta.gui",
                "CFBundleVersion": "1",
                "CFBundleShortVersionString": "0.1.0",
                "CFBundlePackageType": "APPL",
                "CFBundleExecutable": "zeta",
                "CFBundleIconFile": "Zeta.icns",
                "NSHighResolutionCapable": True,
                "LSMinimumSystemVersion": "12.0",
            },
            output,
        )
    return bundle


if __name__ == "__main__":
    print(package_app(Path(__file__).resolve().parent.parent))

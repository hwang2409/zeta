"""Assemble an unsigned developer app with an explicit source server launcher."""

import plistlib
import shlex
import shutil
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
bundle = repo / "dist" / "Zeta.app"
macos = bundle / "Contents" / "MacOS"
resources = bundle / "Contents" / "Resources"
macos.mkdir(parents=True, exist_ok=True)
resources.mkdir(parents=True, exist_ok=True)
shutil.copy2(repo / "gui/target/debug/zeta-gui", macos / "zeta-gui")
uv = shutil.which("uv")
if uv is None:
    raise SystemExit("uv is required for the developer server launcher")
server = resources / "zeta-server"
server.write_text(
    f'#!/bin/sh\nexec {shlex.quote(uv)} run --frozen --directory {shlex.quote(str(repo))} zeta "$@"\n'
)
launcher = macos / "zeta"
launcher.write_text("""#!/bin/sh
APP_MACOS=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export ZETA_BIN="${ZETA_BIN:-$APP_MACOS/../Resources/zeta-server}"
exec "$APP_MACOS/zeta-gui" "$@"
""")
for path in (server, launcher):
    path.chmod(0o755)
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
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
        output,
    )
print(bundle)

from __future__ import annotations

import subprocess
import sys


def test_importing_cli_does_not_load_harness_modules() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zeta.cli; "
                "assert 'zeta.core.context' not in sys.modules; "
                "assert 'zeta.core.store' not in sys.modules; "
                "assert 'zeta.protocol.types' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr or check.stdout


def test_all_exports_resolve_and_cache() -> None:
    import zeta

    for name in zeta.__all__:
        value = getattr(zeta, name)
        assert value is getattr(zeta, name)

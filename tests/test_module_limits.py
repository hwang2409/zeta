from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "zeta"
# the codex stream parser is a legitimate provider module at this size.
MAX_FILE_LINES = 1200
# Built-in todo state and tool modules keep their layer-specific boundaries.
# Bumped on 2026-08-27 for the dedicated TUI agent-card seam.
MAX_FILES_PER_DIRECTORY = 15


def test_module_limits() -> None:
    oversized = [
        f"{path.relative_to(ROOT)}: {len(path.read_text(encoding='utf-8').splitlines())} lines"
        for path in ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
        and len(path.read_text(encoding="utf-8").splitlines()) > MAX_FILE_LINES
    ]
    crowded = []
    for directory in (ROOT, *[path for path in ROOT.rglob("*") if path.is_dir()]):
        if "__pycache__" in directory.parts:
            continue
        count = sum(
            1
            for path in directory.glob("*.py")
            if "__pycache__" not in path.parts
        )
        if count > MAX_FILES_PER_DIRECTORY:
            crowded.append(f"{directory.relative_to(ROOT)}: {count} files")
    assert not oversized and not crowded, "\n".join(oversized + crowded)

from __future__ import annotations

from pathlib import Path


def files_containing(root: Path, needle: bytes) -> list[Path]:
    """Return every file under ``root`` whose bytes contain ``needle``.

    A live process can unlink or replace a file between listing and reading
    (sqlite index rebuilds, atomic ``os.replace`` temp files). A path that
    vanishes makes the walk list the tree once more, so the new version of a
    replaced file is still read; a path gone on that second walk is skipped.
    """
    read: set[Path] = set()
    found: list[Path] = []
    for _ in range(2):
        vanished = False
        for path in root.rglob("*"):
            if path in read or not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                vanished = True
                continue
            read.add(path)
            if needle in data:
                found.append(path)
        if not vanished:
            break
    return found

from pathlib import Path

from zeta.core.path_identity import same_physical_path


def test_same_physical_path_resolves_existing_symlink(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)

    assert same_physical_path(directory, alias)


def test_same_physical_path_resolves_missing_target_through_symlink(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)

    assert same_physical_path(directory / "future", alias / "future")
    assert not same_physical_path(directory / "future", alias / "other")

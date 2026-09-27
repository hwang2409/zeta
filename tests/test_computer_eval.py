"""Keep the disposable-computer file boundary and launch flags honest."""

import io
import tarfile

import pytest

from evals.computer.run import _archive, _container_args, _unarchive


def test_computer_archive_only_round_trips_expected_regular_files() -> None:
    assert _unarchive(_archive({"input.txt": b"hello\n"}), ("input.txt",)) == {
        "input.txt": b"hello\n"
    }
    with pytest.raises(ValueError, match="unsafe"):
        _archive({"../host.txt": b"no"})
    with pytest.raises(ValueError, match="unexpected"):
        _unarchive(_archive({"other.txt": b"no"}), ("input.txt",))

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        link = tarfile.TarInfo("input.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../host.txt"
        archive.addfile(link)
    with pytest.raises(ValueError, match="unexpected"):
        _unarchive(payload.getvalue(), ("input.txt",))


def test_computer_has_no_network_or_host_mounts() -> None:
    args = _container_args("test-computer")
    assert args[:5] == ("run", "-d", "--rm", "--name", "test-computer")
    for pair in (
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--user", "65532:65532"),
    ):
        index = args.index(pair[0])
        assert args[index : index + 2] == pair
    assert "--read-only" in args
    assert not {"-v", "--volume", "--mount"}.intersection(args)

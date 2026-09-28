from __future__ import annotations

import json
import multiprocessing
import os
import stat
from io import StringIO

import pytest

from zeta.cli.main import build_parser
from zeta.cli.project import run
from zeta.project_registry import ProjectRegistry, ProjectRegistryError


def test_lifecycle_and_deterministic_listing(tmp_path):
    registry = ProjectRegistry(tmp_path / "projects")
    beta = registry.create_project("beta", "second")
    alpha = registry.create_project("alpha", "first")
    lane = registry.add_lane(beta.project_id, "build", "build the thing")
    assert [p.name for p in registry.list_projects()] == ["alpha", "beta"]
    assert registry.show_project(name="beta").lanes == (lane,)
    assert registry.show_lane(beta.project_id, lane.lane_id) == lane
    assert alpha.created_at == alpha.updated_at
    assert beta.created_at == beta.updated_at
    assert set(
        json.loads(
            (tmp_path / "projects" / beta.project_id / "project.json").read_text()
        )
    ) == {
        "schema_version",
        "project_id",
        "name",
        "scope",
        "created_at",
        "updated_at",
        "canonical_integration_root",
        "lanes",
    }


def test_bounds_duplicates_and_paths(tmp_path):
    registry = ProjectRegistry(tmp_path / "projects")
    registry.create_project("same", "scope")
    with pytest.raises(ProjectRegistryError):
        registry.create_project("same", "scope")
    with pytest.raises(ProjectRegistryError):
        registry.create_project("x", "")
    with pytest.raises(ProjectRegistryError):
        registry.create_project("x", "scope", "relative/path")
    project = registry.show_project(name="same")
    registry.add_lane(project.project_id, "lane", "scope")
    with pytest.raises(ProjectRegistryError):
        registry.add_lane(project.project_id, "lane", "other")
    with pytest.raises(ProjectRegistryError):
        registry.add_lane(project.project_id, "x", "x\x00")
    with pytest.raises(ProjectRegistryError):
        registry.create_project("normalized", "scope", "/tmp/normalized/")


def test_concurrent_project_creation_is_safe(tmp_path):
    root = tmp_path / "projects"
    ctx = multiprocessing.get_context("fork")

    def create(index: int) -> None:
        ProjectRegistry(root).create_project(f"project-{index}", "scope")

    processes = [ctx.Process(target=create, args=(index,)) for index in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert [p.name for p in ProjectRegistry(root).list_projects()] == [
        f"project-{index}" for index in range(8)
    ]


def test_unknown_schema_symlink_hardlink_and_permissions_refuse(tmp_path):
    root = tmp_path / "projects"
    registry = ProjectRegistry(root)
    project = registry.create_project("safe", "scope")
    record = root / project.project_id / "project.json"
    value = json.loads(record.read_text())
    value["unexpected"] = True
    record.write_text(json.dumps(value))
    with pytest.raises(ProjectRegistryError):
        registry.show_project(project.project_id)
    value.pop("unexpected")
    record.write_text(json.dumps(value))
    os.chmod(record, 0o644)
    with pytest.raises(ProjectRegistryError):
        registry.show_project(project.project_id)
    os.chmod(record, 0o600)
    outside = tmp_path / "outside"
    outside.write_text(record.read_text())
    record.unlink()
    record.symlink_to(outside)
    with pytest.raises(ProjectRegistryError):
        registry.show_project(project.project_id)
    record.unlink()
    os.link(outside, record)
    with pytest.raises(ProjectRegistryError):
        registry.show_project(project.project_id)


def _add(root: str, index: int) -> None:
    ProjectRegistry(root).add_lane(
        next(iter(ProjectRegistry(root).list_projects())).project_id,
        f"lane-{index}",
        "scope",
    )


def test_concurrent_lane_additions_are_not_lost(tmp_path):
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("p", "scope")
    ctx = multiprocessing.get_context("fork")
    processes = [
        ctx.Process(target=_add, args=(str(tmp_path / "projects"), i)) for i in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    assert sorted(lane.name for lane in registry.list_lanes(project.project_id)) == [
        f"lane-{i}" for i in range(8)
    ]


def test_cli_identity_surface_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))
    parser = build_parser()
    output = StringIO()
    args = parser.parse_args(["project", "create", "demo", "--scope", "local"])
    assert run(args, stdout=output) == 0
    project = json.loads(output.getvalue())
    output = StringIO()
    args = parser.parse_args(
        ["project", "add-lane", project["project_id"], "build", "--scope", "deliver"]
    )
    assert run(args, stdout=output) == 0
    lane = json.loads(output.getvalue())
    output = StringIO()
    args = parser.parse_args(["project", "show", project["project_id"]])
    assert run(args, stdout=output) == 0
    shown = json.loads(output.getvalue())
    assert shown["lanes"][0]["lane_id"] == lane["lane_id"]
    assert "session" not in json.dumps(shown).lower()


def test_interrupted_temp_is_ignored_and_standalone_storage_untouched(tmp_path):
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("p", "scope")
    project_dir = tmp_path / "projects" / project.project_id
    (project_dir / ".project.json.crash.tmp").write_text("not json")
    assert registry.show_project(project.project_id).project_id == project.project_id
    assert stat.S_IMODE((tmp_path / "projects").stat().st_mode) == 0o700

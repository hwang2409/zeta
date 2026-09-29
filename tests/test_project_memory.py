from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from zeta.core.project_context import load_project_context, refresh_project_memory
from zeta.core.session import SessionManager
from zeta.project_registry import ProjectRegistry, ProjectRegistryError
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def test_project_memory_workflow_and_automatic_session_linkage(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repository"
    repository.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo", repository)
    registry.initialize_memory(project.project_id)
    (home / "projects" / project.project_id / "memory" / "state.md").write_text(
        "# Current state\nshipping the first version\n", encoding="utf-8"
    )

    context = load_project_context(
        cwd=repository / "nested",
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
    )
    assert "shipping the first version" in context.system_prompt
    assert any(path.name == "state.md" for path in context.files)

    session = SessionManager(home).create(provider="fake", model="fake", cwd=repository)
    assert session.metadata.project_id == project.project_id
    assert session.metadata.parent_session_id is None
    references = (home / "projects" / project.project_id / "sessions.jsonl").read_text()
    assert session.metadata.session_id in references
    assert str(home / "sessions" / session.metadata.session_id) in references
    session.store.close()


def test_project_tool_is_builtin_and_updates_only_standard_memory(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    assert "project" in registry.registered_names
    definition = registry.definitions_by_name["project"]
    assert definition.requires_approval is False
    assert "standard filename-to-text" in definition.description


def test_resume_replaces_only_current_project_memory(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    project = ProjectRegistry(home / "projects").create_project(
        "demo", "scope", repository
    )
    context = load_project_context(
        cwd=repository,
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
    )
    ProjectRegistry(home / "projects").update_memory(
        project.project_id, {"state.md": "# Current state\\nnew state"}
    )
    resumed = refresh_project_memory(context.system_prompt, home=home, cwd=repository)
    assert "new state" in resumed
    assert "# Current state\\n" not in resumed or "new state" in resumed


def test_duplicate_roots_and_update_tool_boundary(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    repository = tmp_path / "repo"
    repository.mkdir()
    registry = ProjectRegistry(root)
    registry.create_project("one", "scope", repository)
    with pytest.raises(ProjectRegistryError, match="canonical integration root"):
        registry.create_project("two", "scope", repository)
    tools = ToolRegistry(repository, skill_catalog=SkillCatalog.empty())
    assert tools.definitions_by_name["project_update"].requires_approval
    assert (
        tools.definitions_by_name["project_update"].parameters["additionalProperties"]
        is False
    )
    assert tools.definitions_by_name["project_update"].parameters["properties"]["name"][
        "enum"
    ] == ["brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md"]


def test_project_memory_is_bounded_and_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    repository = tmp_path / "repo"
    repository.mkdir()
    registry = ProjectRegistry(root)
    project = registry.create_project("demo", "demo", repository)
    registry.initialize_memory(project.project_id)
    memory = root / project.project_id / "memory"
    (memory / "brief.md").write_text("x" * 100, encoding="utf-8")
    assert all(
        name != "brief.md"
        for name, _ in registry.load_memory(project.project_id, byte_cap=10)
    )
    (memory / "state.md").unlink()
    (memory / "state.md").symlink_to(memory / "brief.md")
    with pytest.raises(ProjectRegistryError):
        registry.load_memory(project.project_id)


def test_project_registry_serializes_concurrent_creates_and_rejects_hardlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "projects"
    registry = ProjectRegistry(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        projects = list(
            pool.map(
                lambda index: registry.create_project(f"project-{index}", "scope"),
                range(12),
            )
        )
    assert len({project.project_id for project in projects}) == 12
    project = projects[0]
    memory = root / project.project_id / "memory"
    external = tmp_path / "external.md"
    external.write_text("secret", encoding="utf-8")
    (memory / "brief.md").unlink()
    (memory / "brief.md").hardlink_to(external)
    with pytest.raises(ProjectRegistryError):
        registry.load_memory(project.project_id)

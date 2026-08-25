from pathlib import Path

import pytest

from zeta.skills.loader import SkillMeta, discover_skills, load_skill
from zeta.tools import ToolRegistry
from zeta.types import ToolCall


def test_discover_and_load_skill(tmp_path: Path) -> None:
    skill_path = tmp_path / "skills" / "review.md"
    skill_path.parent.mkdir()
    skill_path.write_text(
        "---\n"
        "name: review\n"
        "description: review code\n"
        "triggers:\n"
        "  - review\n"
        "  - inspect\n"
        "---\n\n"
        "# review\n\ncheck the diff.\n",
        encoding="utf-8",
    )

    skills = discover_skills(tmp_path)

    assert skills == [
        SkillMeta("review", "review code", ["review", "inspect"], skill_path)
    ]
    assert load_skill(skills[0]) == "# review\n\ncheck the diff."


def test_discover_packaged_skill() -> None:
    from zeta.skills.loader import discover_packaged_skills

    catalog = discover_packaged_skills()

    assert [skill.name for skill in catalog.skills] == ["review"]
    assert "keywords: review, inspect, diff" in catalog.index()


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: broken\ndescription: bad list\nkeywords: [broken\n",
        "name: broken\ndescription: no keywords\n",
        "name: broken\nunknown: field\nkeywords: [broken]\n",
    ],
)
def test_malformed_skill_frontmatter_fails_loudly(
    tmp_path: Path, frontmatter: str
) -> None:
    skill_path = tmp_path / "skills" / "broken.md"
    skill_path.parent.mkdir()
    skill_path.write_text(f"---\n{frontmatter}---\nbody", encoding="utf-8")

    with pytest.raises(ValueError, match="skill"):
        discover_skills(tmp_path)


@pytest.mark.asyncio
async def test_skill_tool_loads_and_reports_unknown_name(tmp_path: Path) -> None:
    from zeta.skills.loader import discover_packaged_skills

    registry = ToolRegistry(tmp_path)
    registry.set_skill_loader(discover_packaged_skills().load)

    loaded = await registry.execute(
        ToolCall("skill-load", "skill", {"name": "review"})
    )
    unknown = await registry.execute(
        ToolCall("skill-unknown", "skill", {"name": "missing"})
    )

    assert loaded["isError"] is False
    assert "Review the requested code change." in loaded["content"][0]["text"]
    assert unknown["isError"] is True
    assert "available skills: review" in unknown["content"][0]["text"]

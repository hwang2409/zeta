from pathlib import Path

from zeta.skills.loader import SkillMeta, discover_skills, load_skill


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

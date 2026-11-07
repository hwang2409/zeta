"""Check that a merge did not silently lose main-side Python symbols."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass

ROOTS = ("src/", "scripts/", "tests/")


@dataclass(frozen=True)
class Item:
    path: str
    kind: str
    name: str
    line: int
    value: str = ""


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL)


def files(commit: str) -> set[str]:
    out = git("ls-tree", "-r", "--name-only", commit)
    return {p for p in out.splitlines() if p.endswith(".py") and p.startswith(ROOTS)}


def source(commit: str, path: str) -> str | None:
    try:
        return git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None


def items(path: str, text: str) -> list[Item]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return []
    result: list[Item] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.append(
                Item(
                    path,
                    "async def"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else ("def" if isinstance(node, ast.FunctionDef) else "class"),
                    node.name,
                    node.lineno,
                )
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name):
                    rendered = (
                        ast.unparse(value)
                        if isinstance(value, ast.Constant)
                        and isinstance(value.value, (int, float))
                        else ""
                    )
                    result.append(
                        Item(
                            path,
                            "constant" if rendered else "assignment",
                            target.id,
                            node.lineno,
                            rendered,
                        )
                    )
                if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
                    keys = value.keys if isinstance(value, ast.Dict) else []
                    for key in keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            result.append(
                                Item(path, "string key", key.value, key.lineno)
                            )
        for node2 in ast.walk(node):
            if (
                isinstance(node2, ast.Compare)
                and len(node2.ops) == 1
                and isinstance(node2.ops[0], ast.Eq)
            ):
                left, right = node2.left, node2.comparators[0]
                if (
                    isinstance(left, ast.Name)
                    and isinstance(right, ast.Constant)
                    and isinstance(right.value, str)
                ):
                    result.append(
                        Item(
                            path,
                            "dispatch comparison",
                            f'{left.id} == "{right.value}"',
                            node2.lineno,
                        )
                    )
    return result


def marker(line: str, finding: Item) -> bool:
    prefix = f"merge-resolution: {finding.path} {finding.kind} {finding.name} ->"
    return line.startswith(prefix)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("commit", nargs="?", default="HEAD")
    parser.add_argument("--main-parent", type=int, choices=(1, 2))
    parser.add_argument("--allow", type=argparse.FileType("r"))
    parser.add_argument("--pr-body-file", type=argparse.FileType("r"))
    args = parser.parse_args()
    commit = args.commit
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if len(parents) != 2:
        print(f"{commit} is not a two-parent merge; nothing to check")
        return 0
    p1, p2 = parents
    base = git("merge-base", p1, p2).strip()
    if args.main_parent:
        main, branch = (p1, p2) if args.main_parent == 1 else (p2, p1)
    else:
        chain = set(git("rev-list", "--first-parent", "origin/main").splitlines())
        main, branch = (
            (p1, p2) if p1 in chain else ((p2, p1) if p2 in chain else (p1, p2))
        )
    text = "\n".join(
        [git("log", "-1", "--format=%B", commit)]
        + ([args.allow.read()] if args.allow else [])
        + ([args.pr_body_file.read()] if args.pr_body_file else [])
    )
    changed = set(git("diff", "--name-only", base, main).splitlines())
    # A loss can only occur in a path touched by one side or by the merge.
    all_paths = set()
    for left, right in ((base, main), (base, branch), (commit, main)):
        all_paths.update(git("diff", "--name-only", left, right).splitlines())
    all_paths = {p for p in all_paths if p.endswith(".py") and p.startswith(ROOTS)}
    main_items: dict[tuple[str, str, str], Item] = {}
    old_items: dict[tuple[str, str, str], Item] = {}
    branch_items: dict[tuple[str, str, str], Item] = {}
    merge_items: dict[tuple[str, str, str], Item] = {}
    for path in all_paths:
        for label, rev, target in (
            ("old", base, old_items),
            ("main", main, main_items),
            ("branch", branch, branch_items),
            ("merge", commit, merge_items),
        ):
            content = source(rev, path)
            if content is not None:
                for item in items(path, content):
                    target[(path, item.kind, item.name)] = item
    findings: list[Item] = []
    for key, item in main_items.items():
        path, _kind, _name = key
        # Main additions matter in untouched modules; any symbol in a changed module
        # is part of the module-level merge contract.
        candidate = key not in old_items or path in changed
        if candidate and key not in merge_items:
            # If the branch removed an item that was already in B, that is an
            # intentional branch-side deletion rather than a merge loss.
            if key not in branch_items and key in old_items:
                continue
            findings.append(item)
    for key, item in main_items.items():
        if (
            item.kind == "constant"
            and key in merge_items
            and key in branch_items
            and merge_items[key].value not in {item.value, branch_items[key].value}
        ):
            findings.append(
                Item(item.path, item.kind, item.name, item.line, item.value)
            )
    findings = [
        f
        for i, f in enumerate(findings)
        if f not in findings[:i]
        and not any(marker(line, f) for line in text.splitlines())
    ]
    if findings:
        for f in findings:
            branch_line = branch_items.get((f.path, f.kind, f.name))
            merged = merge_items.get((f.path, f.kind, f.name))
            where = f"main parent line {f.line}; branch parent line {branch_line.line if branch_line else 'absent'}; merge absent"
            if f.kind == "constant" and merged is not None:
                where = f"main parent line {f.line}; branch parent line {branch_line.line if branch_line else 'absent'}; merged value {merged.value!r}"
            print(f"{f.path}: {f.kind} {f.name}: {where}")
            print(
                f"  add: merge-resolution: {f.path} {f.kind} {f.name} -> explain intended drop"
            )
        return 1
    print(
        f"clean merge: {len(main_items)} main-side items checked across {len(all_paths)} Python files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

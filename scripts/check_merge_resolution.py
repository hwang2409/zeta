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


def git(*args: str, input: str | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, input=input, stderr=subprocess.DEVNULL
    )


def source(revision: str, path: str) -> str | None:
    try:
        return git("show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return None


def load_sources(
    revisions: list[str], paths: set[str]
) -> dict[tuple[str, str], str | None]:
    specs = [f"{revision}:{path}" for revision in revisions for path in sorted(paths)]
    if not specs:
        return {}
    output = subprocess.check_output(
        ["git", "cat-file", "--batch"],
        input=("\n".join(specs) + "\n").encode(),
        stderr=subprocess.DEVNULL,
    )
    result: dict[tuple[str, str], str | None] = {}
    offset = 0
    for revision in revisions:
        for path in sorted(paths):
            header_end = output.index(b"\n", offset)
            header = output[offset:header_end].split()
            offset = header_end + 1
            if len(header) != 3:
                result[(revision, path)] = None
                continue
            _object_id, kind, size_text = header
            size = int(size_text)
            body = output[offset : offset + size]
            offset += size
            result[(revision, path)] = body.decode() if kind != b"missing" else None
            if offset < len(output) and output[offset] == ord("\n"):
                offset += 1
    return result


def parse_items(path: str, text: str) -> list[Item]:
    tree = ast.parse(text, filename=path)
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
    return line.startswith(
        f"merge-resolution: {finding.path} {finding.kind} {finding.name} ->"
    )


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _parent_candidates(
    parents: list[str],
) -> tuple[list[int], str, str]:
    try:
        chain = set(git("rev-list", "--first-parent", "origin/main").splitlines())
        origin_main = git("rev-parse", "--verify", "origin/main").strip()
    except subprocess.CalledProcessError:
        chain = set()
        origin_main = ""
    matches = [index for index, parent in enumerate(parents) if parent in chain]
    reason = "the selected parent is on origin/main first-parent ancestry"
    if not matches and origin_main:
        matches = [
            index
            for index, parent in enumerate(parents)
            if _is_ancestor(parent, origin_main)
        ]
        reason = "the selected parent is an ancestor of origin/main"
    return matches, reason, origin_main


def _render_parent_refusal(
    commit: str, parents: list[str], origin_main: str, details: str
) -> None:
    fetch_hint = ""
    if origin_main:
        main_date = int(git("show", "-s", "--format=%ct", origin_main).strip())
        parent_dates = [
            int(git("show", "-s", "--format=%ct", parent).strip()) for parent in parents
        ]
        if main_date < min(parent_dates):
            fetch_hint = "; run git fetch origin"
    print(
        f"cannot identify exactly one main parent for {commit}: {details}; "
        "pass --main-sha <the parent that carries main> "
        "(for a train branch: parent 1 is the side put there by git merge --no-ff)"
        f"{fetch_hint}",
        file=sys.stderr,
    )


def _validate_override(
    commit: str,
    parents: list[str],
    requested: str,
    candidates: list[int],
    origin_main: str,
) -> int | None:
    if len(requested) != 40 or any(
        char not in "0123456789abcdef" for char in requested
    ):
        print(
            f"--main-sha {requested} must be a full 40-hex parent sha", file=sys.stderr
        )
        return None
    if requested not in parents:
        print(
            f"--main-sha {requested} is not a merge parent of {commit}; "
            f"pass one of the merge parents ({parents[0]} or {parents[1]})",
            file=sys.stderr,
        )
        return None
    index = parents.index(requested)
    if len(candidates) == 1:
        if index != candidates[0]:
            print(
                f"--main-sha {requested} disagrees with the identified main parent "
                f"{parents[candidates[0]]}",
                file=sys.stderr,
            )
            return None
        return index
    if candidates:
        _render_parent_refusal(commit, parents, origin_main, "both parents qualify")
        return None
    if not origin_main:
        _render_parent_refusal(commit, parents, origin_main, "neither parent qualifies")
        return None
    base = git("merge-base", "origin/main", commit).strip()
    carries = [_is_ancestor(base, parent) for parent in parents]
    permitted = (
        0
        if carries == [True, True]
        else next((i for i, carries_base in enumerate(carries) if carries_base), None)
    )
    if permitted is None or index != permitted:
        _render_parent_refusal(commit, parents, origin_main, "neither parent qualifies")
        return None
    return index


def resolve_parent(
    commit: str, parents: list[str], main_sha: str | None
) -> tuple[str, str, str] | None:
    p1, p2 = parents
    print(f"parents: 1={p1} 2={p2}")
    candidates, reason, origin_main = _parent_candidates(parents)
    if main_sha is not None:
        index = _validate_override(commit, parents, main_sha, candidates, origin_main)
        if index is None:
            return None
        reason = f"explicit --main-sha {parents[index]}"
    elif len(candidates) != 1:
        _render_parent_refusal(
            commit,
            parents,
            origin_main,
            "neither parent qualifies" if not candidates else "both parents qualify",
        )
        return None
    else:
        index = candidates[0]
    main, branch = (p1, p2) if index == 0 else (p2, p1)
    print(f"main parent: {main} ({reason})")
    return main, branch, reason


def parse_revisions(
    revisions: list[str], paths: set[str], sources: dict[tuple[str, str], str | None]
) -> tuple[
    dict[str, dict[tuple[str, str, str], Item]], tuple[str, str, SyntaxError] | None
]:
    parsed = {revision: {} for revision in revisions}
    for revision in revisions:
        for path in sorted(paths):
            text = sources[(revision, path)]
            if text is None:
                continue
            try:
                found = parse_items(path, text)
            except SyntaxError as error:
                return parsed, (revision, path, error)
            parsed[revision].update(
                {(path, item.kind, item.name): item for item in found}
            )
    return parsed, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("commit", nargs="?", default="HEAD")
    parser.add_argument(
        "--main-sha",
        help="exact object name of the merge parent that carries main",
    )
    parser.add_argument("--allow", type=argparse.FileType("r"))
    parser.add_argument("--pr-body-file", type=argparse.FileType("r"))
    args = parser.parse_args()
    commit = args.commit
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if len(parents) != 2:
        print(f"{commit} is not a two-parent merge; nothing to check")
        return 0
    resolved = resolve_parent(commit, parents, args.main_sha)
    if resolved is None:
        return 2
    main, branch, _reason = resolved
    base = git("merge-base", main, branch).strip()
    text = "\n".join(
        [git("log", "-1", "--format=%B", commit)]
        + ([args.allow.read()] if args.allow else [])
        + ([args.pr_body_file.read()] if args.pr_body_file else [])
    )
    changed = set(git("diff", "--name-only", base, main).splitlines())
    all_paths = set()
    for left, right in ((base, main), (base, branch), (commit, main)):
        all_paths.update(git("diff", "--name-only", left, right).splitlines())
    all_paths = {p for p in all_paths if p.endswith(".py") and p.startswith(ROOTS)}
    revisions = [base, main, branch, commit]
    sources = load_sources(revisions, all_paths)
    parsed, error = parse_revisions(revisions, all_paths, sources)
    if error:
        revision, path, syntax = error
        print(
            f"merge does not parse: {revision}:{path}:{syntax.lineno} (conflict marker?)",
            file=sys.stderr,
        )
        return 2
    old_items, main_items, branch_items, merge_items = (
        parsed[revision] for revision in revisions
    )
    findings: list[Item] = []
    for key, item in main_items.items():
        path, _kind, _name = key
        candidate = key not in old_items or path in changed
        if candidate and key not in merge_items:
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
    lines = text.splitlines()
    findings = [
        f
        for i, f in enumerate(findings)
        if f not in findings[:i] and not any(marker(line, f) for line in lines)
    ]
    if findings:
        for finding in findings:
            key = (finding.path, finding.kind, finding.name)
            branch_item = branch_items.get(key)
            merged = merge_items.get(key)
            where = f"main parent line {finding.line}; branch parent line {branch_item.line if branch_item else 'absent'}; merge absent"
            if finding.kind == "constant" and merged is not None:
                where = f"main parent line {finding.line}; branch parent line {branch_item.line if branch_item else 'absent'}; merged value {merged.value!r}"
            print(f"{finding.path}: {finding.kind} {finding.name}: {where}")
            print(
                f"  add: merge-resolution: {finding.path} {finding.kind} {finding.name} -> explain intended drop"
            )
        return 1
    print(
        f"clean merge: {len(main_items)} main-side items checked across {len(all_paths)} Python files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

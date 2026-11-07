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


def item_key(item: Item) -> tuple[str, str, str, str]:
    return item.path, item.kind, item.name, item.value


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


def _collection_elements(value: ast.expr) -> list[ast.expr] | None:
    if isinstance(value, (ast.Set, ast.Tuple, ast.List)):
        return list(value.elts)
    if isinstance(value, ast.Dict):
        return [key for key in value.keys if key is not None]
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "frozenset"
        and len(value.args) == 1
        and isinstance(value.args[0], (ast.Set, ast.Tuple, ast.List))
    ):
        return list(value.args[0].elts)
    return None


def parse_items(
    path: str, tree: ast.Module, collection_lookup: dict[str, list[ast.expr]]
) -> tuple[list[Item], set[tuple[str, str]]]:
    result: list[Item] = []
    unresolved: set[tuple[str, str]] = set()

    def add_memberships(node: ast.AST, qualname: str) -> None:
        for comparison in ast.walk(node):
            if not (
                isinstance(comparison, ast.Compare)
                and len(comparison.ops) == 1
                and isinstance(comparison.ops[0], (ast.In, ast.NotIn))
            ):
                continue
            right = comparison.comparators[0]
            elements = _collection_elements(right)
            if elements is None:
                name = ast.unparse(right)
                elements = collection_lookup.get(name)
                if elements is None:
                    unresolved.add((path, f"{qualname}|{ast.unparse(comparison.left)}"))
                    continue
            left = ast.unparse(comparison.left)
            for element in elements:
                result.append(
                    Item(
                        path,
                        "membership element",
                        f"{qualname}|{left}",
                        element.lineno,
                        ast.unparse(element),
                    )
                )

    def add_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        add_memberships(node, node.name)
        result.append(
            Item(
                path,
                "async def" if isinstance(node, ast.AsyncFunctionDef) else "def",
                node.name,
                node.lineno,
            )
        )

    def add_collection_items(value: ast.expr, name: str, line: int) -> None:
        rendered = (
            ast.unparse(value)
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
            else ""
        )
        result.append(
            Item(path, "constant" if rendered else "assignment", name, line, rendered)
        )
        elements = _collection_elements(value)
        if elements is not None:
            result.extend(
                Item(
                    path,
                    "collection element",
                    name,
                    element.lineno,
                    ast.unparse(element),
                )
                for element in elements
            )

    def add_assignment(node: ast.Assign | ast.AnnAssign, scope: str) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                add_collection_items(node.value, f"{scope}{target.id}", node.lineno)
        if isinstance(node.value, ast.Dict):
            result.extend(
                Item(path, "string key", key.value, key.lineno)
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )

    def add_dispatch_comparisons(node: ast.AST) -> None:
        for comparison in ast.walk(node):
            if (
                isinstance(comparison, ast.Compare)
                and len(comparison.ops) == 1
                and isinstance(comparison.ops[0], ast.Eq)
                and isinstance(comparison.left, ast.Name)
                and isinstance(comparison.comparators[0], ast.Constant)
                and isinstance(comparison.comparators[0].value, str)
            ):
                result.append(
                    Item(
                        path,
                        "dispatch comparison",
                        f'{comparison.left.id} == "{comparison.comparators[0].value}"',
                        comparison.lineno,
                    )
                )

    def visit_body(body: list[ast.stmt], scope: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_function(node)
            elif isinstance(node, ast.ClassDef):
                result.append(Item(path, "class", node.name, node.lineno))
                visit_body(node.body, f"{node.name}.")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                add_assignment(node, scope)
            add_dispatch_comparisons(node)

    visit_body(tree.body)
    return result, unresolved


def _owner_was_removed(
    key: tuple[str, str, str, str], old_items: dict, side_items: dict
) -> bool:
    _path, kind, name, _value = key
    if kind != "membership element" or "|" not in name:
        return False
    owner = name.split("|", 1)[0]
    return any(
        item.kind in {"def", "async def"} and item.name == owner
        for item_key, item in old_items.items()
        if item_key[0] == key[0]
    ) and not any(
        item.kind in {"def", "async def"} and item.name == owner
        for item_key, item in side_items.items()
        if item_key[0] == key[0]
    )


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


def _module_name(path: str) -> str:
    return path.removeprefix("src/").removesuffix(".py").replace("/", ".")


def _index_collections(
    tree: ast.Module, module: str, collections: dict[str, list[ast.expr]]
) -> None:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        elements = _collection_elements(node.value)
        if elements is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                collections[f"{module}.{target.id}"] = elements


def _revision_index(
    paths: set[str], sources: dict[tuple[str, str], str | None], revision: str
) -> tuple[dict[str, ast.Module], dict[str, list[ast.expr]]]:
    trees: dict[str, ast.Module] = {}
    collections: dict[str, list[ast.expr]] = {}
    for path in sorted(paths):
        text = sources[(revision, path)]
        if text is None:
            continue
        tree = ast.parse(text, filename=path)
        trees[path] = tree
        _index_collections(tree, _module_name(path), collections)
    return trees, collections


def _lookup_import(
    lookup: dict[str, list[ast.expr]],
    alias: ast.alias,
    module_collections: dict[str, list[ast.expr]],
) -> None:
    name = alias.asname or alias.name.rsplit(".", 1)[-1]
    prefix = alias.name + "."
    lookup.update(
        {
            f"{name}.{full[len(prefix) :]}": elements
            for full, elements in module_collections.items()
            if full.startswith(prefix)
        }
    )


def _relative_module(module: str, node: ast.ImportFrom) -> str:
    package = module.split(".")[:-1]
    module_name = ".".join(package[: len(package) - node.level + 1])
    if node.module:
        return f"{module_name}.{node.module}" if module_name else node.module
    return module_name


def _lookup_from_package(
    lookup: dict[str, list[ast.expr]],
    node: ast.ImportFrom,
    module: str,
    module_collections: dict[str, list[ast.expr]],
) -> None:
    module_name = (
        _relative_module(module, node) if node.level else module.rsplit(".", 1)[0]
    )
    for alias in node.names:
        if alias.name == "*":
            continue
        imported_name = alias.asname or alias.name
        prefix = f"{module_name}.{alias.name}" if module_name else alias.name
        lookup.update(
            {
                f"{imported_name}.{full[len(prefix) + 1 :]}": elements
                for full, elements in module_collections.items()
                if full.startswith(f"{prefix}.")
            }
        )


def _lookup_from_module(
    lookup: dict[str, list[ast.expr]],
    node: ast.ImportFrom,
    module: str,
    module_collections: dict[str, list[ast.expr]],
) -> None:
    module_name = node.module or module.rsplit(".", 1)[0]
    for alias in node.names:
        if alias.name == "*":
            continue
        imported_name = alias.asname or alias.name
        full = f"{module_name}.{alias.name}" if module_name else alias.name
        if full in module_collections:
            lookup[imported_name] = module_collections[full]


def _lookup_from_import(
    lookup: dict[str, list[ast.expr]],
    node: ast.ImportFrom,
    module: str,
    module_collections: dict[str, list[ast.expr]],
) -> None:
    if node.module is None:
        _lookup_from_package(lookup, node, module, module_collections)
    else:
        _lookup_from_module(lookup, node, module, module_collections)


def _path_lookup(
    path: str, tree: ast.Module, module_collections: dict[str, list[ast.expr]]
) -> dict[str, list[ast.expr]]:
    module = _module_name(path)
    lookup = {
        full.removeprefix(f"{module}."): elements
        for full, elements in module_collections.items()
        if full.startswith(f"{module}.")
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                _lookup_import(lookup, alias, module_collections)
        elif isinstance(node, ast.ImportFrom):
            _lookup_from_import(lookup, node, module, module_collections)
    return lookup


def parse_revisions(
    revisions: list[str], paths: set[str], sources: dict[tuple[str, str], str | None]
) -> tuple[
    dict[str, dict[tuple[str, str, str, str], Item]],
    set[tuple[str, str]],
    tuple[str, str, SyntaxError] | None,
]:
    parsed = {revision: {} for revision in revisions}
    unresolved: set[tuple[str, str]] = set()
    for revision in revisions:
        try:
            trees, module_collections = _revision_index(paths, sources, revision)
        except SyntaxError as error:
            return parsed, unresolved, (revision, error.filename or "<unknown>", error)
        for path, tree in trees.items():
            found, unresolved_in_file = parse_items(
                path, tree, _path_lookup(path, tree, module_collections)
            )
            unresolved.update(unresolved_in_file)
            parsed[revision].update({item_key(item): item for item in found})
    return parsed, unresolved, None


def _final_paths(revision: str) -> set[str]:
    return {
        path
        for path in git(
            "ls-tree", "-r", "--name-only", revision, "--", *ROOTS
        ).splitlines()
        if path.endswith(".py")
    }


def _final_index(
    revision: str,
) -> tuple[
    dict[tuple[str, str, str, str], Item], dict[tuple[str, str], str | None], str | None
]:
    paths = _final_paths(revision)
    sources = load_sources([revision], paths)
    parsed, _unresolved, error = parse_revisions([revision], paths, sources)
    error_path = error[1] if error else None
    return parsed[revision], sources, error_path


def _parse_error_status(
    error: tuple[str, str, SyntaxError],
    text: str,
    commit: str,
    final_revision: str | None,
    final_sources: dict[tuple[str, str], str | None],
    final_error_path: str | None,
) -> int:
    revision, path, syntax = error
    lines = text.splitlines()
    if any(line.startswith(f"merge-resolution: {path} parse ->") for line in lines):
        print(f"allowed merge parse diagnostic: {revision}:{path}")
        return 0
    final_text = final_sources.get((final_revision, path)) if final_revision else None
    fixed_later = (
        final_revision
        and commit != final_revision
        and final_error_path is None
        and final_text is not None
        and not any(
            line.startswith(("<<<<<<<", "=======", ">>>>>>>"))
            for line in final_text.splitlines()
        )
    )
    if fixed_later:
        print(
            f"info: merge does not parse: {revision}:{path}:{syntax.lineno} "
            f"(conflict marker?); fixed later in {final_revision}:{path}"
        )
        return 0
    print(
        f"merge does not parse: {revision}:{path}:{syntax.lineno} (conflict marker?)",
        file=sys.stderr,
    )
    return 2


def _final_match(finding: Item, item: Item) -> bool:
    if finding.kind != item.kind:
        return False
    if finding.kind in {"collection element", "membership element"}:
        return finding.value == item.value
    if finding.kind == "constant":
        return finding.name == item.name
    if finding.kind in {"def", "async def", "class"}:
        return finding.name.lstrip("_") == item.name.lstrip("_")
    return finding.name == item.name and finding.value == item.value


def _report_final_moves(
    findings: list[Item], final_items: dict[tuple[str, str, str, str], Item]
) -> list[Item]:
    remaining: list[Item] = []
    reported: set[tuple[str, str, str, str]] = set()
    for finding in findings:
        if item_key(finding) in reported:
            continue
        reported.add(item_key(finding))
        match = next(
            (item for item in final_items.values() if _final_match(finding, item)), None
        )
        if match is None:
            remaining.append(finding)
        else:
            print(
                f"info: {finding.path}: {finding.kind} {finding.name}: "
                f"moved/restored later: {match.path}:{match.line}"
            )
    return remaining


def _ordinary_findings(
    main_items: dict,
    old_items: dict,
    branch_items: dict,
    merge_items: dict,
    changed: set[str],
) -> list[Item]:
    findings = []
    for key, item in main_items.items():
        if item.kind in {"collection element", "membership element"}:
            continue
        candidate = key not in old_items or item.path in changed
        if candidate and key not in merge_items:
            if key not in branch_items and key in old_items:
                continue
            findings.append(item)
    return findings


def _constant_findings(
    main_items: dict, branch_items: dict, merge_items: dict
) -> list[Item]:
    return [
        Item(item.path, item.kind, item.name, item.line, item.value)
        for key, item in main_items.items()
        if item.kind == "constant"
        and key in merge_items
        and key in branch_items
        and merge_items[key].value not in {item.value, branch_items[key].value}
    ]


def _element_findings(
    old_items: dict,
    main_items: dict,
    branch_items: dict,
    merge_items: dict,
    unresolved: set[tuple[str, str]],
) -> list[Item]:
    findings = []
    for key, item in {**main_items, **branch_items}.items():
        if item.kind not in {"collection element", "membership element"}:
            continue
        if key in merge_items or (
            item.kind == "membership element" and (item.path, item.name) in unresolved
        ):
            continue
        main_item = main_items.get(key)
        branch_item = branch_items.get(key)
        if (main_item is None or branch_item is None) and key in old_items:
            continue
        if _owner_was_removed(key, old_items, main_items) or _owner_was_removed(
            key, old_items, branch_items
        ):
            continue
        findings.append(item)
    return findings


def _render_finding(
    finding: Item, main_items: dict, branch_items: dict, merge_items: dict
) -> None:
    key = item_key(finding)
    branch_item = branch_items.get(key)
    merged = merge_items.get(key)
    where = f"main parent line {finding.line}; branch parent line {branch_item.line if branch_item else 'absent'}; merge absent"
    if finding.kind == "constant" and merged is not None:
        where = f"main parent line {finding.line}; branch parent line {branch_item.line if branch_item else 'absent'}; merged value {merged.value!r}"
    if finding.kind in {"collection element", "membership element"}:
        parents_with_element = []
        main_item = main_items.get(key)
        if main_item:
            parents_with_element.append(f"main parent line {main_item.line}")
        if branch_item:
            parents_with_element.append(f"branch parent line {branch_item.line}")
        where = "; ".join(parents_with_element) + "; merge absent"
        print(f"{finding.path}: {finding.name} element {finding.value!r}: {where}")
    else:
        print(f"{finding.path}: {finding.kind} {finding.name}: {where}")
    print(
        f"  add: merge-resolution: {finding.path} {finding.kind} {finding.name} -> explain intended drop"
    )


def _parse_merge_data(commit: str, main: str, branch: str, args):
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
    parsed, unresolved, error = parse_revisions(revisions, all_paths, sources)
    return text, changed, all_paths, parsed, unresolved, error, revisions


def _filter_findings(findings: list[Item], text: str) -> list[Item]:
    lines = text.splitlines()
    return [
        finding
        for index, finding in enumerate(findings)
        if finding not in findings[:index]
        and not any(marker(line, finding) for line in lines)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("commit", nargs="?", default="HEAD")
    parser.add_argument(
        "--main-sha",
        help="exact object name of the merge parent that carries main",
    )
    parser.add_argument("--allow", type=argparse.FileType("r"))
    parser.add_argument("--pr-body-file", type=argparse.FileType("r"))
    parser.add_argument(
        "--final", help="revision whose final tree contains later fixes and moves"
    )
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
    text, changed, all_paths, parsed, unresolved, error, revisions = _parse_merge_data(
        commit, main, branch, args
    )
    final_revision = args.final
    if final_revision:
        final_items, final_sources, final_error_path = _final_index(final_revision)
    else:
        final_items = {}
        final_sources = {}
        final_error_path = None
    if error:
        return _parse_error_status(
            error,
            text,
            commit,
            final_revision,
            final_sources,
            final_error_path,
        )
    old_items, main_items, branch_items, merge_items = (
        parsed[revision] for revision in revisions
    )
    findings = _ordinary_findings(
        main_items, old_items, branch_items, merge_items, changed
    )
    findings.extend(_constant_findings(main_items, branch_items, merge_items))
    findings.extend(
        _element_findings(old_items, main_items, branch_items, merge_items, unresolved)
    )
    findings = _filter_findings(findings, text)
    if final_revision:
        findings = _report_final_moves(findings, final_items)
    if findings:
        for finding in findings:
            _render_finding(finding, main_items, branch_items, merge_items)
        return 1
    print(
        f"clean merge: {len(main_items)} main-side items checked across {len(all_paths)} Python files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

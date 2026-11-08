"""Check that a merge did not silently lose main-side Python symbols."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass

ROOTS = ("src/", "scripts/", "tests/")


@dataclass(frozen=True)
class Item:
    path: str
    kind: str
    name: str
    line: int
    value: str = ""
    owner: str = ""
    fingerprint: str = ""


def item_key(item: Item) -> tuple[str, str, str, str, str]:
    return item.path, item.kind, item.name, item.value, item.owner


def same_item(left: Item, right: Item) -> bool:
    return item_key(left) == item_key(right)


def _item_in(items: dict, item: Item) -> bool:
    return item_key(item) in items


def _definition_fingerprint(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


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

    def add_function(node: ast.FunctionDef | ast.AsyncFunctionDef, scope: str) -> None:
        qualname = f"{scope}{node.name}"
        add_memberships(node, qualname)
        result.append(
            Item(
                path,
                "async def" if isinstance(node, ast.AsyncFunctionDef) else "def",
                qualname,
                node.lineno,
                fingerprint=_definition_fingerprint(node),
            )
        )

    def add_collection_items(value: ast.expr, name: str, line: int) -> None:
        rendered = (
            ast.unparse(value)
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
            else ""
        )
        result.append(
            Item(
                path,
                "constant" if rendered else "assignment",
                name,
                line,
                rendered,
                fingerprint=(
                    ast.dump(value, include_attributes=False)
                    if value is not None
                    else ""
                ),
            )
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

    def add_dispatch_comparisons(node: ast.AST, owner: str) -> None:
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
                        owner=owner,
                    )
                )

    def visit_body(body: list[ast.stmt], scope: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_function(node, scope)
            elif isinstance(node, ast.ClassDef):
                qualname = f"{scope}{node.name}"
                result.append(
                    Item(
                        path,
                        "class",
                        qualname,
                        node.lineno,
                        fingerprint=_definition_fingerprint(node),
                    )
                )
                visit_body(node.body, f"{qualname}.")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                add_assignment(node, scope)
            add_dispatch_comparisons(node, scope)

    visit_body(tree.body)
    return result, unresolved


def _owner_was_removed(
    key: tuple[str, str, str, str, str], old_items: dict, side_items: dict
) -> bool:
    _path, kind, name, _value, _owner = key
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


def _owners_by_line(tree: ast.Module, count: int) -> list[str]:
    owners = ["<module>"] * count
    scopes: list[tuple[int, int, str]] = []

    def visit(body: list[ast.stmt], scope: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{scope}{node.name}"
                scopes.append((node.lineno, getattr(node, "end_lineno", node.lineno), name))
                visit(node.body, f"{name}.")

    visit(tree.body)
    for start, end, name in sorted(scopes, key=lambda item: item[1] - item[0], reverse=True):
        for line in range(max(1, start), min(count, end) + 1):
            owners[line - 1] = name
    return owners


def _diff_blocks(base: list[str], side: list[str]) -> list[tuple[int, int, int, int]]:
    matcher = difflib.SequenceMatcher(None, base, side, autojunk=False)
    return [
        (old_start, old_end, new_start, new_end)
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes()
        if tag != "equal"
    ]


def _owner_matches(item: Item, target: dict) -> list[Item]:
    return [
        candidate
        for candidate in target.values()
        if candidate.kind == item.kind and candidate.fingerprint == item.fingerprint
    ]


def _same_name_owner(item: Item, target: dict) -> list[Item]:
    return [
        candidate
        for candidate in target.values()
        if candidate.kind == item.kind and candidate.name == item.name
    ]


def _translated_owner(item: Item, target: dict) -> str | None:
    matches = _owner_matches(item, target)
    if len(matches) == 1:
        return matches[0].name
    same_name = _same_name_owner(item, target)
    return same_name[0].name if len(same_name) == 1 else None


def _owner_translation(source: dict, target: dict) -> dict[str, str]:
    return {
        item.name: translated
        for item in source.values()
        if item.kind in {"def", "async def", "class"}
        for translated in [_translated_owner(item, target)]
        if translated is not None
    }


def _line_items(path: str, text: str, owners: dict[str, str]) -> list[Item]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return []
    lines = text.splitlines()
    names = _owners_by_line(tree, len(lines))
    definitions = {
        item.name: item.fingerprint
        for item in parse_items(path, tree, {})[0]
        if item.kind in {"def", "async def", "class"}
    }
    result = []
    for number, (value, owner) in enumerate(zip(lines, names), 1):
        value = value.strip()
        if not value or value.startswith("#"):
            continue
        owner = owners.get(owner, owner)
        digest = hashlib.sha256(f"{owner}\\0{value}".encode()).hexdigest()[:12]
        result.append(
            Item(
                path, "pr-line", f"{owner}@{digest}", number, value,
                owner=owner, fingerprint=definitions.get(names[number - 1], ""),
            )
        )
    return result


def _main_overlaps(start: int, end: int, blocks: list[tuple[int, int, int, int]]) -> bool:
    for other_start, other_end, _new_start, _new_end in blocks:
        if start == end == other_start == other_end:
            return True
        if start != end and other_start <= start and other_end >= end:
            return True
        if start == end and other_start < start < other_end:
            return True
    return False


def _pr_line_candidates(
    path: str, base: list[str], branch: list[str], owners: dict[str, str]
) -> list[tuple[Item, tuple[int, int]]]:
    branch_items = _line_items(path, "\n".join(branch), owners)
    candidates = []
    for old_start, old_end, new_start, new_end in _diff_blocks(base, branch):
        for line in range(new_start + 1, new_end + 1):
            candidates.extend(
                (item, (old_start, old_end))
                for item in branch_items
                if item.line == line
            )
    return candidates


def _pr_line_items(
    path: str,
    base_text: str,
    branch_text: str,
    main_text: str,
    merge_text: str | None,
    owner_maps: tuple[dict[str, str], dict[str, str], dict[str, str]] = ({}, {}, {}),
) -> list[Item]:
    base, branch, main = (text.splitlines() for text in (base_text, branch_text, main_text))
    base_map, main_map, merge_map = owner_maps
    candidates = _pr_line_candidates(path, base, branch, {})
    main_blocks = _diff_blocks(base, main)
    base_items = _line_items(path, base_text, base_map)
    branch_items = _line_items(path, branch_text, {})
    main_items = _line_items(path, main_text, main_map)
    merge_items = _line_items(path, merge_text or "", merge_map)
    counts = [Counter(item_key(item) for item in items) for items in
              (base_items, branch_items, main_items, merge_items)]
    additions = Counter(item_key(item) for item, _interval in candidates)
    result = []
    for item, (start, end) in candidates:
        if _main_overlaps(start, end, main_blocks):
            continue
        key = item_key(item)
        required = max(0, counts[1][key] + counts[2][key] - counts[0][key])
        lost = min(additions[key], max(0, required - counts[3][key]))
        if lost and sum(item_key(candidate) == key for candidate in result) < lost:
            result.append(item)
    return result


def _group_marker(line: str, findings: list[Item]) -> list[Item]:
    prefix = "merge-resolution: "
    if not line.startswith(prefix) or " pr-lines " not in line or " ->" not in line:
        return []
    head = line[len(prefix) : line.index(" ->")]
    path, rest = head.split(" pr-lines ", 1)
    owner, digest = rest.rsplit("@", 1)
    group = [item for item in findings if item.path == path and item.owner == owner]
    values = sorted(f"{item.name}={item.value}" for item in group)
    actual = hashlib.sha256("\n".join(values).encode()).hexdigest()[:12]
    return group if actual == digest else []


def marker(line: str, finding: Item, findings: list[Item] | None = None) -> bool:
    if line.startswith(f"merge-resolution: {finding.path} {finding.kind} {finding.name} ->"):
        return True
    return bool(findings and finding in _group_marker(line, findings))


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _train_parent_qualifies(parent: str, first_parent_chain: set[str]) -> bool:
    """Return whether a parent is the head of a no-ff train chain."""
    current = parent
    while True:
        try:
            parents = git("rev-list", "--parents", "-n", "1", current).split()[1:]
        except subprocess.CalledProcessError:
            return False
        if len(parents) != 2:
            return False
        if current in first_parent_chain:
            return True
        current = parents[0]


def _direct_main_parent(
    parents: list[str], first_parent_chain: set[str], origin_main: str
) -> tuple[int | None, str]:
    matches = [
        index for index, parent in enumerate(parents) if parent in first_parent_chain
    ]
    if len(matches) == 1:
        return matches[0], "the selected parent is on origin/main first-parent ancestry"
    if len(matches) > 1:
        return None, ""
    matches = [
        index
        for index, parent in enumerate(parents)
        if origin_main and _is_ancestor(parent, origin_main)
    ]
    if len(matches) == 1:
        return matches[0], "the selected parent is an ancestor of origin/main"
    return None, ""


def _train_chain_parent(
    parents: list[str], first_parent_chain: set[str], origin_main: str
) -> tuple[int | None, str]:
    if not origin_main:
        return None, ""
    matches = [
        index
        for index, parent in enumerate(parents)
        if _train_parent_qualifies(parent, first_parent_chain)
    ]
    if len(matches) == 1:
        return matches[0], "the selected parent is the train merge chain"
    return None, ""


def _identify_main_parent(
    parents: list[str],
) -> tuple[int | None, str, str, str]:
    """Identify the main parent, or return evidence for an explicit refusal."""
    try:
        first_parent_chain = set(
            git("rev-list", "--first-parent", "origin/main").splitlines()
        )
        origin_main = git("rev-parse", "--verify", "origin/main").strip()
    except subprocess.CalledProcessError:
        first_parent_chain, origin_main = set(), ""
    index, reason = _direct_main_parent(parents, first_parent_chain, origin_main)
    if index is not None:
        return (
            index,
            reason,
            origin_main,
            "exactly one parent is on origin/main ancestry",
        )
    index, reason = _train_chain_parent(parents, first_parent_chain, origin_main)
    if index is not None:
        return index, reason, origin_main, "exactly one parent is the train merge chain"
    train_matches = [
        parent
        for parent in parents
        if origin_main and _train_parent_qualifies(parent, first_parent_chain)
    ]
    details = (
        "both parents qualify" if len(train_matches) > 1 else "neither parent qualifies"
    )
    return None, "", origin_main, details


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
    identified: int | None,
    origin_main: str,
    details: str,
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
    if identified is None:
        _render_parent_refusal(commit, parents, origin_main, details)
        return None
    index = parents.index(requested)
    if index != identified:
        print(
            f"--main-sha {requested} disagrees with the identified main parent "
            f"{parents[identified]}",
            file=sys.stderr,
        )
        return None
    return index


def resolve_parent(
    commit: str, parents: list[str], main_sha: str | None
) -> tuple[str, str, str] | None:
    p1, p2 = parents
    print(f"parents: 1={p1} 2={p2}")
    index, reason, origin_main, details = _identify_main_parent(parents)
    if main_sha is not None:
        index = _validate_override(
            commit, parents, main_sha, index, origin_main, details
        )
        if index is None:
            return None
        reason = f"explicit --main-sha {parents[index]}"
    elif index is None:
        _render_parent_refusal(commit, parents, origin_main, details)
        return None
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
    dict[tuple[str, str, str, str], Item], dict[tuple[str, str], str | None], str | None,
    list[Item],
]:
    paths = _final_paths(revision)
    sources = load_sources([revision], paths)
    parsed, _unresolved, error = parse_revisions([revision], paths, sources)
    error_path = error[1] if error else None
    return parsed[revision], sources, error_path, _final_line_items(revision, paths, sources, error)


def _final_line_items(
    revision: str, paths: set[str], sources: dict, error: tuple | None
) -> list[Item]:
    if error:
        return []
    return [
        item
        for path in paths
        if (text := sources.get((revision, path))) is not None
        for item in _line_items(path, text, {})
    ]


def _parse_error_status(
    error: tuple[str, str, SyntaxError],
    text: str,
    commit: str,
    final_revision: str | None,
    final_sources: dict[tuple[str, str], str | None],
    final_error_path: str | None,
    base: str,
) -> int:
    revision, path, syntax = error
    # The merge base view is not the merge: a later fix of the merge cannot excuse it.
    label = "merge base" if revision == base else "merge"
    lines = text.splitlines()
    if any(line.startswith(f"merge-resolution: {path} parse ->") for line in lines):
        print(f"allowed {label} parse diagnostic: {revision}:{path}")
        return 0
    final_text = final_sources.get((final_revision, path)) if final_revision else None
    fixed_later = (
        revision != base
        and final_revision
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
        f"{label} does not parse: {revision}:{path}:{syntax.lineno} (conflict marker?)",
        file=sys.stderr,
    )
    return 2


def _final_pr_line_match(finding: Item, item: Item, final_lines: list[Item]) -> bool:
    if finding.value != item.value or finding.fingerprint != item.fingerprint:
        return False
    candidates = [
        candidate for candidate in final_lines
        if candidate.kind == "pr-line"
        and candidate.value == finding.value
        and candidate.fingerprint == finding.fingerprint
    ]
    locations = []
    for candidate in candidates:
        location = (candidate.path, candidate.owner)
        if location not in locations:
            locations.append(location)
    return len(locations) == 1


def _final_match(
    finding: Item,
    item: Item,
    final_items: dict[tuple[str, str, str, str, str], Item],
    final_lines: list[Item] = (),
) -> bool:
    if same_item(finding, item):
        return True
    if finding.kind != item.kind:
        return False
    if finding.kind == "pr-line":
        return _final_pr_line_match(finding, item, final_lines)
    if finding.path == item.path:
        return False
    if finding.kind in {"def", "async def"} and "." in finding.name:
        return _unique_class_member_match(finding, item, final_items)
    if finding.kind in {"def", "async def", "class"}:
        return _unique_definition_match(finding, item, final_items)
    if finding.kind in {"assignment", "constant"}:
        return (_unique_class_member_match(finding, item, final_items)
                if "." in finding.name else _unique_assignment_match(finding, item, final_items))
    return finding.kind == "collection element" and _unique_collection_match(
        finding, item, final_items
    )


def _unique_definition_match(
    finding: Item, item: Item, final_items: dict[tuple[str, str, str, str, str], Item]
) -> bool:
    if finding.name != item.name or finding.fingerprint != item.fingerprint:
        return False
    return (
        sum(
            other.kind == finding.kind
            and other.name == finding.name
            and other.fingerprint == finding.fingerprint
            for other in final_items.values()
        )
        == 1
    )


def _unique_class_member_match(
    finding: Item, item: Item, final_items: dict[tuple[str, str, str, str, str], Item]
) -> bool:
    owner = finding.name.rsplit(".", 1)[0]
    classes = [
        candidate
        for candidate in final_items.values()
        if candidate.kind == "class" and candidate.name == owner
    ]
    if len(classes) != 1 or classes[0].path != item.path:
        return False
    return (
        finding.name == item.name
        and finding.fingerprint == item.fingerprint
        and item.path == classes[0].path
    )


def _unique_assignment_match(
    finding: Item, item: Item, final_items: dict[tuple[str, str, str, str, str], Item]
) -> bool:
    if finding.owner or item.owner or finding.name != item.name:
        return False
    if finding.fingerprint != item.fingerprint:
        return False
    return (
        sum(
            other.kind in {"assignment", "constant"}
            and not other.owner
            and other.name == finding.name
            and other.fingerprint == finding.fingerprint
            for other in final_items.values()
        )
        == 1
    )


def _unique_collection_match(
    finding: Item, item: Item, final_items: dict[tuple[str, str, str, str, str], Item]
) -> bool:
    if finding.name != item.name or finding.value != item.value:
        return False
    collections = [
        candidate
        for candidate in final_items.values()
        if candidate.kind in {"assignment", "constant"}
        and candidate.name == finding.name
        and any(
            element.kind == "collection element"
            and element.name == candidate.name
            and element.path == candidate.path
            for element in final_items.values()
        )
    ]
    return (
        len(collections) == 1
        and item.path == collections[0].path
        and any(
            candidate.kind == "collection element"
            and candidate.name == finding.name
            and candidate.value == finding.value
            and candidate.path == item.path
            for candidate in final_items.values()
        )
    )


def _find_final_match(
    finding: Item,
    final_items: dict[tuple[str, str, str, str], Item],
    final_lines: list[Item],
    consumed_lines: set[int],
) -> Item | None:
    candidates = final_lines if finding.kind == "pr-line" else list(final_items.values())
    for index, item in enumerate(candidates):
        if index not in consumed_lines and _final_match(finding, item, final_items, final_lines):
            if finding.kind == "pr-line":
                consumed_lines.add(index)
            return item
    return None


def _report_final_moves(
    findings: list[Item],
    final_items: dict[tuple[str, str, str, str], Item],
    final_lines: list[Item],
) -> list[Item]:
    remaining: list[Item] = []
    consumed_lines: set[int] = set()
    for finding in findings:
        match = _find_final_match(finding, final_items, final_lines, consumed_lines)
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
    for item in main_items.values():
        if item.kind in {"collection element", "membership element"}:
            continue
        candidate = not _item_in(old_items, item) or item.path in changed
        if candidate and not _item_in(merge_items, item):
            if not _item_in(branch_items, item) and _item_in(old_items, item):
                continue
            findings.append(item)
    return findings


def _constant_findings(
    main_items: dict, branch_items: dict, merge_items: dict
) -> list[Item]:
    return [
        Item(
            item.path,
            item.kind,
            item.name,
            item.line,
            item.value,
            item.owner,
            item.fingerprint,
        )
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
    if finding.kind == "pr-line":
        where = f"pr/branch parent line {finding.line}; merge absent"
    if finding.kind == "constant" and merged is not None:
        where = f"main parent line {finding.line}; branch parent line {branch_item.line if branch_item else 'absent'}; merged value {merged.value!r}"
    if finding.kind == "pr-line":
        print(f"{finding.path}: pr-line {finding.name}: {where}; text {finding.value!r}")
        print(
            f"  add: merge-resolution: {finding.path} pr-lines "
            f"{finding.owner}@<group-digest> -> explain intended drop"
        )
        return
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


def _base_view(main: str, branch: str) -> str:
    bases = git("merge-base", "--all", main, branch).splitlines()
    base = bases[0]
    for index, other in enumerate(bases[1:], 1):
        tree = _merge_bases(base, other, bases[0])
        base = (
            git(
                "-c", "user.name=zeta merge checker",
                "-c", "user.email=zeta@example.invalid",
                "commit-tree", tree, "-p", base, "-p", other,
                input="virtual merge base\n",
            ).strip()
            if index < len(bases) - 1
            else tree
        )
    return base


def _merge_bases(left: str, right: str, real_base: str) -> str:
    result = subprocess.run(
        ["git", "merge-tree", "--write-tree", "-z", "--name-only", "--no-messages", left, right],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    tree, *conflicted = [field for field in result.stdout.split("\0") if field]
    if not conflicted:
        return tree
    # A real common ancestor can only over-report a loss; Git's conflicted tree has markers.
    paths = sorted(set(conflicted))
    print(f"merge bases conflict on {', '.join(paths)}; using base {real_base} for them")
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "index")}

        def index_git(*args: str, input: str = "") -> str:
            return subprocess.check_output(
                ["git", "--literal-pathspecs", *args], text=True, input=input, env=env
            )

        index_git("read-tree", tree)
        listed = "".join(f"{path}\0" for path in paths)
        index_git("update-index", "--force-remove", "-z", "--stdin", input=listed)
        entries = index_git("ls-tree", "-r", "-z", real_base, "--", *paths)
        index_git("update-index", "-z", "--index-info", input=entries)
        return index_git("write-tree").strip()


def _merge_paths(base: str, main: str, branch: str, commit: str) -> tuple[set[str], set[str]]:
    changed = set(git("diff", "--name-only", base, main).splitlines())
    all_paths = {
        path
        for left, right in ((base, main), (base, branch), (commit, main))
        for path in git("diff", "--name-only", left, right).splitlines()
        if path.endswith(".py") and path.startswith(ROOTS)
    }
    return changed, all_paths


def _path_line_findings(path: str, revisions: list[str], sources: dict, parsed: dict) -> list[Item]:
    texts = [sources.get((revision, path)) for revision in revisions]
    if any(text is None or not text for text in texts[:3]) or texts[3] is None:
        return []
    path_items = [
        {key: item for key, item in parsed[revision].items() if item.path == path}
        for revision in revisions
    ]
    maps = (
        _owner_translation(path_items[0], path_items[2]),
        _owner_translation(path_items[1], path_items[2]),
        _owner_translation(path_items[3], path_items[2]),
    )
    return _pr_line_items(path, texts[0], texts[2], texts[1], texts[3], maps)


def _parse_merge_data(commit: str, main: str, branch: str, args):
    base = _base_view(main, branch)
    text = "\n".join(
        [git("log", "-1", "--format=%B", commit)]
        + ([args.allow.read()] if args.allow else [])
        + ([args.pr_body_file.read()] if args.pr_body_file else [])
    )
    changed, all_paths = _merge_paths(base, main, branch, commit)
    revisions = [base, main, branch, commit]
    sources = load_sources(revisions, all_paths)
    parsed, unresolved, error = parse_revisions(revisions, all_paths, sources)
    if error:
        return text, changed, all_paths, parsed, unresolved, error, revisions, []
    line_findings = [
        finding
        for path in all_paths
        for finding in _path_line_findings(path, revisions, sources, parsed)
    ]
    return text, changed, all_paths, parsed, unresolved, error, revisions, line_findings

def _print_group_markers(findings: list[Item]) -> None:
    groups: dict[tuple[str, str], list[Item]] = {}
    for finding in findings:
        if finding.kind == "pr-line":
            groups.setdefault((finding.path, finding.owner), []).append(finding)
    for (path, owner), group in sorted(groups.items()):
        values = sorted(f"{item.name}={item.value}" for item in group)
        digest = hashlib.sha256("\n".join(values).encode()).hexdigest()[:12]
        print(f"suggested override: merge-resolution: {path} pr-lines {owner}@{digest} -> <reason>")


def _filter_findings(findings: list[Item], text: str) -> list[Item]:
    lines = text.splitlines()
    return [
        finding
        for index, finding in enumerate(findings)
        if finding not in findings[:index]
        and not any(marker(line, finding, findings) for line in lines)
    ]


def _borrow_skip_reason(pull: dict, author: str) -> str | None:
    head_repo = (pull.get("head") or {}).get("repo") or {}
    base = pull.get("base") or {}
    if head_repo.get("full_name") != (base.get("repo") or {}).get("full_name"):
        return "fork head"
    if base.get("ref") != "main":
        return f"base {base.get('ref')}"
    if (pull.get("user") or {}).get("login") != author:
        return "other author"
    return None


def _candidate_lines(
    data: dict, findings: list[Item], remaining: list[Item]
) -> tuple[dict[int, list[str]], list[str]]:
    found: dict[int, list[str]] = {}
    skipped = []
    # The PR being checked always contains its own merges; its body is already applied.
    for pull in (pull for pull in data["pulls"] if pull.get("number") != data["pr"]):
        skip = _borrow_skip_reason(pull, data["author"])
        lines = sorted(
            {
                line
                for line in (pull.get("body") or "").splitlines()
                if any(marker(line, item, findings) for item in remaining)
            }
        )
        if skip:
            skipped.append(f"#{pull.get('number')} ({skip})")
        elif lines:
            found[pull["number"]] = lines
    return found, skipped


def borrowed_overrides(
    lookup: str, commit: str, findings: list[Item], remaining: list[Item]
) -> tuple[str, str]:
    """Return override lines borrowed from the PR that introduced this merge, and why.

    A stacked PR contains its base PR's merge commits; the base PR body already
    explains them. Any doubt returns no text, so the check is never weaker.
    """
    try:
        data = json.loads(lookup)
        if data.get("commit") != commit:
            return "", f"lookup is for commit {data.get('commit')}, not {commit}"
        pulls = data.get("pulls")
        if data.get("error") or not isinstance(pulls, list):
            message = pulls.get("message") if isinstance(pulls, dict) else pulls
            return "", f"PR lookup failed: {data.get('error')}: {message}"
        found, skipped = _candidate_lines(data, findings, remaining)
    except (ValueError, AttributeError, KeyError, TypeError) as exc:
        return "", f"PR lookup is unreadable: {exc!r}"
    numbers = ", ".join(f"#{number}" for number in sorted(found))
    if len({tuple(lines) for lines in found.values()}) > 1:
        return "", f"conflicting override lines in {numbers}"
    if not found:
        return "", "no PR with a matching override line; skipped: " + (", ".join(skipped) or "none")
    return "\n".join(next(iter(found.values()))), f"override lines from {numbers}"


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
        "--borrow-file",
        type=argparse.FileType("r"),
        help="JSON {commit, pr, author, pulls: <GET commits/<sha>/pulls>} for stacked PRs",
    )
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
    text, changed, all_paths, parsed, unresolved, error, revisions, line_findings = _parse_merge_data(
        commit, main, branch, args
    )
    final_revision = args.final
    if (
        final_revision
        and git("rev-parse", f"{commit}^{{commit}}").strip()
        == git("rev-parse", f"{final_revision}^{{commit}}").strip()
    ):
        final_revision = None
    final_items = {}
    final_sources = {}
    final_error_path = None
    final_lines = []
    if error and final_revision:
        final_items, final_sources, final_error_path, final_lines = _final_index(final_revision)
    if error:
        return _parse_error_status(
            error,
            text,
            commit,
            final_revision,
            final_sources,
            final_error_path,
            revisions[0],
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
    findings.extend(line_findings)
    if final_revision and findings:
        final_items, final_sources, final_error_path, final_lines = _final_index(final_revision)
        findings = _report_final_moves(findings, final_items, final_lines)
    all_findings = findings
    findings = _filter_findings(all_findings, text)
    if findings and args.borrow_file:
        full = git("rev-parse", f"{commit}^{{commit}}").strip()
        extra, reason = borrowed_overrides(args.borrow_file.read(), full, all_findings, findings)
        print(f"{'borrowed' if extra else 'did not borrow'} overrides: {reason}")
        findings = _filter_findings(all_findings, f"{text}\n{extra}")
    if findings:
        for finding in findings:
            _render_finding(finding, main_items, branch_items, merge_items)
        _print_group_markers(findings)
        return 1
    print(
        f"clean merge: {len(main_items)} main-side items checked across {len(all_paths)} Python files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

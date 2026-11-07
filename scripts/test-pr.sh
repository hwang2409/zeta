#!/bin/sh
set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"
record=
args=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --record)
            [ "$#" -ge 2 ] || { echo "test-pr: --record needs a path" >&2; exit 2; }
            record=$2
            shift 2
            ;;
        --record=*)
            record=${1#*=}
            shift
            ;;
        *)
            args="$args
$1"
            shift
            ;;
    esac
done

if [ -z "$record" ]; then
    branch=$(git branch --show-current | tr '/' '-')
    short_head=$(git rev-parse --short HEAD)
    record="${ZETA_TEST_RECORD_DIR:-$HOME/.zeta/scratch/test-runs}/$branch-$short_head.txt"
fi
mkdir -p "$(dirname "$record")"

imported=$(uv run python -c 'import zeta; print(zeta.__file__)')
source=$(cd "$root/src" && pwd -P)
case "$(realpath "$imported")" in
    "$source"/*) ;;
    *)
        echo "test-pr: refusing: zeta imports from $imported, outside $source" >&2
        exit 2
        ;;
esac

head=$(git rev-parse HEAD)
dirty_digest=$( {
    git diff HEAD
    git ls-files --others --exclude-standard -- tests | while IFS= read -r file; do
        printf '\n--- %s ---\n' "$file"
        cat "$file"
    done
} | sha256sum | cut -d' ' -f1 )

changed=$( {
    git diff --diff-filter=ACMRTUXB --name-only "$(git merge-base origin/main HEAD)"
    git diff --diff-filter=ACMRTUXB --name-only
    git ls-files --others --exclude-standard -- tests
} | awk '$0 ~ "^tests/.*test_[^/]*[.]py$"' | sort -u )

if [ -n "$args" ]; then
    targets=$(printf '%s\n' "$args" | sed '/^$/d')
elif [ -n "$changed" ]; then
    targets=$changed
else
    targets=
fi

{
    printf 'head sha: %s\n' "$head"
    printf 'dirty-tree digest: %s\n' "$dirty_digest"
    printf 'zeta.__file__: %s\n' "$imported"
    printf 'record: %s\n' "$record"
} >"$record"

if [ -z "$targets" ]; then
    echo "test-pr: no changed tests; no tests run"
    printf 'result: no changed tests; no tests run\n' >>"$record"
    exit 0
fi

run() {
    label=$1
    shift
    command="$*"
    printf '\ncommand: %s\n' "$command" | tee -a "$record"
    set +e
    output=$("$@" 2>&1)
    status=$?
    set -e
    printf '%s\n' "$output"
    printf '%s\n' "$output" | tee -a "$record" >/dev/null
    counts=$(printf '%s\n' "$output" | grep -Eo '[0-9]+ (passed|failed|skipped|xfailed|xpassed|error|errors)' | tr '\n' ' ' || true)
    printf 'run: %s\ncounts: %s\nexit code: %s\n' "$label" "${counts:-not reported}" "$status" >>"$record"
    if [ "$status" -ne 0 ]; then
        echo "test-pr: failed: $label (exit $status)" >&2
        exit "$status"
    fi
}

set --
while IFS= read -r target; do
    [ -n "$target" ] || continue
    set -- "$@" "$target"
    run "file $target" uv run pytest -q "$target"
done <<EOF
$targets
EOF
run "all changed tests" uv run pytest -q "$@"
echo "test-pr: passed"

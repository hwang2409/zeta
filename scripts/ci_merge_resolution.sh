#!/usr/bin/env bash
set -eu

body_file=""
cleanup() { [ -z "$body_file" ] || rm -f "$body_file"; }
trap cleanup EXIT

if [ "$EVENT_NAME" = pull_request ]; then
  body_file=$(mktemp)
  gh pr view "$PR_NUMBER" --json body -q .body > "$body_file"
  git fetch origin main --no-tags
  commits=$(git rev-list --merges --first-parent origin/main.."$HEAD_SHA")
  if git rev-parse --verify HEAD^2 >/dev/null 2>&1; then
    commits="$commits $(git rev-parse HEAD)"
  fi
else
  FINAL_SHA=$(git rev-parse HEAD)
  commits="$(git rev-parse HEAD)"
  subject=$(git log -1 --format=%s)
  pr_number=$(printf '%s\n' "$subject" | sed -nE 's/^Merge pull request #([0-9]+).*/\1/p')
  if [ -n "$pr_number" ] && body_file=$(mktemp) && gh pr view "$pr_number" --json body -q .body > "$body_file"; then
    :
  else
    [ -z "$body_file" ] || rm -f "$body_file"
    body_file=""
    echo "::notice title=merge resolution::PR explanations were not loaded"
  fi
fi

commits=$(printf '%s\n' "$commits" | sed '/^$/d' | sort -u)
failed=0
for commit in $commits; do
  echo "checking merge resolution: $commit"
  args=("$commit" --final "$FINAL_SHA")
  [ -z "$body_file" ] || args+=(--pr-body-file "$body_file")
  # train/<lane>-<n> is built from origin/main with git merge --no-ff of each PR head.
  case "${HEAD_REF:-}" in train/*) args+=(--train-head "$HEAD_SHA") ;; esac
  set +e
  output=$(python "${CHECK_MERGE_RESOLUTION_SCRIPT:-scripts/check_merge_resolution.py}" "${args[@]}" 2>&1)
  status=$?
  set -e
  printf '%s\n' "$output"
  if [ "$status" -eq 0 ]; then
    continue
  fi
  if [ "$status" -eq 2 ] && printf '%s\n' "$output" | grep -q 'cannot identify exactly one main parent'; then
    echo "::warning title=merge resolution::skipping $commit because the main parent is ambiguous"
    continue
  fi
  echo "::error title=merge resolution::$commit failed (exit $status)"
  failed=1
done
exit "$failed"

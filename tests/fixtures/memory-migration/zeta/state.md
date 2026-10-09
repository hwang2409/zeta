# Current state

As of 2026-10-09, the latest project state includes:
- PR #415 merged into `main` as `756e3be5`, from exact head `60d77e15c7e0a8ccd825af89b06445628cdc8e8a`. The repair enforces one strict ASCII canonical-path rule across sender preparation, local publication, the SSH verifier, transfer digests, and project CAS digests. NFC/NFD ambiguity is rejected, case-only archive collisions fail closed, and no project payload is published for rejected paths.
- PR #415 validation passed exact-head CI 4/4, the full suite with 5,374 passed and 1 skipped, focused memory/project suites with 396 passed, and the six new NFC/NFD regression cases. Final review found no blockers; the worktree was clean after merge.
- PR #415 also merged the dormant memory-rewrite storage, publication, transfer, schema, and migration implementation; no project uses the new format yet.
- PR #423 merged into `main` as `f368df57` after repair head `290c93e701f86d022e3690895f1271541e52aa4a`. Review found no blockers; exact-head CI passed, including Ubuntu's successful retry with 5,275 passed and 2 skipped.
- PR #422 remains merged into `main` as `6473eeaf`.
- As of 2026-10-09, CI on `756e3be5` passed all four checks.
- As of 2026-10-09, the earlier no-background-worker state is superseded: a background worker is active in worktree `.worktrees/at-refs-text` on branch `fix/at-refs-never-block-send`, investigating the composer behavior for non-file `@` references.
- As of 2026-10-09, an overflow-recovery worker is active in `.worktrees/overflow-recovery` on branch `fix/eviction-nested-markers-overflow-recovery`. It is investigating nested eviction markers incorrectly invalidated by #423, robust provider-overflow recovery, and provider-limit safety margins; no PR, merge, or validation completion is recorded.
- As of 2026-10-09, memory-rewrite PR 6 activation work is on branch `feat/memory-core-pr6-activation` in a separate worktree as PR #425 at head `9d15e9097270c3d954ab67153c3a5cde879bf080`; exact-head CI passed 4/4 and the full suite passed 5,381 tests with 1 skipped, but the PR is not merge-ready and is not merged.
- PR #425 review confirmed the shared lock, atomic publication, format dispatch, stale format-1 write rejection, and mixed-format sync refusal. It found a blocker because profile/schema transactions cannot currently be undone, plus major issues with self-fulfilling capability reporting and missing real migration/updater race and publication-failure coverage. A fix is in progress; no real-project migration has occurred.
- As of 2026-10-09, investigation of session `9bd2840975ef42e490c268b433ea0096` found a high-confidence compound regression on build `756e3be5`: PR #420 undercounted the provider context after PR #423 invalidated the resumed session's earlier eviction view, making forced eviction a no-op. Zeta estimated 831,610 tokens versus Anthropic's 1,000,386, an actual/estimate ratio of 1.20295×; the pre-#420 estimate was 1,046,517. The session failed with `context_length_exceeded`, and `/compact` did not unstick it (`compacted=false`). Generic overflow-recovery tests passed 2/2; no repair is merged yet.
- The user is not resuming session `9bd2840975ef42e490c268b433ea0096`; they started a new session, so the incident is not urgent for their current work.

## Active follow-up
- As of 2026-10-09, no active PR #415 blocker remains. Its shared `tree-v2:` digest and strict ASCII project-path rule are merged and validated. Unsupported non-ASCII project paths intentionally fail closed during synchronization.
- As of 2026-10-09, activation of the merged memory-rewrite format is approved by the user but remains pending PR 6 completion, review repair, and per-project migration validation.
- As of 2026-10-09, removal of the model-facing `project_update` tool remains tracked separately in deslop PR 4 and is homelab-gated.
- As of 2026-10-09, the older risk that existing eviction transformations can grow under an adversarial supplied `token_counter` remains tracked.
- As of 2026-10-09, short receipts for `inbox` results remain a user-supported follow-up; the minimum-gain rule with summary fallback remains an open design discussion.
- As of 2026-10-09, literal user messages containing text such as `@get` remain an unresolved submission UX issue when the referenced path is not a real local file; investigation is active in `.worktrees/at-refs-text` on branch `fix/at-refs-never-block-send`.
- As of 2026-10-09, the context-overflow regression remains an unmerged follow-up. The ranked remediation findings are accepting valid nested compaction markers and rebuilding invalid views, lower-target forced eviction with one retry, a provider-limit safety margin, and calibration/persistence of estimator-versus-provider usage.

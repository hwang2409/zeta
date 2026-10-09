# Changelog

## 2026-10-09
- PR #415 merged into `main` as `756e3be5` from exact head `60d77e15c7e0a8ccd825af89b06445628cdc8e8a`.
- PR #415's final repair uses one strict ASCII canonical-path rule across sender preparation, local publication, the shipped SSH verifier, transfer authentication, and project CAS digesting. Unsupported NFC/NFD names are rejected, case-only archive collisions fail closed, and rejected paths publish no payload.
- Six NFC/NFD regression cases passed. Focused memory/project validation passed 396 tests, the full suite passed 5,374 tests with 1 skipped and 8 warnings, and exact-head Ruff, browser, Ubuntu, and macOS CI passed 4/4. Final review found no blockers.
- PR #415's merged memory-rewrite implementation includes versioned atomic memory saving, shared local/SSH publication code, single-pass transfer archive preparation and fingerprinting, schema validation, and migration support. The new format remains dormant because no project uses it yet.
- PR #423 merged into `main` as `f368df57`; persisted-view preparation applies deterministic receipt constructors to eligible raw rows while preserving protection and provider-payload invariants.
- Investigation of session `9bd2840975ef42e490c268b433ea0096` found a high-confidence compound context-overflow regression on build `756e3be5`: the post-#420 estimator undercounted the provider request by 16.87% (831,610 estimated versus 1,000,386 reported), and PR #423 invalidated the resumed session's earlier eviction view. `/compact` did not recover the session (`compacted=false`). Generic overflow-recovery tests passed 2/2; no code repair was merged.

## 2026-10-08
- PR #422 merged into `main` as `6473eeaf`. Recallable receipts cover `inbox`, `project`, `recall_history`, and `run_background` results while omitting message bodies; eviction replaces a receipt only when it is strictly smaller under the supplied `token_counter`.
- PR #418 merged into `main` as `7b9f3224`; passive sent-message claim-status visibility records one quiet note on the sender's next turn without wake-up or polling.
- PR #421 merged into `main` as `396d0223`; the merge removed the `compaction: summary` line from `/status`.
- PR #420 merged into `main` as `cad2582d`; eviction-only normalization preserves provider-bound payload bytes.
- PR #419 merged into `main` as `0570b672`; focused validation passed 442 tests and the full suite passed 5,197 tests with 1 skipped.

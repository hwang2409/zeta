# Decisions

## 2026-10-09: PR #415 ASCII canonical-path repair is merged
The earlier decision that PR #415 remained blocked by cross-platform Unicode path canonicalization is Superseded on 2026-10-09 by merge commit `756e3be5`, from exact head `60d77e15c7e0a8ccd825af89b06445628cdc8e8a`. One strict ASCII canonical-path function is used by sender preparation, local publication, the shipped SSH verifier, transfer authentication, and project CAS digests. NFC and NFD names are rejected, case-only archive collisions fail closed, and rejected paths do not publish project payloads. Final review found no blockers; exact-head CI passed 4/4 and the full suite passed with 5,374 passed and 1 skipped.

## 2026-10-09: PR #415 Unicode path canonicalization remains a merge blocker — Superseded
The earlier decision that PR #415 at `63d67ba8162ccff80d153b63b8b006d102620411` was not merge-ready because raw NFC and NFD filesystem spellings produced different transfer and project digests is Superseded on 2026-10-09 by the merged ASCII canonical-path repair at `60d77e15c7e0a8ccd825af89b06445628cdc8e8a`.

## 2026-10-09: PR #415 framed digest and compatibility repair validated
The shared tagged `tree-v2:` digest at `63d67ba8162ccff80d153b63b8b006d102620411` used unambiguous file boundaries for transfer authentication and project CAS state. Whole-project CAS values were ephemeral within one synchronization transaction, and legacy baselines used a safe compatibility path. This history remains superseded in readiness status by the merged path-canonicalization repair.

## 2026-10-09: PR #423 is merged
The earlier decision that PR #423 at `1eaef5a` had fixed its three safety blockers but remained under investigation is Superseded on 2026-10-09 by repair head `290c93e701f86d022e3690895f1271541e52aa4a`, merged into `main` as `f368df57`. Persisted-view preparation applies the normal deterministic receipt constructors to eligible raw rows, retains protection and provider-payload invariants, and leaves a second pass byte-identical.

## 2026-10-08: PR #422 is merged
The earlier decision that PR #422 was merge-ready but unmerged at `a8c23c577917558b5736ef1affab35c45a9a332d` is Superseded on 2026-10-08 by merge commit `6473eeaf`. Its size policy uses the eviction-supplied `token_counter`, replacing a receipt only when it is strictly smaller than the original.

## 2026-10-08: Old eviction receipts are combined into range receipts
The user agreed that “combining old receipts into one range receipt would be effective as well.” The agreed design combines consecutive old eviction receipts into a deterministic range receipt containing the sequence range, counts by kind, and the corresponding `recall_history` range; it does not use a model summary and leaves the protected recent window unchanged.

## 2026-10-08: Short `inbox` receipts are the favored follow-up
The user said, “i do think we should do the short receipts for inbox results though.” PR #422 delivered bounded, recallable receipts covering `inbox`, `project`, `recall_history`, and `run_background` results while omitting message bodies.

## 2026-10-09: New memory format activation is approved
The user said, “zeta is restarted now, we can switch the new memory format, works for me.” Activation of the new memory format is approved after the per-project capability and migration implementation is available; this supersedes the prior pending-only authorization for switching the format.

## 2026-10-09: Literal `@get` text is expected to remain sendable
The user said, “i tried to paste a message that had an @get text, but i couldn't send it because it wasn't a real file on my laptop. i feel like that is kind of weird behaviour to not let the user send messages like that”. This records the user's expectation that literal message text containing `@get` should not be rejected merely because the referenced path is absent.

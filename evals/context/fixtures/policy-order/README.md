# Gate policy

Fix policy evaluation and audit rendering. Before code, read `POLICY.md` and all `evidence/`. The historical compatibility rule near the top of POLICY.md is normative even though one trace proposes a simpler first-match implementation. Keep `decide(rules, subject)` returning `(allowed, reason)`.

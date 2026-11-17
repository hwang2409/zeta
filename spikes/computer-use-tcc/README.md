# computer-use slice 0 — TCC spike (branch note)

**This directory is a branch note on `feature/computer-use`. It will be removed before
the port to main.** It exists to answer four questions on henry's mac before we commit
to a signing/architecture choice for the computer-use tool.

Everything here is harmless: capture writes one screenshot to a temp file and prints its
size; the input test nudges the cursor 1 point and moves it straight back. Nothing clicks
or types into any app, nothing is uploaded, and no `tccutil` is run. Read-only checks use
`CGPreflightScreenCaptureAccess` / `AXIsProcessTrusted`, which never prompt.

## what it decides
1. can the ad-hoc python zeta runs as (under tmux → launchd) capture the screen and post a
   `CGEvent` after a grant, and **which process name** owns the System Settings entry?
2. does that grant survive a python change / `uv` reinstall?
3. ad-hoc vs a local **self-signed code-signing identity**: after granting, does a rebuild
   (new cdhash) keep the grant? this decides the signing identity for the feature branch.
4. (optional) does it work from a LaunchAgent-like context?

## files
- `run.sh` — the one entry point (subcommands below).
- `cuspike.swift` — tiny signed helper; `preflight` (read-only), `request` (prompts), `version`.
- `cuspike.py` — python (pyobjc) path; `preflight`, `request`, `capture`, `input`.
- `build/` — compiled binary + generated sources (gitignored).

Results append to `~/.zeta/scratch/computer-use/slice0-results.md`. End with `./run.sh report`
and send that file (plus the entry names you note in step 4) back to the lead.

## important: run it from your normal tmux/zeta terminal
Run every command from the same terminal zeta runs in (inside `tmux`), **not** a fresh
Terminal.app window. The whole point of Q1 is which process macOS blames under
tmux → launchd, and that differs by terminal.

Each `./run.sh check` / `capture` uses `uv run`, which resolves to the **same real binary**
your `uv`-tool zeta python points at today (`miniforge .../python3.13`), so the grant is
representative of real zeta sessions.

---

## procedure (≈ 10–15 min)

```
cd spikes/computer-use-tcc
```

1. **facts** — `./run.sh env`  (~5 s). Records the process tree, the python TCC identity, and
   that there are 0 code-signing identities today.
2. **build the helper (ad-hoc)** — `./run.sh build`  (~10 s). Compiles + ad-hoc-signs `cuspike`
   with the stable identifier `com.henry.zeta.cuspike`.
3. **check before granting** — `./run.sh check`. Expect all four lines `[FAIL] ... not granted`.
4. **grant** — `./run.sh grant`. Two prompts appear and System Settings opens. In
   **System Settings → Privacy & Security**, enable the python entry **and** the `cuspike`
   entry under **both** *Screen Recording* and *Accessibility*.
   → **Q1 answer:** write down the **exact name** each list shows (e.g. `python3.13`, `tmux`,
   `cuspike`, or the terminal). This is the key observation; the script cannot read it.
5. **check after granting** — `./run.sh check`. Expect `[PASS] ... GRANTED` for what you enabled.
6. **capture** — `./run.sh capture`. Expect `PASS capture: WxH px, N bytes -> .../cuspike-*.png`.
   The screenshot is written under a private mode-700 temporary directory.
7. **input** — `./run.sh input`. Expect `PASS input: moved cursor ...`.
   → Q1 confirmed: the ad-hoc python under tmux can capture + inject after the grant.

**Q2 — does the grant survive a python change / reinstall?**

8. **different python** — `CUSPIKE_PY=3.12 ./run.sh check`. uv fetches a *different* real python
   binary (homebrew `python@3.12`, different path + cdhash). Expect `[FAIL] not granted` even
   though the default python is granted. This is a proxy observation that grants are keyed to the
   exact binary; it does not itself prove an in-place reinstall loses the original grant.
   The in-place reinstall question remains unmeasured. *(Optional, more faithful but
   disruptive: reinstall zeta's python with `uv tool install --reinstall ...`, then re-run
   `./run.sh check`; you will need to re-grant afterwards.)*

**Q3 — ad-hoc vs self-signed (the signing decision):**

9. **ad-hoc rebuild** — `./run.sh rebuild adhoc`. Rebuilds with a fresh nonce (real cdhash change).
   Expect `[FAIL] grant LOST after rebuild` → ad-hoc re-keys TCC on every build.
10. **make a self-signed code-signing cert** — `./run.sh cert` prints the exact steps
    (Keychain Access → Certificate Assistant; **needs your clicks and maybe your login password**;
    ~2–3 min). On macOS 26.6, set Trust → Code Signing to **Always Trust** because a
    Certificate Assistant self-signed root is not trusted by default. Verify with
    `security find-identity -v -p codesigning`.
11. **sign with the cert + grant** — `./run.sh build cert` (first use may pop
    "codesign wants to sign… → **Always Allow**"), then `./run.sh grant`, enable the `cuspike`
    entries again, then `./run.sh check` → expect `[PASS]`.
12. **cert rebuild** — `./run.sh rebuild cert`. Expect `[PASS] grant SURVIVED rebuild` → a stable
    self-signed identity keeps the grant across rebuilds. **This is the decision: self-signed over
    ad-hoc if it survives.**

**Q4 (optional) — LaunchAgent context.** Only if you have time. A LaunchAgent that runs
`cuspike preflight` and logs it confirms whether a login-agent context holds the grant; a
headless/daemon context (no active GUI login) cannot capture. Skip unless asked.

13. **report** — `./run.sh report`. Send the printed results file to the lead, plus the entry
    names you noted in step 4.

## cleanup
`./run.sh cleanup` prints these steps (it runs nothing). Do whichever you did:

1. Remove the build dir: `rm -rf build`.
2. If you explicitly added a trust override, export the certificate from Keychain Access,
   run `security remove-trusted-cert -d <exported-certificate>`, and set Trust → Code Signing
   to **Use System Defaults** before deleting the identity:
   `security remove-trusted-cert -d <exported-certificate>`.
   Set Trust to "Use System Defaults" in Keychain Access if no override exists.
3. If you made the self-signed cert (`./run.sh cert`), delete the identity:
   `security delete-identity -c "Zeta CU Spike" ~/Library/Keychains/login.keychain-db`
   (or delete `Zeta CU Spike` and its key in Keychain Access → `login`).
4. Reset the TCC grants **for this spike's helper only**, scoped by its bundle id:
   ```
   tccutil reset ScreenCapture com.henry.zeta.cuspike
   tccutil reset Accessibility com.henry.zeta.cuspike
   ```
   Never omit the bundle identifier from either command.
5. Remove screenshots, including files from older runs:
   `rm -rf "${TMPDIR:-/tmp}/cuspike-${UID}"; rm -f "${TMPDIR:-/tmp}"/cuspike-*.png`.
6. The python and `tmux`/terminal grants have no bundle id, so remove them by hand in
   **System Settings → Privacy & Security**: delete the python entry (e.g. `python3.13`) and
   the `tmux`/terminal entry you added, under **both** Screen Recording and Accessibility.

## expected conclusions (to confirm or refute)
- Q1: ad-hoc python under tmux **can** capture + inject after a grant; the entry is owned by the
  responsible process (likely `tmux` or the resolved python binary, not the terminal).
- Q2 proxy: a different python binary does not inherit the grant; the in-place reinstall case
  remains unmeasured (ad-hoc identity includes path + cdhash).
- Q3: a **self-signed code-signing identity keeps the grant across rebuilds**; ad-hoc does not →
  sign the helper with a stable self-signed identity on the branch.
- Q3 therefore supports the design's "small signed helper holds the grants" plan without needing a
  paid Apple Developer ID on the feature branch.

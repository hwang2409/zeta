#!/usr/bin/env bash
# TCC spike for computer-use slice 0. One entry point; harmless actions only.
#
#   ./run.sh env               environment facts (no TCC, no desktop)
#   ./run.sh build [adhoc|cert]  compile + sign the swift helper (default adhoc)
#   ./run.sh check             read-only preflight for python + swift (no prompt)
#   ./run.sh grant             trigger the TCC prompts + register in System Settings
#   ./run.sh capture           python: grab one screenshot to a temp file, print size
#   ./run.sh input             python: nudge the cursor 1 point and back
#   ./run.sh rebuild [adhoc|cert]  rebuild (new cdhash) then re-check the swift grant
#   ./run.sh cert              print how to make a local self-signed code-signing cert
#   ./run.sh cleanup           print how to undo everything this spike set up (prints only)
#   ./run.sh report            print the collected results
#
# Results append to ~/.zeta/scratch/computer-use/slice0-results.md.
# See README.md for the numbered procedure henry runs at his desk.

set -u

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SPIKE_DIR/build"
BIN="$BUILD_DIR/cuspike"
IDENT="com.henry.zeta.cuspike"
CERTNAME="${CUSPIKE_CERT:-Zeta CU Spike}"
RESULTS="${CUSPIKE_RESULTS:-$HOME/.zeta/scratch/computer-use/slice0-results.md}"
CAPTURE_DIR="${CUSPIKE_CAPTURE_DIR:-${TMPDIR:-/tmp}/cuspike-${UID}}"
AGENT_LABEL="com.henry.zeta.cuspike.q4"
AGENT_DIR=""
AGENT_ACTIVE=0
AGENT_DRY_RUN=0

agent_cleanup() {
  local dir="$AGENT_DIR" active="$AGENT_ACTIVE" dry_run="$AGENT_DRY_RUN"
  if [ "$active" -eq 1 ] && [ "$dry_run" -eq 0 ]; then
    launchctl bootout "gui/$UID/$AGENT_LABEL" >/dev/null 2>&1 || true
  fi
  if [ -n "$dir" ]; then
    rm -rf "$dir"
    if [ "$dry_run" -eq 1 ]; then
      printf 'dry-run: rm -rf %s\n' "$dir"
    fi
  fi
  if [ "$active" -eq 1 ] && [ "$dry_run" -eq 1 ]; then
    printf 'dry-run: launchctl bootout gui/%s/%s\n' "$UID" "$AGENT_LABEL"
  fi
  AGENT_ACTIVE=0
  AGENT_DIR=""
}

agent_finish() {
  local rc="$1"
  agent_cleanup
  return "$rc"
}

agent_signal() {
  local rc="$1"
  trap - EXIT INT TERM
  agent_cleanup
  exit "$rc"
}
trap agent_cleanup EXIT
trap 'agent_signal 130' INT
trap 'agent_signal 143' TERM

# The python we grant must be the SAME real binary zeta runs as. zeta's uv-tool
# python is a symlink; resolve it so the grant is representative of real sessions.
ZETA_TOOL_PY="$HOME/.local/share/uv/tools/zeta/bin/python"
if [ -n "${CUSPIKE_PY:-}" ]; then
  ZETA_PY="$CUSPIKE_PY"   # override: point at a different python (Q2 reinstall test)
elif [ -e "$ZETA_TOOL_PY" ]; then
  ZETA_PY="$(/usr/bin/python3 -c "import os;print(os.path.realpath('$ZETA_TOOL_PY'))")"
else
  ZETA_PY="3.13"
fi

pyrun() {
  uv run --no-project --python "$ZETA_PY" \
    --with pyobjc-framework-Quartz \
    --with pyobjc-framework-ApplicationServices \
    python3 "$SPIKE_DIR/cuspike.py" "$@"
}

log() {
  mkdir -p "$(dirname "$RESULTS")"
  printf '%s\n' "$*" | tee -a "$RESULTS"
}

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

need_bin() {
  if [ ! -x "$BIN" ]; then
    log "[FAIL] swift helper not built. run: ./run.sh build"
    return 1
  fi
}

cmd_env() {
  log ""
  log "## env @ $(ts)"
  log "macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  log "host context: run this from your normal tmux/zeta terminal, not a fresh Terminal.app."
  log "process tree (this shell up to launchd):"
  local p=$PPID
  for _ in 1 2 3 4 5 6 7 8; do
    [ "$p" = 1 ] || [ -z "$p" ] && break
    log "  $(ps -o pid=,ppid=,command= -p "$p" 2>/dev/null | sed 's/  */ /g')"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
  log "zeta uv-tool python: $ZETA_TOOL_PY -> $ZETA_PY"
  log "python TCC identity (codesign of resolved binary):"
  codesign -dv "$ZETA_PY" 2>&1 | sed 's/^/  /' | tee -a "$RESULTS"
  if [ -x "$BIN" ]; then
    log "swift helper codesign:"
    codesign -dvvv "$BIN" 2>&1 | grep -E 'Identifier|Signature|flags|Authority' | sed 's/^/  /' | tee -a "$RESULTS"
  else
    log "swift helper: not built yet"
  fi
  log "code-signing identities on this machine:"
  security find-identity -v -p codesigning 2>&1 | sed 's/^/  /' | tee -a "$RESULTS"
}

cmd_build() {
  local mode="${1:-adhoc}"
  mkdir -p "$BUILD_DIR"
  # Fresh nonce per build so the Mach-O (and cdhash) really changes on rebuild,
  # mirroring a real code change to the helper.
  local nonce; nonce="$(date +%s)-$$-${RANDOM}"
  printf 'let buildNonce = "%s"\n' "$nonce" > "$BUILD_DIR/buildinfo.swift"
  # swift top-level code is only allowed in a file named main.swift; copy there.
  cp "$SPIKE_DIR/cuspike.swift" "$BUILD_DIR/main.swift"
  echo "compiling cuspike.swift (nonce=$nonce) ..."
  if ! swiftc -O "$BUILD_DIR/main.swift" "$BUILD_DIR/buildinfo.swift" -o "$BIN"; then
    log "[FAIL] swiftc failed"; return 1
  fi
  case "$mode" in
    adhoc) codesign -f -s - -i "$IDENT" "$BIN" || { log "[FAIL] ad-hoc sign failed"; return 1; } ;;
    cert)  codesign -f -s "$CERTNAME" -i "$IDENT" "$BIN" || { log "[FAIL] cert sign failed (is '$CERTNAME' in your keychain? run ./run.sh cert)"; return 1; } ;;
    *) echo "unknown build mode: $mode (use adhoc|cert)"; return 2 ;;
  esac
  log "[build:$mode] signed $BIN"
  codesign -dvvv "$BIN" 2>&1 | grep -E 'Identifier|Signature|flags|CDHash|Authority' | sed 's/^/  /' | tee -a "$RESULTS"
}

python_identity() {
  local pyinfo resolved
  pyinfo="$(pyrun whoami 2>&1)"
  resolved="$(printf '%s\n' "$pyinfo" | sed -nE 's/^realpath[[:space:]]*= //p' | head -1)"
  log "python resolved path: ${resolved:-unknown}"
  if [ -n "$resolved" ]; then
    codesign -dvvv "$resolved" 2>&1 | grep -E 'Identifier|CDHash' | sed 's/^/  /' | tee -a "$RESULTS"
  fi
  printf '%s\n' "$pyinfo" | tee -a "$RESULTS"
}

swift_identity() {
  codesign -dvvv "$BIN" 2>&1 | grep -E 'Identifier|CDHash' | sed 's/^/  /' | tee -a "$RESULTS"
}

# parse "screen_recording=<bool> accessibility=<bool>" portably on BSD and GNU sed.
_preflight_values() {
  printf '%s\n' "$1" | sed -nE 's/.*screen_recording=(true|false).*accessibility=(true|false).*/\1 \2/p' | head -1
}
classify_rebuild_result() {
  local values="$1" before="$2" after="$3" build_rc="$4" sr ax
  if [ "$build_rc" -ne 0 ] || [ -z "$before" ] || [ -z "$after" ] || [ "$before" = "$after" ] || [ -z "$values" ]; then
    echo INCONCLUSIVE; return 2
  fi
  sr="${values%% *}"; ax="${values#* }"
  if [ "$sr" = true ] && [ "$ax" = true ]; then echo SURVIVED; return 0; fi
  if [ "$sr" = false ] && [ "$ax" = false ]; then echo LOST; return 1; fi
  echo INCONCLUSIVE; return 2
}
_report_line() {
  local who="$1" out="$2" values sr ax
  values="$(_preflight_values "$out")"; sr="${values%% *}"; ax="${values#* }"
  if [ -z "$values" ]; then
    log "[FAIL] $who: could not read preflight. raw: $out"; return
  fi
  if [ "$sr" = "true" ]; then log "[PASS] $who screen-recording = GRANTED"; else log "[FAIL] $who screen-recording = not granted"; fi
  if [ "$ax" = "true" ]; then log "[PASS] $who accessibility   = GRANTED"; else log "[FAIL] $who accessibility   = not granted"; fi
}

cmd_selftest() {
  local real_results="$HOME/.zeta/scratch/computer-use/slice0-results.md"
  local real_results_path results_path selftest_results
  real_results_path="$(cd "$(dirname "$real_results")" && pwd -P)/$(basename "$real_results")"
  results_path="$(cd "$(dirname "$RESULTS")" 2>/dev/null && pwd -P)/$(basename "$RESULTS")" || results_path="$RESULTS"
  if [ -n "${CUSPIKE_RESULTS+x}" ] && [ "$results_path" = "$real_results_path" ]; then
    printf 'FAIL selftest refuses real results file: %s\n' "$RESULTS"
    return 1
  fi
  selftest_results="$(mktemp "${TMPDIR:-/tmp}/cuspike-selftest-results.XXXXXX")" || return 1
  chmod 700 "$selftest_results"
  trap 'agent_cleanup; rm -f "${selftest_results:-}"' EXIT
  RESULTS="$selftest_results"

  local name input expected expected_status values actual status removed_dir
  while IFS='|' read -r name input expected expected_status; do
    values="$(_preflight_values "$input")"
    if actual="$(classify_rebuild_result "$values" old new 0)"; then status=0; else status=$?; fi
    printf '%-18s %-12s status=%s\n' "$name" "$actual" "$status"
    if [ "$actual" != "$expected" ] || [ "$status" -ne "$expected_status" ]; then
      printf 'FAIL classify %s: expected %s/status=%s, got %s/status=%s\n' "$name" "$expected" "$expected_status" "$actual" "$status"
      return 1
    fi
  done <<'EOF'
both-true|screen_recording=true accessibility=true|SURVIVED|0
both-false|screen_recording=false accessibility=false|LOST|1
split|screen_recording=true accessibility=false|INCONCLUSIVE|2
malformed|screen_recording=yes accessibility=false|INCONCLUSIVE|2
empty||INCONCLUSIVE|2
EOF
  # Keep the helper's responsible-process line in this fixture: round 1 saw the
  # real two-line output rejected as malformed.
  local real_helper_output='responsible_pid=123 executable=/usr/bin/cuspike bundle_id=com.henry.zeta.cuspike
screen_recording=true accessibility=true'
  local command_output
  if command_output="$(CUSPIKE_SELFTEST_HELPER_OUTPUT="$real_helper_output" AGENT_DRY_RUN=1 cmd_rebuild adhoc)"; then status=0; else status=$?; fi
  actual="$(printf '%s\n' "$command_output" | grep -o 'SURVIVED' | tail -1)"
  printf '%-18s %-12s status=%s\n' real-helper-old-rebuild "$actual" "$status"
  if [ "$actual" != SURVIVED ] || [ "$status" -ne 0 ]; then
    printf 'FAIL classify real-helper-old-rebuild: expected SURVIVED/status=0, got %s/status=%s\n' "$actual" "$status"
    return 1
  fi
  if command_output="$(CUSPIKE_SELFTEST_HELPER_OUTPUT="$real_helper_output" AGENT_DRY_RUN=1 cmd_agent_rebuild adhoc)"; then status=0; else status=$?; fi
  actual="$(printf '%s\n' "$command_output" | grep -o 'SURVIVED' | tail -1)"
  printf '%-18s %-12s status=%s\n' real-helper-agent-rebuild "$actual" "$status"
  if [ "$actual" != SURVIVED ] || [ "$status" -ne 0 ]; then
    printf 'FAIL classify real-helper-agent-rebuild: expected SURVIVED/status=0, got %s/status=%s\n' "$actual" "$status"
    return 1
  fi
  # The last two cases exercise the same failure gates used by rebuild commands.
  if actual="$(classify_rebuild_result "true true" same same 0)"; then status=0; else status=$?; fi
  printf '%-18s %-12s status=%s\n' unchanged-cdhash "$actual" "$status"
  if [ "$actual" != INCONCLUSIVE ] || [ "$status" -ne 2 ]; then
    printf 'FAIL classify unchanged-cdhash: expected INCONCLUSIVE/status=2, got %s/status=%s\n' "$actual" "$status"
    return 1
  fi
  if actual="$(classify_rebuild_result "true true" old new 1)"; then status=0; else status=$?; fi
  printf '%-18s %-12s status=%s\n' build-failure "$actual" "$status"
  if [ "$actual" != INCONCLUSIVE ] || [ "$status" -ne 2 ]; then
    printf 'FAIL classify build-failure: expected INCONCLUSIVE/status=2, got %s/status=%s\n' "$actual" "$status"
    return 1
  fi
  local path mode output status_line expected_status
  for path in agent agent-rebuild; do
    for mode in success failure timeout; do
      output="${TMPDIR:-/tmp}/com.henry.zeta.cuspike-selftest-output.$$"
      AGENT_DRY_RUN=1 AGENT_DRY_RUN_MODE="$mode"
      if [ "$path" = agent ]; then
        if cmd_agent check >"$output" 2>&1; then status_line=0; else status_line=$?; fi
      else
        if cmd_agent_rebuild adhoc >"$output" 2>&1; then status_line=0; else status_line=$?; fi
      fi
      case "$path:$mode" in
        agent:success|agent-rebuild:success) expected_status=0 ;;
        agent:failure) expected_status=7 ;;
        agent:timeout) expected_status=124 ;;
        agent-rebuild:failure|agent-rebuild:timeout) expected_status=2 ;;
      esac
      if [ "$status_line" -ne "$expected_status" ]; then
        printf 'FAIL %s %s: expected status=%s, got status=%s\n' "$path" "$mode" "$expected_status" "$status_line"
        return 1
      fi
      if ! awk '
        /dry-run: rm -rf / { removed = NR }
        /dry-run: launchctl bootout / { booted = NR }
        END { exit !(removed && booted && booted > removed) }
      ' "$output"; then
        printf 'FAIL %s %s: cleanup order is invalid\n' "$path" "$mode"
        return 1
      fi
      removed_dir="$(sed -n 's/^dry-run: rm -rf //p' "$output")"
      if [ -z "$removed_dir" ] || [ -e "$removed_dir" ]; then
        printf 'FAIL %s %s: cleanup directory remains\n' "$path" "$mode"
        return 1
      fi
      if [ -n "$AGENT_DIR" ]; then
        printf 'FAIL %s %s: AGENT_DIR remains set\n' "$path" "$mode"
        return 1
      fi
      printf '%-18s %-12s status=%s cleanup=ok\n' "$path" "$mode" "$status_line"
      rm -f "$output"
    done
  done
  AGENT_DRY_RUN=0
}

cmd_check() {
  log ""
  log "## check @ $(ts)   (PASS = GRANTED, FAIL = not granted; read-only, no prompt)"
  python_identity
  local pyout; pyout="$(pyrun preflight 2>&1)"; printf '%s\n' "$pyout" | tee -a "$RESULTS"
  _report_line "python" "$pyout"
  if need_bin; then
    swift_identity
    local swout; swout="$("$BIN" preflight 2>&1)"; printf '%s\n' "$swout" | tee -a "$RESULTS"
    _report_line "swift($IDENT)" "$swout"
  fi
}

cmd_grant() {
  log ""
  log "## grant @ $(ts)"
  log "python request (will prompt + open System Settings):"
  python_identity; pyrun request 2>&1 | tee -a "$RESULTS"
  if need_bin; then
    log "swift request (will prompt + open System Settings):"
    "$BIN" request 2>&1 | tee -a "$RESULTS"
  fi
  log "ACTION: in System Settings > Privacy & Security, enable both the python"
  log "entry and the '$IDENT' (cuspike) entry under Screen Recording AND"
  log "Accessibility. NOTE the EXACT name each entry shows. Then run: ./run.sh check"
}

cmd_capture() {
  log ""
  log "## capture @ $(ts)   (python, harmless: private temp file + size only)"
  umask 077
  mkdir -p "$CAPTURE_DIR"
  chmod 700 "$CAPTURE_DIR"
  python_identity; CUSPIKE_CAPTURE_DIR="$CAPTURE_DIR" pyrun capture 2>&1 | tee -a "$RESULTS"
}

cmd_input() {
  log ""
  log "## input @ $(ts)   (python, harmless: +1px then back)"
  python_identity; pyrun input 2>&1 | tee -a "$RESULTS"
}

cmd_rebuild() {
  local mode="${1:-adhoc}"
  if [ -z "${CUSPIKE_SELFTEST_HELPER_OUTPUT:-}" ]; then
    need_bin || return 1
  fi
  log ""
  log "## rebuild @ $(ts)   mode=$mode  (does the grant survive a cdhash change?)"
  local before after build_rc=0
  if [ -n "${CUSPIKE_SELFTEST_HELPER_OUTPUT:-}" ]; then
    before=old; after=new
  else
    before="$(codesign -dvvv "$BIN" 2>&1 | sed -nE 's/.*CDHash=(.*)/\1/p' | head -1)"
    cmd_build "$mode" >/dev/null || build_rc=$?
    after="$(codesign -dvvv "$BIN" 2>&1 | sed -nE 's/.*CDHash=(.*)/\1/p' | head -1)"
  fi
  log "cdhash before=$before after=$after"
  if [ -z "${CUSPIKE_SELFTEST_HELPER_OUTPUT:-}" ]; then
    swift_identity
  fi
  local swout values outcome outcome_rc
  if [ -n "${CUSPIKE_SELFTEST_HELPER_OUTPUT:-}" ]; then
    swout="$CUSPIKE_SELFTEST_HELPER_OUTPUT"
  else
    swout="$([ -x "$BIN" ] && "$BIN" preflight 2>&1 || true)"
  fi
  values="$(_preflight_values "$swout")"
  if outcome="$(classify_rebuild_result "$values" "$before" "$after" "$build_rc")"; then outcome_rc=0; else outcome_rc=$?; fi
  log "[$([ "$outcome" = SURVIVED ] && echo PASS || echo "$outcome")] grant $outcome after rebuild (mode=$mode): ${values:-malformed}"
  return "$outcome_rc"
}

cmd_cert() {
  cat <<EOF
How to create a local self-signed CODE-SIGNING identity (needed for build mode 'cert').
There are 0 code-signing identities on this machine today; this adds one.

PRIMARY (GUI, reliable) -- needs your clicks and maybe your login password:
  1. Open Keychain Access.app.
  2. Menu bar: Keychain Access > Certificate Assistant > Create a Certificate...
  3. Name:            $CERTNAME
     Identity Type:   Self Signed Root
     Certificate Type: Code Signing
  4. Click Create, keep defaults, Done. It is stored in your 'login' keychain
     with its private key. On macOS 26.6, set Trust > Code Signing to Always Trust
     in Keychain Access; Certificate Assistant does not trust this root by default.
  5. Verify:  security find-identity -v -p codesigning
     '$CERTNAME' must appear as a valid identity.

The first time codesign uses this key it may pop "codesign wants to sign using
key ... in your keychain" -> click **Always Allow** (may ask your password).

The GUI path is the only supported path. It avoids leaving private-key files on disk.

Then:  ./run.sh build cert   (signs with '$CERTNAME' and identifier $IDENT)
EOF
}

cmd_cleanup() {
  cat <<EOF
How to UNDO everything this spike set up. This command only PRINTS the commands;
it runs nothing. Review each line, then run the ones that apply to what you did.

1. Remove the build dir (compiled helper + generated sources):
     rm -rf "$BUILD_DIR"

2. If you explicitly added a trust override for this certificate, recover it
   before deleting the identity. Export the certificate from Keychain Access first,
   then run this against the exported file (admin trust store):
     security remove-trusted-cert -d <exported-certificate>
   Set the certificate to "Use System Defaults" in Keychain Access before deleting it.

3. Delete the self-signed code-signing identity (only if you ran './run.sh cert'
   and created '$CERTNAME'):
     security delete-identity -c "$CERTNAME" ~/Library/Keychains/login.keychain-db
   Or in Keychain Access.app: 'login' keychain -> find '$CERTNAME' (the cert and
   its private key) -> delete both. Verify it is gone:
     security find-identity -v -p codesigning

4. Reset the TCC grants for THIS SPIKE's helper only, scoped by its bundle id
   ($IDENT). This touches only cuspike, nothing else:
     tccutil reset ScreenCapture com.henry.zeta.cuspike
     tccutil reset Accessibility com.henry.zeta.cuspike
   Never omit the bundle identifier from either command.

5. Remove capture screenshots, including any files from an older run:
     rm -rf "$CAPTURE_DIR"
     rm -f "${TMPDIR:-/tmp}"/cuspike-*.png

6. Keep $RESULTS. It is the spike result; henry may delete it by hand after sending or archiving it.

7. The python and tmux/terminal grants have no bundle id, so remove them by hand
   them. Remove them by hand in System Settings > Privacy & Security:
     - Screen Recording: remove the python entry you enabled in step 4 of the
       procedure (e.g. 'python3.13') and the 'tmux' / terminal entry if you added
       one. Select the row and click the '-' button.
     - Accessibility: remove the same python and tmux/terminal entries.
   Leave entries you rely on for other apps.
EOF
}

cmd_agent() {
  local command="${1:-}"; [ -n "$command" ] || { echo "usage: $0 agent <check|capture|grant>"; return 2; }
  if [ "$AGENT_DRY_RUN" -eq 0 ]; then need_bin || return 1; fi
  case "$command" in check) command=preflight ;; grant) command=request ;; capture) command=preflight ;; *) echo "unknown agent command: $command"; return 2 ;; esac
  AGENT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/com.henry.zeta.cuspike.XXXXXX")" || return 1
  chmod 700 "$AGENT_DIR"
  local plist="$AGENT_DIR/com.henry.zeta.cuspike.q4.plist"
  local wrapper="$AGENT_DIR/com.henry.zeta.cuspike.q4-wrapper.sh"
  cat >"$wrapper" <<'EOF'
#!/usr/bin/env bash
"$1" "$2" >"$3" 2>"$4"
printf '%s\n' "$?" >"$5"
EOF
  chmod 700 "$wrapper"
  cat >"$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$AGENT_LABEL</string>
<key>ProgramArguments</key><array><string>$wrapper</string><string>$BIN</string><string>$command</string><string>$AGENT_DIR/stdout</string><string>$AGENT_DIR/stderr</string><string>$AGENT_DIR/done</string></array>
<key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>$AGENT_DIR/stdout</string>
<key>StandardErrorPath</key><string>$AGENT_DIR/stderr</string>
</dict></plist>
EOF
  AGENT_ACTIVE=1
  if [ "$AGENT_DRY_RUN" -eq 1 ]; then
    case "${AGENT_DRY_RUN_MODE:-success}" in
      helper) printf '%s\n' "$CUSPIKE_SELFTEST_HELPER_OUTPUT" >"$AGENT_DIR/stdout"; printf '0\n' >"$AGENT_DIR/done" ;;
      success) printf 'screen_recording=true accessibility=true\n' >"$AGENT_DIR/stdout"; printf '0\n' >"$AGENT_DIR/done" ;;
      failure) printf 'helper failed\n' >"$AGENT_DIR/stderr"; printf '7\n' >"$AGENT_DIR/done" ;;
      timeout) : ;;
    esac
  else
    launchctl bootstrap "gui/$UID" "$plist" || { agent_finish 1; return $?; }
  fi
  local status=124 done_code
  if [ "$AGENT_DRY_RUN" -eq 1 ]; then
    [ -s "$AGENT_DIR/done" ] && status=0
  else
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if [ -s "$AGENT_DIR/done" ]; then status=0; break; fi
      sleep 1
    done
  fi
  if [ "$status" -eq 0 ]; then
    done_code="$(cat "$AGENT_DIR/done")"
    case "$done_code" in 0|[1-9]|[1-9][0-9]) status="$done_code" ;; *) status=125 ;; esac
  fi
  cat "$AGENT_DIR/stdout" "$AGENT_DIR/stderr" 2>/dev/null | tee -a "$RESULTS"
  agent_finish "$status"
  return "$?"
}

cmd_agent_rebuild() {
  local mode="${1:-adhoc}"; case "$mode" in adhoc|cert) ;; *) echo "usage: $0 agent-rebuild <adhoc|cert>"; return 2 ;; esac
  if [ "$AGENT_DRY_RUN" -eq 0 ]; then need_bin || return 1; fi
  local before="old" after="new" build_rc=0
  if [ "$AGENT_DRY_RUN" -eq 0 ]; then
    before="$(codesign -dvvv "$BIN" 2>&1 | sed -nE 's/.*CDHash=(.*)/\1/p' | head -1)"
    cmd_build "$mode" >/dev/null || build_rc=$?
    after="$(codesign -dvvv "$BIN" 2>&1 | sed -nE 's/.*CDHash=(.*)/\1/p' | head -1)"
  fi
  log "## agent-rebuild @ $(ts) mode=$mode"
  log "cdhash before=$before after=$after"
  local out values outcome outcome_rc
  out="${TMPDIR:-/tmp}/com.henry.zeta.cuspike-rebuild-output.$$"
  : >"$out"
  if [ -n "${CUSPIKE_SELFTEST_HELPER_OUTPUT:-}" ]; then
    AGENT_DRY_RUN=1 AGENT_DRY_RUN_MODE=helper cmd_agent check >"$out" 2>&1 || true
  else
    cmd_agent check >"$out" 2>&1 || true
  fi
  values="$(_preflight_values "$(cat "$out")")"
  cat "$out"
  rm -f "$out"
  if outcome="$(classify_rebuild_result "$values" "$before" "$after" "$build_rc")"; then outcome_rc=0; else outcome_rc=$?; fi
  log "[$([ "$outcome" = SURVIVED ] && echo PASS || echo "$outcome")] grant $outcome after agent rebuild (mode=$mode): ${values:-malformed}"
  return "$outcome_rc"
}

cmd_agent_status() {
  if launchctl print "gui/$UID/$AGENT_LABEL" >/dev/null 2>&1; then
    echo "loaded: $AGENT_LABEL"; return 1
  fi
  echo "not loaded: $AGENT_LABEL"
  if find "${TMPDIR:-/tmp}" -maxdepth 1 -type d -name 'com.henry.zeta.cuspike.*' -print -quit 2>/dev/null | grep -q .; then
    echo "leftover temp dir found"; return 1
  fi
  echo "no leftover temp dir"
}

cmd_report() {
  if [ -f "$RESULTS" ]; then cat "$RESULTS"; else echo "no results at $RESULTS yet"; fi
}

case "${1:-}" in
  env) cmd_env ;;
  build) shift; cmd_build "$@" ;;
  check) cmd_check ;;
  grant) cmd_grant ;;
  capture) cmd_capture ;;
  input) cmd_input ;;
  rebuild) shift; cmd_rebuild "$@" ;;
  cert) cmd_cert ;;
  cleanup) cmd_cleanup ;;
  selftest) cmd_selftest ;;
  agent) shift; cmd_agent "$@" ;;
  agent-rebuild) shift; cmd_agent_rebuild "$@" ;;
  agent-status) cmd_agent_status ;;
  report) cmd_report ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac

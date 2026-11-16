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

agent_cleanup() {
  if [ "$AGENT_ACTIVE" -eq 1 ]; then
    launchctl bootout "gui/$UID/$AGENT_LABEL" >/dev/null 2>&1 || true
  fi
  if [ -n "$AGENT_DIR" ]; then
    rm -rf "$AGENT_DIR"
  fi
  AGENT_ACTIVE=0
  AGENT_DIR=""
}
trap agent_cleanup EXIT INT TERM

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
_classify_preflight() {
  local values="$1" sr="${1%% *}" ax="${1#* }"
  [ -z "$values" ] && { echo malformed; return; }
  if [ "$sr" = true ] && [ "$ax" = true ]; then echo granted
  elif [ "$sr" = false ] && [ "$ax" = false ]; then echo not-granted
  else echo split; fi
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
  local name input expected actual category
  while IFS='|' read -r name input expected; do
    actual="$(_preflight_values "$input")"
    category="$(_classify_preflight "$actual")"
    printf '%-10s %-10s %s\n' "$name" "$category" "${actual:-empty}"
    [ "$category" = "$expected" ] || return 1
  done <<'EOF'
both-true|screen_recording=true accessibility=true|granted
both-false|screen_recording=false accessibility=false|not-granted
split|screen_recording=true accessibility=false|split
malformed|screen_recording=yes accessibility=false|malformed
empty||malformed
EOF
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
  need_bin || return 1
  log ""
  log "## rebuild @ $(ts)   mode=$mode  (does the grant survive a cdhash change?)"
  local before; before="$(codesign -dvvv "$BIN" 2>&1 | sed -n 's/.*CDHash=\(.*\)/\1/p' | head -1)"
  if ! cmd_build "$mode" >/dev/null; then
    log "[INCONCLUSIVE] rebuild failed; grant result is invalid"
    return 1
  fi
  local after; after="$(codesign -dvvv "$BIN" 2>&1 | sed -n 's/.*CDHash=\(.*\)/\1/p' | head -1)"
  log "cdhash before=$before after=$after"
  if [ -z "$before" ] || [ -z "$after" ] || [ "$before" = "$after" ]; then
    log "[INCONCLUSIVE] rebuild did not change cdhash, result invalid"
    return 1
  fi
  swift_identity
  local swout; swout="$("$BIN" preflight 2>&1)"
  local values sr ax; values="$(_preflight_values "$swout")"; sr="${values%% *}"; ax="${values#* }"
  if [ -z "$values" ] || [ "$sr" = "$values" ]; then
    log "[INCONCLUSIVE] malformed preflight output: $swout"
    return 1
  elif [ "$sr" = "true" ] && [ "$ax" = "true" ]; then
    log "[PASS] grant SURVIVED rebuild (mode=$mode): screen_recording=$sr accessibility=$ax"
    log "       -> this signing mode keeps both grants across rebuilds."
  elif [ "$sr" = "false" ] && [ "$ax" = "false" ]; then
    log "[FAIL] grant LOST after rebuild (mode=$mode): screen_recording=$sr accessibility=$ax"
    log "       -> this signing mode re-keys TCC on every rebuild; re-grant needed."
  else
    log "[INCONCLUSIVE] split grant after rebuild (mode=$mode): screen_recording=$sr accessibility=$ax"
    return 1
  fi
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
  need_bin || return 1
  case "$command" in check) command=preflight ;; grant) command=request ;; capture) command=preflight ;; *) echo "unknown agent command: $command"; return 2 ;; esac
  AGENT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cuspike-agent.XXXXXX")" || return 1
  chmod 700 "$AGENT_DIR"
  cat >"$AGENT_DIR/agent.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$AGENT_LABEL</string>
<key>ProgramArguments</key><array><string>$BIN</string><string>$command</string></array>
<key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>$AGENT_DIR/stdout</string>
<key>StandardErrorPath</key><string>$AGENT_DIR/stderr</string>
</dict></plist>
EOF
  launchctl bootstrap "gui/$UID" "$AGENT_DIR/agent.plist" || return 1
  AGENT_ACTIVE=1
  local status=1
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! launchctl print "gui/$UID/$AGENT_LABEL" >/dev/null 2>&1; then status=0; break; fi
    sleep 1
  done
  launchctl bootout "gui/$UID/$AGENT_LABEL" >/dev/null 2>&1 || true
  AGENT_ACTIVE=0
  cat "$AGENT_DIR/stdout" "$AGENT_DIR/stderr" 2>/dev/null | tee -a "$RESULTS"
  return "$status"
}

cmd_agent_rebuild() {
  local mode="${1:-adhoc}"; case "$mode" in adhoc|cert) ;; *) echo "usage: $0 agent-rebuild <adhoc|cert>"; return 2 ;; esac
  need_bin || return 1
  local before; before="$(codesign -dvvv "$BIN" 2>&1 | sed -nE 's/.*CDHash=(.*)/\1/p' | head -1)"
  cmd_build "$mode" >/dev/null || return 1
  local after; after="$(codesign -dvvv "$BIN" 2>&1 | sed -nE 's/.*CDHash=(.*)/\1/p' | head -1)"
  log "## agent-rebuild @ $(ts) mode=$mode"
  log "cdhash before=$before after=$after"
  if [ -z "$before" ] || [ -z "$after" ] || [ "$before" = "$after" ]; then
    log "[INCONCLUSIVE] agent rebuild did not change cdhash"; return 1
  fi
  local out values sr ax
  out="$(cmd_agent check 2>&1)"
  values="$(_preflight_values "$out")"; sr="${values%% *}"; ax="${values#* }"
  if [ "$sr" = true ] && [ "$ax" = true ]; then
    log "[PASS] grant SURVIVED agent rebuild (mode=$mode): screen_recording=$sr accessibility=$ax"
  elif [ "$sr" = false ] && [ "$ax" = false ]; then
    log "[FAIL] grant LOST after agent rebuild (mode=$mode): screen_recording=$sr accessibility=$ax"
  else
    log "[INCONCLUSIVE] split or malformed agent preflight (mode=$mode): ${values:-$out}"; return 1
  fi
}

cmd_agent_status() {
  if launchctl print "gui/$UID/$AGENT_LABEL" >/dev/null 2>&1; then
    echo "loaded: $AGENT_LABEL"; return 1
  fi
  echo "not loaded: $AGENT_LABEL"
  if find "${TMPDIR:-/tmp}" -maxdepth 1 -type d -name 'cuspike-agent.*' -print -quit 2>/dev/null | grep -q .; then
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

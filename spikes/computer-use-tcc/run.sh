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

cmd_check() {
  log ""
  log "## check @ $(ts)   (PASS = GRANTED, FAIL = not granted; read-only, no prompt)"
  echo "python preflight (identity: $ZETA_PY) ..."
  local pyout; pyout="$(pyrun preflight 2>&1)"
  _report_line "python" "$pyout"
  if need_bin; then
    local swout; swout="$("$BIN" preflight 2>&1)"
    _report_line "swift($IDENT)" "$swout"
  fi
}

# parse "screen_recording=<bool> accessibility=<bool>" into PASS/FAIL lines
_report_line() {
  local who="$1" out="$2"
  local sr ax
  sr="$(printf '%s' "$out" | sed -n 's/.*screen_recording=\([a-z]*\).*/\1/p' | head -1)"
  ax="$(printf '%s' "$out" | sed -n 's/.*accessibility=\([a-z]*\).*/\1/p' | head -1)"
  if [ -z "$sr$ax" ]; then
    log "[FAIL] $who: could not read preflight. raw: $out"
    return
  fi
  if [ "$sr" = "true" ]; then log "[PASS] $who screen-recording = GRANTED"; else log "[FAIL] $who screen-recording = not granted"; fi
  if [ "$ax" = "true" ]; then log "[PASS] $who accessibility   = GRANTED"; else log "[FAIL] $who accessibility   = not granted"; fi
}

cmd_grant() {
  log ""
  log "## grant @ $(ts)"
  log "python request (will prompt + open System Settings):"
  pyrun request 2>&1 | tee -a "$RESULTS"
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
  CUSPIKE_CAPTURE_DIR="$CAPTURE_DIR" pyrun capture 2>&1 | tee -a "$RESULTS"
}

cmd_input() {
  log ""
  log "## input @ $(ts)   (python, harmless: +1px then back)"
  pyrun input 2>&1 | tee -a "$RESULTS"
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
  local swout; swout="$("$BIN" preflight 2>&1)"
  local sr; sr="$(printf '%s' "$swout" | sed -n 's/.*screen_recording=\(true\|false\).*/\1/p' | head -1)"
  local ax; ax="$(printf '%s' "$swout" | sed -n 's/.*accessibility=\(true\|false\).*/\1/p' | head -1)"
  if [ -z "$sr" ] || [ -z "$ax" ]; then
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
     with its private key (a self-signed root is trusted in login by default).
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
   Set the certificate to "Use System Defaults" in Keychain Access if removal
   reports that no matching trust override exists.

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
  report) cmd_report ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac

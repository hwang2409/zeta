#!/bin/sh
# build the zeta computer-use macOS helper with plain swiftc, then code-sign it.
# ZETA_HELPER_SIGN_IDENTITY set -> sign with that identity; otherwise ad-hoc. never Developer ID.
# usage: scripts/build-computer-helper.sh [output-dir]
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
src="$root/helpers/macos-computer"
out="${1:-$src/build}"
bin="$out/zeta-computer-helper"

mkdir -p "$out"

# embed Info.plist (stable bundle id + usage strings) into the Mach-O so TCC attributes the grant
# to a stable identity rather than the mutable python path.
swiftc -O \
	-o "$bin" \
	"$src/main.swift" \
	-framework ScreenCaptureKit \
	-framework AppKit \
	-framework Carbon \
	-Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$src/Info.plist"

if [ -n "${ZETA_HELPER_SIGN_IDENTITY:-}" ]; then
	codesign --force --sign "$ZETA_HELPER_SIGN_IDENTITY" --options runtime "$bin"
else
	codesign --force --sign - "$bin"
fi

codesign --verify --verbose "$bin"
echo "built $bin"

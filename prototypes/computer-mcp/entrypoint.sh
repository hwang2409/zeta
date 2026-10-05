#!/bin/sh
set -eu
export DISPLAY=:99
export HOME=/home/zeta
export XDG_RUNTIME_DIR=/tmp/runtime-zeta
mkdir -p "$XDG_RUNTIME_DIR" "$HOME/.config" "$HOME/notes"
chmod 700 "$XDG_RUNTIME_DIR"
Xvfb :99 -screen 0 1280x800x24 -nolisten tcp -noreset >/tmp/xvfb.log 2>&1 &
for _attempt in $(seq 1 50); do
    if xdpyinfo -display :99 >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
openbox >/tmp/openbox.log 2>&1 &
tint2 >/tmp/tint2.log 2>&1 &
dbus-run-session -- mousepad >/tmp/mousepad.log 2>&1 &
exec sleep 1800

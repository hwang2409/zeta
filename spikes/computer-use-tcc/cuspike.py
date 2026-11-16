#!/usr/bin/env python3
"""TCC spike helper for the python (pyobjc) identity path.

Run only through run.sh, which supplies pyobjc via `uv run --with`.
Harmless by design: `capture` writes one screenshot to a temp file and prints
its size; `input` nudges the mouse 1 point and moves it straight back.
Read-only checks use CGPreflightScreenCaptureAccess / AXIsProcessTrusted and
never prompt.
"""

import os
import sys
import tempfile
import time

import Quartz

try:
    import ApplicationServices as _AS

    AXIsProcessTrusted = _AS.AXIsProcessTrusted
    AXIsProcessTrustedWithOptions = _AS.AXIsProcessTrustedWithOptions
    kAXTrustedCheckOptionPrompt = _AS.kAXTrustedCheckOptionPrompt
except (ImportError, AttributeError):  # fallback if umbrella layout differs
    from HIServices import (  # type: ignore
        AXIsProcessTrusted,
        AXIsProcessTrustedWithOptions,
        kAXTrustedCheckOptionPrompt,
    )


def _preflight():
    """Read-only: does THIS process already hold the grants? No prompt."""
    sr = bool(Quartz.CGPreflightScreenCaptureAccess())
    ax = bool(AXIsProcessTrusted())
    return sr, ax


def cmd_whoami():
    exe = sys.executable
    print(f"sys.executable = {exe}")
    print(f"realpath       = {os.path.realpath(exe)}")


def cmd_preflight():
    sr, ax = _preflight()
    print(f"screen_recording={str(sr).lower()} accessibility={str(ax).lower()}")


def cmd_request():
    """Trigger the TCC prompts and register this process in System Settings."""
    sr_before, ax_before = _preflight()
    print(f"before: screen_recording={str(sr_before).lower()} "
          f"accessibility={str(ax_before).lower()}")
    # Adds this process to the Screen Recording list and prompts if not set.
    sr = bool(Quartz.CGRequestScreenCaptureAccess())
    # Adds this process to the Accessibility list, opens System Settings, prompts.
    opts = {kAXTrustedCheckOptionPrompt: True}
    ax = bool(AXIsProcessTrustedWithOptions(opts))
    print(f"CGRequestScreenCaptureAccess returned {str(sr).lower()}")
    print(f"AXIsProcessTrustedWithOptions(prompt=True) returned {str(ax).lower()}")
    print("now toggle the switches in System Settings, then re-run `check`.")


def cmd_capture():
    """Harmless: grab the main display once, save to a temp PNG, print its size."""
    sr, _ = _preflight()
    if not sr:
        print("FAIL capture: Screen Recording not granted (preflight=false)")
        return 1
    img = Quartz.CGDisplayCreateImage(Quartz.CGMainDisplayID())
    if img is None:
        print("FAIL capture: CGDisplayCreateImage returned None (no grant / empty)")
        return 1
    w = Quartz.CGImageGetWidth(img)
    h = Quartz.CGImageGetHeight(img)
    fd, path = tempfile.mkstemp(prefix="cuspike-", suffix=".png")
    os.close(fd)
    url = Quartz.CFURLCreateWithFileSystemPath(
        None, path, Quartz.kCFURLPOSIXPathStyle, False
    )
    dest = Quartz.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
    if dest is None:
        print("FAIL capture: could not create PNG destination")
        return 1
    Quartz.CGImageDestinationAddImage(dest, img, None)
    ok = Quartz.CGImageDestinationFinalize(dest)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    if not ok or size == 0:
        print(f"FAIL capture: wrote 0 bytes to {path}")
        return 1
    print(f"PASS capture: {w}x{h} px, {size} bytes -> {path}")
    return 0


def cmd_input():
    """Harmless: move the cursor 1 point and move it straight back."""
    _, ax = _preflight()
    if not ax:
        print("FAIL input: Accessibility not granted (AXIsProcessTrusted=false)")
        return 1
    here = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    x, y = here.x, here.y

    def move(px, py):
        ev = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved,
            Quartz.CGPointMake(px, py), Quartz.kCGMouseButtonLeft,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    move(x + 1, y + 1)
    time.sleep(0.05)
    move(x, y)
    after = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    print(f"PASS input: moved cursor ({x:.0f},{y:.0f}) -> (+1,+1) -> "
          f"({after.x:.0f},{after.y:.0f})")
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    table = {
        "whoami": cmd_whoami,
        "preflight": cmd_preflight,
        "request": cmd_request,
        "capture": cmd_capture,
        "input": cmd_input,
    }
    fn = table.get(cmd)
    if fn is None:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    rc = fn()
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())

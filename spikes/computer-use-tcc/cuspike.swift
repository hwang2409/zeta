// TCC spike helper for the signed-binary identity path (Q3).
//
// A tiny Mach-O with a STABLE signing identifier. It answers one question:
// does the Screen Recording / Accessibility grant stay attached to this code
// identity when the binary is rebuilt (cdhash changes)?
//
//   preflight  read-only: print whether this process holds the grants (no prompt)
//   request    trigger the TCC prompts and register this binary in System Settings
//   version    print the build nonce (proves a rebuild changed the binary/cdhash)
//
// It never captures the screen and never posts an input event; the read-only
// preflight result is the official "do I have the grant" signal, which is all
// the persistence test needs.
//
// `buildNonce` is defined in a generated buildinfo.swift that run.sh writes
// fresh on every build, so each rebuild produces a different Mach-O (and cdhash)
// exactly as a real code change to the helper would.

import ApplicationServices
import CoreGraphics
import Foundation
import Darwin

func responsibleProcess() -> String {
    guard let symbol = dlsym(dlopen(nil, RTLD_LAZY), "responsibility_get_pid_responsible_for_pid") else {
        return "responsible_process=unknown"
    }
    typealias Responsible = @convention(c) (pid_t) -> pid_t
    let responsible = unsafeBitCast(symbol, to: Responsible.self)(getpid())
    var path = [CChar](repeating: 0, count: Int(MAXPATHLEN))
    let pathText = proc_pidpath(responsible, &path, UInt32(path.count)) > 0
        ? String(cString: path) : "unknown"
    let bundle = Bundle(path: pathText)?.bundleIdentifier ?? "unknown"
    return "responsible_pid=\(responsible) executable=\(pathText) bundle_id=\(bundle)"
}

func preflight() -> (Bool, Bool) {
    let sr = CGPreflightScreenCaptureAccess()
    let ax = AXIsProcessTrusted()
    return (sr, ax)
}

let cmd = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "preflight"

switch cmd {
case "preflight":
    print(responsibleProcess())
    let (sr, ax) = preflight()
    print("screen_recording=\(sr) accessibility=\(ax)")

case "request":
    print(responsibleProcess())
    let (srBefore, axBefore) = preflight()
    print("before: screen_recording=\(srBefore) accessibility=\(axBefore)")
    // Registers this binary in the Screen Recording list and prompts if unset.
    let sr = CGRequestScreenCaptureAccess()
    // Registers this binary in the Accessibility list, opens System Settings.
    let opts = [kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary
    let ax = AXIsProcessTrustedWithOptions(opts)
    print("CGRequestScreenCaptureAccess returned \(sr)")
    print("AXIsProcessTrustedWithOptions(prompt=true) returned \(ax)")
    print("now toggle the switches in System Settings, then re-run `check`.")

case "version":
    print("build_nonce=\(buildNonce)")

default:
    FileHandle.standardError.write("unknown command: \(cmd)\n".data(using: .utf8)!)
    exit(2)
}

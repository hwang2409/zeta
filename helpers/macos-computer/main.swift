// zeta computer-use helper: JSON-lines stdio protocol for native macOS capture + input.
// one request per line: {"id":int,"op":str,...args}
// one response per line: {"id":int,"ok":true,...} or {"id":int,"ok":false,"error":{"code","message"}}
// capture = ScreenCaptureKit; input = CGEvent. refuses while locked / secure input.
// kill switch: ctrl+opt+cmd+esc makes every later op fail with "killed" until restart.

import AppKit
import ApplicationServices
import Carbon.HIToolbox
import CoreGraphics
import Foundation
import ScreenCaptureKit

let helperVersion = "0.1.0"

// MARK: - errors

enum HelperError: Error {
    case locked
    case secureInput
    case notPermitted(String)
    case killed
    case badRequest(String)
    case internalError(String)

    var code: String {
        switch self {
        case .locked: return "locked"
        case .secureInput: return "secure_input"
        case .notPermitted: return "not_permitted"
        case .killed: return "killed"
        case .badRequest: return "bad_request"
        case .internalError: return "internal"
        }
    }

    var message: String {
        switch self {
        case .locked: return "screen is locked"
        case .secureInput: return "secure input is enabled"
        case .notPermitted(let m): return m
        case .killed: return "kill switch engaged; restart the helper"
        case .badRequest(let m): return m
        case .internalError(let m): return m
        }
    }
}

// MARK: - kill switch flag (set from the event-tap thread, read from the main loop)

final class Flag {
    private let lock = NSLock()
    private var value = false
    func set() { lock.lock(); value = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
}

let killed = Flag()

func killCallback(
    proxy: CGEventTapProxy, type: CGEventType, event: CGEvent, userInfo: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    if type == .keyDown {
        let keycode = event.getIntegerValueField(.keyboardEventKeycode)
        let flags = event.flags
        if keycode == Int64(kVK_Escape)
            && flags.contains(.maskControl)
            && flags.contains(.maskAlternate)
            && flags.contains(.maskCommand)
        {
            killed.set()
        }
    }
    return Unmanaged.passUnretained(event)
}

// best-effort: needs Accessibility. if it cannot be created (headless CI, no grant) the helper
// still runs; input ops just fail with not_permitted on their own.
func installKillSwitch() {
    let thread = Thread {
        let mask = CGEventMask(1 << CGEventType.keyDown.rawValue)
        guard
            let tap = CGEvent.tapCreate(
                tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly,
                eventsOfInterest: mask, callback: killCallback, userInfo: nil)
        else {
            FileHandle.standardError.write(
                Data("zeta-computer-helper: kill switch unavailable (grant Accessibility)\n".utf8))
            return
        }
        let source = CFMachPortCreateRunLoopSource(nil, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        CFRunLoopRun()
    }
    thread.name = "zeta-computer-helper.killswitch"
    thread.start()
}

// MARK: - system state

func sessionDictionary() -> [String: Any] {
    guard let dict = CGSessionCopyCurrentDictionary() as? [String: Any] else { return [:] }
    return dict
}

func screenIsLocked() -> Bool {
    let dict = sessionDictionary()
    if let locked = dict["CGSSessionScreenIsLocked"] as? Int, locked == 1 { return true }
    if let onConsole = dict[kCGSessionOnConsoleKey as String] as? Bool, onConsole == false {
        return true
    }
    return false
}

func secureInputEnabled() -> Bool {
    return IsSecureEventInputEnabled()
}

func frontmostBundleID() -> String {
    return NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? ""
}

// cua (github.com/trycua/cua, MIT) refuses input while a macOS security/consent prompt owns the
// input — platform-macos/src/security_prompt.rs (KEYCHAIN_PROMPT_OWNERS). we borrow the idea in a
// minimal form: if the frontmost app is one of those prompt owners (keychain / admin password /
// Touch ID panel), refuse input. complements IsSecureEventInputEnabled, which cua lacks.
let securityPromptBundleIDs: Set<String> = [
    "com.apple.SecurityAgent",
    "com.apple.LocalAuthentication.UIAgent",
]

func securityPromptFrontmost() -> Bool {
    guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else {
        return false
    }
    return securityPromptBundleIDs.contains(bundleID)
}

// MARK: - capture

struct Capture {
    let pngBase64: String
    let width: Int
    let height: Int
    let scale: Double
    let originX: Int
    let originY: Int
}

func captureDisplay(_ index: Int) throws -> Capture {
    let sem = DispatchSemaphore(value: 0)
    var result: Capture?
    var failure: Error?
    Task {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            let displays = content.displays
            guard index >= 0, index < displays.count else {
                throw HelperError.badRequest(
                    "display \(index) out of range (have \(displays.count))")
            }
            let display = displays[index]
            let bounds = CGDisplayBounds(display.displayID)  // origin in global points
            let filter = SCContentFilter(display: display, excludingWindows: [])
            // use ScreenCaptureKit's own point->pixel scale for this display; do not assume 2x.
            let scale = Double(filter.pointPixelScale)
            let rect = filter.contentRect
            let pxWidth = Int((rect.width * CGFloat(scale)).rounded())
            let pxHeight = Int((rect.height * CGFloat(scale)).rounded())
            let config = SCStreamConfiguration()
            config.width = pxWidth
            config.height = pxHeight
            config.showsCursor = true
            let cgImage = try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: config)
            let rep = NSBitmapImageRep(cgImage: cgImage)
            guard let png = rep.representation(using: .png, properties: [:]) else {
                throw HelperError.internalError("png encode failed")
            }
            result = Capture(
                pngBase64: png.base64EncodedString(),
                width: cgImage.width, height: cgImage.height, scale: scale,
                originX: Int(bounds.origin.x), originY: Int(bounds.origin.y))
        } catch {
            failure = error
        }
        sem.signal()
    }
    sem.wait()
    if let failure { throw failure }
    guard let result else { throw HelperError.internalError("capture produced no image") }
    return result
}

// MARK: - input

let eventSource = CGEventSource(stateID: .combinedSessionState)

func mouseButton(_ name: String) throws -> (down: CGEventType, up: CGEventType, button: CGMouseButton)
{
    switch name {
    case "left": return (.leftMouseDown, .leftMouseUp, .left)
    case "right": return (.rightMouseDown, .rightMouseUp, .right)
    case "middle": return (.otherMouseDown, .otherMouseUp, .center)
    default: throw HelperError.badRequest("unknown button: \(name)")
    }
}

func postMouse(_ type: CGEventType, _ point: CGPoint, _ button: CGMouseButton, clickState: Int = 1) {
    guard
        let event = CGEvent(
            mouseEventSource: eventSource, mouseType: type, mouseCursorPosition: point,
            mouseButton: button)
    else { return }
    if clickState > 1 { event.setIntegerValueField(.mouseEventClickState, value: Int64(clickState)) }
    event.post(tap: .cghidEventTap)
}

func moveMouse(_ point: CGPoint) {
    postMouse(.mouseMoved, point, .left)
}

func click(_ point: CGPoint, button name: String, count: Int) throws {
    let b = try mouseButton(name)
    let n = max(1, min(count, 3))
    for i in 1...n {
        postMouse(b.down, point, b.button, clickState: i)
        postMouse(b.up, point, b.button, clickState: i)
    }
}

func drag(_ path: [CGPoint]) throws {
    guard let start = path.first else { throw HelperError.badRequest("drag path is empty") }
    postMouse(.leftMouseDown, start, .left)
    for point in path.dropFirst() {
        postMouse(.leftMouseDragged, point, .left)
    }
    postMouse(.leftMouseUp, path.last ?? start, .left)
}

func scroll(at point: CGPoint, dx: Int, dy: Int) {
    moveMouse(point)
    guard
        let event = CGEvent(
            scrollWheelEvent2Source: eventSource, units: .pixel, wheelCount: 2, wheel1: Int32(dy),
            wheel2: Int32(dx), wheel3: 0)
    else { return }
    event.post(tap: .cghidEventTap)
}

func typeText(_ text: String) {
    for ch in text {
        var utf16 = Array(String(ch).utf16)
        guard
            let down = CGEvent(keyboardEventSource: eventSource, virtualKey: 0, keyDown: true),
            let up = CGEvent(keyboardEventSource: eventSource, virtualKey: 0, keyDown: false)
        else { continue }
        down.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: &utf16)
        up.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: &utf16)
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
    }
}

// name -> ANSI virtual keycode. covers letters, digits, and common named keys.
let keyCodes: [String: CGKeyCode] = [
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
    "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "o": 31, "u": 32, "i": 34, "p": 35,
    "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29,
    "=": 24, "-": 27, "]": 30, "[": 33, "'": 39, ";": 41, "\\": 42, ",": 43, "/": 44, ".": 47,
    "`": 50,
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51,
    "escape": 53, "esc": 53, "forwarddelete": 117, "fdelete": 117,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111,
]

func keyCombo(_ combo: String) throws {
    let parts = combo.lowercased().split(separator: "+").map(String.init)
    guard !parts.isEmpty else { throw HelperError.badRequest("empty key combo") }
    var flags: CGEventFlags = []
    var keycode: CGKeyCode?
    for part in parts {
        switch part {
        case "cmd", "command", "meta", "super": flags.insert(.maskCommand)
        case "shift": flags.insert(.maskShift)
        case "ctrl", "control": flags.insert(.maskControl)
        case "opt", "option", "alt": flags.insert(.maskAlternate)
        case "fn", "function": flags.insert(.maskSecondaryFn)
        default:
            guard let code = keyCodes[part] else {
                throw HelperError.badRequest("unknown key: \(part)")
            }
            keycode = code
        }
    }
    guard let code = keycode else { throw HelperError.badRequest("no key in combo: \(combo)") }
    guard
        let down = CGEvent(keyboardEventSource: eventSource, virtualKey: code, keyDown: true),
        let up = CGEvent(keyboardEventSource: eventSource, virtualKey: code, keyDown: false)
    else { throw HelperError.internalError("could not create key event") }
    down.flags = flags
    up.flags = flags
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
}

// MARK: - request handling

func intArg(_ req: [String: Any], _ key: String) throws -> Int {
    if let n = req[key] as? Int { return n }
    if let n = req[key] as? Double { return Int(n) }
    throw HelperError.badRequest("missing or invalid int arg: \(key)")
}

func pointArg(_ req: [String: Any]) throws -> CGPoint {
    return CGPoint(x: Double(try intArg(req, "x")), y: Double(try intArg(req, "y")))
}

func guardInput() throws {
    if screenIsLocked() { throw HelperError.locked }
    if secureInputEnabled() || securityPromptFrontmost() { throw HelperError.secureInput }
}

func handle(_ op: String, _ req: [String: Any]) throws -> [String: Any] {
    switch op {
    case "hello":
        return [
            "version": helperVersion,
            "screen_recording": CGPreflightScreenCaptureAccess(),
            "accessibility": AXIsProcessTrusted(),
        ]
    case "state":
        return [
            "locked": screenIsLocked(),
            "secure_input": secureInputEnabled(),
            "frontmost_bundle_id": frontmostBundleID(),
        ]
    case "cursor_position":
        let loc = CGEvent(source: nil)?.location ?? .zero
        return ["x": Int(loc.x), "y": Int(loc.y)]
    case "screenshot":
        if screenIsLocked() { throw HelperError.locked }
        let display = (req["display"] as? Int) ?? 0
        let cap = try captureDisplay(display)
        return [
            "png_b64": cap.pngBase64, "width": cap.width, "height": cap.height,
            "scale": cap.scale, "origin_x": cap.originX, "origin_y": cap.originY,
        ]
    case "move":
        try guardInput()
        moveMouse(try pointArg(req))
        return [:]
    case "click":
        try guardInput()
        let button = (req["button"] as? String) ?? "left"
        let count = (req["count"] as? Int) ?? 1
        try click(try pointArg(req), button: button, count: count)
        return [:]
    case "mouse_down":
        try guardInput()
        let b = try mouseButton((req["button"] as? String) ?? "left")
        postMouse(b.down, try pointArg(req), b.button)
        return [:]
    case "mouse_up":
        try guardInput()
        let b = try mouseButton((req["button"] as? String) ?? "left")
        postMouse(b.up, try pointArg(req), b.button)
        return [:]
    case "drag":
        try guardInput()
        guard let raw = req["path"] as? [[Any]] else {
            throw HelperError.badRequest("drag needs path:[[x,y],...]")
        }
        let path = try raw.map { pair -> CGPoint in
            guard pair.count == 2 else { throw HelperError.badRequest("drag point needs [x,y]") }
            func num(_ v: Any) throws -> Double {
                if let i = v as? Int { return Double(i) }
                if let d = v as? Double { return d }
                throw HelperError.badRequest("drag coord not a number")
            }
            return CGPoint(x: try num(pair[0]), y: try num(pair[1]))
        }
        try drag(path)
        return [:]
    case "type":
        try guardInput()
        guard let text = req["text"] as? String else {
            throw HelperError.badRequest("type needs text")
        }
        typeText(text)
        return [:]
    case "key":
        try guardInput()
        guard let combo = req["combo"] as? String else {
            throw HelperError.badRequest("key needs combo")
        }
        try keyCombo(combo)
        return [:]
    case "scroll":
        try guardInput()
        scroll(at: try pointArg(req), dx: try intArg(req, "dx"), dy: try intArg(req, "dy"))
        return [:]
    default:
        throw HelperError.badRequest("unknown op: \(op)")
    }
}

// MARK: - stdio loop

let stdout = FileHandle.standardOutput

func writeResponse(_ dict: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: dict) else { return }
    stdout.write(data)
    stdout.write(Data([0x0a]))
}

func errorResponse(_ id: Int, _ error: Error) -> [String: Any] {
    let he = (error as? HelperError) ?? .internalError("\(error)")
    return ["id": id, "ok": false, "error": ["code": he.code, "message": he.message]]
}

installKillSwitch()

while let line = readLine(strippingNewline: true) {
    if line.isEmpty { continue }
    guard let data = line.data(using: .utf8),
        let req = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else {
        writeResponse([
            "id": 0, "ok": false,
            "error": ["code": "bad_request", "message": "malformed request line"],
        ])
        continue
    }
    let id = (req["id"] as? Int) ?? 0
    guard let op = req["op"] as? String else {
        writeResponse(errorResponse(id, HelperError.badRequest("missing op")))
        continue
    }
    if killed.isSet {
        writeResponse(errorResponse(id, HelperError.killed))
        continue
    }
    do {
        var payload = try handle(op, req)
        payload["id"] = id
        payload["ok"] = true
        writeResponse(payload)
    } catch {
        writeResponse(errorResponse(id, error))
    }
}

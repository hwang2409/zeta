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
    private var observers: [() -> Void] = []

    func observe(_ observer: @escaping () -> Void) {
        lock.lock()
        observers.append(observer)
        let alreadySet = value
        lock.unlock()
        if alreadySet { observer() }
    }

    func set() {
        lock.lock()
        guard !value else { lock.unlock(); return }
        value = true
        let callbacks = observers
        lock.unlock()
        callbacks.forEach { $0() }
    }

    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }

    // The reply commit is the linearization point. stdout is written after this lock is released.
    func commitReply() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !value
    }
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
            activeKill.set()
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
            // The desktop can become protected while ScreenCaptureKit awaits content or pixels.
            try guardSafety()
            let cgImage = try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: config)
            try guardSafety()
            let rep = NSBitmapImageRep(cgImage: cgImage)
            guard let png = rep.representation(using: .png, properties: [:]) else {
                throw HelperError.internalError("png encode failed")
            }
            // Refuse bytes if protection began during encoding as well.
            try guardSafety()
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

final class EventGate {
    private let lock = NSLock()
    private let killed: Flag
    private let poster: (CGEvent) -> Void
    private var held: Set<CGMouseButton> = []
    private var heldKeys: [CGKeyCode: CGEventFlags] = [:]
    private var cursor = CGPoint.zero

    init(killed: Flag, poster: @escaping (CGEvent) -> Void) {
        self.killed = killed
        self.poster = poster
    }

    func post(_ event: CGEvent, type: CGEventType, point: CGPoint, button: CGMouseButton) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !killed.isSet else { throw HelperError.killed }
        poster(event)
        if type == .mouseMoved || type == .leftMouseDown || type == .leftMouseDragged ||
            type == .leftMouseUp || type == .rightMouseDown || type == .rightMouseDragged ||
            type == .rightMouseUp || type == .otherMouseDown || type == .otherMouseDragged ||
            type == .otherMouseUp || type == .scrollWheel {
            cursor = point
        }
        if type == .leftMouseDown || type == .rightMouseDown || type == .otherMouseDown {
            held.insert(button)
        } else if type == .leftMouseUp || type == .rightMouseUp || type == .otherMouseUp {
            held.remove(button)
        }
    }

    func postKey(_ event: CGEvent, keycode: CGKeyCode, flags: CGEventFlags, down: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !killed.isSet else { throw HelperError.killed }
        poster(event)
        if down { heldKeys[keycode] = flags } else { heldKeys.removeValue(forKey: keycode) }
    }

    func postKeyPair(_ down: CGEvent, _ up: CGEvent, keycode: CGKeyCode, flags: CGEventFlags) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !killed.isSet else { throw HelperError.killed }
        poster(down)
        heldKeys[keycode] = flags
        poster(up)
        heldKeys.removeValue(forKey: keycode)
    }

    func kill() {
        lock.lock()
        defer { lock.unlock() }
        for button in held {
            let type: CGEventType = button == .left ? .leftMouseUp :
                button == .right ? .rightMouseUp : .otherMouseUp
            if let event = CGEvent(mouseEventSource: eventSource, mouseType: type,
                                   mouseCursorPosition: cursor, mouseButton: button) {
                // cleanup is committed under the same lock, at the current cursor position.
                poster(event)
            }
        }
        held.removeAll()
        for (keycode, flags) in heldKeys {
            if let event = CGEvent(keyboardEventSource: eventSource, virtualKey: keycode, keyDown: false) {
                event.flags = flags
                poster(event)
            }
        }
        heldKeys.removeAll()
    }
}

var eventGate = EventGate(killed: killed) { $0.post(tap: .cghidEventTap) }
var activeKill = killed
killed.observe { DispatchQueue.global().async { eventGate.kill() } }
func currentKill() -> Flag { activeKill }

func modifierFlags(_ raw: Any?) -> CGEventFlags {
    guard let names = raw as? [String] else { return [] }
    var flags: CGEventFlags = []
    for name in names {
        switch name.lowercased() {
        case "cmd", "command", "meta", "super": flags.insert(.maskCommand)
        case "shift": flags.insert(.maskShift)
        case "ctrl", "control": flags.insert(.maskControl)
        case "opt", "option", "alt": flags.insert(.maskAlternate)
        case "fn", "function": flags.insert(.maskSecondaryFn)
        default: break
        }
    }
    return flags
}

func postMouse(_ type: CGEventType, _ point: CGPoint, _ button: CGMouseButton, clickState: Int = 1, flags: CGEventFlags = []) throws {
    guard let event = CGEvent(mouseEventSource: eventSource, mouseType: type,
                              mouseCursorPosition: point, mouseButton: button) else { return }
    if clickState > 1 { event.setIntegerValueField(.mouseEventClickState, value: Int64(clickState)) }
    event.flags = flags
    try eventGate.post(event, type: type, point: point, button: button)
}

func moveMouse(_ point: CGPoint) throws {
    try postMouse(.mouseMoved, point, .left)
}

func checkKilled(_ source: Flag = killed) throws {
    if source.isSet { throw HelperError.killed }
}

func runInterruptibleSteps<Step>(
    _ steps: some Sequence<Step>, killSource: Flag, post: (Step) throws -> Void
) throws {
    for step in steps {
        try checkKilled(killSource)
        try post(step)
    }
}

func click(_ point: CGPoint, button name: String, count: Int, flags: CGEventFlags = []) throws {
    let b = try mouseButton(name)
    let n = max(1, min(count, 3))
    try runInterruptibleSteps(1...n, killSource: currentKill()) { i in
        try postMouse(b.down, point, b.button, clickState: i, flags: flags)
        try postMouse(b.up, point, b.button, clickState: i, flags: flags)
    }
}

func drag(_ path: [CGPoint], killSource: Flag? = nil) throws {
    let killSource = killSource ?? currentKill()
    guard let start = path.first else { throw HelperError.badRequest("drag path is empty") }
    try checkKilled(killSource)
    try postMouse(.leftMouseDown, start, .left)
    var lastPostedPoint = start
    defer { try? postMouse(.leftMouseUp, lastPostedPoint, .left) }
    try runInterruptibleSteps(path.dropFirst(), killSource: killSource) { point in
        try postMouse(.leftMouseDragged, point, .left)
        lastPostedPoint = point
    }
    try checkKilled(killSource)
}

func scroll(at point: CGPoint, dx: Int, dy: Int, flags: CGEventFlags = []) throws {
    try moveMouse(point)
    guard
        let event = CGEvent(
            scrollWheelEvent2Source: eventSource, units: .line, wheelCount: 2, wheel1: Int32(dy),
            wheel2: Int32(dx), wheel3: 0)
    else { return }
    event.flags = flags
    try eventGate.post(event, type: .scrollWheel, point: point, button: .left)
}

func typeText(_ text: String, killSource: Flag? = nil) throws {
    try runInterruptibleSteps(Array(text), killSource: killSource ?? currentKill()) { ch in
        var utf16 = Array(String(ch).utf16)
        guard
            let down = CGEvent(keyboardEventSource: eventSource, virtualKey: 0, keyDown: true),
            let up = CGEvent(keyboardEventSource: eventSource, virtualKey: 0, keyDown: false)
        else { return }
        down.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: &utf16)
        up.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: &utf16)
        try eventGate.postKeyPair(down, up, keycode: 0, flags: down.flags)
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

func keyCombo(_ combo: String, repeatCount: Int = 1, hold: Bool = false, release: Bool = false) throws {
    try checkKilled(currentKill())
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
    let count = max(1, min(repeatCount, 100))
    if hold || release {
        try eventGate.postKey(hold ? down : up, keycode: code, flags: flags, down: hold)
    } else {
        for _ in 0..<count { try eventGate.postKeyPair(down, up, keycode: code, flags: flags) }
    }
}

// MARK: - request handling

func intArg(_ req: [String: Any], _ key: String) throws -> Int {
    if let n = req[key] as? Int { return n }
    if let n = req[key] as? Double { return Int(n) }
    throw HelperError.badRequest("missing or invalid int arg: \(key)")
}

func pointArg(_ req: [String: Any]) throws -> CGPoint {
    if req["x"] == nil && req["y"] == nil {
        return CGEvent(source: nil)?.location ?? .zero
    }
    return CGPoint(x: Double(try intArg(req, "x")), y: Double(try intArg(req, "y")))
}

var selfTestMode = false

func guardSafety() throws {
    if screenIsLocked() { throw HelperError.locked }
    if secureInputEnabled() || securityPromptFrontmost() { throw HelperError.secureInput }
}

func guardInput() throws {
    if selfTestMode { return }
    try guardSafety()
    if !AXIsProcessTrusted() { throw HelperError.notPermitted("Accessibility permission is required") }
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
            "secure_input": secureInputEnabled() || securityPromptFrontmost(),
            "frontmost_bundle_id": frontmostBundleID(),
        ]
    case "cursor_position":
        let loc = CGEvent(source: nil)?.location ?? .zero
        // Use the same nearest-integer rule as all integer input coordinates.
        return ["x": Int(loc.x.rounded()), "y": Int(loc.y.rounded())]
    case "screenshot":
        try guardSafety()
        if !CGPreflightScreenCaptureAccess() {
            throw HelperError.notPermitted(
                "Screen Recording permission is required; turn it on in System Settings > " +
                "Privacy & Security, then retry. macOS can take up to a minute to apply it")
        }
        let display = (req["display"] as? Int) ?? 0
        let cap = try captureDisplay(display)
        return [
            "png_b64": cap.pngBase64, "width": cap.width, "height": cap.height,
            "scale": cap.scale, "origin_x": cap.originX, "origin_y": cap.originY,
        ]
    case "move":
        try guardInput()
        try moveMouse(try pointArg(req))
        return [:]
    case "click":
        try guardInput()
        let button = (req["button"] as? String) ?? "left"
        let count = (req["count"] as? Int) ?? 1
        try click(try pointArg(req), button: button, count: count, flags: modifierFlags(req["modifiers"]))
        return [:]
    case "mouse_down":
        try guardInput()
        let b = try mouseButton((req["button"] as? String) ?? "left")
        try postMouse(b.down, try pointArg(req), b.button)
        return [:]
    case "mouse_up":
        try guardInput()
        let b = try mouseButton((req["button"] as? String) ?? "left")
        try postMouse(b.up, try pointArg(req), b.button)
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
        try typeText(text)
        return [:]
    case "key":
        try guardInput()
        guard let combo = req["combo"] as? String else {
            throw HelperError.badRequest("key needs combo")
        }
        let repeatCount = (req["repeat"] as? Int) ?? 1
        let hold = (req["hold"] as? Bool) ?? false
        let release = (req["release"] as? Bool) ?? false
        if hold && release { throw HelperError.badRequest("hold and release are exclusive") }
        try keyCombo(combo, repeatCount: repeatCount, hold: hold, release: release)
        return [:]
    case "scroll":
        try guardInput()
        try scroll(at: try pointArg(req), dx: try intArg(req, "dx"), dy: try intArg(req, "dy"), flags: modifierFlags(req["modifiers"]))
        return [:]
    default:
        throw HelperError.badRequest("unknown op: \(op)")
    }
}

// MARK: - indicator and application lifecycle

final class Indicator {
    private var windows: [NSWindow] = []
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

    init() {
        let menu = NSMenu()
        menu.addItem(withTitle: "Zeta controls the desktop", action: nil, keyEquivalent: "")
        menu.addItem(withTitle: "kill: ctrl+opt+cmd+esc", action: nil, keyEquivalent: "")
        statusItem.menu = menu
        statusItem.button?.title = "Zeta controls the desktop"
        killed.observe { [weak self] in
            DispatchQueue.main.async { self?.showKilled() }
        }
    }

    func showActive() {
        guard !killed.isSet else { return }
        windows = NSScreen.screens.map { screen in
            let window = NSWindow(contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false)
            window.backgroundColor = .clear
            window.isOpaque = false
            window.ignoresMouseEvents = true
            window.level = .screenSaver
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            let view = NSView(frame: window.contentView?.bounds ?? screen.frame)
            view.wantsLayer = true
            view.layer?.borderColor = NSColor.systemRed.cgColor
            view.layer?.borderWidth = 4
            view.autoresizingMask = [.width, .height]
            window.contentView = view
            window.orderFrontRegardless()
            return window
        }
    }

    func hide() {
        windows.forEach { $0.orderOut(nil) }
        windows.removeAll()
    }

    private func showKilled() {
        hide()
        statusItem.button?.title = "Zeta desktop control killed"
        statusItem.menu?.items.forEach { $0.isEnabled = false }
    }
}

// MARK: - stdio loop

let stdout = FileHandle.standardOutput
let responseLock = NSLock()
let actionQueue = DispatchQueue(label: "zeta-computer-helper.actions")

func writeResponse(_ dict: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: dict) else { return }
    responseLock.lock()
    stdout.write(data)
    stdout.write(Data([0x0a]))
    responseLock.unlock()
}

func errorResponse(_ id: Int, _ error: Error) -> [String: Any] {
    let he = (error as? HelperError) ?? .internalError("\(error)")
    return ["id": id, "ok": false, "error": ["code": he.code, "message": he.message]]
}

func runRequest(
    id: Int, op: String, request: [String: Any], indicator: Indicator,
    beforeCommit: (() -> Void)? = nil, sink: @escaping ([String: Any]) -> Void
) {
    if currentKill().isSet {
        sink(errorResponse(id, HelperError.killed))
        return
    }
    if !selfTestMode { DispatchQueue.main.sync { indicator.showActive() } }
    do {
        var payload = try handle(op, request)
        payload["id"] = id
        payload["ok"] = true
        beforeCommit?()
        sink(currentKill().commitReply() ? payload : errorResponse(id, HelperError.killed))
    } catch {
        sink(errorResponse(id, error))
    }
    if !selfTestMode { DispatchQueue.main.sync { if !killed.isSet { indicator.hide() } } }
}

func selfTestKill() {
    selfTestMode = true
    let cases: [(String, String, Int)] = [
        ("drag", "drag", 2), ("type", "type", 1), ("mouse_down", "mouse_down", 1),
        ("kill_after_last_step", "type", 2), ("kill_before_commit", "type", 2),
        ("kill_after_commit", "type", 2),
    ]
    func triggerKill() {
        let event = CGEvent(keyboardEventSource: nil, virtualKey: 53, keyDown: true)!
        event.flags = [.maskControl, .maskAlternate, .maskCommand]
        _ = killCallback(proxy: OpaquePointer(bitPattern: 1)!, type: .keyDown, event: event, userInfo: nil)
    }
    for (name, op, blockAfter) in cases {
        let flag = Flag()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let cleanup = DispatchSemaphore(value: 0)
        var events: [String] = []
        var posts = 0
        let gate = EventGate(killed: flag) { event in
            posts += 1
            if event.type == .leftMouseDown { events.append("mouse_down") }
            else if event.type == .leftMouseDragged { events.append("drag_\(posts - 1)") }
            else if event.type == .leftMouseUp { events.append("mouse_up"); cleanup.signal() }
            else if event.type == .keyDown { events.append("key_down") }
            else if event.type == .keyUp { events.append("key_up") }
            if posts == blockAfter { entered.signal(); release.wait() }
        }
        eventGate = gate
        activeKill = flag
        flag.observe { DispatchQueue.global().async { gate.kill() } }
        let replies = DispatchSemaphore(value: 0)
        var reply = ""
        actionQueue.async {
            let req: [String: Any] = op == "drag"
                ? ["path": [[1, 1], [2, 2], [3, 3]]]
                : op == "mouse_down" ? ["x": 4, "y": 5, "button": "left"]
                : ["text": "ab"]
            runRequest(
                id: 1, op: op, request: req, indicator: indicator,
                beforeCommit: name == "kill_before_commit" ? triggerKill : nil
            ) { response in
                reply = (response["ok"] as? Bool == true) ? "ok" :
                    ((response["error"] as? [String: Any])?["code"] as? String ?? "internal")
                replies.signal()
            }
        }
        entered.wait()
        if name == "kill_after_commit" {
            release.signal(); replies.wait(); triggerKill(); actionQueue.sync {}
        } else if name == "kill_before_commit" {
            release.signal(); actionQueue.sync {}
        } else {
            triggerKill()
            release.signal(); actionQueue.sync {}
            if name == "mouse_down" { cleanup.wait() }
        }
        var later = ""
        runRequest(id: 2, op: "type", request: ["text": "z"], indicator: indicator) { response in
            later = (response["ok"] as? Bool == true) ? "ok" :
                ((response["error"] as? [String: Any])?["code"] as? String ?? "internal")
        }
        if name != "kill_after_commit" { replies.wait() }
        print((try? JSONSerialization.data(withJSONObject: [
            "name": name, "events": events, "reply": reply, "later_reply": later,
        ])).flatMap { String(data: $0, encoding: .utf8) } ?? "{}")
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let indicator = Indicator()
killed.observe { DispatchQueue.main.async { indicator.hide() } }
if CommandLine.arguments.contains("--selftest-kill") {
    selfTestKill()
    exit(0)
}
installKillSwitch()

Thread {
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
        actionQueue.async {
            runRequest(id: id, op: op, request: req, indicator: indicator, sink: writeResponse)
        }
    }
    // Shutdown owns the boundary: stop new posts, drain accepted work, then release again.
    activeKill.set()
    actionQueue.sync {}
    eventGate.kill()
    DispatchQueue.main.sync {
        indicator.hide()
        app.terminate(nil)
    }
}.start()

app.run()

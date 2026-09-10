// Draw the icon as vector paths; no font or image dependencies.
import AppKit

let size = 1024
let bitmap = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
    isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
NSColor(calibratedWhite: 0.12, alpha: 1).setFill()
NSBezierPath(roundedRect: NSRect(x: 64, y: 64, width: 896, height: 896),
             xRadius: 200, yRadius: 200).fill()

// Lowercase Greek zeta: a sweeping top, curved bowl, and descending tail.
let glyph = NSBezierPath()
glyph.move(to: NSPoint(x: 340, y: 780))
glyph.curve(to: NSPoint(x: 710, y: 755),
            controlPoint1: NSPoint(x: 400, y: 722),
            controlPoint2: NSPoint(x: 590, y: 726))
glyph.curve(to: NSPoint(x: 318, y: 412),
            controlPoint1: NSPoint(x: 548, y: 664),
            controlPoint2: NSPoint(x: 320, y: 556))
glyph.curve(to: NSPoint(x: 628, y: 318),
            controlPoint1: NSPoint(x: 310, y: 296),
            controlPoint2: NSPoint(x: 512, y: 350))
glyph.curve(to: NSPoint(x: 562, y: 185),
            controlPoint1: NSPoint(x: 738, y: 287),
            controlPoint2: NSPoint(x: 669, y: 202))
glyph.lineWidth = 70
glyph.lineCapStyle = .round
glyph.lineJoinStyle = .round
NSColor(calibratedWhite: 0.95, alpha: 1).setStroke()
glyph.stroke()
NSGraphicsContext.restoreGraphicsState()
try bitmap.representation(using: .png, properties: [:])!.write(
    to: URL(fileURLWithPath: CommandLine.arguments[1]))

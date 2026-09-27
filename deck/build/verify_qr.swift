// Decode the deck's QR PNG with Apple's own detectors, as an independent check.
//
// deck/build/qrgen.py is a from-scratch QR encoder written for this deck. It
// verifies itself by reversing its own encoding, which cannot catch a convention
// error that its own reader repeats. This decodes the rendered PNG with Core
// Image's QR detector and with the Vision framework -- the readers behind the
// iOS and macOS camera -- so a PASS here means a judge's phone will read it.
//
// Run:
//   swiftc -O -o /tmp/verify_qr deck/build/verify_qr.swift
//   /tmp/verify_qr deck/figures/fig7_live_demo_qr.png https://surface-vision.github.io

import Foundation
import CoreImage
import Vision

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: verify_qr <png> <expected-url>\n".data(using: .utf8)!)
    exit(64)
}
let url = URL(fileURLWithPath: args[1])
let expected = args[2]
guard let image = CIImage(contentsOf: url) else {
    print("could not load \(url.path)")
    exit(65)
}
print("png          \(url.path)")
print("pixels       \(Int(image.extent.width)) x \(Int(image.extent.height))")
print("expected     \(expected)")

let detector = CIDetector(ofType: CIDetectorTypeQRCode, context: nil,
                          options: [CIDetectorAccuracy: CIDetectorAccuracyHigh])!
let coreImage = detector.features(in: image).compactMap { ($0 as? CIQRCodeFeature)?.messageString }
print("core image   \(coreImage.count) code(s): \(coreImage.joined(separator: " | "))")

let request = VNDetectBarcodesRequest()
if #available(macOS 12.0, *) { request.symbologies = [.qr] }
try VNImageRequestHandler(ciImage: image, options: [:]).perform([request])
let vision = (request.results ?? []).compactMap { $0.payloadStringValue }
print("vision       \(vision.count) code(s): \(vision.joined(separator: " | "))")

let pass = coreImage == [expected] && vision == [expected]
print("verdict      \(pass ? "PASS" : "FAIL")")
exit(pass ? 0 : 1)

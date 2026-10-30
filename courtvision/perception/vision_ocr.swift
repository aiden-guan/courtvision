// Text recognition with Apple's Vision framework (macOS), for perception/vision_ocr.py.
//
// Protocol on stdin / stdout, one request at a time:
//   request:  4-byte big-endian length, then an encoded image (PNG or JPEG)
//   response: one JSON line, a list of {"text", "conf", "box": [x, y, w, h]} with the box in image
//             pixels, origin top-left.
// A zero length ends the session. Built by vision_ocr.py with `swiftc -O`.
import AppKit
import Foundation
import Vision

func readExactly(_ n: Int) -> Data? {
    var out = Data()
    while out.count < n {
        let chunk = FileHandle.standardInput.readData(ofLength: n - out.count)
        if chunk.isEmpty { return nil }
        out.append(chunk)
    }
    return out
}

let fast = CommandLine.arguments.contains("--fast")
let out = FileHandle.standardOutput

while true {
    guard let header = readExactly(4) else { break }
    let n = header.reduce(0) { ($0 << 8) | Int($1) }
    if n == 0 { break }
    guard let data = readExactly(n) else { break }
    var results: [[String: Any]] = []
    if let rep = NSBitmapImageRep(data: data), let cg = rep.cgImage {
        let w = Double(cg.width), h = Double(cg.height)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = fast ? .fast : .accurate
        request.usesLanguageCorrection = false
        request.minimumTextHeight = 0.0
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        try? handler.perform([request])
        for obs in request.results ?? [] {
            guard let cand = obs.topCandidates(1).first else { continue }
            let b = obs.boundingBox  // normalised, origin bottom-left
            results.append([
                "text": cand.string,
                "conf": Double(cand.confidence),
                "box": [b.origin.x * w, (1.0 - b.origin.y - b.size.height) * h, b.size.width * w, b.size.height * h],
            ])
        }
    }
    let json = (try? JSONSerialization.data(withJSONObject: results)) ?? Data("[]".utf8)
    out.write(json)
    out.write(Data("\n".utf8))
}

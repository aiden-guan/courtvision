"""The browser check of a new main model (research note §10.15, §13): verify.html runs the app's
courtvision/server/static/nba-student.onnx on ONNX Runtime Web (WASM, one thread) on the deterministic input of
nba-validation.json and compares its 640 probed outputs with the reference the export wrote (tolerance 0.003). Static
files with cross-origin isolation (COOP/COEP), from the app's static folder. usage: serve.py [port]; then open
http://127.0.0.1:<port>/verify.html"""
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "courtvision/server/static"
FILES = {"verify.html": Path(__file__).parent / "verify.html",
         **{name: STATIC / name for name in ("ort.min.js", "ort-wasm-simd-threaded.jsep.mjs",
                                             "ort-wasm-simd-threaded.jsep.wasm", "nba-student.onnx",
                                             "nba-validation.json")}}
TYPES = {".json": "application/json", ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
         ".wasm": "application/wasm", ".onnx": "application/octet-stream"}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        f = FILES.get(self.path.split("?")[0].lstrip("/"))
        if f is None or not f.is_file():
            self.send_error(404)
            return
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(f.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), Handler).serve_forever()

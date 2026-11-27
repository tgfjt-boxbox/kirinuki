"""Serve the browser latency page on 127.0.0.1 and wait for its result.

usage: python3 bench/browser/serve_latency.py OUT.json [--port 8765] [--timeout 300] [--mask-dir DIR]
Serves bench/, models/ and data/ from the repository, so the page loads the vendored @mediapipe/tasks-vision and models/ over http
(file:// blocks the module imports). The page POSTs its result JSON to /result; the server writes it to OUT.json
and stops. GET /hold answers only after the result arrives: the page puts it in an <img>, which keeps the load
event pending, so `chrome --dump-dom` exits after the measurement instead of right after the first paint.
With --mask-dir, POST /mask/NAME.png writes the body to DIR/NAME.png (litert_masks.html sends one mask per image).
Exits 1 if no result comes before the timeout or the page reports an error.
"""
import argparse
import functools
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVED = ("/bench/", "/models/", "/data/")
PIXEL = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        name = self.path.removeprefix("/mask/")
        if name != self.path and self.server.mask_dir and name.endswith(".png") and "/" not in name and ".." not in name:
            (self.server.mask_dir / name).write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(204)
            self.end_headers()
            return
        if self.path != "/result":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.result = json.loads(body)
        self.send_response(204)
        self.end_headers()
        self.server.done.set()

    def do_GET(self):
        if self.path != "/hold":
            if self.path.startswith(SERVED) and ".." not in self.path:
                super().do_GET()
            else:
                self.send_error(404)
            return
        self.server.done.wait(self.server.timeout_s)
        self.send_response(200)
        self.send_header("Content-Type", "image/gif")
        self.send_header("Content-Length", str(len(PIXEL)))
        self.end_headers()
        self.wfile.write(PIXEL)

    def log_message(self, fmt, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--mask-dir", type=Path)
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(ROOT)))
    server.done, server.result, server.timeout_s = threading.Event(), None, args.timeout
    server.mask_dir = args.mask_dir
    if args.mask_dir:
        args.mask_dir.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"serving {ROOT} on http://127.0.0.1:{args.port}/", flush=True)
    got = server.done.wait(args.timeout)
    server.shutdown()
    if not got:
        sys.exit(f"no result within {args.timeout:.0f} s")
    args.out.write_text(json.dumps(server.result, indent=1) + "\n")
    print(f"wrote {args.out}")
    if "error" in server.result:
        sys.exit(f"page error: {server.result['error']}")


if __name__ == "__main__":
    main()

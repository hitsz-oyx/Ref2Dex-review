"""Rewrite the broken HRDexDB hf-mirror Xet download hostname.

The mirror's Git-LFS batch endpoint is reachable from the training host, but
currently returns signed object URLs under ``us.aws.cdn.hf-mirror.org``, whose
DNS record is unavailable.  The same signed URL is accepted by Hugging Face's
canonical ``us.aws.cdn.hf.co`` endpoint.  This local proxy only forwards LFS
batch POSTs and rewrites that hostname in the JSON response; object bytes are
still downloaded directly by git-lfs.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BROKEN_HOST = b"us.aws.cdn.hf-mirror.org"
WORKING_HOST = b"us.aws.cdn.hf.co"


class BatchProxyHandler(BaseHTTPRequestHandler):
    upstream: str

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") != "/objects/batch":
            self.send_error(404, "Only Git-LFS batch requests are supported")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request = Request(
            self.upstream,
            data=body,
            method="POST",
            headers={
                "Accept": self.headers.get("Accept", "application/vnd.git-lfs+json"),
                "Content-Type": self.headers.get("Content-Type", "application/vnd.git-lfs+json"),
                "User-Agent": self.headers.get("User-Agent", "git-lfs"),
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                payload = response.read().replace(BROKEN_HOST, WORKING_HOST)
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/vnd.git-lfs+json"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("Content-Type", "text/plain"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument(
        "--upstream",
        default="https://hf-mirror.com/datasets/HRDexDB/HRDexDB.git/info/lfs/objects/batch",
    )
    args = parser.parse_args()
    BatchProxyHandler.upstream = args.upstream
    server = ThreadingHTTPServer((args.host, args.port), BatchProxyHandler)
    print(f"Git-LFS batch proxy listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

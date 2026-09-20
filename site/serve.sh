#!/bin/sh
# Preview the review site at http://localhost:8322/ (no-store, so edits show up on reload).
cd "$(dirname "$0")"
exec python3 - <<'PY'
import http.server, socketserver

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()
    def log_message(self, *a): pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", 8322), NoCache) as srv:
    print("serving http://localhost:8322/ (no-store)")
    srv.serve_forever()
PY

import http.server, urllib.request, sys, time, json
LOG = open(__import__("os").environ.get("PROXY_LOG", "ollama_proxy.log"), "a", buffering=1)
class H(http.server.BaseHTTPRequestHandler):
    def _fwd(self, method):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else None
        t0 = time.time()
        req = urllib.request.Request("http://127.0.0.1:11434" + self.path, data=body, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read(); code = r.status
        except Exception as e:
            data = str(e).encode(); code = 502
        LOG.write(f"{time.strftime('%H:%M:%S')} {method} {self.path} {code} {time.time()-t0:.1f}s req={len(body or b'')}B resp={len(data)}B\n")
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self): self._fwd("GET")
    def do_POST(self): self._fwd("POST")
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", 11500), H).serve_forever()

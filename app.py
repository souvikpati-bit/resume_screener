"""Hiring dashboard server. Locally: python app.py -> http://127.0.0.1:8765. On Vercel: api/index.py.

When DASHBOARD_PASSWORD is set, every page and API call needs a login. It must be set
anywhere the app is reachable by other people (e.g. Vercel).
"""
import base64
import csv
import hashlib
import hmac
import io
import json
import mimetypes
import re
import sys
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote

import screener as S

STATIC = Path(__file__).parent / "static"
PORT = int(S.env("PORT", "8765"))
# Vercel caps a request body at 4.5 MB, and uploads arrive base64-encoded (+33%).
MAX_UPLOAD = (3 if S.SERVERLESS else 15) * 1024 * 1024
COOKIE = "screener_auth"


def session_token():
    pw = S.env("DASHBOARD_PASSWORD")
    return hmac.new(pw.encode(), b"screener-session-v1", hashlib.sha256).hexdigest() if pw else None


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in</title><style>
:root{--bg:#0e100f;--card:#1a1c1b;--line:#2c2f2d;--ink:#fffce1;--muted:#a8a693;--green:#0ae448}
@media (prefers-color-scheme:light){:root{--bg:#fffce1;--card:#fff;--line:#e6e1c4;--ink:#0e100f;--muted:#5e5c4f;--green:#08a838}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px}
form{width:min(380px,100%);background:var(--card);border:1px solid var(--line);border-radius:24px;padding:30px}
h1{font-size:26px;letter-spacing:-.03em;margin:0 0 4px}p{color:var(--muted);margin:0 0 20px}
input{width:100%;padding:12px 14px;border-radius:14px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit;outline:none}
input:focus{border-color:var(--green)}button{margin-top:14px;width:100%;padding:12px;border:0;border-radius:99px;background:var(--green);color:#0e100f;font:600 15px system-ui,sans-serif;cursor:pointer}
.err{color:#ff5c7a;margin:12px 0 0;font-size:14px}</style></head><body>
<form method="post" action="/login"><h1>Hiring Dashboard</h1><p>Enter the dashboard password.</p>
<input type="password" name="password" autofocus autocomplete="current-password" aria-label="Password" required>
<button type="submit">Sign in</button>__ERR__</form></body></html>"""


def config():
    provider = S.env("LLM_PROVIDER", "claude").lower()
    model = S.env("GEMINI_MODEL", "gemini-3.5-flash") if provider == "gemini" else S.env("CLAUDE_MODEL", "claude-opus-5")
    key_ok = bool(S.env("GEMINI_API_KEY")) if provider == "gemini" else bool(S.env("ANTHROPIC_API_KEY"))
    return {"provider": provider, "model": model, "llm_key_set": key_ok, "smtp_ready": S.smtp_ready(), "email_via": S.email_provider(), "test_to": S.test_recipient(),
            "demo": bool(S.env("DEMO_MODE")), "sender": S.env("SENDER_NAME", "Arjun"), "company": S.env("COMPANY_NAME", "Kargo"),
            "params": [{"key": k, "label": l, "weight": w} for k, l, w in S.PARAMS], "max": S.MAX_WEIGHTED}


def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "name", "email", "role", "band", "total_24", "pct"] +
               [f"{k.upper()} {l}" for k, l, _ in S.PARAMS] + ["decision", "emails_sent", "why_ranked", "file"])
    done = sorted((c for c in S.store.all() if c["status"] == "done"), key=lambda c: -c["result"]["total"])
    for i, c in enumerate(done, 1):
        r = c["result"]
        w.writerow([i, r["candidate"]["name"], r["candidate"]["email"], r["candidate"]["role_assessed"], r["band"],
                    r["total"], r["pct"]] + [p["score"] for p in r["params"]] +
                   [c["decision"], "; ".join(f'{s["kind"]} {s["at"]}' for s in c["sent"]), r["why_ranked"], c["filename"]])
    return buf.getvalue().encode("utf-8-sig")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "/api/candidates " not in (str(args[0]) if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def _send(self, code, body=b"", ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD * 2:
            raise ValueError("Request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _route(self):
        m = re.fullmatch(r"/api/candidates/([0-9a-f]+)(?:/(\w+))?", self.path.split("?")[0])
        return (m.group(1), m.group(2)) if m else (None, None)

    # ---------- login ----------

    def _secure(self):
        return S.SERVERLESS or self.headers.get("X-Forwarded-Proto") == "https"

    def _authed(self):
        tok = session_token()
        if not tok:
            return True  # no password configured (local use only)
        jar = SimpleCookie(self.headers.get("Cookie") or "")
        return COOKIE in jar and hmac.compare_digest(jar[COOKIE].value, tok)

    def _cookie(self, value, max_age):
        flags = "; Secure" if self._secure() else ""
        return f"{COOKIE}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}{flags}"

    def _login_page(self, err=""):
        html = LOGIN_PAGE.replace("__ERR__", f'<p class="err">{err}</p>' if err else "")
        self._send(401 if err else 200, html.encode(), "text/html; charset=utf-8")

    def _guard(self):
        """Let the request through only when logged in; on Vercel also load fresh data from Supabase."""
        path = self.path.split("?")[0]
        if path == "/login" or path == "/logout":
            return False
        if not self._authed():
            if path.startswith("/api/"):
                self._send(401, {"error": "Please sign in again"})
            else:
                self._send(302, b"", headers={"Location": "/login"})
            return False
        if S.SERVERLESS:
            S.store.refresh()
        return True

    def _login(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(n, 4096)).decode("utf-8", "replace")
        pw = (parse_qs(raw).get("password") or [""])[0]
        real = S.env("DASHBOARD_PASSWORD") or ""
        if real and hmac.compare_digest(pw.encode(), real.encode()):
            return self._send(302, b"", headers={"Location": "/", "Set-Cookie": self._cookie(session_token(), 30 * 86400)})
        time.sleep(1)  # slow down password guessing
        self._login_page("Wrong password. Try again.")

    def do_HEAD(self):
        self._send(200, b"", "text/html")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/login":
            return self._send(302, b"", headers={"Location": "/"}) if self._authed() and session_token() else self._login_page()
        if path == "/logout":
            return self._send(302, b"", headers={"Location": "/login", "Set-Cookie": self._cookie("", 0)})
        if not self._guard():
            return
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/candidates":
            return self._send(200, {"candidates": S.store.all(), "config": config()})
        if path == "/api/export.csv":
            return self._send(200, export_csv(), "text/csv; charset=utf-8",
                              {"Content-Disposition": 'attachment; filename="ranked_candidates.csv"'})
        cid, action = self._route()
        if cid and action == "file":
            c = S.store.get(cid)
            if not c:
                return self._send(404, {"error": "not found"})
            ctype = mimetypes.guess_type(c["filename"])[0] or "application/octet-stream"
            return self._send(200, S.resume_path(c).read_bytes(), ctype,
                              {"Content-Disposition": f"inline; filename*=UTF-8''{quote(c['filename'])}"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] == "/login":
            return self._login()
        if not self._guard():
            return
        try:
            body = self._json()
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        if self.path == "/api/upload":
            try:
                blob = base64.b64decode(body["data"])
            except Exception:
                return self._send(400, {"error": "bad file data"})
            if len(blob) > MAX_UPLOAD:
                return self._send(400, {"error": f"File over {MAX_UPLOAD // (1024 * 1024)} MB"})
            if Path(body.get("name", "")).suffix.lower() not in (".pdf", ".docx", ".txt"):
                return self._send(400, {"error": "Only PDF, DOCX or TXT resumes"})
            prior = S.store.get(S.store.by_hash(hashlib.sha256(blob).hexdigest()) or "")
            sent_before = len(prior["sent"]) if prior else 0
            try:
                cid, dup = S.ingest(body["name"], blob)
            except Exception as e:
                return self._send(503, {"error": f"Could not save this resume: {str(e)[:200]}. Try again in a minute."})
            return self._send(200, {"id": cid, "duplicate": dup, "sent_before": sent_before})

        cid, action = self._route()
        c = S.store.get(cid) if cid else None
        if not c:
            return self._send(404, {"error": "not found"})
        if action == "rescore":
            role = body.get("role") if body.get("role") in ("PM", "Sr PM") else None
            return self._send(200, S.rescore(cid, role))
        if action == "send":
            try:
                return self._send(200, S.email_candidate(cid, body.get("kind"), to=body.get("to") or "",
                                                         subject=body.get("subject", ""), body=body.get("body", ""),
                                                         force=bool(body.get("force"))))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except PermissionError as e:
                return self._send(409, {"error": str(e)})
            except Exception as e:
                return self._send(502, {"error": str(e)[:300]})
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        if not self._guard():
            return
        cid, _ = self._route()
        c = S.store.get(cid) if cid else None
        if not c:
            return self._send(404, {"error": "not found"})
        body = self._json()
        fields = {}
        if body.get("decision") in ("invite", "decline", "hold"):
            fields["decision"] = body["decision"]
        if isinstance(body.get("drafts"), dict):
            fields["drafts"] = {**(c["drafts"] or {}), **body["drafts"]}
        if "email_override" in body:
            fields["email_override"] = (body["email_override"] or "").strip()
        self._send(200, S.store.update(cid, **fields))

    def do_DELETE(self):
        if not self._guard():
            return
        cid, _ = self._route()
        if not cid or not S.store.get(cid):
            return self._send(404, {"error": "not found"})
        S.store.delete(cid)
        self._send(200, {"ok": True})


if __name__ == "__main__":
    cfg = config()
    print(f"Scoring with {cfg['provider']} / {cfg['model']}  (API key {'found' if cfg['llm_key_set'] else 'MISSING - edit .env'})")
    print(f"Email sending: {'ready' if cfg['smtp_ready'] else 'not configured (drafts only) - edit .env'}")
    print(f"Storage: {'Supabase ' + S.env('SUPABASE_URL') if S.store.sb else 'local file data/candidates.json'}")
    print(f"Login: {'password required' if session_token() else 'none (DASHBOARD_PASSWORD not set; local use only)'}")
    print(f"Dashboard: http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

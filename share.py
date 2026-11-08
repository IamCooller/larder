#!/usr/bin/env python3
"""The `./larder share` control service: a device broadcasts itself on the LAN, hands out a 12-character
pairing code and runs model operations for the launcher that pairs with it."""
import argparse
import base64
import binascii
import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import engine_version
import machine

HERE = Path(__file__).resolve().parent

# The UDP port every beacon goes to and the magic word that marks it as this feature.
BEACON_PORT = 43890
BEACON_MAGIC = "moe-share"
# A pairing code lives this many seconds; this many wrong tries lock pairing until sharing restarts, so a code cannot be guessed by rotation.
CODE_TTL = 600
CODE_TRIES = 5
# A pairing code is this many characters from a base32 alphabet without I, L, O and U (60 bits), so a recorded proof cannot be brute-forced offline.
CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
CODE_LEN = 12
# Letters a person may type for the digits they look like.
_CODE_READS = str.maketrans("OIL", "011")
# The session key is stretched from the code with this many PBKDF2 rounds.
PBKDF2_ITER = 200_000
# An authenticated call is accepted only within this many seconds of its timestamp.
REPLAY_WINDOW = 60
# Unfinished pairing sessions are kept at most this long and this many at a time.
SESSION_TTL = 120
MAX_SESSIONS = 32
# The largest control-call body read; pairing and control calls are small JSON, so a bigger one is refused unread.
MAX_BODY = 64 << 10


def _b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _nonce():
    return _b64(os.urandom(16))


def new_code():
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def normalize_code(text):
    """The code as both sides use it, from what a person typed: case, spaces and dashes ignored."""
    return re.sub(r"[\s-]", "", text or "").upper().translate(_CODE_READS)


def format_code(code):
    """The code in groups of four, as the device shows it."""
    return "-".join(code[i:i + 4] for i in range(0, len(code), 4))


def write_private(path, text):
    """Write text to a file only this user can read, its mode set when it is created and before any byte is written."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)          # a file an older version made with a wider mode
        f.write(text)


def derive_key(code, salt):
    """The shared secret both sides stretch from the code and the salt the device picked."""
    return hashlib.pbkdf2_hmac("sha256", code.encode(), salt, PBKDF2_ITER, 32)


def hostname():
    return socket.gethostname() or "this computer"


LOCAL_NAMES = ("127.0.0.1", "localhost", "::1")


def local_refusal(client_ip, headers):
    """Why a request to a local-only port is refused, or None: it must come from this computer, and a cross-site page (its Origin) or a DNS-rebinding page (its Host) must name this computer too, while programs send no Origin."""
    if client_ip not in ("127.0.0.1", "::1"):
        return "local connections only"
    host = urlsplit("//" + (headers.get("Host") or "")).hostname
    origin = headers.get("Origin")
    if ((host is not None and host not in LOCAL_NAMES)
            or (origin is not None and urlsplit(origin).hostname not in LOCAL_NAMES)):
        return "open this server as http://127.0.0.1:PORT or http://localhost:PORT"
    return None


def openssl_bin():
    """The openssl executable when this computer has one, else None."""
    return shutil.which("openssl")


def cert_fingerprint(cert_path):
    """The sha256 fingerprint of a PEM certificate, as the hex string the other side pins."""
    try:
        pem = Path(cert_path).read_text()
    except OSError:
        return ""
    m = re.search(r"-----BEGIN CERTIFICATE-----\s*(.+?)\s*-----END CERTIFICATE-----", pem, re.S)
    if not m:
        return ""
    try:
        der = base64.b64decode("".join(m.group(1).split()))
    except (ValueError, binascii.Error):
        return ""
    return hashlib.sha256(der).hexdigest()


def make_cert(cache):
    """The self-signed certificate made at the first share, or (None, None, "") without openssl."""
    openssl = openssl_bin()
    if not openssl:
        return None, None, ""
    cache = Path(cache)
    cert, key = cache / "share-cert.pem", cache / "share-key.pem"
    fp = cert_fingerprint(cert) if cert.is_file() and key.is_file() else ""
    if not fp:
        cache.mkdir(parents=True, exist_ok=True)
        cmd = [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
               "-keyout", str(key), "-out", str(cert), "-days", "3650",
               "-subj", "/CN=moe-share", "-sha256"]
        old_umask = os.umask(0o077)  # the key is born private, the chmod below only confirms it
        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        except (OSError, subprocess.SubprocessError):
            return None, None, ""
        finally:
            os.umask(old_umask)
        try:
            key.chmod(0o600)
        except OSError:
            pass
        fp = cert_fingerprint(cert)
    if not fp:
        return None, None, ""
    return cert, key, fp


def server_ssl_context(cert, key):
    """A TLS server context for the self-signed certificate."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))
    return ctx


def _tls_client_context():
    """A client context that accepts the self-signed cert; the fingerprint check comes next."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_fingerprint(host, port, timeout=10):
    """The device certificate's fingerprint from one TLS handshake, for pinning at first pairing."""
    conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=_tls_client_context())
    try:
        conn.connect()
        der = conn.sock.getpeercert(binary_form=True)
        return hashlib.sha256(der).hexdigest() if der else ""
    except (OSError, ssl.SSLError, http.client.HTTPException):
        return ""
    finally:
        conn.close()


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # a client closing mid-TLS or a stop is not a traceback


def pairings_path(cache=None):
    """Where this laptop keeps the tokens for the devices it paired with."""
    return (Path(cache) if cache else machine.CACHE) / "paired.json"


def load_pairings(cache=None):
    p = pairings_path(cache)
    if p.is_file():
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}
    return {}


def save_pairing(cache, host, port, name, token, https=False, fingerprint=""):
    """Remember a paired device's token, keyed by host and port, readable only by this user."""
    data = load_pairings(cache)
    data[f"{host}:{port}"] = {"host": host, "port": port, "name": name, "token": token,
                              "https": bool(https), "fingerprint": fingerprint or ""}
    path = pairings_path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_private(path, json.dumps(data, indent=1))
    return data


def pairing_for(host, port, cache=None):
    return load_pairings(cache).get(f"{host}:{port}") or {}


def token_for(host, port, cache=None):
    row = load_pairings(cache).get(f"{host}:{port}") or {}
    return row.get("token")


def sign(token, method, path, body=b"", nonce=None, ts=None):
    """The auth headers for one call: an HMAC over the method, path, body hash, nonce and time."""
    nonce = nonce or _nonce()
    ts = ts if ts is not None else int(time.time())
    line = f"{method}\n{path}\n{nonce}\n{ts}\n{hashlib.sha256(body).hexdigest()}".encode()
    mac = hmac.new(token.encode(), line, hashlib.sha256).digest()
    return {"X-Moe-Auth": _b64(mac), "X-Moe-Nonce": nonce, "X-Moe-Time": str(ts)}


def _http(host, port, method, path, body=None, headers=None, https=False, fingerprint=None,
          timeout=10):
    """One JSON call to the control service, returning (status, body); https pins the cert."""
    if https:
        return _https(host, port, method, path, body, headers, fingerprint, timeout)
    url = f"http://{host}:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw.decode("utf-8", "replace")}


def _https(host, port, method, path, body=None, headers=None, fingerprint=None, timeout=10):
    """One JSON call over TLS, comparing the peer certificate's fingerprint before sending."""
    conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=_tls_client_context())
    try:
        conn.connect()
        der = conn.sock.getpeercert(binary_form=True)
        if not der:
            return 502, {"error": "no TLS certificate"}
        fp = hashlib.sha256(der).hexdigest()
        if fingerprint and not hmac.compare_digest(fp, fingerprint):
            return 502, {"error": "TLS fingerprint mismatch"}
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data,
                     headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        raw = resp.read()
        try:
            return resp.status, json.loads(raw) if raw else {}
        except ValueError:
            return resp.status, {"error": raw.decode("utf-8", "replace")}
    except (OSError, ssl.SSLError, http.client.HTTPException) as err:
        return 502, {"error": f"cannot reach the device: {err}"}
    finally:
        conn.close()


def request(host, port, token, method, path, body=None, cache=None):
    """An authenticated call to a paired device, signing the method, path and body."""
    row = pairing_for(host, port, cache)
    data = json.dumps(body).encode() if body is not None else b""
    return _http(host, port, method, path, body=body, headers=sign(token, method, path, data),
                 https=row.get("https"), fingerprint=row.get("fingerprint"))


def pair(host, port, code, https=False, fingerprint=None):
    """The full pairing handshake: both sides prove they know the code, then the device hands out a
    token; over https it also pins the certificate. Returns (token, fingerprint)."""
    if https and not fingerprint:
        fingerprint = fetch_fingerprint(host, port)
        if not fingerprint:
            raise RuntimeError("the device has no TLS certificate to pin")
    status, start = _http(host, port, "POST", "/pair/start", body={},
                          https=https, fingerprint=fingerprint)
    if status != 200:
        raise RuntimeError(start.get("error") or "pairing refused")
    try:
        sid = start["session"]
        salt = _unb64(start["salt"])
        device_challenge = _unb64(start["challenge"])
    except (KeyError, ValueError, binascii.Error):
        raise RuntimeError("bad pairing start") from None
    key = derive_key(normalize_code(code), salt)
    client_challenge = os.urandom(16)
    # The certificate the client sees is inside both proofs, so a relay with its own certificate cannot pair.
    bind = (fingerprint or "").encode()
    proof = hmac.new(key, b"client|" + device_challenge + bind, hashlib.sha256).digest()
    status, ans = _http(host, port, "POST", "/pair/answer", body={
        "session": sid, "challenge": _b64(client_challenge), "proof": _b64(proof)},
        https=https, fingerprint=fingerprint)
    if status != 200:
        raise RuntimeError(ans.get("error") or "wrong code")
    try:
        device_proof = _unb64(ans["proof"])
        token = ans["token"]
    except (KeyError, ValueError, binascii.Error):
        raise RuntimeError("bad pairing answer") from None
    if not hmac.compare_digest(hmac.new(key, b"device|" + client_challenge + bind, hashlib.sha256).digest(),
                               device_proof):
        raise RuntimeError("the device did not prove it knows the code")
    return token, fingerprint or ""


class ShareHandler(BaseHTTPRequestHandler):
    service = None  # the ShareService, set before serving

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _route(self):
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _read_body(self):
        """The body, or None after a 413 when its Content-Length is not a number up to MAX_BODY."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 <= n <= MAX_BODY:
            self.close_connection = True
            self._json(413, {"error": f"the body must be at most {MAX_BODY} bytes"})
            return None
        return self.rfile.read(n) if n else b""

    def _read_json(self):
        raw = self._read_body()
        return None if raw is None else self._parse(raw)

    @staticmethod
    def _parse(raw):
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {}

    def _authed(self, method, path):
        """The request body when its signature checks out, or None after the error is written."""
        body = self._read_body()
        if body is None:
            return None
        status, err = self.service.check_auth(self.headers, method, path, body)
        if err:
            self._json(status, err)
            return None
        return body

    def do_GET(self):
        path = self._route()
        if path == "/":
            self._json(200, self.service.hello())
        elif path == "/info":
            body = self._authed("GET", path)
            if body is not None:
                code, resp = self.service.info()
                self._json(code, resp)
        elif path == "/models":
            body = self._authed("GET", path)
            if body is not None:
                code, resp = self.service.models()
                self._json(code, resp)
        elif path.startswith("/prepare/"):
            body = self._authed("GET", path)
            if body is not None:
                code, resp = self.service.prepare_status(path[len("/prepare/"):])
                self._json(code, resp)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self._route()
        if path == "/pair/start":
            code, body = self.service.pair_start()
            self._json(code, body)
        elif path == "/pair/answer":
            ans = self._read_json()
            if ans is not None:
                code, body = self.service.pair_answer(ans)
                self._json(code, body)
        elif path == "/prepare":
            raw = self._authed("POST", path)
            if raw is not None:
                code, resp = self.service.prepare(self._parse(raw))
                self._json(code, resp)
        elif path == "/start":
            raw = self._authed("POST", path)
            if raw is not None:
                code, resp = self.service.start_model(self._parse(raw))
                self._json(code, resp)
        elif path == "/stop":
            body = self._authed("POST", path)
            if body is not None:
                code, resp = self.service.stop_model()
                self._json(code, resp)
        elif path == "/stop-share":
            body = self._authed("POST", path)
            if body is not None:
                self._json(200, {"stopped": True})
                threading.Thread(target=self.service.stop, daemon=True).start()
        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self):
        self._json(404, {"error": "not found"})


class ShareService:
    """The control service state: the code, the paired clients, and the HTTP server and beacon
    threads."""

    def __init__(self, host="0.0.0.0", port=0, cache=None, name=None, tls=False):
        self.name = name or hostname()
        self.cache = Path(cache) if cache else machine.CACHE
        self.tokens = self._load_clients()
        self.seen_nonces = {}
        self.sessions = {}
        self.wrong_tries = 0
        self.code_expires_at = 0.0
        self.locked = False
        self.stopped = False
        self.code = ""
        self.jobs = {}
        self.model_proc = None
        self.model_port = 0
        self.model_api_key = ""
        self.https = False
        self.cert = self.key = self.fingerprint = None
        self._rotate_code()
        ShareHandler.service = self
        self.httpd = QuietServer((host, port), ShareHandler)
        self.port = self.httpd.server_address[1]
        if tls:
            self.cert, self.key, self.fingerprint = make_cert(self.cache)
            if self.cert and self.key and self.fingerprint:
                try:
                    self.httpd.socket = server_ssl_context(self.cert, self.key).wrap_socket(
                        self.httpd.socket, server_side=True)
                    self.https = True
                except (OSError, ssl.SSLError):
                    self.https = False
        self._beacon_stop = threading.Event()
        self._started = False

    def _clients_path(self):
        return self.cache / "shared-clients.json"

    def _load_clients(self):
        p = self._clients_path()
        if p.is_file():
            try:
                data = json.loads(p.read_text())
                return data.get("tokens", []) if isinstance(data, dict) else []
            except ValueError:
                return []
        return []

    def _save_clients(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        write_private(self._clients_path(), json.dumps({"tokens": self.tokens}, indent=1))

    def _rotate_code(self):
        self.code = new_code()
        self.code_expires_at = time.time() + CODE_TTL
        self.wrong_tries = 0
        self.sessions = {}

    def _ensure_code(self):
        if not self.locked and time.time() >= self.code_expires_at:
            self._rotate_code()

    def note_wrong(self):
        """Count one wrong try and lock pairing after the allowed number, until sharing restarts."""
        self.wrong_tries += 1
        if self.wrong_tries >= CODE_TRIES:
            self.locked, self.code, self.sessions = True, "", {}

    def hello(self):
        self._ensure_code()
        return {"name": self.name, "sharing": True, "port": self.port,
                "https": self.https, "fingerprint": self.fingerprint or "",
                "code_ok": not self.locked and time.time() < self.code_expires_at, "locked": self.locked}

    def pair_start(self):
        """Begin pairing: a fresh salt and challenge for the code the menu is about to prove."""
        if self.locked:
            return 403, {"error": "pairing is locked after wrong codes: restart ./larder share on that computer"}
        self._ensure_code()
        now = time.time()
        self.sessions = {k: v for k, v in self.sessions.items() if now - v["t"] < SESSION_TTL}
        if len(self.sessions) >= MAX_SESSIONS:
            return 429, {"error": "too many pairing attempts at once"}
        sid = secrets.token_urlsafe(16)
        salt = os.urandom(16)
        challenge = os.urandom(16)
        self.sessions[sid] = {"salt": salt, "challenge": challenge, "t": time.time()}
        return 200, {"session": sid, "salt": _b64(salt), "challenge": _b64(challenge)}

    def pair_answer(self, body):
        """Finish pairing: verify the client's proof, prove back and issue a long token."""
        if self.locked:
            return 403, {"error": "pairing is locked after wrong codes: restart ./larder share on that computer"}
        self._ensure_code()
        sid = body.get("session", "")
        sess = self.sessions.get(sid)
        if sess is None:
            return 403, {"error": "no such pairing session"}
        key = derive_key(self.code, sess["salt"])
        try:
            client_challenge = _unb64(body.get("challenge", ""))
            proof = _unb64(body.get("proof", ""))
        except (ValueError, binascii.Error):
            return 403, {"error": "bad pairing message"}
        bind = (self.fingerprint or "").encode() if self.https else b""
        want = hmac.new(key, b"client|" + sess["challenge"] + bind, hashlib.sha256).digest()
        if not hmac.compare_digest(want, proof):
            self.note_wrong()
            self.sessions.pop(sid, None)
            return 403, {"error": "wrong code"}
        token = secrets.token_urlsafe(32)
        if token not in self.tokens:
            self.tokens.append(token)
            self._save_clients()
        self.sessions.pop(sid, None)
        device_proof = hmac.new(key, b"device|" + client_challenge + bind, hashlib.sha256).digest()
        return 200, {"proof": _b64(device_proof), "token": token}

    def check_auth(self, headers, method, path, body):
        """(status, error or None): verify a signed call against one of the paired tokens, with a
        fresh nonce and a timestamp inside the replay window."""
        auth = headers.get("X-Moe-Auth")
        nonce = headers.get("X-Moe-Nonce")
        ts = headers.get("X-Moe-Time")
        if not (auth and nonce and ts):
            return 403, {"error": "missing auth"}
        try:
            ts_i = int(ts)
        except ValueError:
            return 403, {"error": "bad timestamp"}
        if abs(time.time() - ts_i) > REPLAY_WINDOW:
            return 403, {"error": "replay or clock skew"}
        if nonce in self.seen_nonces:
            return 403, {"error": "replay"}
        try:
            mac = _unb64(auth)
        except (ValueError, binascii.Error):
            return 403, {"error": "bad auth"}
        line = f"{method}\n{path}\n{nonce}\n{ts_i}\n{hashlib.sha256(body).hexdigest()}".encode()
        for token in self.tokens:
            if hmac.compare_digest(hmac.new(token.encode(), line, hashlib.sha256).digest(), mac):
                self.seen_nonces[nonce] = ts_i
                self._prune_nonces(ts_i)
                return 200, None
        return 403, {"error": "bad token"}

    def _prune_nonces(self, now):
        cut = now - REPLAY_WINDOW * 2
        for k in [k for k, v in self.seen_nonces.items() if v < cut]:
            del self.seen_nonces[k]

    def info(self):
        """The machine facts and engine version, so the menu can show what it is pairing with."""
        return 200, {"machine": machine.facts(), "engine": engine_version.version()}

    def models(self):
        """The same JSON the device's own launcher prints for --list --json."""
        cmd = [sys.executable, str(HERE / "launch.py"), "--list", "--json"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=HERE)
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": f"could not list models ({type(err).__name__})"}
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()
            return 500, {"error": tail[-1] if tail else "the launcher could not list models"}
        start, end = r.stdout.find("{"), r.stdout.rfind("}")
        if start < 0 or end <= start:
            return 500, {"error": "the launcher printed no list"}
        try:
            return 200, json.loads(r.stdout[start:end + 1])
        except ValueError as err:
            return 500, {"error": f"bad list ({err})"}

    def prepare(self, data):
        """Start a background download/pack for one model and hand back a job id to poll."""
        model_id = data.get("id") or data.get("model")
        variant = data.get("variant") or "published"
        if not model_id:
            return 400, {"error": "no model id"}
        job_id = secrets.token_urlsafe(12)
        self.jobs[job_id] = {"state": "running", "log": [], "error": "", "started": time.time()}
        threading.Thread(target=self._prepare_job, args=(job_id, model_id, variant), daemon=True).start()
        return 200, {"job": job_id}

    def _prepare_job(self, job_id, model_id, variant):
        job = self.jobs[job_id]
        cmd = [sys.executable, str(HERE / "launch.py"), "--auto", "--model", model_id,
               "--variant", variant, "--prepare-only", "--yes"]
        try:
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
        except OSError as err:
            job.update(state="failed", error=str(err))
            return
        for line in proc.stdout:
            job["log"].append(line.rstrip("\n"))
            job["log"] = job["log"][-40:]
        code = proc.wait()
        job["state"] = "done" if code == 0 else "failed"
        if code != 0:
            job["error"] = job["log"][-1] if job["log"] else f"prepare failed (exit {code})"

    def prepare_status(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return 404, {"error": "no such job"}
        return 200, {"job": job_id, "state": job["state"], "error": job["error"], "log": job["log"]}

    def _free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def start_model(self, data):
        """Start the model server on a fresh LAN port behind a fresh api key, and report both."""
        if self.model_proc and self.model_proc.poll() is None:
            return 409, {"error": "a model is already running"}
        model_id = data.get("id") or data.get("model")
        variant = data.get("variant") or "published"
        if not model_id:
            return 400, {"error": "no model id"}
        port = self._free_port()
        api_key = secrets.token_urlsafe(24)
        self.cache.mkdir(parents=True, exist_ok=True)
        log = self.cache / f"shared-server-{time.strftime('%Y%m%d-%H%M%S')}.log"
        cmd = [sys.executable, str(HERE / "launch.py"), "--auto", "--model", model_id,
               "--variant", variant, "--port", str(port), "--host", "0.0.0.0",
               "--api-key", api_key, "--no-browser", "--no-chat"]
        if self.cert and self.key:
            cmd += ["--tls-cert", str(self.cert), "--tls-key", str(self.key)]
        try:
            self.model_proc = subprocess.Popen(cmd, cwd=HERE, stdout=open(log, "w"),
                                               stderr=subprocess.STDOUT)
        except OSError as err:
            return 500, {"error": str(err)}
        self.model_port, self.model_api_key = port, api_key
        return 200, {"port": port, "api_key": api_key, "model": model_id,
                     "https": bool(self.cert), "fingerprint": self.fingerprint or ""}

    def stop_model(self):
        if not (self.model_proc and self.model_proc.poll() is None):
            return 200, {"stopped": False}
        try:
            self.model_proc.terminate()
            self.model_proc.wait(30)
        except subprocess.TimeoutExpired:
            self.model_proc.kill()
        except OSError:
            pass
        self.model_proc = None
        return 200, {"stopped": True}

    def start(self):
        self._started = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._beacon_loop, daemon=True).start()

    def stop(self):
        self.stopped = True
        self._beacon_stop.set()
        if self.model_proc and self.model_proc.poll() is None:
            try:
                self.model_proc.terminate()
                self.model_proc.wait(10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.model_proc = None
        if self._started:
            self.httpd.shutdown()
        self.httpd.server_close()
        self._started = False

    def _beacon_payload(self):
        self._ensure_code()
        return json.dumps({"magic": BEACON_MAGIC, "name": self.name, "port": self.port,
                           "https": self.https, "v": 1}).encode()

    def _beacon_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not self._beacon_stop.is_set():
            payload = self._beacon_payload()
            for addr in ("<broadcast>", "255.255.255.255"):
                try:
                    s.sendto(payload, (addr, BEACON_PORT))
                except OSError:
                    pass
            self._beacon_stop.wait(2.0)
        s.close()


class ProxyHandler(BaseHTTPRequestHandler):
    target = None       # (host, port) of the shared device's model server
    api_key = ""
    https = False       # the model server speaks TLS
    fingerprint = ""    # the pinned certificate fingerprint, when https

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _forward(self):
        why = local_refusal(self.client_address[0], self.headers)
        if why:                              # the key added below must not reach a foreign page
            self._json(403, {"error": {"message": why, "type": "invalid_request_error"}})
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        host, port = self.target
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "connection", "content-length", "transfer-encoding",
                                        "authorization", "x-api-key", "accept-encoding")}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        conn = (http.client.HTTPSConnection(host, port, timeout=7200, context=_tls_client_context())
                if self.https else http.client.HTTPConnection(host, port, timeout=7200))
        try:
            conn.connect()
            if self.https:
                der = conn.sock.getpeercert(binary_form=True)
                fp = hashlib.sha256(der).hexdigest() if der else ""
                if not self.fingerprint or not hmac.compare_digest(fp, self.fingerprint):
                    self._json(502, {"error": "TLS fingerprint mismatch"})
                    return
            conn.request(self.command, self.path, body=body if body else None, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("content-length", "transfer-encoding", "connection"):
                    continue
                self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, ssl.SSLError, http.client.HTTPException) as err:
            self._json(502, {"error": f"cannot reach the device: {err}"})
        finally:
            conn.close()

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def do_DELETE(self):
        self._forward()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


class Proxy:
    """A tiny local server that forwards the chat port to a shared device and adds its api key."""

    def __init__(self, port, target_host, target_port, api_key, https=False, fingerprint=""):
        ProxyHandler.target = (target_host, target_port)
        ProxyHandler.api_key = api_key
        ProxyHandler.https = https
        ProxyHandler.fingerprint = fingerprint
        self.httpd = QuietServer(("127.0.0.1", port), ProxyHandler)
        self.port = self.httpd.server_address[1]

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def start_proxy(port, target_host, target_port, api_key, https=False, fingerprint=""):
    """Start the proxy on the local chat port, returning it ready to forward."""
    proxy = Proxy(port, target_host, target_port, api_key, https, fingerprint)
    proxy.start()
    return proxy


def main():
    ap = argparse.ArgumentParser(description="Share this computer with the launcher on another one (experimental).")
    ap.add_argument("--port", type=int, default=0, help="the control port (0 picks a free one)")
    ap.add_argument("--host", default="0.0.0.0", help="the address to bind (default 0.0.0.0)")
    ap.add_argument("--no-tls", action="store_true",
                    help="serve plain HTTP: the pairing, the token and every control call then travel in clear")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)     # the code must show at once, under nohup or ssh too
    svc = ShareService(host=args.host, port=args.port, tls=not args.no_tls)
    if not args.no_tls and not svc.https:
        svc.httpd.server_close()
        raise SystemExit("share: no TLS (openssl " + ("could not make a certificate" if openssl_bin() else "is not installed")
                         + "); install openssl, or run ./larder share --no-tls to send the pairing and the token in clear")
    svc.start()
    print(f"sharing this computer as {svc.name} (experimental)")
    print(f"pairing code: {format_code(svc.code)}  (valid {CODE_TTL // 60} minutes, "
          f"{CODE_TRIES} wrong tries lock pairing until you restart this)")
    if svc.https:
        print(f"certificate: {svc.fingerprint[:16]}  (the menu shows it beside the question for the code: "
              "pair only when the two match)")
    print(f"control service on port {svc.port} ({'https' if svc.https else 'http'}); "
          "Ctrl+C to stop sharing")
    if not svc.https:
        print("--no-tls: the pairing, the token and every control call travel in clear on this network")
    shown, was_locked = svc.code, False
    try:
        while not svc.stopped:
            time.sleep(1)
            svc._ensure_code()
            if svc.locked and not was_locked:
                print("pairing locked: someone entered a wrong code 5 times; restart ./larder share for a new code")
                was_locked = True
            elif not svc.locked and svc.code != shown:
                print(f"new pairing code: {format_code(svc.code)}  (the last one expired)")
                shown = svc.code
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()


if __name__ == "__main__":
    main()

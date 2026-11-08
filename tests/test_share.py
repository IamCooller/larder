#!/usr/bin/env python3
"""Checks the share control service without a real network: the pairing code lifecycle and the
UDP beacon, every packet on loopback."""
import hashlib
import hmac
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import devices
import share

# Fewer PBKDF2 rounds keep the loopback tests fast; both sides read the same constant.
share.PBKDF2_ITER = 1000


def check_code_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        code = svc.code
        assert len(code) == 12 and set(code) <= set(share.CODE_ALPHABET), code
        assert len(share.CODE_ALPHABET) == 32 and not set("ILOU") & set(share.CODE_ALPHABET)
        shown = share.format_code(code)
        assert shown.count("-") == 2 and all(len(g) == 4 for g in shown.split("-")), shown
        assert share.normalize_code(shown.lower()) == code, "entry ignores case and the dashes"
        assert share.normalize_code(" ".join(shown.split("-"))) == code, "and spaces"
        assert share.normalize_code("oil0-1234-abcd") == "01101234ABCD", "O, I and L read as the digits"
        hello = svc.hello()
        assert hello["name"] == "demo-device" and hello["code_ok"] is True, hello
        for _ in range(share.CODE_TRIES - 1):
            svc.note_wrong()
        assert svc.code == code, "fewer than the allowed wrong tries keep the code"
        svc.note_wrong()                     # the fifth wrong try locks pairing until sharing restarts
        assert svc.locked and svc.code == "" and svc.pair_start()[0] == 403
        svc.stop()
        fresh = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        fresh._rotate_code()
        first = fresh.code
        fresh.code_expires_at = time.time() - 1
        fresh._ensure_code()                 # an expired code rotates while pairing is open
        assert fresh.code != first and fresh.hello()["code_ok"] is True
        fresh.sessions = {str(i): {"t": time.time()} for i in range(share.MAX_SESSIONS)}
        assert fresh.pair_start()[0] == 429, "unfinished pairings are capped"
    print("1. the pairing code is 12 characters (60 bits) shown in groups of four, read in any case; it "
          "survives four wrong tries, the fifth locks pairing until sharing restarts; an expired code "
          "rotates; unfinished pairings are capped")


def check_beacon():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        payload = svc._beacon_payload()
        assert json.loads(payload) == {"magic": share.BEACON_MAGIC, "name": "demo-device",
                                       "port": svc.port, "https": False, "v": 1}
        result = []

        def listen():
            result.append(devices.beacons(timeout=1.2))

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        time.sleep(0.2)                      # let the listener bind before the packet is sent
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(payload, ("127.0.0.1", share.BEACON_PORT))
        s.close()
        t.join(3.0)
        assert result and result[0] == [{"name": "demo-device", "host": "127.0.0.1",
                                         "port": svc.port, "https": False}], result
        svc.stop()
    print("2. the beacon carries the name and control port, and devices.beacons picks it up")


def check_pairing():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        svc.start()
        old_mask = os.umask(0)                      # a wide umask must not widen the files below
        try:
            typed = share.format_code(svc.code).lower().replace("-", " ")
            token, fingerprint = share.pair("127.0.0.1", svc.port, typed)
            assert token and token in svc.tokens, "the device stores the token it issued"
            assert fingerprint == "", "a plain-HTTP pairing pins no certificate"
            clients = Path(tmp) / "shared-clients.json"
            assert clients.stat().st_mode & 0o777 == 0o600, oct(clients.stat().st_mode)
            share.save_pairing(tmp, "127.0.0.1", svc.port, "demo-device", token)
            assert share.token_for("127.0.0.1", svc.port, cache=tmp) == token
            assert share.pairings_path(tmp).stat().st_mode & 0o777 == 0o600
            # a file an older version left readable by others is narrowed before anything is written to it
            share.pairings_path(tmp).chmod(0o644)
            share.save_pairing(tmp, "127.0.0.1", svc.port, "demo-device", token)
            assert share.pairings_path(tmp).stat().st_mode & 0o777 == 0o600
        finally:
            os.umask(old_mask)
            svc.stop()
    print("3. pairing proves the code (typed in lower case, spaced) both ways and issues a token stored "
          "on both sides, each file created readable by this user only")


def check_wrong_code():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        svc.start()
        try:
            code = svc.code
            wrong = "0" * 12 if code != "0" * 12 else "1" * 12
            for _ in range(share.CODE_TRIES):
                try:
                    share.pair("127.0.0.1", svc.port, wrong)
                    raise AssertionError("a wrong code must be refused")
                except RuntimeError as err:
                    assert str(err) == "wrong code", err
            assert svc.locked, "five wrong tries lock pairing"
            try:
                share.pair("127.0.0.1", svc.port, code)
                raise AssertionError("a locked device must refuse even the right code")
            except RuntimeError as err:
                assert "locked" in str(err), err
        finally:
            svc.stop()
    print("4. a wrong code is refused, and five of them lock pairing, even for the right code")


def check_token_auth():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        svc.start()
        try:
            token, _ = share.pair("127.0.0.1", svc.port, svc.code)
            st, err = svc.check_auth({}, "GET", "/info", b"")
            assert st == 403 and err["error"] == "missing auth"
            st, err = svc.check_auth(share.sign("wrong-token", "GET", "/info"), "GET", "/info", b"")
            assert st == 403 and err["error"] == "bad token"
            st, err = svc.check_auth(share.sign(token, "GET", "/info"), "GET", "/info", b"")
            assert st == 200 and err is None
            headers = share.sign(token, "POST", "/start", b'{"id": "a"}')
            st, err = svc.check_auth(headers, "POST", "/start", b'{"id": "b"}')
            assert st == 403 and err["error"] == "bad token", "the HMAC covers the body"
        finally:
            svc.stop()
    print("5. calls need a paired token, and the HMAC covers the method, path and body")


def check_replay():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        svc.start()
        try:
            token, _ = share.pair("127.0.0.1", svc.port, svc.code)
            headers = share.sign(token, "GET", "/info")
            assert svc.check_auth(headers, "GET", "/info", b"")[0] == 200
            st, err = svc.check_auth(headers, "GET", "/info", b"")
            assert st == 403 and err["error"] == "replay", "the same nonce is rejected"
            old = share.sign(token, "GET", "/info", ts=int(time.time()) - share.REPLAY_WINDOW - 5)
            st, err = svc.check_auth(old, "GET", "/info", b"")
            assert st == 403 and err["error"] == "replay or clock skew"
        finally:
            svc.stop()
    print("6. a reused nonce or a timestamp outside the window is a replay")


def check_control_calls():
    with tempfile.TemporaryDirectory() as tmp:
        fake_dir = Path(tmp)
        listing = {"machine": {"name": "demo"}, "engine": {"protocol": 1, "commit": "x"},
                   "models": [{"id": "m1", "name": "Demo", "variants": [
                       {"key": "published", "label": "published", "size": 1, "ok": True,
                        "ready": False, "download": 1, "disk": 1, "expected": 1,
                        "entry_id": "m1", "path": "", "own": False}]}]}
        (fake_dir / "launch.py").write_text("import json\nprint(json.dumps(" + repr(listing) + "))\n")
        old_here = share.HERE
        share.HERE = fake_dir
        try:
            svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
            svc.start()
            try:
                token, _ = share.pair("127.0.0.1", svc.port, svc.code)
                share.save_pairing(tmp, "127.0.0.1", svc.port, "demo-device", token)
                st, info = share.request("127.0.0.1", svc.port, token, "GET", "/info", cache=tmp)
                assert st == 200 and isinstance(info["machine"], dict) and "engine" in info, info
                st, data = share.request("127.0.0.1", svc.port, token, "GET", "/models", cache=tmp)
                assert st == 200 and data["models"][0]["id"] == "m1", data
                st, err = share.request("127.0.0.1", svc.port, "wrong", "GET", "/info", cache=tmp)
                assert st == 403 and err["error"] == "bad token", err
            finally:
                svc.stop()
        finally:
            share.HERE = old_here
    print("7. /info and /models answer signed calls, /models reads the launcher, and a wrong token is refused")


def check_api_key_guard():
    import io
    import server as srv

    class Fake:
        api_key = "sekret"

        def __init__(self, headers):
            self.headers = headers
            self.wfile = io.BytesIO()
            self.status = None
            self.sent = []

        def send_response(self, code):
            self.status = code

        def send_header(self, key, value):
            self.sent.append((key, value))

        def end_headers(self):
            pass

    assert srv.Handler._check_key(Fake({"Authorization": "Bearer sekret"})) is True
    assert srv.Handler._check_key(Fake({"X-Api-Key": "sekret"})) is True
    bad = Fake({"Authorization": "Bearer nope"})
    assert srv.Handler._check_key(bad) is False
    assert bad.status == 401 and ("X-Moe-Api-Key", "required") in bad.sent
    assert "missing or wrong api key" in bad.wfile.getvalue().decode()
    none = Fake({})
    assert srv.Handler._check_key(none) is False and none.status == 401
    page = Fake({})
    page.api_key = "sekret"
    assert srv.Handler._page_allowed(page) is True, "the page loads without the key so it can ask"
    print("8. server.py accepts the key as Bearer or x-api-key and marks a wrong one as required")


def check_tls():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device", tls=True)
        svc.start()
        try:
            if not svc.https:
                print("9. (skipped: no openssl on this computer)")
                return
            assert len(svc.fingerprint) == 64 and set(svc.fingerprint) <= set("0123456789abcdef")
            assert (Path(tmp) / "share-key.pem").stat().st_mode & 0o077 == 0, "the private key is owner-only"
            token, fingerprint = share.pair("127.0.0.1", svc.port, svc.code, https=True)
            assert token and fingerprint == svc.fingerprint
            # the menu reads the certificate first and shows it, then pairs against exactly that one
            assert share.fetch_fingerprint("127.0.0.1", svc.port) == svc.fingerprint
            try:
                share.pair("127.0.0.1", svc.port, svc.code, https=True, fingerprint="0" * 64)
                raise AssertionError("a pairing pinned to another certificate went through")
            except RuntimeError as err:
                assert "fingerprint mismatch" in str(err), err
            assert svc.wrong_tries == 0, "a mismatched certificate never reaches the code check"
            share.save_pairing(tmp, "127.0.0.1", svc.port, "demo-device", token,
                               https=True, fingerprint=fingerprint)
            st, info = share.request("127.0.0.1", svc.port, token, "GET", "/info", cache=tmp)
            assert st == 200 and isinstance(info["machine"], dict), info
            row = share.load_pairings(tmp)
            row[f"127.0.0.1:{svc.port}"]["fingerprint"] = "0" * 64
            share.pairings_path(tmp).write_text(json.dumps(row))
            st, err = share.request("127.0.0.1", svc.port, token, "GET", "/info", cache=tmp)
            assert st == 502 and err["error"] == "TLS fingerprint mismatch", err
            # A relay with its own certificate: the right code fails unless the device's own fingerprint is inside the proof.
            for seen in (b"f" * 64, b""):
                _, start = svc.pair_start()
                key = share.derive_key(svc.code, share._unb64(start["salt"]))
                relayed = hmac.new(key, b"client|" + share._unb64(start["challenge"]) + seen,
                                   hashlib.sha256).digest()
                st, err = svc.pair_answer({"session": start["session"], "challenge": share._b64(b"c" * 16),
                                           "proof": share._b64(relayed)})
                assert st == 403, (seen, st, err)
                svc.wrong_tries = 0              # these two are the test's own tries, not a lockout
        finally:
            svc.stop()
        assert svc.stopped, "stopping the service tells the command to exit"
    print("9. https pins the certificate fingerprint, refuses a different one, binds it into the "
          "pairing proofs, and a stopped service ends the command")


def check_body_cap():
    with tempfile.TemporaryDirectory() as tmp:
        svc = share.ShareService(host="127.0.0.1", port=0, cache=tmp, name="demo-device")
        svc.start()
        try:
            for length in (str(10 ** 9), "abc", "-1"):
                c = http.client.HTTPConnection("127.0.0.1", svc.port, timeout=10)
                c.putrequest("POST", "/pair/answer")
                c.putheader("Content-Length", length)
                c.endheaders()                       # no body follows: the cap must answer before reading
                r = c.getresponse()
                assert r.status == 413, (length, r.status)
                c.close()
            assert share._http("127.0.0.1", svc.port, "GET", "/")[0] == 200, "the service still answers"
        finally:
            svc.stop()
    print("10. a control call's body over MAX_BODY or a bad Content-Length gets a 413 before anything is read")


def check_proxy_guard():
    seen = []

    class Target(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    threading.Thread(target=target.serve_forever, daemon=True).start()
    proxy = share.start_proxy(0, "127.0.0.1", target.server_address[1], "sekret")
    try:
        def status(headers):
            c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
            c.request("GET", "/v1/models", headers=headers)
            code = c.getresponse().status
            c.close()
            return code

        assert status({}) == 200 and seen == ["Bearer sekret"], seen
        assert status({"Origin": f"http://127.0.0.1:{proxy.port}"}) == 200 and len(seen) == 2
        assert status({"Origin": "http://evil.example"}) == 403, "a foreign page may not use the key"
        assert status({"Host": "evil.example"}) == 403, "a DNS-rebinding page may not use the key"
        assert len(seen) == 2, "refused requests never reach the device"
    finally:
        proxy.stop()
        target.shutdown()
        target.server_close()
    print("11. the share proxy refuses a foreign Origin or Host as the server does, before adding the key")


def main():
    check_code_lifecycle()
    check_beacon()
    check_pairing()
    check_wrong_code()
    check_token_auth()
    check_replay()
    check_control_calls()
    check_api_key_guard()
    check_tls()
    check_body_cap()
    check_proxy_guard()
    print("test_share: ok")


if __name__ == "__main__":
    main()

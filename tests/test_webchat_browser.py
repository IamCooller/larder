#!/usr/bin/env python3
"""The chat page in headless Chrome, against the stub engine (no model): the
Thinking switch and its collapsed block, images by button, paste and drop, the
the live indicator and the Status link, conversations in the sidebar (rename, delete, reload), the theme
switch, code highlighting and its copy button, the stop button, light and dark,
phone and desktop width, and no console errors.

    pip install playwright          # drives the Google Chrome that is installed
    STUB_PYTHON=python3 python tests/test_webchat_browser.py [--shots DIR]

The stub servers (tests/test_webchat.py --serve) are started with STUB_PYTHON,
which has to be a Python that can import server.py; this script's own Python
only needs playwright. --shots DIR saves a PNG of each step.
"""

import base64
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError, sync_playwright

HERE = Path(__file__).resolve().parent
STUB_PYTHON = os.environ.get("STUB_PYTHON", sys.executable)
PHONE = dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
DESKTOP = dict(viewport={"width": 1100, "height": 760})
NO_VISION = "this model has no vision part; images are not supported"


def png(w=96, h=96, tint=(200, 80, 60)):
    """A small gradient PNG, so the thumbnails in the screenshots are pictures."""
    raw = b"".join(b"\0" + b"".join(bytes(((tint[0] + x) % 256, (tint[1] + y * 2) % 256, tint[2]))
                                    for x in range(w)) for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def start_stubs():
    """Four stubs together: one that cannot see, one that can, one that answers
    with a fenced code block, and one that is slow to start and thinks for
    seconds, for the animations."""
    procs = [subprocess.Popen([STUB_PYTHON, str(HERE / "test_webchat.py"), "--serve", *flags],
                              stdout=subprocess.PIPE, text=True)
             for flags in ([], ["--vision"], ["--code"], ["--first-delay", "3.0", "--long-think"])]
    ports = []
    for p in procs:
        line = p.stdout.readline()
        assert line.startswith("PORT"), f"the stub did not start: {line!r}"
        ports.append(int(line.split()[1]))
    return procs, ports


def rows(page, sel):
    return page.evaluate("""s => Object.fromEntries([...document.querySelectorAll(s + ' dt')]
        .map(t => [t.textContent, t.nextElementSibling.textContent]))""", sel)


def until(page, js, timeout=30):
    """Poll a JS expression until it is true. page.wait_for_function would eval
    it in the page, which the page's own CSP (no unsafe-eval) refuses, and the
    test keeps that CSP on."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            if page.evaluate("() => " + js):
                return
        except PlaywrightError:         # an element that is not there yet, or a page that is changing
            pass
        page.wait_for_timeout(50)
    raise AssertionError(f"timed out waiting for: {js}")


def reply_done(page, count):
    until(page, f"document.querySelectorAll('.msg.assistant').length >= {count} && "
                "document.querySelector('#send').textContent === 'Send'")


def send(page, text, count, enter=False):
    page.fill("#box", text)
    page.press("#box", "Enter") if enter else page.click("#send")
    reply_done(page, count)


def anim(page, selector):
    """The CSS animation a live element runs, or 'none'."""
    return page.evaluate("s => getComputedStyle(document.querySelector(s)).animationName", selector)


def drop_files(page, kind, data):
    """Dispatch a paste or a drop carrying one PNG, as a browser would."""
    return page.evaluate("""([kind, b64]) => {
        const dt = new DataTransfer();
        dt.items.add(new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], "x.png", {type: "image/png"}));
        const ev = kind === "paste"
            ? new ClipboardEvent("paste", {clipboardData: dt, bubbles: true, cancelable: true})
            : new DragEvent(kind, {dataTransfer: dt, bubbles: true, cancelable: true});
        document.dispatchEvent(ev);
        return ev.defaultPrevented;
    }""", [kind, base64.b64encode(data).decode()])


def open_drawer(page, desktop):
    if desktop:
        return
    page.click("#menuBtn")
    until(page, "document.body.classList.contains('drawer-open')")


def close_drawer(page, desktop):
    if desktop:
        return
    page.click("#closeSidebar")
    until(page, "!document.body.classList.contains('drawer-open')")

def run_context(browser, scheme, size, name, shots):
    procs, (plain, seeing, code, slow) = start_stubs()
    ctx = browser.new_context(color_scheme=scheme, permissions=["clipboard-read", "clipboard-write"], **size)
    page = ctx.new_page()
    errors, bodies = [], []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: bodies.append(json.loads(r.post_data))
            if r.url.endswith("/v1/chat/completions") else None)

    def shot(label):
        if shots:
            page.wait_for_timeout(250)                       # let the color transitions settle
            page.screenshot(path=str(shots / f"{name}-{label}.png"))

    desktop = size is DESKTOP       # a computer sends with Enter, a touch screen with the button

    def fits():
        return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    def bg():
        return page.evaluate("getComputedStyle(document.body).backgroundColor")

    try:
        # -- empty state, sidebar, theme ------------------------------------------
        page.goto(f"http://127.0.0.1:{plain}/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        assert bg() == ("rgb(20, 20, 22)" if scheme == "dark" else "rgb(255, 255, 255)"), bg()
        assert page.is_disabled("#attach") and not page.is_checked("#think") and page.is_hidden("#thumbs")
        assert page.inner_text("#liveMain") in ("-", "ready", "offline") or "tok/s" in page.inner_text("#liveMain")
        # the icons the page names exist (the empty data: icon is gone)
        for sel in ("link[rel=icon]", "link[rel=apple-touch-icon]"):
            href = page.get_attribute(sel, "href")
            assert href.startswith("/") and page.evaluate("h => fetch(h).then(r => r.ok)", href), href
        assert "no vision part" in page.get_attribute("#attach", "title")
        assert page.locator(".empty h2").inner_text() == "How can I help?"
        assert fits()
        shot("empty")

        if desktop:
            assert page.is_visible("#sidebar") and page.is_hidden("#menuBtn")
            shot("sidebar")
        else:
            assert page.is_visible("#menuBtn") and not page.evaluate(
                "document.body.classList.contains('drawer-open')")
            open_drawer(page, desktop)
            shot("sidebar")
            close_drawer(page, desktop)

        # -- thinking: live, collapsed, open -------------------------------------
        page.click("label:has(#think)")
        page.reload()
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        assert page.is_checked("#think"), "the Thinking switch is not remembered"
        page.fill("#box", "What is the answer?")
        page.click("#send")
        live = page.locator(".msg.assistant .think.is-thinking")
        live.wait_for(state="visible")                       # reasoning has started, no answer yet
        assert live.locator("summary").inner_text().startswith("Thinking") and not live.evaluate("e => e.open")
        shot("thinking-live")
        reply_done(page, 1)
        want = {"enable_thinking": True}
        assert bodies[-1]["chat_template_kwargs"] == want and bodies[-1]["max_tokens"] == 8192, bodies[-1]
        last = page.locator(".msg.assistant").last
        think = last.locator("details.think")
        assert think.is_visible() and not think.evaluate("e => e.open") and "is-thinking" not in think.get_attribute("class")
        assert last.locator(".body").inner_text() == "It is 42." and last.locator(".body strong").inner_text() == "42"
        meta = last.locator(".meta").inner_text()
        assert re.search(r"first token \d+\.\d s", meta) and re.search(r"\d+\.\d tok/s", meta), meta
        assert meta.endswith("tokens"), meta
        shot("thinking-collapsed")
        think.locator("summary").click()
        assert think.locator(".tbody").inner_text().startswith("Let me see.")
        shot("thinking-open")

        page.click("label:has(#think)")                      # off again
        send(page, "And without thinking?", 2, desktop)
        assert bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False} and bodies[-1]["max_tokens"] == 2048
        assert page.locator(".msg.assistant").last.locator("details.think").is_hidden()
        assert page.locator(".msg.assistant").last.locator(".body").inner_text() == "Hello world from a fake"
        # the first answer's thinking stayed open while the second reply was added
        assert page.locator(".msg.assistant").first.locator("details.think").evaluate("e => e.open")
        assert [m["content"] for m in bodies[-1]["messages"]] == [
            "What is the answer?", "It is **42**.", "And without thinking?"], bodies[-1]["messages"]

        # -- the live indicator in the header and the way to the Status page ------
        until(page, "/tok\\/s/.test(document.querySelector('#liveMain').textContent)")
        pill = page.locator("#liveBtn")
        assert pill.get_attribute("href") == "/monitor"
        assert re.fullmatch(r"\d+\.\d tok/s", page.inner_text("#liveMain")), page.inner_text("#liveMain")
        if page.is_visible("#liveSub"):
            assert re.fullmatch(r"\d+\.\d+ GB", page.inner_text("#liveSub")), page.inner_text("#liveSub")
        assert "Open the Status page" in pill.get_attribute("aria-label")
        assert not page.evaluate("document.querySelector('#liveBtn').classList.contains('off')")
        assert fits()
        shot("status")
        send(page, "One more", 3)                            # the speed follows the reply
        page.reload()
        until(page, "/tok\\/s/.test(document.querySelector('#liveMain').textContent)")
        page.click("#liveBtn")
        until(page, "location.pathname === '/monitor'")
        until(page, "document.querySelector('#tReplies, #tSpeedSub') && document.querySelector('#modelStats dd')")
        assert page.title() == "Status - Local chat"
        assert rows(page, "#chatStats")["Messages"] == "6" and rows(page, "#chatStats")["Prompt tokens"] == "24"
        assert rows(page, "#chatStats")["Context now"] == "13 of 8.2k (0.2%)", rows(page, "#chatStats")
        page.click("#backLink")
        until(page, "location.pathname === '/'")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        assert page.locator(".msg.assistant").count() == 3, "the chat is as it was"

        # -- theme: manual light/dark, then back to the system setting -------------
        def theme_color():
            return page.get_attribute("meta[name=theme-color]", "content")
        page.click("#themeBtn")
        assert page.get_attribute("#themeBtn", "aria-expanded") == "true"
        page.click("[data-theme-choice='light']")
        assert bg() == "rgb(255, 255, 255)", bg()
        assert theme_color() == "#ffffff", theme_color()
        page.click("#themeBtn")
        page.click("[data-theme-choice='dark']")
        assert bg() == "rgb(20, 20, 22)", bg()
        assert theme_color() == "#141416", theme_color()
        page.click("#themeBtn")
        page.click("[data-theme-choice='system']")
        assert bg() == ("rgb(20, 20, 22)" if scheme == "dark" else "rgb(255, 255, 255)"), bg()

        # -- no vision part: the attach button is off, a drop says why ------------
        assert drop_files(page, "drop", png())
        assert "no vision part" in page.inner_text("#note") and page.is_hidden("#thumbs")
        shot("no-vision")

        # -- the same page told the model can see: the server's 400 shows ---------
        def claim_vision(route):
            r = route.fetch()
            route.fulfill(response=r, json=dict(r.json(), vision=True))
        page.route("**/status", claim_vision)
        page.reload()
        until(page, "!document.querySelector('#attach').disabled")
        page.set_input_files("#file", dict(name="a.png", mimeType="image/png", buffer=png()))
        page.wait_for_selector("#thumbs .thumb")
        n_errors = len(errors)
        page.fill("#box", "look")
        page.click("#send")
        reply_done(page, 4)
        assert page.locator(".msg.assistant").last.locator(".err .err-text").inner_text() == NO_VISION
        errors[n_errors:] = [e for e in errors[n_errors:] if "status of 400" not in e]   # Chrome logs the 400
        shot("error")
        page.unroute("**/status")

        # -- a model that can see --------------------------------------------------
        page.goto(f"http://127.0.0.1:{seeing}/")
        until(page, "!document.querySelector('#attach').disabled")
        assert page.get_attribute("#attach", "title") == "Attach images"
        page.set_input_files("#file", dict(name="a.png", mimeType="image/png", buffer=png(tint=(40, 120, 220))))
        until(page, "document.querySelectorAll('#thumbs .thumb').length === 1")
        assert drop_files(page, "paste", png(tint=(220, 160, 30))), "an image paste must be taken over"
        until(page, "document.querySelectorAll('#thumbs .thumb').length === 2")
        drop_files(page, "dragenter", png())
        assert page.is_visible("#drop")
        shot("drop")
        drop_files(page, "drop", png(tint=(60, 200, 90)))
        until(page, "document.querySelectorAll('#thumbs .thumb').length === 3")
        assert page.is_hidden("#drop")
        page.set_input_files("#file", dict(name="d.png", mimeType="image/png", buffer=png(tint=(120, 40, 160))))
        until(page, "document.querySelectorAll('#thumbs .thumb').length === 4")
        assert fits()
        shot("images")
        page.set_input_files("#file", dict(name="e.png", mimeType="image/png", buffer=png()))
        assert "At most 4" in page.inner_text("#note")
        page.set_input_files("#file", dict(name="t.txt", mimeType="text/plain", buffer=b"hi"))
        assert "Only PNG, JPEG, WebP and GIF" in page.inner_text("#note")
        with tempfile.TemporaryDirectory() as tmp:       # by path: a buffer of this size crawls
            big = Path(tmp) / "big.png"
            big.write_bytes(b"\0" * (11 * 2**20))
            page.set_input_files("#file", str(big))
        assert "big.png is larger than 10 MB" in page.inner_text("#note")
        assert page.locator("#thumbs .thumb").count() == 4
        assert fits()
        shot("images-limit")
        page.locator("#thumbs .thumb button").nth(0).click()
        page.locator("#thumbs .thumb button").nth(0).click()
        assert page.locator("#thumbs .thumb").count() == 2 and fits()
        send(page, "what is this", 1)
        parts = bodies[-1]["messages"][0]["content"]
        assert [p["type"] for p in parts] == ["image_url", "image_url", "text"], parts
        assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts[:2])
        assert parts[2]["text"] == "what is this" and page.is_hidden("#thumbs")
        assert page.locator(".msg.user img").count() == 2
        shot("images-sent")
        page.reload()                                        # pictures are not kept, the message is
        page.wait_for_selector(".msg.user .kept")
        assert "2 images were attached" in page.inner_text(".msg.user .kept")
        send(page, "and now?", 2)
        assert bodies[-1]["messages"][0]["content"] == "what is this", "a reloaded message goes as text"
        assert fits()

        # -- a reply with code: highlight, copy, and the sidebar -------------------
        page.goto(f"http://127.0.0.1:{code}/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        send(page, "Show a loop", 1, desktop)
        assert page.locator(".code").count() == 1
        assert page.locator(".code .lang").first.inner_text() == "python"
        assert page.locator(".code code.hljs").count() == 1
        assert "hljs-keyword" in page.locator(".code code").first.inner_html()
        assert fits()
        shot("code")
        page.locator(".code .copy").first.click()
        until(page, "document.querySelector('.code .copy').textContent === 'Copied'")
        assert "for i in range(3)" in page.evaluate("navigator.clipboard.readText()")

        open_drawer(page, desktop)
        page.locator(".conv-item.active [aria-label='Rename conversation']").click()
        page.fill(".conv-title-input", "Renamed chat")
        page.press(".conv-title-input", "Enter")
        until(page, "document.querySelector('.conv-item.active .conv-open').textContent === 'Renamed chat'")
        page.click("#new")
        until(page, "document.querySelectorAll('.conv-item').length === 2")
        assert page.locator(".empty").is_visible()
        send(page, "A second conversation", 1, desktop)
        open_drawer(page, desktop)
        assert "A second conversation" in page.locator("#convList").inner_text()
        assert page.locator(".conv-item.active .conv-open").inner_text() == "A second conversation"
        shot("conversations")
        page.locator(".conv-item.active [aria-label='Delete conversation']").click()
        page.wait_for_selector("#confirmDialog[open]")
        page.click("#confirmOk")
        until(page, "document.querySelectorAll('.conv-item').length === 1")
        assert page.locator(".conv-item.active .conv-open").inner_text() == "Renamed chat"
        assert page.locator(".code").count() == 1
        page.reload()
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        assert page.locator(".conv-item.active .conv-open").inner_text() == "Renamed chat"
        assert page.locator(".code").count() == 1

        # -- the animations: waiting, thinking, typing -----------------------------
        page.goto(f"http://127.0.0.1:{slow}/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        page.click("label:has(#think)")
        page.fill("#box", "Think about it")
        page.click("#send")
        wait = page.locator(".msg.assistant.working .wait")
        wait.wait_for(state="visible")
        assert "Waiting for the first token" in wait.inner_text()
        assert page.locator(".msg.assistant .body").inner_text() == ""
        assert anim(page, ".wait .dots i") == "wave"
        until(page, "/ 2 s/.test(document.querySelector('.wait-time').textContent)")
        shot("waiting")
        live = page.locator(".msg.assistant .think.is-thinking")
        live.wait_for(state="visible")
        assert page.locator(".msg.working .wait").is_hidden()
        assert anim(page, ".think.is-thinking .t-dot") == "breathe"
        assert anim(page, ".think.is-thinking .t-label") == "shimmer"
        assert anim(page, ".dots i") == "wave" and not live.evaluate("e => e.open")
        until(page, "/^Thinking\\s+\\S\\s+[1-9] s$/.test(document.querySelector('.think.is-thinking summary').innerText.trim())")
        shot("thinking-live")
        until(page, "document.querySelector('.msg.working .body .caret')")
        assert page.locator(".msg.working .think.is-thinking").count() == 0, "the answer started"
        assert page.locator(".msg.working .body .caret").count() == 1
        assert anim(page, ".msg.working .body .caret") == "caret"
        shot("typing")
        reply_done(page, 1)
        assert page.locator(".caret").count() == 0 and page.locator(".wait").is_hidden()
        label = page.locator(".msg.assistant .think .t-label").inner_text()
        assert re.fullmatch(r"Thought for \d+(\.\d)? s", label), label
        assert "It is 42, and here is why" in page.locator(".msg.assistant .body").inner_text()
        shot("thought")
        page.reload()                                        # the time spent thinking is kept
        until(page, "document.querySelector('.msg.assistant .t-label')")
        assert page.locator(".msg.assistant .think .t-label").inner_text() == label

        # -- stop a streaming reply ------------------------------------------------
        page.goto(f"http://127.0.0.1:{plain}/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        open_drawer(page, desktop)
        page.click("#new")
        page.fill("#box", "Please stop")
        page.click("#send")
        until(page, "document.querySelector('#send').textContent === 'Stop'")
        page.wait_for_timeout(800)
        page.click("#send")
        until(page, "document.querySelector('#send').textContent === 'Send'")
        assert "stopped" in page.locator(".msg.assistant").last.locator(".meta").inner_text()
        assert fits()
        assert not errors, errors
    finally:
        ctx.close()
        for p in procs:
            p.terminate()
    print(f"  {name}: thinking switch and block, live indicator, images (button, paste, drop), "
          "conversations (rename, delete, reload), theme, code copy, stop, no console errors")


def check_reduced_motion(browser):
    """With prefers-reduced-motion the cues stay but nothing moves."""
    procs, ports = start_stubs()
    ctx = browser.new_context(reduced_motion="reduce", **DESKTOP)
    page = ctx.new_page()
    try:
        page.goto(f"http://127.0.0.1:{ports[3]}/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        page.click("label:has(#think)")
        page.fill("#box", "Think about it")
        page.click("#send")
        page.locator(".msg.assistant.working .wait").wait_for(state="visible")
        assert anim(page, ".wait .dots i") == "none"
        assert "Waiting for the first token" in page.inner_text(".wait")
        page.locator(".msg.assistant .think.is-thinking").wait_for(state="visible")
        for sel in (".think.is-thinking .t-dot", ".think.is-thinking .t-label"):
            assert anim(page, sel) == "none", sel
        until(page, "document.querySelector('.msg.working .body .caret')")
        assert anim(page, ".msg.working .body .caret") == "none"
        assert page.evaluate("getComputedStyle(document.querySelector('.caret')).opacity") == "1"
        reply_done(page, 1)
    finally:
        ctx.close()
        for p in procs:
            p.terminate()
    print("  reduced motion: the waiting dots, the thinking cue and the caret are still, the words stay")


def main():
    shots = None
    if "--shots" in sys.argv:
        shots = Path(sys.argv[sys.argv.index("--shots") + 1])
        shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        try:
            for scheme in ("light", "dark"):
                for label, size in (("phone", PHONE), ("desktop", DESKTOP)):
                    run_context(browser, scheme, size, f"{label}-{scheme}", shots)
            check_reduced_motion(browser)
        finally:
            browser.close()
    print("ok")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The chat's context bar and what hangs off it, in headless Chrome against the
stub engine (no model): how full the context is and the warning past 80 percent,
Compact (the model summarises, a marker and the folded messages stay), Clear,
Regenerate, Edit and resend, the welcome screen's example prompts, the keyboard
shortcuts (Cmd/Ctrl+K, Esc), and the model and computer saved with each chat.

    pip install playwright
    STUB_PYTHON=python3 python3 tests/test_chat_features_browser.py [--shots DIR]

The stub is tests/test_webchat.py --serve --grow --ctx 300 --fast: its prompt
tokens grow with the conversation (two a word) and its model reads 300 tokens,
so the bar moves by whole numbers a test can state.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_webchat_browser import DESKTOP, PHONE, reply_done, until          # noqa: E402

STUB_PYTHON = os.environ.get("STUB_PYTHON", sys.executable)
REPLY = "Hello world from a fake"           # as shown
RAW = "Hello **world** from a fake"         # as sent back


def start_stub(*flags):
    p = subprocess.Popen([STUB_PYTHON, str(HERE / "test_webchat.py"), "--serve", *flags],
                         stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.startswith("PORT"), f"the stub did not start: {line!r}"
    return p, int(line.split()[1])


def words(n, word="word"):
    return " ".join([word] * n)


def run_context(browser, scheme, size, name, shots):
    stub, port = start_stub("--grow", "--ctx", "300", "--fast")
    slow, slow_port = start_stub("--first-delay", "3.0", "--ctx", "300")
    ctx = browser.new_context(color_scheme=scheme, permissions=["clipboard-read", "clipboard-write"], **size)
    page = ctx.new_page()
    errors, bodies = [], []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: bodies.append(json.loads(r.post_data))
            if r.url.endswith("/v1/chat/completions") else None)
    desktop = size is DESKTOP

    def shot(label):
        if shots:
            page.wait_for_timeout(250)
            page.screenshot(path=str(shots / f"{name}-{label}.png"))

    def fits():
        return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    def ctx_text():
        return page.inner_text("#ctxText")

    def send(text, count):
        page.fill("#box", text)
        page.click("#send")
        reply_done(page, count)

    def stored_chats():
        return page.evaluate("JSON.parse(localStorage.getItem('chat.v2')).conversations")

    try:
        # -- the welcome screen and its example prompts ----------------------------------
        page.goto(f"http://127.0.0.1:{port}/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        assert page.locator(".empty h2").inner_text() == "How can I help?"
        until(page, "document.querySelector('.empty-sub').textContent.includes('tiny-chat on test-host')")
        assert page.locator(".example").count() == 4
        assert page.locator(".ex-title").all_inner_texts() == ["Explain", "Write code", "Compare", "Plan"]
        if desktop:
            assert page.locator(".hints").is_visible() and page.locator(".hints kbd").count() == 4
        assert ctx_text() == "0 / 300 · 0%", ctx_text()
        assert page.is_disabled("#compactBtn") and page.is_disabled("#clearBtn")
        assert fits()
        shot("welcome")
        first_example = page.locator(".example").first.locator(".ex-text").inner_text()
        page.locator(".example").first.click()
        reply_done(page, 1)
        assert bodies[-1]["messages"][0]["content"] == first_example, bodies[-1]
        assert page.locator(".msg.user .user-text").inner_text() == first_example
        # the chat is saved with the model and the computer that answered it
        chat = stored_chats()[0]
        assert chat["model"] == "tiny-chat" and chat["host"] == "test-host", chat
        assert page.locator(".conv-item.active .conv-sub").inner_text() == "tiny-chat"
        assert page.get_attribute("#footLink", "href") == "/settings#sec-chats"

        # -- how full the context is -----------------------------------------------------
        page.keyboard.press("Control+k")                       # a new chat, kept apart from the first
        until(page, "document.querySelectorAll('.conv-item').length === 2")
        assert page.locator(".empty").is_visible() and ctx_text() == "0 / 300 · 0%"
        if desktop:                                             # Shift+Enter is a new line, Enter alone sends
            page.focus("#box")
            page.keyboard.type("line one")
            page.keyboard.press("Shift+Enter")
            page.keyboard.type("line two")
            assert page.input_value("#box") == "line one\nline two" and page.locator(".msg").count() == 0
            page.fill("#box", "")
        send(words(20), 1)                                      # 5 + 2 * 20 prompt tokens, 3 cached, 5 written
        assert ctx_text() == "53 / 300 · 18%", ctx_text()
        assert page.get_attribute("#ctxBar", "aria-valuenow") == "18"
        assert "53 of 300 tokens" in page.get_attribute("#ctx", "title")
        assert not page.evaluate("document.querySelector('#ctx').classList.contains('warn')")
        assert page.is_disabled("#compactBtn"), "one exchange is not enough to compact"
        assert "Nothing to compact yet" in page.get_attribute("#compactBtn", "title")
        calm = page.evaluate("getComputedStyle(document.querySelector('#ctxFill')).backgroundColor")
        send(words(60), 2)
        assert ctx_text() == "183 / 300 · 61%", ctx_text()
        fill_width = page.evaluate("document.querySelector('#ctxFill').style.width")
        assert fill_width == "61%", fill_width
        shot("context")
        assert page.is_enabled("#compactBtn"), "two exchanges can be compacted"
        send(words(40), 3)                                      # past 80 percent: the bar turns amber
        assert ctx_text() == "273 / 300 · 91%", ctx_text()
        assert page.evaluate("document.querySelector('#ctx').classList.contains('warn')")
        assert "The context is 91% full" in page.inner_text("#note")
        page.wait_for_timeout(400)                              # the colour fades in
        amber = page.evaluate("getComputedStyle(document.querySelector('#ctxFill')).backgroundColor")
        assert amber != calm, (amber, calm)
        assert page.evaluate("document.querySelector('#compactBtn').classList.contains('attention')")
        assert fits()
        shot("context-warning")

        # -- Compact: the model summarises, a marker and the folded messages stay --------
        page.click("#compactBtn")
        card = page.locator(".msg.compacting")
        card.wait_for(state="visible")
        assert "Summarising 4 earlier messages" in card.text_content()
        assert page.is_disabled("#compactBtn") and page.inner_text("#send") == "Stop"
        until(page, "!document.querySelector('.msg.compacting')")
        ask = bodies[-1]
        assert ask["chat_template_kwargs"] == {"enable_thinking": False} and ask["max_tokens"] == 1200, ask
        assert ask["messages"][-1]["content"].startswith("Write a summary of our conversation so far")
        assert [m["role"] for m in ask["messages"]] == ["user", "assistant", "user", "assistant", "user"], ask
        assert page.inner_text("#send") == "Send" and not page.is_disabled("#send")
        marker = page.locator(".compaction .marker .label")
        assert marker.inner_text() == "Conversation compacted · 4 messages summarised"
        archive = page.locator("details.archive")
        assert page.locator("details.archive > summary").inner_text() == "4 earlier messages, folded"
        assert not archive.evaluate("e => e.open") and page.locator("details.archive .msg").count() == 4
        assert page.locator(".archive .msg").first.is_hidden(), "the old messages are collapsed"
        summary = page.locator("details.summary-view")
        assert not summary.evaluate("e => e.open")
        assert "Compacted 4 messages" in page.inner_text("#note")
        visible = page.locator("#msgs > .msg")
        assert visible.count() == 2, "the last exchange stays word for word"
        used = ctx_text()
        assert used.startswith("~") and int(re.search(r"(\d+)%", used).group(1)) < 50, used
        assert fits()
        shot("compacted")
        page.locator("details.archive > summary").click()
        page.locator("details.summary-view > summary").click()
        assert page.locator(".archive .msg").first.is_visible()
        assert REPLY in summary.inner_text()
        assert page.locator(".archive .actions [aria-label^='Edit']").count() == 0, "folded messages are not editable"
        shot("compacted-open")
        page.locator("details.archive > summary").click()

        # the model is sent the summary and the kept messages, not the folded ones
        send("And what comes next?", 3)
        sent = bodies[-1]["messages"]
        assert sent[0]["role"] == "system" and sent[0]["content"].startswith("Summary of the earlier part")
        assert RAW in sent[0]["content"]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"], sent
        assert sent[1]["content"] == words(40) and sent[3]["content"] == "And what comes next?"
        page.reload()                                           # the marker, the folded part and the numbers survive
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        assert page.locator(".compaction .marker").count() == 1 and page.locator("details.archive .msg").count() == 4
        assert page.locator("#msgs > .msg").count() == 4
        until(page, "document.querySelector('#ctxText').textContent.length > 0")

        # -- Regenerate: the same question, a new reply -------------------------------------
        before = len(bodies)
        page.locator("#msgs > .msg.assistant").last.hover()
        page.locator("#msgs > .msg.assistant").last.locator("[aria-label='Regenerate the reply']").click()
        until(page, "document.querySelector('#send').textContent === 'Send' && document.querySelectorAll('#msgs > .msg.assistant').length === 2")
        assert len(bodies) == before + 1
        assert [m["role"] for m in bodies[-1]["messages"]] == ["system", "user", "assistant", "user"]
        assert bodies[-1]["messages"][-1]["content"] == "And what comes next?"
        assert page.locator("#msgs > .msg.assistant").count() == 2 and page.locator("#msgs > .msg").count() == 4
        # only the last reply can be regenerated
        assert page.locator("#msgs > .msg.assistant").first.locator("[aria-label='Regenerate the reply']").count() == 0

        # -- Edit and resend: the message changes and what came after it is dropped ----------
        page.locator("#msgs > .msg.user").nth(1).hover()
        page.locator("#msgs > .msg.user").nth(1).locator("[aria-label='Edit and resend']").click()
        area = page.locator(".edit-box")
        area.wait_for(state="visible")
        assert area.input_value() == "And what comes next?"
        assert "replies after this message are removed" in page.inner_text(".edit-actions")
        shot("edit")
        page.keyboard.press("Escape")                           # cancels, nothing changed
        assert page.locator(".edit-box").count() == 0 and page.locator("#msgs > .msg").count() == 4
        page.locator("#msgs > .msg.user").nth(1).hover()
        page.locator("#msgs > .msg.user").nth(1).locator("[aria-label='Edit and resend']").click()
        page.fill(".edit-box", "Say it differently")
        page.click("text=Save and resend")
        until(page, "document.querySelector('#send').textContent === 'Send' && document.querySelectorAll('#msgs > .msg.assistant').length === 2")
        assert bodies[-1]["messages"][-1]["content"] == "Say it differently"
        assert [m["content"] for m in bodies[-1]["messages"]][1:3] == [words(40), RAW]
        assert page.locator("#msgs > .msg.user").nth(1).inner_text().startswith("Say it differently")
        assert page.locator("#msgs > .msg").count() == 4

        # -- Clear: a new chat with the same settings ----------------------------------------
        page.click("label:has(#think)")
        assert page.is_checked("#think")
        chats = page.locator(".conv-item").count()
        page.click("#clearBtn")
        until(page, f"document.querySelectorAll('.conv-item').length === {chats + 1}")
        assert page.locator(".empty").is_visible() and page.is_checked("#think")
        assert ctx_text() == "0 / 300 · 0%" and page.is_disabled("#clearBtn")
        assert "Started a new chat" in page.inner_text("#note")
        page.click("label:has(#think)")
        assert fits()
        assert not errors, errors

        # -- Esc stops a reply, and a leaving link asks first ---------------------------------
        page.goto(f"http://127.0.0.1:{slow_port}/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        page.fill("#box", "Please wait")
        page.click("#send")
        page.locator(".msg.working .wait").wait_for(state="visible")
        page.click("#settingsBtn")                              # a reply is being written: ask first
        page.wait_for_selector("#confirmDialog[open]")
        assert "Leave while a reply is being written?" in page.inner_text("#confirmTitle")
        assert page.inner_text("#confirmOk") == "Leave"
        shot("leave")
        page.keyboard.press("Escape")                           # the dialog closes, the reply goes on
        until(page, "!document.querySelector('#confirmDialog').open")
        assert page.inner_text("#send") == "Stop" and page.url.endswith(f"{slow_port}/")
        page.keyboard.press("Escape")                           # Esc stops the reply
        until(page, "document.querySelector('#send').textContent === 'Send'")
        assert page.locator(".msg.assistant").count() == 0, "a reply stopped before its first token leaves nothing"
        assert "Stopped before the first token" in page.inner_text("#note")
        assert not errors, errors
    finally:
        ctx.close()
        stub.terminate()
        slow.terminate()
    print(f"  {name}: context bar and warning, Compact with marker and folded messages, "
          "Regenerate, Edit and resend, Clear, example prompts, shortcuts, chat tags")


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
                    run_context(browser, scheme, size, f"chat-{label}-{scheme}", shots)
        finally:
            browser.close()
    print("ok")


if __name__ == "__main__":
    main()

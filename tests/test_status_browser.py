#!/usr/bin/env python3
"""The Status page in headless Chrome, against the stub engine (no model): the
empty state, the tiles with units and rounding, the two per-reply charts and the
memory chart, the tooltip on hover and on arrow keys, the token bar, the machine
and model lists, the table view, the offline state, light and dark, phone and
desktop width.

    pip install playwright
    STUB_PYTHON=python3 python3 tests/test_status_browser.py [--shots DIR]

The stub is tests/test_webchat.py --serve --vary --fast: every reply has its own
speed, first-token time and a prompt that grows with the chat.
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
from test_webchat_browser import DESKTOP, PHONE, rows, until                  # noqa: E402

STUB_PYTHON = os.environ.get("STUB_PYTHON", sys.executable)
REPLIES = 9


def start_stub():
    p = subprocess.Popen([STUB_PYTHON, str(HERE / "test_webchat.py"), "--serve", "--vary", "--fast",
                          "--grow", "--ctx", "4096"], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.startswith("PORT"), f"the stub did not start: {line!r}"
    return p, int(line.split()[1])


def run_context(browser, scheme, size, name, shots):
    proc, port = start_stub()
    ctx = browser.new_context(color_scheme=scheme, **size)
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("requestfailed", lambda r: errors.append(f"{r.url} {r.failure}"))
    desktop = size is DESKTOP

    def shot(label):
        if shots:
            page.wait_for_timeout(250)
            page.screenshot(path=str(shots / f"{name}-{label}.png"), full_page=True)

    def fits():
        return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    try:
        # -- before any reply: honest empty states ----------------------------------
        page.goto(f"http://127.0.0.1:{port}/monitor")
        until(page, "document.querySelector('#modelStats dd')")
        assert page.title() == "Status - Local chat"
        until(page, "document.querySelector('#liveText').textContent.startsWith('Live')")
        assert page.inner_text("#tSpeed") == "-" and page.inner_text("#tSpeedSub") == "no replies yet"
        assert "No replies yet" in page.inner_text("#chartSpeed")
        assert page.locator("#chartSpeed svg").is_hidden() and page.locator(".lc-empty").count() == 3
        model = rows(page, "#modelStats")
        assert model["Model"] == "tiny-chat" and model["Context length"] == "4.1k tokens", model
        assert model["Images"] == "not supported" and model["Replies served"] == "0", model
        machine = rows(page, "#machineStats")
        assert machine["Computer"] == "test-host" and machine["Chip"] == "Test chip", machine
        assert machine["Memory"] == "16 GB" and machine["Engine"] == "Metal, MLX 0.0", machine
        assert rows(page, "#chatStats")["Messages"] == "0"
        assert fits()
        shot("empty")

        # -- replies, then the page follows them -----------------------------------
        for i in range(REPLIES):
            ok = page.evaluate("""n => fetch('/v1/chat/completions', {method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({messages: [{role: 'user', content: 'question number ' + n}]})}).then(r => r.ok)""", i)
            assert ok
        until(page, f"document.querySelectorAll('#chartSpeed .lc-dot').length === {REPLIES}", timeout=15)
        until(page, f"[...document.querySelectorAll('#modelStats dd')].pop().textContent === '{REPLIES}'")
        # the tiles say the units and round: 13.8 tok/s, 0.62 s, 4.21 GB
        assert re.fullmatch(r"\d+\.\d tok/s", page.inner_text("#tSpeed")), page.inner_text("#tSpeed")
        assert re.fullmatch(r"average \d+\.\d tok/s . best \d+\.\d tok/s", page.inner_text("#tSpeedSub").replace("\n", " ")), page.inner_text("#tSpeedSub")
        assert re.fullmatch(r"0\.\d s", page.inner_text("#tFirst")), page.inner_text("#tFirst")
        assert re.fullmatch(r"average 0\.\d s", page.inner_text("#tFirstSub"))
        assert re.fullmatch(r"\d+\.\d\d GB", page.inner_text("#tMem")), page.inner_text("#tMem")
        assert "no fixed budget" in page.inner_text("#tMemSub") and "16 GB computer" in page.inner_text("#tMemSub")
        assert re.fullmatch(r"\d+ s|\d+ min", page.inner_text("#tUp")), page.inner_text("#tUp")
        assert page.inner_text("#tUpSub").startswith("since ")
        assert float(page.get_attribute("#gaugeFill", "style").split("width:")[1].strip(" %;")) > 0
        # three charts, each with a line, a dot per reply and a text summary
        for host in ("#chartSpeed", "#chartFirst", "#chartMemory"):
            assert page.locator(f"{host} path.lc-line").count() == 1, host
            label = page.get_attribute(f"{host} svg", "aria-label")
            assert re.search(r"\d+ replies, latest ", label), label
        assert page.locator("#chartSpeed .lc-dot").count() == REPLIES and page.locator("#chartSpeed .lc-last").count() == 1
        assert page.locator("#chartSpeed .lc-avg").count() == 1
        assert "tok/s" in page.get_attribute("#chartSpeed svg", "aria-label")
        assert re.fullmatch(r"\d+\.\d tok/s", page.text_content("#chartSpeed .lc-last-text"))
        assert re.fullmatch(r"\d\.\d\d s", page.text_content("#chartFirst .lc-last-text")) or \
            re.fullmatch(r"0\.\d s", page.text_content("#chartFirst .lc-last-text")), page.text_content("#chartFirst .lc-last-text")
        assert not page.evaluate("document.querySelector('#gauge').classList.contains('warn')")
        # tokens: a bar of three parts and a legend with the same numbers
        assert page.locator("#tokenBar .seg").count() == 3
        legend = page.locator("#tokenLegend li").all_inner_texts()
        assert [x.replace("\n", " ").split()[0] for x in legend] == ["new", "cached", "reply"], legend
        tokens = rows(page, "#tokenStats")
        assert tokens["Served from cache"].endswith("%)"), tokens
        assert re.fullmatch(r"\d+ in, \d+ out", tokens["Per reply"]), tokens
        assert rows(page, "#modelStats")["Replies served"] == str(REPLIES)
        assert fits()
        shot("charts")

        # -- what else is using the computer, and what the engine's own store holds -----
        cond = rows(page, "#condStats")
        assert cond["Power"] == "on battery" and cond["Low Power Mode"] == "off", cond
        assert cond["Thermal limit"] == "none reported" and cond["Memory free"] == "1.30 GB", cond
        assert cond["Other apps"] == "2.1 CPU cores (cmux 2.1)", cond
        assert cond["GPU"] == "4% in use by others", cond
        assert page.inner_text("#condLine") == "on battery · other apps use 2.1 CPU cores (cmux): expect slower"
        mem = rows(page, "#memStats")
        assert mem["Expert store"] == "60 slots, 4.25 GB", mem
        assert mem["Expert reads"] == "132 rows a token from the SSD", mem
        assert fits()
        shot("conditions")

        # -- the tooltip, by pointer and by keys -----------------------------------
        box = page.locator("#chartSpeed svg").bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.locator("#chartSpeed .lc-tip").wait_for(state="visible")
        tip = page.inner_text("#chartSpeed .lc-tip")
        assert re.search(r"Reply \d+ at \d\d:\d\d:\d\d", tip) and "tok/s decode" in tip and "to the first token" in tip, tip
        assert page.locator("#chartSpeed .lc-hot").is_visible()
        tip_box = page.locator("#chartSpeed .lc-tip").bounding_box()
        assert tip_box["x"] >= box["x"] - 1 and tip_box["x"] + tip_box["width"] <= box["x"] + box["width"] + 1, tip_box
        shot("tooltip")
        page.mouse.move(5, 5)
        page.locator("#chartSpeed .lc-tip").wait_for(state="hidden")
        page.focus("#chartFirst")
        first_tip = page.inner_text("#chartFirst .lc-tip")
        assert f"Reply {REPLIES} at" in first_tip, first_tip                   # focus lands on the latest reply
        page.keyboard.press("ArrowLeft")
        assert f"Reply {REPLIES - 1} at" in page.inner_text("#chartFirst .lc-tip")
        page.keyboard.press("Home")
        assert "Reply 1 at" in page.inner_text("#chartFirst .lc-tip")
        page.keyboard.press("Escape")
        assert page.locator("#chartFirst .lc-tip").is_hidden()

        # -- the numbers as a table ------------------------------------------------
        page.click("#tableView summary")
        assert page.locator("#replyTable tbody tr").count() == REPLIES
        first_row = page.locator("#replyTable tbody tr").first.all_inner_texts()[0].split("\t")
        assert first_row[0] == str(REPLIES) and first_row[7].endswith(" GB"), first_row   # newest first
        assert fits()
        shot("table")

        # -- the server goes away ---------------------------------------------------
        page.route("**/status", lambda route: route.abort())
        until(page, "document.querySelector('#liveText').textContent === 'Server not reachable'", timeout=10)
        assert page.evaluate("document.querySelector('#liveBadge').classList.contains('off')")
        assert re.fullmatch(r"\d+\.\d tok/s", page.inner_text("#tSpeed")), "the last numbers stay"
        page.unroute("**/status")
        errors[:] = [e for e in errors if "ERR_FAILED" not in e and "Failed to load resource" not in e]
        until(page, "document.querySelector('#liveText').textContent.startsWith('Live')", timeout=10)
        assert fits()
        assert not errors, errors
    finally:
        ctx.close()
        proc.terminate()
    print(f"  {name}: empty state, tiles with units, three charts, tooltip by pointer and keys, "
          "tokens, conditions, model and machine lists, table, offline")


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
                    run_context(browser, scheme, size, f"status-{label}-{scheme}", shots)
        finally:
            browser.close()
    print("ok")


if __name__ == "__main__":
    main()

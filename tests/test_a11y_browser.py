#!/usr/bin/env python3
"""Accessibility checks on the three pages in headless Chrome, against the stub
engine (no model): every control has a name, the page has a language, the skip
link is the first stop of the Tab key and shows itself, focus is visible, the
text of each kind of element has a contrast ratio of at least 4.5 to 1 against
what is behind it, in light and dark, and the pages work with only the keyboard
for the main actions.

    pip install playwright
    STUB_PYTHON=python3 python3 tests/test_a11y_browser.py

The stub is tests/test_webchat.py --serve --settings --fast --grow --ctx 300.
"""

import os
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_webchat_browser import DESKTOP, PHONE, reply_done, until          # noqa: E402

STUB_PYTHON = os.environ.get("STUB_PYTHON", sys.executable)
MIN_RATIO = 4.5

# What each page shows as text, by selector. Elements that are not on screen or are disabled are skipped.
TEXT = {
    "/": [".brand-name", ".new-chat span", ".conv-label", ".conv-open", ".conv-sub", ".sidebar-foot span", ".foot-link",
          "#model", ".model-host", ".live-pill", ".ghost-btn span", ".msg.user .user-text", ".msg.assistant .body p",
          ".meta", ".think summary", ".ctx-label", ".ctx-text", ".switch-text", ".mini-btn:not(:disabled)", "#note",
          ".empty h2", ".empty-sub", ".ex-title", ".ex-text", ".hints", "kbd", ".compact-head", ".marker .label",
          ".summary-view summary", ".archive summary", ".code .lang", ".code .copy", "#send", ".composer textarea"],
    "/settings": [".back-link span", ".page-tabs a", ".side-nav a", ".page-title", ".panel h2", ".panel-lead", ".card h3",
                  ".stats dt", ".stats dd", ".set-hint", ".model-name", ".model-about", ".v-label", ".v-meta", ".v-note",
                  ".v-badge", ".v-meas", ".v-meta.short", ".model-tools input", ".mini-btn:not(:disabled)", ".opt-label", ".opt-note", ".switch-text", ".where dt", ".where dd",
                  ".cg-name", ".cg-n", ".primary-btn:not(:disabled)", "#applyNote"],
    "/monitor": [".page-tabs a", ".page-title", ".live-badge", ".tile-label", ".tile-value", ".tile-sub", ".chart-head h2",
                 ".chart-sub", ".card-h", ".legend li", ".lg-val", ".stats dt", ".stats dd", ".lc-tick", ".lc-last-text",
                 ".lc-avg-text", ".table-view summary", ".table-view th", ".table-view td"],
}

CONTRAST_JS = """
(selectors) => {
  const num = (s) => s.split(/[ ,\\/]+/).filter(Boolean);
  function parse(c) {
    let m = c.match(/^rgba?\\(([^)]+)\\)/);
    if (m) { const p = num(m[1]).map(Number); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; }
    m = c.match(/^color\\(srgb ([^)]+)\\)/);
    if (m) { const p = num(m[1]).map(Number); return { r: p[0] * 255, g: p[1] * 255, b: p[2] * 255, a: p.length > 3 ? p[3] : 1 }; }
    return { r: 0, g: 0, b: 0, a: 0 };
  }
  const over = (top, under) => ({
    r: top.r * top.a + under.r * (1 - top.a), g: top.g * top.a + under.g * (1 - top.a),
    b: top.b * top.a + under.b * (1 - top.a), a: 1 });
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const out = [];
  for (const sel of selectors) {
    for (const el of document.querySelectorAll(sel)) {
      const box = el.getBoundingClientRect();
      if (!box.width || !box.height || !(el.innerText || el.textContent || el.value || "").trim()) continue;
      const style = getComputedStyle(el);
      if (style.visibility === "hidden" || style.display === "none") continue;
      if (box.right < 0 || box.left > innerWidth) continue;      // a drawer that is off screen
      // what is behind the text: the layers above the first opaque one, painted over it
      const layers = []; let opacity = 1;
      for (let n = el; n; n = n.parentElement) {
        const cs = getComputedStyle(n);
        opacity *= Number(cs.opacity);
        const bg = parse(cs.backgroundColor);
        if (bg.a > 0) layers.push(bg);
        if (bg.a >= 1) break;
      }
      let back = { r: 255, g: 255, b: 255, a: 1 };
      for (let i = layers.length - 1; i >= 0; i--) back = over(layers[i], back);
      let fg = parse(style.color);
      if (style.webkitTextFillColor && style.webkitTextFillColor !== style.color) {
        const f = parse(style.webkitTextFillColor);
        if (f.a === 0) continue;                                // painted by a gradient: not measured here
        fg = f;
      }
      fg = over({ r: fg.r, g: fg.g, b: fg.b, a: fg.a * opacity }, back);
      const l1 = lum(fg), l2 = lum(back);
      const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
      out.push({ sel, text: (el.innerText || el.textContent || "").trim().slice(0, 30), ratio: Math.round(ratio * 100) / 100 });
    }
  }
  return out;
}
"""

NAMES_JS = """
() => {
  const bad = [];
  for (const el of document.querySelectorAll("a[href], button, input, select, textarea, summary, [role=button], [role=menuitemradio]")) {
    const box = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (el.type === "hidden" || el.hidden || st.display === "none") continue;
    if (el.closest("[hidden]")) continue;
    let name = (el.getAttribute("aria-label") || "").trim();
    if (!name && el.labels && el.labels.length) name = [...el.labels].map(l => l.textContent).join(" ").trim();
    if (!name && el.getAttribute("aria-labelledby")) name = el.getAttribute("aria-labelledby");
    if (!name) name = (el.textContent || "").trim();
    if (!name && el.tagName !== "INPUT") name = (el.getAttribute("title") || "").trim();
    if (!name) bad.push(el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + "." + el.className);
  }
  return bad;
}
"""


def start_stub():
    p = subprocess.Popen([STUB_PYTHON, str(HERE / "test_webchat.py"), "--serve", "--settings", "--fast", "--grow",
                          "--ctx", "300", "--code"], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.startswith("PORT"), f"the stub did not start: {line!r}"
    return p, int(line.split()[1])


def check_page(page, path, label):
    page.mouse.move(0, 0)
    page.wait_for_timeout(450)                                  # colour transitions have ended
    bad = page.evaluate(NAMES_JS)
    assert not bad, f"{label}: controls without a name: {bad}"
    assert page.get_attribute("html", "lang") == "en"
    assert page.title().strip(), label
    found = page.evaluate(CONTRAST_JS, TEXT[path])
    assert found, f"{label}: nothing was measured"
    worst = sorted(found, key=lambda r: r["ratio"])[:4]
    low = [r for r in found if r["ratio"] < MIN_RATIO]
    assert not low, f"{label}: text below {MIN_RATIO}:1: {low[:8]}"
    return len(found), worst[0]


def run(browser, scheme, size, name):
    proc, port = start_stub()
    ctx = browser.new_context(color_scheme=scheme, **size)
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    base = f"http://127.0.0.1:{port}"
    try:
        # the chat, with a reply in code, a compaction marker and the warning state
        page.goto(base + "/")
        until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        # the skip link is hidden until it has the focus, then shows itself and takes you to the message box
        hidden_y = page.locator(".skip-link").bounding_box()["y"]
        assert hidden_y < 0, "the skip link is off screen until focused"
        page.evaluate("document.activeElement.blur()")
        for _ in range(60):                                      # Tab from the message box wraps round to the first stop
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement.className") == "skip-link":
                break
        assert page.evaluate("document.activeElement.className") == "skip-link"
        page.wait_for_timeout(250)
        box = page.locator(".skip-link").bounding_box()
        assert box and box["y"] >= 0 and box["x"] >= 0, box
        page.keyboard.press("Enter")
        assert page.evaluate("document.activeElement.id") == "box", "the skip link goes to the message box"
        counts = [check_page(page, "/", name + " welcome")]
        for text in ("Show a loop", "Another question with a few more words in it", "A third one with some more words again",
                     "A fourth one that makes the chat long enough to compact"):
            before = page.locator(".msg.assistant").count()
            page.fill("#box", text)
            page.click("#send")
            reply_done(page, before + 1)
        page.click("#compactBtn")
        until(page, "document.querySelector('.compaction')")
        page.locator("details.summary-view > summary").click()
        page.locator("details.archive > summary").click()
        page.locator(".msg.assistant").last.hover()
        counts.append(check_page(page, "/", name + " chat"))
        # settings and status
        page.goto(base + "/settings")
        until(page, "document.querySelectorAll('#modelList .model').length === 2")
        until(page, "document.querySelector('#aboutList dd')")
        counts.append(check_page(page, "/settings", name + " settings"))
        page.keyboard.press("Tab")
        assert page.evaluate("document.activeElement.className") == "skip-link"
        page.goto(base + "/monitor")
        until(page, "document.querySelectorAll('#chartSpeed .lc-dot').length > 2")
        counts.append(check_page(page, "/monitor", name + " status"))
        assert not errors, errors
    finally:
        ctx.close()
        proc.terminate()
    worst = min((c[1] for c in counts), key=lambda r: r["ratio"])
    print(f"  {name}: every control named, skip link, {sum(c[0] for c in counts)} texts all at {MIN_RATIO}:1 or better "
          f"(lowest {worst['ratio']}:1, {worst['sel']})")


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        try:
            for scheme in ("light", "dark"):
                for label, size in (("desktop", DESKTOP), ("phone", PHONE)):
                    run(browser, scheme, size, f"a11y-{label}-{scheme}")
        finally:
            browser.close()
    print("ok")


if __name__ == "__main__":
    main()

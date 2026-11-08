#!/usr/bin/env python3
"""The Settings page in headless Chrome, against the stub engine and a fake
launcher (no model, no download, no Hugging Face): the link in the chat's
header, the page's sections, the models with their variants and actions, a
prepare with its bar and Stop, Use, the options and Apply, and the errors,
light and dark, phone and desktop width.

    pip install playwright
    STUB_PYTHON=python3 python3 tests/test_settings_browser.py [--shots DIR]

The stub server is tests/test_webchat.py --serve --settings, which installs the
fake launcher behind web_settings; STUB_PYTHON has to be a Python that can
import server.py, this script's own only needs playwright.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_webchat_browser import DESKTOP, PHONE, open_drawer, until                       # noqa: E402

STUB_PYTHON = os.environ.get("STUB_PYTHON", sys.executable)


def start_stub():
    p = subprocess.Popen([STUB_PYTHON, str(HERE / "test_webchat.py"), "--serve", "--settings"],
                         stdout=subprocess.PIPE, text=True)
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
            page.wait_for_timeout(250)                   # let the color transitions settle
            page.screenshot(path=str(shots / f"{name}-{label}.png"))

    def fits():
        return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    try:
        page.goto(f"http://127.0.0.1:{port}/")
        try:
            until(page, "document.querySelector('#model').textContent === 'tiny-chat'")
        except AssertionError as err:
            raise AssertionError(f"{err}; the page reported {errors}") from None

        # -- the header's link opens the page ---------------------------------------
        assert page.get_attribute("#settingsBtn", "href") == "/settings"
        page.click("#settingsBtn")
        until(page, "location.pathname === '/settings'")
        until(page, "document.querySelectorAll('#modelList .model').length === 2")
        assert page.title() == "Settings - Local chat"
        assert page.is_visible("#settings") and page.is_visible("#backLink")
        cur = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#setCurrent dt')]
            .map(t => [t.textContent, t.nextElementSibling.textContent]))""")
        assert cur == {"Model": "Tiny Chat", "Variant": "4-bit", "Speed": "no reply yet",
                       "Images": "not supported", "Memory": "no fixed limit"}, cur
        machine = page.inner_text("#setMachine")
        assert machine.count("a test chip") == 1, machine
        assert "16 GB memory" in machine and "88 GB free" in machine, machine
        assert "a test chip · 16 GB memory · 88 GB free for models" == machine, machine
        assert fits()
        shot("open")

        # -- the models, their variants and what each button says ---------------------
        tiny, big = page.locator(".model").nth(0), page.locator(".model").nth(1)
        assert tiny.locator(".model-name").inner_text() == "Tiny Chat"
        assert "dense 4-bit" in tiny.locator(".model-about").inner_text()
        labels = [tiny.locator(".v-label").nth(i).inner_text() for i in range(4)]
        assert labels == ["4-bit", "3-bit", "8-bit", "16-bit"], labels
        buttons = [tiny.locator(".v-action").nth(i).inner_text() for i in range(4)]
        assert buttons == ["In use", "Make 3-bit copy", "Can't run here", "Not enough disk"], buttons
        # A full disk is its own state: amber status and button, never "can't run".
        short = tiny.locator(".variant").nth(3)
        assert short.locator(".v-action").is_disabled()
        assert short.locator(".v-meta").inner_text().endswith("needs 41 GB free on disk, 12.6 GB free")
        assert "short" in short.locator(".v-meta").get_attribute("class")
        assert "can't run" not in short.inner_text().lower(), short.inner_text()
        assert short.inner_text().count("needs 41 GB") == 1, short.inner_text()
        assert tiny.locator(".v-action").nth(0).is_disabled()
        assert tiny.locator(".v-action").nth(1).is_enabled()
        assert tiny.locator(".v-action").nth(2).is_disabled()
        assert "needs 12 GB" in tiny.locator(".v-action").nth(2).get_attribute("title")
        assert tiny.locator(".v-badge").inner_text() == "IN USE"
        current = tiny.locator(".variant").nth(0)
        assert "your own copy at " in current.inner_text(), current.inner_text()
        assert current.locator(".v-meta").inner_text().endswith("running now"), current.inner_text()
        assert "2.0 GB" in tiny.locator(".v-meta").nth(0).inner_text()
        assert "~42 tok/s" in tiny.locator(".v-meta").nth(0).inner_text()
        # A switch counts the engine's own memory: the estimate is the one a switch would see.
        assert "~32 tok/s" in big.locator(".v-meta").inner_text(), big.inner_text()
        assert "after this model is unloaded" in big.inner_text(), big.inner_text()
        # A reason that cannot run is said once, not in the meta line and again below.
        cannot = tiny.locator(".variant").nth(2)
        assert cannot.inner_text().count("needs 12 GB") == 1, cannot.inner_text()
        assert "needs 12 GB" not in cannot.locator(".v-meta").inner_text(), cannot.inner_text()
        assert big.locator(".v-action").inner_text() == "Download"
        assert "Not made here: a 2-bit copy" in big.inner_text()
        # Speed measured on this computer sits under the estimate, also for a model not on disk.
        assert tiny.locator(".v-meas").count() == 1
        assert tiny.locator(".v-meas").inner_text() == "measured here: 13.8 tok/s (12 replies, 2026-10-27)"
        assert big.locator(".v-meas").inner_text() == "measured here: 7.9 tok/s (3 replies, 2026-10-20)"
        shot("models")
        # Refresh asks the server to look at the disks again and lists once more.
        with page.expect_request("**/admin/models?refresh=1"):
            page.click("#modelRefresh")
        until(page, "document.querySelectorAll('#modelList .model').length === 2")

        # Search: a link goes straight to its repo, and the results sit apart.
        link = "https://huggingface.co/owner/Foo-4bit/tree/main"
        page.fill("#modelSearch", link)
        page.click("#modelSearchBtn")
        until(page, "document.querySelectorAll('#foundList .model').length === 1")
        assert page.inner_text("#foundTitle") == "Results for “" + link + "”"
        found = page.locator("#foundList .model")
        assert found.locator(".model-name").inner_text() == "owner/Foo-4bit (found)"
        assert found.locator(".v-action").inner_text() == "Download" and found.locator(".v-action").is_enabled()
        assert "download 3.0 GB" in found.locator(".v-meta").inner_text()
        assert page.locator("#modelList .model").count() == 2 and fits()
        shot("search")
        page.fill("#modelSearch", "nope/model")                       # a repo this computer cannot use says why
        page.click("#modelSearchBtn")
        until(page, "document.querySelector('#searchNote').classList.contains('err')")
        assert page.inner_text("#searchNote") == "no safetensors weights in the repo"
        errors[:] = [e for e in errors if "status of 422" not in e]
        page.fill("#modelSearch", "found models")                      # words: a list, with what was skipped
        page.click("#modelSearchBtn")
        until(page, "document.querySelectorAll('#foundList .model').length === 2")
        assert "3 more skipped" in page.inner_text("#foundNote") and page.inner_text("#searchNote") == ""
        page.click("#foundClear")
        until(page, "document.querySelector('#foundBox').hidden")
        assert page.locator("#foundList .model").count() == 0

        # -- a download: bytes, speed, time left, Stop, then pick it up again ---------
        big.locator(".v-action").click()
        until(page, "document.querySelector('#modelList .progress-text')")
        first = page.inner_text("#modelList .progress-text")
        page.wait_for_timeout(900)
        second = page.inner_text("#modelList .progress-text")
        assert first != second, (first, second)
        assert second.endswith(" MB of 12 MB"), second
        until(page, "document.querySelector('#modelList .progress-text').textContent.includes('MB/s')",
              timeout=10)
        rate_text = page.inner_text("#modelList .progress-text")
        assert "MB/s ·" in rate_text and "left" in rate_text, rate_text
        assert page.locator("#modelList .progress .bar > span").evaluate("e => e.style.width")
        shot("progress")
        page.click("#modelList .progress .mini-btn")
        until(page, "document.querySelector('#modelList .progress-text').textContent.includes('Stopped')")
        shot("stopped")
        big.locator(".v-action").click()
        until(page, "document.querySelector('#modelList .progress-text').textContent === 'Ready to use.'",
              timeout=40)
        until(page, "document.querySelectorAll('#modelList .model')[1]"
                    ".querySelector('.v-action').textContent === 'Use'")
        shot("downloaded")

        # Delete: a confirm step first, then the folder goes.
        assert tiny.locator(".v-del").count() == 0, "the model in use and a copy outside the folder have none"
        assert big.locator(".v-del .mini-btn").inner_text() == "Delete"
        big.locator(".v-del .mini-btn").click()
        question = big.locator(".v-del-text").inner_text()
        assert question.startswith("Delete 4-bit at ") and "big-moe" in question and question.endswith("(9.0 GB)?"), question
        assert [b.inner_text() for b in big.locator(".v-del .mini-btn").all()] == ["Yes, delete", "Cancel"]
        shot("delete")
        big.locator(".v-del .mini-btn").nth(1).click()                       # Cancel: nothing happens
        assert big.locator(".v-del-text").count() == 0 and big.locator(".v-action").inner_text() == "Use"
        big.locator(".v-del .mini-btn").click()
        big.locator(".v-del .mini-btn").nth(0).click()
        until(page, "document.querySelectorAll('#modelList .model')[1].querySelector('.v-action')"
                    ".textContent === 'Download'")
        assert page.inner_text("#applyNote").startswith("deleted "), page.inner_text("#applyNote")
        assert big.locator(".v-del").count() == 0 and big.locator(".progress").count() == 0
        shot("deleted")
        big.locator(".v-action").click()                                     # and the download again, for Use
        until(page, "document.querySelector('#modelList .progress-text')?.textContent === 'Ready to use.'",
              timeout=40)
        until(page, "document.querySelectorAll('#modelList .model')[1]"
                    ".querySelector('.v-action').textContent === 'Use'")

        # -- Use: the launcher's own flags, then the card says so --------------------
        big.locator(".v-action").click()
        until(page, "document.querySelector('#setCurrent dd').textContent === 'Big MoE'")
        cur = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#setCurrent dt')]
            .map(t => [t.textContent, t.nextElementSibling.textContent]))""")
        assert cur["Model"] == "Big MoE" and cur["Variant"] == "4-bit", cur
        assert cur["Memory"] == "6.0 GB", cur
        assert page.is_visible("#planNote") and "--prefill-step 0" in page.inner_text("#planNote")
        assert not page.is_checked("#optFast"), "this model runs with exact routing"
        assert page.input_value("#memLimit") == "auto"
        assert "the launcher's choice" in page.inner_text("#memNote")
        assert fits()
        shot("switched")

        # -- the options and Apply ---------------------------------------------------
        page.select_option("#memLimit", "fixed")
        assert page.is_visible("#memGb") and page.is_visible("#memUnit")
        page.fill("#memGb", "9.5")
        page.locator("#optFast").scroll_into_view_if_needed()
        page.click("label:has(#optFast)")
        before = page.evaluate("window.scrollY")
        page.click("label:has(#optThink)")
        assert page.is_checked("#optFast") and page.is_checked("#optThink")
        assert "keeps its own peak under" in page.inner_text("#memNote")
        assert abs(page.evaluate("window.scrollY") - before) < 2, "a switch scrolled the page"
        shot("options")
        page.click("#applyBtn")
        until(page, "document.querySelector('#applyNote').textContent.includes('with these settings')")
        until(page, "document.querySelectorAll('#setCurrent dd')[4].textContent === '9.5 GB'")
        assert page.input_value("#memLimit") == "fixed" and page.input_value("#memGb") == "9.5"
        assert page.is_checked("#optFast") and page.is_checked("#optThink")
        shot("applied")

        # -- an error from the server, and a number it cannot take -------------------
        def fail(route):
            route.fulfill(status=503, json={"error": {
                "message": "the new model did not fit in memory", "type": "server_error"}})
        page.route("**/admin/load", fail)
        n_errors = len(errors)
        page.click("#applyBtn")
        until(page, "document.querySelector('#applyNote').classList.contains('err')")
        assert page.inner_text("#applyNote") == "Could not switch: the new model did not fit in memory"
        errors[n_errors:] = [e for e in errors[n_errors:] if "status of 503" not in e]
        shot("error")
        page.unroute("**/admin/load")
        page.fill("#memGb", "")
        page.click("#applyBtn")
        until(page, "document.querySelector('#applyNote').textContent.includes('Write the memory limit')")
        page.fill("#memGb", "9.5")
        page.click("#applyBtn")
        until(page, "document.querySelector('#applyNote').textContent.includes('with these settings')")

        # -- a reply after the switch: the speed in the card is this model's ---------
        page.click("#backLink")
        until(page, "location.pathname === '/'")
        until(page, "document.querySelector('#model').textContent !== 'Loading'")
        page.fill("#box", "Hello")
        page.click("#send")
        until(page, "document.querySelectorAll('.msg.assistant').length === 1 && "
                    "document.querySelector('#send').textContent === 'Send'")
        page.click("#settingsBtn")
        until(page, "document.querySelectorAll('#setCurrent dd')[2].textContent === '10.0 tok/s'")
        shot("speed")

        # -- memory now, about this machine -----------------------------------------------
        until(page, "document.querySelector('#memNow dd')")
        mem = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#memNow dt')]
            .map(t => [t.textContent, t.nextElementSibling.textContent]))""")
        assert mem["Limit in force"] == "9.5 GB" or "9.50 GB" in mem["Limit in force"], mem
        assert mem["This computer"] == "16 GB" and "GB" in mem["In use"], mem
        about = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#aboutList dt')]
            .map(t => [t.textContent, t.nextElementSibling.textContent]))""")
        assert about["Computer"] == "test-host" and about["Chip"] == "a test chip", about
        assert about["Memory"] == "16 GB" and about["Free disk for models"] == "88 GB", about
        assert about["Engine"] == "Metal, MLX 0.0" and about["Context length"] == "not in the model's config", about   # the switched-to model has no folder here
        assert about["This page is served at"].startswith("127.0.0.1:"), about

        # -- chats and data: what is stored where, export, import --------------------------
        where = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#whereList dt')]
            .map(t => [t.textContent, t.nextElementSibling.textContent]))""")
        assert "In this browser, for the address 127.0.0.1:" in where["Your chats"], where
        assert "1 chat" in where["Your chats"] and "The server does not keep them" in where["Your chats"]
        assert "not saved in your browser" in where["The model and its options"]
        assert page.inner_text("#chatSummary").startswith("1 chat with 2 messages")
        groups = page.locator("#chatGroups li").all_inner_texts()
        assert len(groups) == 1 and "on test-host" in groups[0] and "this server" in groups[0].lower(), groups
        with page.expect_download() as got:
            page.click("#exportBtn")
        assert got.value.suggested_filename.startswith("local-chat-") and got.value.suggested_filename.endswith(".json")
        exported = json.loads(Path(got.value.path()).read_text())
        assert exported["app"] == "local-chat" and exported["format"] == 2 and exported["from"]["host"] == "test-host"
        mine = exported["conversations"][0]
        assert mine["host"] == "test-host" and mine["model"] and [m["role"] for m in mine["messages"]] == ["user", "assistant"]
        assert "Exported 1 chat" in page.inner_text("#chatNote")
        shot("chats")
        tmp = Path(tempfile.mkdtemp(prefix="moe-import-"))
        newer = dict(mine, updated=mine["updated"] + 5000, title="Same chat, newer copy")
        older = dict(mine, updated=mine["updated"] - 5000, title="Same chat, older copy")
        other = {"id": "from-another-port", "title": "A chat from another model", "created": 1, "updated": 2,
                 "model": "qwen-demo", "host": "other-host",
                 "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello **there**"}]}
        (tmp / "a.json").write_text(json.dumps({"conversations": [other, older, {"id": "junk", "messages": []}]}))
        page.set_input_files("#importFile", str(tmp / "a.json"))
        until(page, "document.querySelector('#chatNote').textContent.startsWith('Imported')")
        assert page.inner_text("#chatNote") == "Imported 1 chat, skipped 1 already here.", page.inner_text("#chatNote")
        assert page.inner_text("#chatSummary").startswith("2 chats with 4 messages")
        assert len(page.locator("#chatGroups li").all_inner_texts()) == 2
        (tmp / "b.json").write_text(json.dumps([newer]))
        page.set_input_files("#importFile", str(tmp / "b.json"))
        until(page, "document.querySelector('#chatNote').textContent.includes('replaced 1')")
        (tmp / "c.txt").write_text("not json at all")
        page.set_input_files("#importFile", str(tmp / "c.txt"))
        until(page, "document.querySelector('#chatNote').textContent === 'That file is not JSON.'")
        assert page.evaluate("document.querySelector('#chatNote').classList.contains('err')")
        (tmp / "d.json").write_text(json.dumps({"hello": 1}))
        page.set_input_files("#importFile", str(tmp / "d.json"))
        until(page, "document.querySelector('#chatNote').textContent === 'That file has no chats in it.'")
        assert fits()
        shot("chats-imported")
        # the imported chat is in the chat page's list, tagged with its own model and computer
        page.click("#backLink")
        until(page, "location.pathname === '/'")
        until(page, "document.querySelectorAll('.conv-item').length === 2")
        until(page, "document.querySelector('.conv-sub') && [...document.querySelectorAll('.conv-sub')].some(e => e.textContent.includes('other-host'))")
        open_drawer(page, desktop)
        sub = page.locator(".conv-item", has_text="A chat from another model").locator(".conv-sub").inner_text()
        assert sub == "qwen-demo \u00b7 other-host", sub
        assert page.locator(".conv-item", has_text="Same chat, newer copy").count() == 1
        page.locator(".conv-item", has_text="A chat from another model").locator(".conv-open").click()
        assert page.locator(".msg.assistant .body strong").inner_text() == "there"
        page.click("#settingsBtn")
        until(page, "location.pathname === '/settings'")

        # -- the layout: the section list, the sticky Apply, the way back -----------
        labels = page.locator("#sideNav a").all_inner_texts()
        assert labels == ["Model", "Memory", "Replies", "Chats and data", "About this machine"], labels
        page.click("#sideNav a[href='#sec-replies']")
        until(page, "document.querySelector('#sideNav a.active').getAttribute('href') === '#sec-replies'")
        nav = page.locator("#sideNav").bounding_box()
        if desktop:
            assert nav["width"] < 260 and nav["x"] < 200, nav       # a column beside the sections
            assert page.locator("#settings").bounding_box()["x"] > nav["x"] + nav["width"] - 1
        else:
            assert nav["width"] >= 380 and nav["height"] < 70, nav  # a row of pills under the bar
        apply_box = page.locator(".apply-bar").bounding_box()
        assert apply_box["y"] + apply_box["height"] <= size["viewport"]["height"], apply_box
        shot("sections")
        page.evaluate("window.scrollTo(0, 0)")
        shot("top")
        assert fits()
        assert not errors, errors
    finally:
        ctx.close()
        proc.terminate()
    print(f"  {name}: the link, the sections, models and actions (disk short, measured here), search, a download with "
          "Stop and resume, delete with a confirm, Use with the launcher's plan, options and Apply, errors, the sections")


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
                    run_context(browser, scheme, size, f"settings-{label}-{scheme}", shots)
        finally:
            browser.close()
    print("ok")


if __name__ == "__main__":
    main()
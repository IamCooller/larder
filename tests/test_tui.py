#!/usr/bin/env python3
"""Checks the full-screen menu without a terminal, ssh or a model, from its device-first flow and
status lines to its colours, speeds, start and stop, back navigation and frozen animation."""
import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tui


def settle(app, timeout=8):
    end = time.time() + timeout
    while time.time() < end:
        app.tick()
        busy = any([app.boot_job, app.probe_job, app.scan_job, app.refresh_job, app.cert_job,
                    app.models_job, app.search_job, app.delete_job,
                    app.job and not app.job.done, app.stop_job and not app.stop_job.done,
                    app.dev_job and not app.dev_job.done, app.pair_job and not app.pair_job.done])
        if not busy:
            app.tick()
            return
        time.sleep(0.002)
    raise AssertionError(f"the menu did not settle (mode {app.mode})")


def screen(app, w=100, h=30):
    cv = tui.TextCanvas(w, h)
    app.draw(cv)
    return cv


def find(cv, needle):
    for y, line in enumerate(cv.text().splitlines()):
        x = line.find(needle)
        if x >= 0:
            return y, x
    raise AssertionError(f"{needle!r} is not on the screen:\n{cv.text()}")


def start():
    b = tui.DemoBackend(delay=0.001)
    app = tui.App(b, frozen=True)
    settle(app)
    assert app.mode == "devices", app.mode
    return app, b


def pick_device(app, name):
    app.dev_cursor = [d.name for d in app.dev_rows].index(name)
    app.key("ENTER")
    settle(app)


def open_model(app, model_id):
    rows = app.rows()
    model = None
    for r in rows:
        if r[0] == "model" and r[1].id == model_id:
            model = r[1]
    if model is None:
        raise AssertionError(f"no model {model_id} in {[r[1].id for r in rows if r[0] == 'model']}")
    if app.open is not model:
        app.cursor = app.row_of(model)
        app.key("ENTER")
    assert app.open is model, model_id
    return model


def choose(app, model_id, key):
    m = open_model(app, model_id)
    v = next(v for v in m.variants if v.key == key)
    app.cursor = app.row_of(m, v)
    app.key("ENTER")


def check_devices_first():
    b = tui.DemoBackend(delay=0.001)
    app = tui.App(b, frozen=True)
    boot = screen(app)
    assert "l  a  r  d  e  r" in boot.text() and "run big language models on any of your computers" in boot.text()
    assert "a key skips the title" in boot.text()
    again = screen(app)
    assert boot.text() == again.text(), "frozen frames differ"
    settle(app)
    assert app.mode == "devices"
    names = [d.name for d in app.dev_rows]
    assert names == ["this computer", "studio", "rig", "desk", "shed", "attic.local", "lamp (shared)"], names
    cv = screen(app)
    for name in names:
        assert name in cv.text(), name
    assert "no ssh key" in cv.text() and "offline" in cv.text()
    assert "Enter models" in cv.text() and "r refresh or scan" in cv.text()
    # statuses each in their own colour (the selected row is the exception: it is highlighted)
    app.dev_cursor = names.index("desk")
    cv = screen(app)
    y, x = find(cv, "no ssh key")
    assert cv.style_at(y, x) == "bad"
    y, x = find(cv, "offline")
    assert cv.style_at(y, x) == "bad"
    y, x = find(cv, "ready")
    assert cv.style_at(y, x) == "good"
    # the selected device's detail line: where its engine is, or the fix
    app.dev_cursor = names.index("studio")
    cv = screen(app)
    assert "engine at ~/engine" in cv.text()
    assert "Enter shows the models this computer can run." in cv.text()
    app.dev_cursor = names.index("rig")
    cv = screen(app)
    assert "ssh-copy-id me@rig" in cv.text()
    app.dev_cursor = names.index("desk")
    cv = screen(app)
    assert "engine at ~/larder" in cv.text()
    print("1. devices first: this computer, presets, ssh hosts, peers, statuses in colour, the fix line")


def check_blocked_and_scan():
    app, b = start()
    pick_device(app, "rig")
    assert app.mode == "devices" and "ssh-copy-id" in app.flash[0]
    pick_device(app, "attic.local")
    assert app.mode == "devices" and "no answer on ssh" in app.flash[0]
    # r reads this computer and the known devices again, then asks before any sweep
    studio = app.dev_rows[1]
    studio.models = ["a cached list"]
    b.probed.clear()
    app.key("r")
    assert app.mode == "input"
    assert "Scan 254 addresses on 192.168.1.0/24 for other computers? y/n" in screen(app).text()
    app.key("ESC")                                   # no answer: no sweep
    settle(app)
    assert b.scanned == 0 and app.mode == "devices"
    assert app.dev_rows[1] is studio and studio.models == ["a cached list"], "a refreshed row keeps its list"
    assert "this computer" in b.probed and "studio" in b.probed, b.probed
    assert "attic.local" not in b.probed, "a scan row is never checked over ssh by a refresh"
    app.key("r")
    app.input["buf"] = "n"
    app.key("ENTER")                                 # a no: no sweep either
    settle(app)
    assert b.scanned == 0 and app.mode == "devices"
    # y sweeps; the host only the sweep found is listed and never checked over ssh
    app.key("r")
    settle(app)                                      # the refresh is done, the question still waits
    assert app.mode == "input" and b.scanned == 0
    b.probed.clear()
    app.input["buf"] = "y"
    app.key("ENTER")
    settle(app)
    assert b.scanned == 1
    assert "studio.local" in [d.name for d in app.dev_rows]
    assert b.probed == [], b.probed
    cv = screen(app)
    assert "studio.local" in cv.text() and "ssh open" in cv.text()
    print("2. no key / offline devices stay put with the fix; r reads the known devices again and asks "
          "before it sweeps; a host only the sweep found is listed and never checked over ssh")


def check_update_engine():
    app, b = start()
    d = app.dev_rows[3]                      # desk
    d.status, d.engine_state = "engine-old", "older"
    app.dev_cursor = 3
    cv = screen(app)
    assert "engine is older" in cv.text() and "update its engine" in cv.text(), cv.text()
    app.key("u")
    settle(app)
    assert b.updated == ["desk"], b.updated
    assert d.status == "ready" and d.engine_state == "ok", (d.status, d.engine_state)
    print("3. u updates the engine on a device whose protocol is behind, then its status is read again")


def check_install_engine():
    app, b = start()
    d = app.dev_rows[2]                      # rig
    d.status, d.engine_state = "no-engine", ""
    d.detail = "engine not installed: i installs it"
    app.dev_cursor = 2
    cv = screen(app)
    assert "install the engine" in cv.text(), cv.text()
    app.key("i")
    settle(app)
    assert b.installed == ["rig"], b.installed
    assert d.status == "ready" and d.engine_state == "ok", (d.status, d.engine_state)
    print("4. i installs the engine on a device that has none, then its status is read again")


def check_measured_vs_estimate():
    app, b = start()
    pick_device(app, "studio")
    assert app.mode == "models"
    assert [m.id for m in app.models] == ["flash-next-exl3", "qwen3.6-35b"], [m.id for m in app.models]
    cv = screen(app)
    assert "engine at ~/engine" in cv.text()
    flash = open_model(app, "flash-next-exl3")
    app.cursor = app.row_of(flash)                 # the model row: no variant under the highlight
    cv = screen(app)
    y, x = find(cv, "13.8-14.2 tok/s measured")
    assert cv.style_at(y, x) == "good"
    y, x = find(cv, "~17 tok/s (estimate)")
    assert cv.style_at(y, x) == "dim"
    y, x = find(cv, "~40 min")
    assert cv.style_at(y, x) == "warn"
    # the speed bar runs red -> amber with speed
    y, x = find(cv, "####")
    styles = {cv.style_at(y, x + i) for i in range(8) if x + i < cv.w}
    assert "bar_lo" in styles and "bar_mid" in styles, styles
    y, x = find(cv, "ready")
    assert cv.style_at(y, x) == "good"
    # the details pane sets the measured number apart from the estimate
    pub = next(v for v in flash.variants if v.key == "published")
    app.cursor = app.row_of(flash, pub)
    cv = screen(app)
    assert "measured on studio: 13.8-14.2 tok/s at its preset" in cv.text()
    print("3. models: measured speed in green and set apart, estimate dim, bar gradient, status colours")


def check_remote_start_stop():
    app, b = start()
    pick_device(app, "studio")
    choose(app, "flash-next-exl3", "published")
    assert app.mode == "start", app.mode                  # ready there: starts without a confirm
    settle(app)
    assert app.mode == "ready"
    assert b.started == [("studio", "flash-next-exl3", "published")], b.started
    cv = screen(app)
    assert "Ready:" in cv.text() and "on studio" in cv.text()
    assert "http://127.0.0.1:8800/" in cv.text() and "13.8-14.2 tok/s measured on studio" in cv.text()
    assert "the same for every model and computer" in cv.text()
    assert "stop it" in cv.text() and "devices" in cv.text()
    act = app.key("c")
    assert act == ("chat", "http://127.0.0.1:8800"), act
    app.key("o")
    assert b.opened == ["http://127.0.0.1:8800"]
    app.key("m")
    assert app.mode == "devices" and app.running is not None
    cv = screen(app)
    assert "running: Qwen3.8-Flash-Next" in cv.text()
    app.key("g")
    assert app.mode == "ready"
    app.key("s")
    settle(app)
    assert b.stopped and app.running is None and app.mode == "models"
    cv = screen(app)
    assert "stopped" in cv.text() and "ready" in cv.text()
    print("4. a ready model on a preset device starts at once, the local chat URL and measured speed show, s stops it")


def check_not_ready_paths():
    app, b = start()
    pick_device(app, "studio")
    choose(app, "flash-next-exl3", "q3")
    assert app.mode == "models"
    assert "can't prepare" in app.flash[0]
    cv = screen(app)
    assert "can't prepare models" in cv.text()
    # a generic computer: its own launcher can download and start, so the menu asks first
    app.key("ESC")
    app.key("ESC")
    assert app.mode == "devices"
    pick_device(app, "desk")
    choose(app, "demo-4b", "published")
    assert app.mode == "confirm"
    cv = screen(app)
    assert "Prepare and start on desk" in cv.text() and "Download   2.3 GB" in cv.text()
    assert "on desk" in cv.text()
    app.key("ENTER")
    settle(app)
    assert app.mode == "ready"
    assert ("desk", "demo-4b", "published") in b.started
    print("5. a preset cannot prepare and says so; another computer's launcher may download and start")


def check_switch_stops_previous():
    app, b = start()
    pick_device(app, "studio")
    choose(app, "flash-next-exl3", "published")
    settle(app)
    assert app.mode == "ready" and b.started[-1] == ("studio", "flash-next-exl3", "published")
    app.key("m")                                     # back to the devices, the studio row runs
    assert app.mode == "devices" and app.running is not None
    pick_device(app, "desk")
    choose(app, "demo-4b", "published")              # not ready there, so it asks first
    assert app.mode == "confirm" and app.running is None
    assert b.stopped, "the previous model was stopped before the next one starts"
    print("6. starting another model stops the previous one first, so the chat port stays the same")


def check_pair_shared():
    app, b = start()
    pick_device(app, "lamp (shared)")
    assert app.mode == "input"
    cv = screen(app)
    # the question carries the certificate's first 16 hex characters, to compare with the device's screen
    assert "Code shown on lamp (shared) (certificate 3f9a1c0b7d2e4a65: check that computer shows the same):" \
        in " ".join(cv.text().split()), cv.text()
    app.input["buf"] = "0000-0000-0000"
    app.key("ENTER")                                 # a wrong code comes back to the devices
    settle(app)
    assert app.mode == "devices" and "pairing failed: wrong code" in app.flash[0]
    pick_device(app, "lamp (shared)")
    app.input["buf"] = "7k3m 9qxd r2hf"
    app.key("ENTER")                                 # the right code, typed in lower case, pairs and enters its models
    settle(app)
    assert app.mode == "models" and b.paired == [("lamp (shared)", "0000-0000-0000"),
                                                 ("lamp (shared)", "7k3m 9qxd r2hf")], b.paired
    assert app.dev.name == "lamp (shared)" and app.dev.status == "ready"
    print("7. a shared device shows its certificate beside the question for the code, a wrong code is "
          "refused, the right one pairs and opens its models")


def check_local_flow():
    app, b = start()
    pick_device(app, "this computer")
    open_model(app, "demo-35b")
    cv = screen(app)
    y, x = find(cv, "can't run:")
    assert cv.style_at(y, x) == "bad_dim"
    choose(app, "demo-35b", "published")
    assert app.mode == "confirm"
    cv = screen(app)
    assert "Install and start" in cv.text() and "Download   20.4 GB" in cv.text()
    with contextlib.redirect_stdout(io.StringIO()):          # the install's own progress lines
        app.key("ENTER")
        assert app.mode == "install"
        settle(app)
    assert app.mode == "ready"
    cv = screen(app)
    assert "Ready: as published, 4-bit" in cv.text()
    app.key("s")
    settle(app)
    assert b.stopped and app.mode == "models"
    # search and a model folder exist only on this computer
    app.key("/")
    assert app.mode == "input"
    app.key("x")
    app.key("ENTER")
    settle(app)
    assert app.mode == "models"
    assert any(m.id == "found-x" for m in app.found)
    app.key("f")
    for ch in "nope":
        app.key(ch)
    app.key("ENTER")
    assert "can't use it" in app.flash[0]
    app.key("f")
    for ch in "here":
        app.key(ch)
    app.key("ENTER")
    assert app.found and app.found[0].id == "local"
    print("6. this computer: confirm, install, start, stop, search and a model folder")


def check_disk_short():
    app, b = start()
    pick_device(app, "this computer")
    m = open_model(app, "demo-35b")
    cv = screen(app)
    # a disk too full is amber and says what is needed, a memory limit stays "can't run"
    y, x = find(cv, "needs 58 GB free on disk")
    assert cv.style_at(y, x) == "warn" and "can't run: needs" not in cv.text()
    y, x = find(cv, "can't run: the whole")
    assert cv.style_at(y, x) == "bad_dim"
    short = next(v for v in m.variants if v.key == "hf-8")
    app.cursor = app.row_of(m, short)
    cv = screen(app)
    assert "needs 58 GB free on disk, 12.6 GB free" in cv.text()
    assert "It would run here; only the disk is short" in cv.text()
    app.key("ENTER")
    assert app.mode == "models" and app.flash[0].startswith("needs 58 GB free on disk, 12.6 GB free")
    assert app.flash[1] == "warn" and "can't run" not in app.flash[0]
    big = next(v for v in m.variants if v.key == "hf-16")
    app.cursor = app.row_of(m, big)
    app.key("ENTER")
    assert app.flash[0].startswith("can't run that one there") and app.flash[1] == "bad"
    print("9. a disk too full is amber with what it needs; only memory and hardware limits say can't run")


def check_found_copy():
    app, b = start()
    pick_device(app, "this computer")
    m = open_model(app, "demo-exl3")
    pub = next(v for v in m.variants if v.key == "published")
    app.cursor = app.row_of(m, pub)
    cv = screen(app)
    # a copy found outside the models folder reads "on disk at <path>", in green, with no download
    find(cv, "on disk at /home/demo/.cache/demo-exl3")                 # the details pane has it whole
    app.cursor = app.row_of(m)
    cv = screen(app)
    y, x = find(cv, "on disk at /hom")                                  # the row has what fits
    assert cv.style_at(y, x) == "good" and "download" not in cv.line(y)
    app.cursor = app.row_of(m, pub)
    app.key("ENTER")                                    # ready where it is: confirm shows the folder, no install
    assert app.mode == "confirm"
    cv = screen(app)
    assert "(already there)" in cv.text() and "/home/demo/.cache/demo-exl3" in cv.text()
    assert "Download" not in cv.text()
    app.key("ESC")
    # r looks at the disks again, then lists again
    assert b.rescanned == []
    app.key("r")
    settle(app)
    assert b.rescanned == ["this computer"] and app.mode == "models"
    print("10. a copy found outside the models folder reads on disk at <path> and starts without a download; "
          "r looks at the disks again")


def check_measured_here():
    app, b = start()
    pick_device(app, "this computer")
    m = open_model(app, "demo-exl3")
    pub = next(v for v in m.variants if v.key == "published")
    other = next(v for v in m.variants if v.key == "hf-4")
    # the row leads with what this computer measured, green; the estimate stays in the details beside it
    app.cursor = app.row_of(m, other)
    cv = screen(app)
    y, x = find(cv, "13.8 tok/s measured here")
    assert cv.style_at(y, x) == "good"
    assert "~7 tok/s (estimate)" in cv.text()                       # a variant with no record keeps its estimate
    app.cursor = app.row_of(m, pub)
    cv = screen(app)
    assert "~11 tok/s (estimate for this machine)" in cv.text()
    assert "measured here: 13.8 tok/s (12 replies, 2026-10-27)" in cv.text()
    # the speed bar follows the measured speed, not the estimate
    assert tui.shown_tps(pub) == 13.8 and tui.shown_tps(other) == 7
    app.key("ENTER")
    cv = screen(app)
    assert app.mode == "confirm" and "Measured   13.8 tok/s on this computer (12 replies, 2026-10-27)" in cv.text()
    app.key("ESC")
    print("11. measured speed: the row leads with it, the details say measured here (n replies, date) next to "
          "the estimate, a variant without a record keeps its estimate")


def type_in(app, text):
    for ch in text:
        app.key(ch)
    app.key("ENTER")
    settle(app)


def check_delete():
    app, b = start()
    pick_device(app, "this computer")
    m = open_model(app, "demo-exl3")
    elsewhere, own, absent = (next(v for v in m.variants if v.key == k) for k in ("published", "hf-4", "hf-8"))
    cv = screen(app)
    assert "d delete" in cv.text() and "/ search" in cv.text() and "f folder" in cv.text()
    # a copy outside the launcher's folder: where it is, and nothing is deleted
    app.cursor = app.row_of(m, elsewhere)
    app.key("d")
    assert app.mode == "models" and "/home/demo/.cache/demo-exl3" in app.flash[0], app.flash
    assert "deletes only what it downloaded or made" in app.flash[0] and app.flash[1] == "warn"
    app.cursor = app.row_of(m, absent)
    app.key("d")
    assert "nothing to delete" in app.flash[0]
    # the launcher's own folder: asks with the folder and size, and only a yes deletes it
    app.cursor = app.row_of(m, own)
    app.key("d")
    assert app.mode == "input"
    cv = screen(app)
    want = "Delete Demo-EXL3 (3-bit experts) 4-bit at /demo/models/demo-exl3-4bit (111.0 GB)? type yes: "
    assert want in cv.text(), cv.text()
    narrow = screen(app, 60, 20)                       # a long question gets its own line
    assert "/demo/models/demo-exl3-4bit" in narrow.text() and "type yes:" in narrow.text(), narrow.text()
    assert narrow.line(19) == " > _", repr(narrow.line(19))
    type_in(app, "no")
    assert app.mode == "models" and b.deleted == [] and app.flash[0] == "not deleted"
    app.key("d")
    app.key("ESC")
    assert app.mode == "models" and b.deleted == []
    app.key("d")
    type_in(app, "yes")
    assert b.deleted == [("this computer", "demo-exl3", "hf-4")], b.deleted
    assert app.flash[0] == "deleted /demo/models/demo-exl3-4bit (111.0 GB)" and app.flash[1] == "good"
    assert b.rescanned == ["this computer"] and app.mode == "models"       # and the list is made again
    # the model that is running is not asked about
    app.running = {"name": "Demo-EXL3", "model": "/demo/models/demo-exl3-4bit", "url": "u", "api": "a"}
    m = open_model(app, "demo-exl3")
    app.cursor = app.row_of(m, next(v for v in m.variants if v.key == "hf-4"))
    app.key("d")
    assert app.mode == "models" and "is running" in app.flash[0]
    app.running = None
    # the backend's refusal is shown, not hidden
    b.delete = lambda dev, model, v: (_ for _ in ()).throw(RuntimeError("qwen3-35b was made from it: delete that first"))
    app.key("d")
    type_in(app, "yes")
    assert app.flash[0] == "qwen3-35b was made from it: delete that first" and app.flash[1] == "bad"
    # another computer deletes through its own launcher
    app.key("ESC")
    app.key("ESC")
    del b.delete
    pick_device(app, "desk")
    m = open_model(app, "qwen3.6-35b")
    app.cursor = app.row_of(m, m.variants[0])
    app.key("d")
    assert app.mode == "input" and "at ~/larder/models/qwen3.6-35b" in screen(app).text()
    type_in(app, "yes")
    assert b.deleted[-1] == ("desk", "qwen3.6-35b", "published")
    print("12. d deletes a variant the launcher made after a typed yes, says where a copy elsewhere is, "
          "refuses the running model, shows the backend's refusal, works through another computer's launcher")


def search_for(app, text):
    app.key("/")
    assert app.mode == "input"
    type_in(app, text)


def check_search_everywhere():
    app, b = start()
    # / asks on every computer, and the footer lists it
    pick_device(app, "desk")
    assert "/ search" in screen(app).text()
    app.key("/")
    assert app.mode == "input" and "Search Hugging Face for desk (words, owner/name or a link): " in screen(app).text()
    type_in(app, "mlx-community/Foo-4bit")
    assert b.searched == [("desk", "mlx-community/Foo-4bit")], b.searched
    found = app.found[0]
    assert found.found and found.repo == "mlx-community/Foo-4bit" and app.open is found
    # a repo goes straight to its variants, the cursor on the first, judged by that computer
    cur = app.current()
    assert cur[0] == "variant" and cur[2] is found.variants[0]
    cv = screen(app)
    assert "download 8.0 GB" in cv.text() and "1 found, 2 skipped" in cv.text(), cv.text()
    # Enter prepares and starts it through that computer's launcher
    app.key("ENTER")
    assert app.mode == "confirm" and "Prepare and start on desk" in screen(app).text()
    app.key("ENTER")
    settle(app)
    assert app.mode == "ready" and b.started[-1] == ("desk", "mlx-community--Foo-4bit", "published"), b.started
    app.key("s")
    settle(app)
    # another computer judges for itself: here the disk is short, in amber, and Enter says so
    app.key("ESC")
    app.key("ESC")
    pick_device(app, "shed")
    search_for(app, "https://huggingface.co/mlx-community/Foo-4bit/tree/main")
    assert b.searched[-1][0] == "shed"
    app.cursor = app.row_of(app.found[0])                  # off the variant, so its colour shows
    cv = screen(app)
    y, x = find(cv, "needs 18 GB free on disk")
    assert cv.style_at(y, x) == "warn" and "can't run" not in cv.text()
    app.cursor = app.row_of(app.found[0], app.found[0].variants[0])
    app.key("ENTER")
    assert app.mode == "models" and app.flash[0].startswith("needs 18 GB free on disk, 12.9 GB free")
    # a repo that cannot run says why, from that computer
    search_for(app, "owner/nope")
    assert app.flash[0] == "search failed: model type 'demo_exp' is not in the installed mlx-lm", app.flash
    # a list of another computer does not keep this one's results
    app.key("ESC")
    app.key("ESC")
    pick_device(app, "desk")
    assert app.found == []
    # this computer still searches as before
    app.key("ESC")
    app.key("ESC")
    pick_device(app, "this computer")
    search_for(app, "tiny")
    assert b.searched[-1] == ("this computer", "tiny") and app.found[0].id == "found-tiny"
    print("13. / searches Hugging Face on every computer, a name or link goes straight to its variants judged "
          "by that computer, Enter prepares and starts it there")


def check_navigation():
    app, b = start()
    pick_device(app, "this computer")
    assert app.mode == "models" and app.open is not None   # the recommended model opens
    app.key("ESC")
    assert app.mode == "models" and app.open is None
    app.key("ESC")
    assert app.mode == "devices"
    app.key("LEFT")
    assert app.mode == "devices"                           # nothing above the first screen
    app.key("q")
    assert app.key("ESC") == "quit"
    small = screen(app, 50, 12)
    assert "Make the terminal larger" in small.text()
    wide = screen(app, 60, 16)
    assert "this computer" in wide.text() and "Enter models" in wide.text()
    app.key("DOWN")
    app.key("ENTER")
    settle(app)
    smaller = screen(app, 60, 16)
    assert "no models" not in smaller.text()
    print("7. Esc/Left go back one level, q quits, small terminals shrink without breaking")


def check_palette_and_frozen():
    p256 = tui.palette(256, True)
    assert p256[0][:3] == ("topbar", 16, 51), p256[0]
    assert all(0 <= fg <= 255 and (bg == -1 or 0 <= bg <= 255) for _, fg, bg, _ in p256)
    p8 = tui.palette(8, True)
    assert p8[0][:3] == ("topbar", tui.BLACK, tui.CYAN)
    assert all(fg <= 7 and (bg == -1 or bg <= 7) for _, fg, bg, _ in p8)
    assert tui.palette(256, False) == []
    assert tui.colors_enabled(True) and not tui.colors_enabled(True, {"NO_COLOR": "1"})
    assert not tui.colors_enabled(False, {})
    assert tui.frozen_by_env({"MOE_TUI_FROZEN": "1"}) and tui.frozen_by_env({"NO_COLOR": ""})
    assert not tui.frozen_by_env({})
    assert tui.status_style("download 20.4 GB") == "blue"
    assert tui.status_style("can't run: too big") == "bad_dim"
    assert tui.status_style("ready") == "good"
    assert tui.status_style("running: a model") == "run"
    # the same frozen clock gives the same frame, twice
    app = tui.App(tui.DemoBackend(delay=0.001), frozen=True)
    a, b = screen(app), screen(app)
    assert a.text() == b.text()
    assert tui.spin(True, 0.0) == tui.spin(True, 9.9) and tui.pulse(True, 0.0) == tui.pulse(True, 9.9)
    print("8. 256/8-colour palettes, NO_COLOR and a plain terminal honoured, frozen frames identical")


def check_live_ready():
    app, b = start()
    pick_device(app, "this computer")
    m = open_model(app, "demo-35b")
    pub = next(v for v in m.variants if v.key == "published")
    app.cursor = app.row_of(m, pub)
    cv = screen(app)
    # the model screen says what else is using the computer and that the model's own head is used
    y, x = find(cv, "other apps use 2.1 CPU cores (demo): expect slower")
    assert cv.style_at(y, x) == "warn"
    assert "~14-24 tok/s (estimate for this machine), its MTP head is used: about 25% more" in cv.text()
    app.key("ENTER")
    with contextlib.redirect_stdout(io.StringIO()):
        assert app.mode == "confirm"
        app.key("ENTER")
        settle(app)
    assert app.mode == "ready"
    cv = screen(app)
    # the ready screen shows the speed /status measured, next to the estimate it was started on
    y, x = find(cv, "12.8 tok/s measured over 3 replies (the estimate was ~24)")
    assert cv.style_at(y, x) == "good"
    y, x = find(cv, "other apps use 2.1 CPU cores (demo): expect slower")
    assert cv.style_at(y, x) == "warn"
    y, x = find(cv, "arena 21 slots, 1.5 GB · 132 expert rows a token from the SSD")
    assert cv.style_at(y, x) == "dim"
    # a server that stops answering falls back to the estimate, never to a stale number
    b.live = lambda: None
    app.live, app.live_at = None, 0.0
    app.tick()
    cv = screen(app)
    assert "~14-24 tok/s (estimate for this machine), no reply timed yet" in cv.text()
    assert "12.8 tok/s" not in cv.text()
    print("14. the ready screen: speed measured from /status over n replies beside the estimate, the "
          "conditions and the engine's own store; the model screen says how the head is used")


def main():
    check_devices_first()
    check_blocked_and_scan()
    check_update_engine()
    check_install_engine()
    check_measured_vs_estimate()
    check_remote_start_stop()
    check_not_ready_paths()
    check_switch_stops_previous()
    check_pair_shared()
    check_local_flow()
    check_navigation()
    check_palette_and_frozen()
    check_disk_short()
    check_found_copy()
    check_measured_here()
    check_delete()
    check_search_everywhere()
    check_live_ready()
    print("test_tui: ok")


if __name__ == "__main__":
    main()
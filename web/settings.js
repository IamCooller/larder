(function () {
  "use strict";
  // The Settings page: the model in use, the launcher's models, the engine's options and Apply.
  var $ = LC.$, el = LC.el, dl = LC.dl, errorText = LC.errorText;

  var modelListEl = $("#modelList"), setListNote = $("#setListNote");
  var setCurrentEl = $("#setCurrent"), setMachineEl = $("#setMachine"), memNote = $("#memNote");
  var memLimit = $("#memLimit"), memGb = $("#memGb"), memUnit = $("#memUnit");
  var optFast = $("#optFast"), optThink = $("#optThink");
  var applyBtn = $("#applyBtn"), applyNoteEl = $("#applyNote");
  var stg = { open: true, data: null, busy: false, dirty: false, error: "", timer: 0, confirm: null, delMsg: null };

  function fmtSize(b) {
    if (!b) return "-";
    return b >= 1e9 ? (b / 1e9).toFixed(1) + " GB" : Math.max(1, Math.round(b / 1e6)) + " MB";
  }
  function fmtRate(r) { return r ? Math.round(r / 1e6) + " MB/s" : ""; }
  function fmtEta(s) {
    if (s == null) return "";
    if (s < 90) return Math.max(1, Math.round(s)) + " s left";
    return Math.round(s / 60) + " min left";
  }

  function setNote(text, bad) {
    applyNoteEl.textContent = text || "";
    applyNoteEl.classList.toggle("err", !!bad);
  }

  $("#modelRefresh").addEventListener("click", function () { loadModels(true); });
  var foundBox = $("#foundBox"), foundList = $("#foundList"), foundTitle = $("#foundTitle"), foundNote = $("#foundNote");
  var searchBtn = $("#modelSearchBtn"), searchInput = $("#modelSearch"), searchNote = $("#searchNote");
  $("#modelSearchForm").addEventListener("submit", function (e) {
    e.preventDefault();
    var q = searchInput.value.trim();
    if (q) runSearch(q);
  });
  $("#foundClear").addEventListener("click", function () {
    fetch("/admin/search", { method: "DELETE" }).then(function () { return loadModels(); });
  });

  function loadModels(refresh) {
    return fetch("/admin/models" + (refresh ? "?refresh=1" : ""), { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        stg.data = d;
        stg.error = "";
        renderSettings();
        if (!d.unavailable && (d.building || !d.models || !d.models.length)) {
          if (stg.open && !d.unavailable) setTimeout(function () { if (stg.open) loadModels(); }, 1500);
        }
      })
      .catch(function () {
        stg.error = "Cannot reach the server.";
        renderSettings();
      });
  }

  function renderSettings() {
    var d = stg.data;
    if (!d) {
      setCurrentEl.textContent = "";
      modelListEl.textContent = "";
      setListNote.hidden = !stg.error;
      setListNote.textContent = stg.error;
      return;
    }
    var cur = d.current || {};
    var set = d.settings || {};
    dl(setCurrentEl, [
      ["Model", cur.name || "-"],
      ["Variant", cur.label || "-"],
      ["Speed", cur.tps == null ? "no reply yet" : cur.tps.toFixed(1) + " tok/s"],
      ["Images", cur.vision ? "supported" : "not supported"],
      ["Memory", set.memory_gb == null ? "no fixed limit" : set.memory_gb.toFixed(1) + " GB"]
    ]);
    var m = d.machine;
    setMachineEl.textContent = m
      ? [m.chip, (m.ram_gb || 0).toFixed(0) + " GB memory",
         m.gpu && m.gpu !== m.chip ? m.gpu : "",
         (m.free_gb || 0).toFixed(0) + " GB free for models"]
          .filter(Boolean).join(" · ")
      : "";
    if (!stg.dirty) syncOptions(set);
    renderModels(d);
    applyBtn.disabled = stg.busy || !cur.path;
    if (stg.busy) applyBtn.textContent = "Working...";
    else applyBtn.textContent = "Apply";
    paintAbout();
  }

  function syncOptions(set) {
    var fixed = set.memory_auto === false;
    memLimit.value = fixed ? "fixed" : "auto";
    memGb.hidden = !fixed;
    memUnit.hidden = !fixed;
    if (set.memory_gb != null) memGb.value = String(set.memory_gb);
    optFast.checked = !!set.fast;
    optThink.checked = !!set.thinking;
    paintMemNote();
  }

  function paintMemNote() {
    var d = stg.data || {};
    var cur = d.current || {};
    var plan = cur.memory || {};
    if (memLimit.value === "fixed") {
      memNote.textContent = "The engine keeps its own peak under this many GB. A smaller number leaves less " +
        "room for the experts in memory, so replies can be slower; the model still loads.";
    } else if (plan.gb != null) {
      memNote.textContent = "Auto: the launcher's choice for this computer and this model right now, " +
        plan.gb.toFixed(1) + " GB" + (plan.max_gb && plan.max_gb > plan.gb
          ? " and up to " + plan.max_gb.toFixed(1) + " GB while memory allows" : "") + ".";
    } else {
      memNote.textContent = "Auto: the launcher's own choice for this model, which sets no fixed limit here.";
    }
  }

  function currentJobFor(model, variant) {
    var job = stg.data && stg.data.job;
    if (!job || job.state === "idle") return null;
    return job.model === model.id && job.variant === variant.key ? job : null;
  }

  function renderModels(d) {
    modelListEl.textContent = "";
    if (d.unavailable) {
      setListNote.hidden = false;
      setListNote.textContent = "The model list is not available here: " + d.unavailable;
      return;
    }
    if (d.error) {
      setListNote.hidden = false;
      setListNote.textContent = "The launcher could not check the models: " + d.error;
    } else if (!d.models.length) {
      setListNote.hidden = false;
      setListNote.textContent = "Checking the models for this computer, their sizes and the speed here...";
    } else {
      setListNote.hidden = true;
      setListNote.textContent = "";
    }
    d.models.forEach(function (m) { modelListEl.appendChild(modelCard(m)); });
    renderFound(d.found);
  }

  function modelCard(m) {
    var card = el("div", "model");
    var head = el("div", "model-head");
    head.appendChild(el("div", "model-name", m.name + (m.tested ? "" : " (found)")));
    var about = [m.kind, m.about, m.license ? "licence " + m.license : ""].filter(Boolean).join(" · ");
    if (about) head.appendChild(el("div", "model-about", about));
    card.appendChild(head);
    if (m.unavailable) {
      card.appendChild(el("div", "model-off", m.unavailable));
      return card;
    }
    (m.not_made || []).forEach(function (line) {
      card.appendChild(el("div", "model-off", "Not made here: " + line));
    });
    m.variants.forEach(function (v) {
      card.appendChild(variantRow(m, v));
    });
    return card;
  }

  // The last search: its results as cards judged for this computer, above the tested models.
  function renderFound(found) {
    foundList.textContent = "";
    foundBox.hidden = !found;
    if (!found) return;
    foundTitle.textContent = "Results for “" + found.query + "”";
    found.models.forEach(function (m) { foundList.appendChild(modelCard(m)); });
    foundNote.textContent = found.skipped
      ? found.skipped + " more skipped: not MLX safetensors, an unknown model type, or gated."
      : "";
  }

  function runSearch(q) {
    searchBtn.disabled = true;
    setNote("");
    stg.searchNote = "Searching Hugging Face and judging the results for this computer...";
    paintSearchNote(false);
    return fetch("/admin/search?q=" + encodeURIComponent(q), { cache: "no-store" }).then(function (r) {
      if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function () {
      stg.searchNote = "";
      return loadModels(true);
    }).catch(function (e) {
      stg.searchNote = e instanceof TypeError ? "Cannot reach the server." : e.message;
      paintSearchNote(true);
    }).then(function () {
      searchBtn.disabled = false;
      if (!stg.searchNote) paintSearchNote(false);
    });
  }

  function paintSearchNote(bad) {
    searchNote.textContent = stg.searchNote || "";
    searchNote.classList.toggle("err", !!bad);
  }

  function variantRow(m, v) {
    var row = el("div", "variant" + (v.current ? " current" : ""));
    var head = el("div", "v-head");
    head.appendChild(el("span", "v-label", v.label));
    if (v.current) head.appendChild(el("span", "v-badge", "in use"));
    var speed = v.tps == null ? "" : v.tps_adjusted
      ? "~" + Math.round(v.tps) + " tok/s, adjusted to this computer's measured runs"
      : v.tps_lo && v.tps_hi && v.tps_lo < v.tps_hi * 0.85
        ? "~" + Math.round(v.tps_lo) + "-" + Math.round(v.tps_hi) + " tok/s (estimate)"
        : "~" + Math.round(v.tps) + " tok/s (estimate)";
    var bits = [fmtSize(v.size_bytes), speed, v.ok || v.disk_short ? v.status : ""].filter(Boolean).join(" · ");
    row.appendChild(head);
    row.appendChild(el("div", "v-meta" + (v.disk_short ? " short" : ""), bits));
    if (v.mtp) row.appendChild(el("p", "v-note", "its own MTP head is on this computer and is used: about 25% more"));
    var seen = v.measured_here;
    if (seen && seen.tps) {
      row.appendChild(el("p", "v-meas", "measured here: " + seen.tps.toFixed(1) + " tok/s (" + seen.n +
        (seen.n === 1 ? " reply, " : " replies, ") + seen.date + ")"));
    }
    if (v.own_path) row.appendChild(el("p", "v-note", "your own copy at " + v.own_path));
    if (v.note) row.appendChild(el("p", "v-note", v.note));
    if (!v.ok && v.why && !v.disk_short) row.appendChild(el("p", "v-note", v.why));
    if (v.after_unload) {
      row.appendChild(el("p", "v-note", "Fit and speed: after this model is unloaded."));
    }
    var job = currentJobFor(m, v);
    var btn = el("button", "mini-btn v-action");
    btn.type = "button";
    if (v.current) {
      btn.textContent = "In use";
      btn.disabled = true;
    } else if (v.disk_short) {
      btn.textContent = "Not enough disk";
      btn.disabled = true;
      btn.title = v.why || "";
    } else if (!v.ok) {
      btn.textContent = "Can't run here";
      btn.disabled = true;
      btn.title = v.why || "";
    } else if (job && job.state === "running") {
      btn.textContent = "Working";
      btn.disabled = true;
    } else if (v.ready) {
      btn.textContent = "Use";
      btn.classList.add("primary");
      btn.disabled = stg.busy;
      btn.addEventListener("click", function () { switchModel(m, v); });
    } else if (v.make_bits) {
      btn.textContent = "Make " + v.make_bits + "-bit copy";
      btn.disabled = !!(job && job.state === "running");
      btn.addEventListener("click", function () { startPrepare(m, v); });
    } else {
      btn.textContent = "Download";
      btn.disabled = !!(job && job.state === "running");
      btn.addEventListener("click", function () { startPrepare(m, v); });
    }
    row.appendChild(btn);
    // Only what the launcher downloaded or made can go, and never the model in use.
    if (v.disk_path && v.own && !v.current && !(job && job.state === "running")) row.appendChild(deleteRow(m, v));
    // The last prepare stays in view on its own row, never on the model in use.
    var show = job && !v.current && job.state !== "idle";
    if (show) row.appendChild(progressRow(job));
    return row;
  }

  function deleteRow(m, v) {
    var key = m.id + "/" + v.key;
    var wrap = el("div", "v-del");
    if (stg.delMsg && stg.delMsg.key === key) wrap.appendChild(el("span", "v-del-text err", stg.delMsg.text));
    if (stg.confirm !== key) {
      var ask = el("button", "mini-btn danger", "Delete");
      ask.type = "button";
      ask.addEventListener("click", function () { stg.confirm = key; stg.delMsg = null; renderSettings(); });
      wrap.appendChild(ask);
      return wrap;
    }
    wrap.appendChild(el("span", "v-del-text", "Delete " + v.label + " at " + v.disk_path + " (" +
      fmtSize(v.size_bytes) + ")?"));
    var yes = el("button", "mini-btn danger", "Yes, delete");
    yes.type = "button";
    yes.addEventListener("click", function () { deleteVariant(m, v, key); });
    var no = el("button", "mini-btn", "Cancel");
    no.type = "button";
    no.addEventListener("click", function () { stg.confirm = null; renderSettings(); });
    wrap.appendChild(yes);
    wrap.appendChild(no);
    return wrap;
  }

  function deleteVariant(m, v, key) {
    stg.confirm = null;
    fetch("/admin/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: m.id, variant: v.key })
    }).then(function (r) {
      if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function (d) {
      stg.delMsg = null;
      setNote(d.message || "Deleted.");
      return loadModels(true);
    }).catch(function (e) {
      stg.delMsg = { key: key, text: e instanceof TypeError ? "Cannot reach the server." : e.message };
      renderSettings();
    });
    renderSettings();
  }

  function progressRow(job) {
    var wrap = el("div", "progress");
    var bar = el("div", "bar");
    var fill = el("span");
    var ratio = job.total ? Math.min(1, job.bytes / job.total) : 0;
    fill.style.width = (ratio * 100).toFixed(1) + "%";
    bar.appendChild(fill);
    var text = el("span", "progress-text", progressText(job));
    var cancel = null;
    if (job.state === "running") {
      cancel = el("button", "mini-btn", "Stop");
      cancel.type = "button";
      cancel.addEventListener("click", cancelPrepare);
    }
    wrap.appendChild(bar);
    wrap.appendChild(text);
    if (cancel) wrap.appendChild(cancel);
    return wrap;
  }

  function progressText(job) {
    if (job.state === "done") return "Ready to use.";
    if (job.state === "cancelled") return "Stopped; the same choice picks it up where it stopped.";
    if (job.state === "failed") return "Stopped with an error: " + (job.error || "see the log");
    var parts = [fmtSize(job.bytes) + " of " + fmtSize(job.total)];
    var rate = fmtRate(job.rate_bps);
    if (rate) parts.push(rate);
    var eta = fmtEta(job.eta_s);
    if (eta) parts.push(eta);
    return parts.join(" · ");
  }

  function startPrepare(m, v) {
    setNote("");
    fetch("/admin/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: m.id, variant: v.key })
    }).then(function (r) {
      if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function (j) {
      if (stg.data) stg.data.job = j.job;
      renderSettings();
      pollPrepare();
    }).catch(function (e) {
      setNote(e instanceof TypeError ? "Cannot reach the server." : e.message, true);
    });
  }

  function pollPrepare() {
    clearTimeout(stg.timer);
    stg.timer = 0;
    if (!stg.open) return;
    stg.timer = setTimeout(function () {
      fetch("/admin/prepare", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (d) {
        var job = d.job || {};
        var was = stg.data && stg.data.job ? stg.data.job.state : "idle";
        if (stg.data) stg.data.job = job;
        if (job.state !== was) renderSettings();
        else if (job.state === "running") paintProgress(job);
        if (job.state === "running") pollPrepare();
        else if (job.state === "done" && was === "running") loadModels(true);
      }).catch(function () { if (stg.open) pollPrepare(); });
    }, 800);
  }

  function paintProgress(job) {
    var rows = document.querySelectorAll("#modelList .progress");
    Array.prototype.forEach.call(rows, function (row) {
      var text = row.querySelector(".progress-text");
      var fill = row.querySelector(".bar > span");
      if (text) text.textContent = progressText(job);
      if (fill) fill.style.width = (job.total ? Math.min(1, job.bytes / job.total) * 100 : 0).toFixed(1) + "%";
    });
  }

  function cancelPrepare() {
    fetch("/admin/prepare", { method: "DELETE" }).then(function (r) { return r.json(); }).then(function (d) {
      if (stg.data) stg.data.job = d.job;
      renderSettings();
    }).catch(function () { setNote("Cannot reach the server.", true); });
  }

  function optionBody() {
    var memory = memLimit.value === "fixed" ? Number(memGb.value) : "auto";
    return { memory: memory, fast: optFast.checked, thinking: optThink.checked };
  }

  function switchModel(m, v) {
    if (stg.busy) return;
    stg.busy = true;
    setNote("Loading " + m.name + "...");
    renderSettings();
    // The launcher's own flags for this model, not the options above, as its menu would start it.
    postLoad({ model: v.path, plan: true }, m.name + ", " + v.label);
  }

  function postLoad(body, what) {
    stg.busy = true;
    return fetch("/admin/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function (d) {
      stg.busy = false;
      stg.dirty = false;
      setNote("Running " + what + (what.indexOf("with these settings") < 0 ? "." : ""));
      planHint(d.plan);
      LC.refreshStatus();
      return loadModels(true);
    }).catch(function (e) {
      stg.busy = false;
      var msg = e instanceof TypeError ? "Cannot reach the server." : e.message;
      setNote("Could not switch: " + msg, true);
      renderSettings();
    });
  }

  function planHint(plan) {
    var at = plan && plan.at_start;
    var box = $("#planNote");
    if (!at || !at.length) { box.hidden = true; box.textContent = ""; return; }
    box.hidden = false;
    box.textContent = "The launcher would also start this model with " + at.join(", ") +
      ", which this server can only set when it starts.";
  }

  applyBtn.addEventListener("click", function () {
    var d = stg.data || {};
    var cur = d.current || {};
    if (!cur.path) { setNote("No model is running to apply this to.", true); return; }
    var body = optionBody();
    if (memLimit.value === "fixed" && !(Number(memGb.value) > 0)) {
      setNote("Write the memory limit in GB, or choose Auto.", true);
      return;
    }
    body.model = cur.path;
    body.plan = true;
    setNote("Restarting the engine...");
    postLoad(body, "with these settings");
  });

  memLimit.addEventListener("change", function () {
    stg.dirty = true;
    var fixed = memLimit.value === "fixed";
    memGb.hidden = !fixed;
    memUnit.hidden = !fixed;
    if (fixed && !(Number(memGb.value) > 0)) {
      var cur = (stg.data && stg.data.settings) || {};
      memGb.value = cur.memory_gb ? String(cur.memory_gb) : "8";
    }
    paintMemNote();
  });
  [optFast, optThink, memGb].forEach(function (c) {
    c.addEventListener("input", function () { stg.dirty = true; });
    c.addEventListener("change", function () { stg.dirty = true; });
  });


  // ---- memory now, chats and data, about this machine ------------------------------
  var srv = null;
  var chatNoteEl = $("#chatNote");

  function paintMemory() {
    var used = srv && srv.memory_gb, peak = srv && srv.memory_peak_gb, budget = srv && srv.memory_budget_gb;
    var ram = srv && srv.machine && srv.machine.ram_gb;
    var scale = budget || ram || Math.max(used || 0, peak || 0) * 1.25 || 1;
    dl($("#memNow"), [
      ["In use", used == null ? "-" : LC.fmtGB(used)],
      ["Peak since the server started", peak == null ? "-" : LC.fmtGB(peak)],
      ["Limit in force", budget ? LC.fmtGB(budget) + (srv.memory_budget_max_gb && srv.memory_budget_max_gb > budget
        ? ", up to " + LC.fmtGB(srv.memory_budget_max_gb) : "") : "none: the engine takes what it needs"],
      ["This computer", ram ? Math.round(ram) + " GB" : "-"]
    ]);
    var pct = used == null ? 0 : Math.min(100, used / scale * 100);
    $("#memGaugeFill").style.width = pct.toFixed(1) + "%";
    $("#memGauge").setAttribute("aria-valuenow", String(Math.round(pct)));
    $("#memGauge").classList.toggle("warn", !!budget && used != null && used > budget * 0.9);
  }

  function paintAbout() {
    var launcher = (stg.data && stg.data.machine) || {}, m = (srv && srv.machine) || {};
    var backend = { metal: "Metal", cuda: "CUDA", cpu: "CPU" }[m.backend] || m.backend;
    var rows = [["Computer", (srv && srv.host) || "-"], ["Chip", launcher.chip || m.chip || "-"],
                ["Memory", (launcher.ram_gb || m.ram_gb) ? Math.round(launcher.ram_gb || m.ram_gb) + " GB" : "-"]];
    var gpu = launcher.gpu || m.gpu;
    if (gpu && gpu !== (launcher.chip || m.chip)) rows.push(["GPU", gpu]);
    rows.push(["System", launcher.os || m.os || "-"], ["Engine", (backend || "-") + (m.mlx ? ", MLX " + m.mlx : "")]);
    if (launcher.free_gb != null) rows.push(["Free disk for models", Math.round(launcher.free_gb) + " GB"]);
    if (launcher.ssd_gbps) rows.push(["Disk read speed", launcher.ssd_gbps.toFixed(1) + " GB/s"]);
    if (launcher.conditions_text) rows.push(["Conditions now", launcher.conditions_text]);
    if (launcher.models_dir) rows.push(["Models folder", launcher.models_dir]);
    rows.push(["This page is served at", location.host]);
    if (srv) {
      rows.push(["Model folder", srv.model || "-"],
                ["Context length", srv.context_length ? LC.fmtTokens(srv.context_length) + " tokens" : "not in the model's config"]);
    }
    dl($("#aboutList"), rows);
  }

  function where() {
    var launcher = (stg.data && stg.data.machine) || {}, host = (srv && srv.host) || "the computer that runs the server";
    var store = LC.loadStore(), raw = LC.stored(LC.KEY) || "";
    var n = store.conversations.filter(function (c) { return c.messages.length; }).length;
    var kb = Math.max(1, Math.round(new Blob([raw]).size / 1024));
    dl($("#whereList"), [
      ["Your chats", "In this browser, for the address " + location.host + ": " + n + (n === 1 ? " chat, " : " chats, ") + kb +
        " KB. The server does not keep them."],
      ["Which model answered", "Saved with each chat: the model and the computer, shown under its title in the chat list."],
      ["Page choices", "The theme and the Thinking switch, in this browser, for this address."],
      ["The model and its options", "On " + host + ", in the running server. They are not saved in your browser."],
      ["Downloaded and made models", launcher.models_dir ? "On " + host + ", in " + launcher.models_dir + "." : "On " + host + ", in the launcher's models folder."],
      ["Reply statistics", "On " + host + ", in the server's memory: the last 200 replies, gone when it restarts."]
    ]);
  }

  function chats() {
    var store = LC.loadStore();
    var list = store.conversations.filter(function (c) { return c.messages.length; });
    var count = list.reduce(function (a, c) { return a + c.messages.filter(function (m) { return m.role !== "marker"; }).length; }, 0);
    $("#chatSummary").textContent = list.length
      ? list.length + (list.length === 1 ? " chat" : " chats") + " with " + LC.fmtNum(count) + " messages are saved in this browser."
      : "No chats are saved in this browser yet.";
    var groups = {};
    list.forEach(function (c) {
      var key = (c.model || "an unknown model") + (c.host ? " on " + c.host : "");
      groups[key] = groups[key] || { n: 0, mine: !!srv && !!c.model && c.model === LC.basename(srv.model) && (!c.host || c.host === srv.host) };
      groups[key].n++;
    });
    var ul = $("#chatGroups");
    ul.textContent = "";
    Object.keys(groups).sort().forEach(function (key) {
      var li = el("li");
      li.appendChild(el("span", "cg-name", key));
      li.appendChild(el("span", "cg-n", groups[key].n + (groups[key].n === 1 ? " chat" : " chats")));
      if (groups[key].mine) li.appendChild(el("span", "v-badge", "this server"));
      ul.appendChild(li);
    });
  }

  function chatNote(text, bad) {
    chatNoteEl.textContent = text || "";
    chatNoteEl.classList.toggle("err", !!bad);
  }

  $("#exportBtn").addEventListener("click", function () {
    var store = LC.loadStore();
    var list = store.conversations.filter(function (c) { return c.messages.length; }).map(LC.cleanConv);
    if (!list.length) { chatNote("There are no chats to export yet."); return; }
    var payload = {
      app: "local-chat", format: 2, exported_at: new Date().toISOString(),
      from: { address: location.host, model: srv ? LC.basename(srv.model) : null, host: srv ? srv.host : null },
      conversations: list
    };
    var url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    var a = el("a");
    a.href = url;
    a.download = "local-chat-" + new Date().toISOString().slice(0, 10) + ".json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
    chatNote("Exported " + list.length + (list.length === 1 ? " chat." : " chats."));
  });
  $("#importBtn").addEventListener("click", function () { $("#importFile").click(); });
  $("#importFile").addEventListener("change", function (e) {
    var file = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > 64 * 1024 * 1024) { chatNote("That file is larger than 64 MB.", true); return; }
    file.text().then(function (text) {
      var incoming = LC.readExport(text), store = LC.loadStore();
      var r = LC.mergeChats(store, incoming);
      LC.saveStore(store);
      chatNote("Imported " + r.added + (r.added === 1 ? " chat" : " chats") +
        (r.updated ? ", replaced " + r.updated + " with a newer copy" : "") +
        (r.kept ? ", skipped " + r.kept + " already here" : "") + ".");
      chats();
      where();
    }).catch(function (err) { chatNote(err.message || "Could not read that file.", true); });
  });

  function pollStatus() {
    LC.refreshStatus().then(function (s) {
      srv = s;
      paintMemory();
      paintAbout();
      chats();
      where();
    });
  }
  setInterval(function () { if (!document.hidden) pollStatus(); }, 5000);
  pollStatus();

  // The section list follows the section in view, and a clicked link stays marked while the page scrolls to it.
  (function () {
    var links = Array.prototype.slice.call(document.querySelectorAll("#sideNav a"));
    var panels = links.map(function (a) { return $(a.getAttribute("href")); }).filter(Boolean);
    var lockUntil = 0;
    function show(id) {
      links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + id); });
    }
    function mark() {
      if (Date.now() < lockUntil) return;
      var line = window.innerHeight * 0.3, cur = panels[0];
      panels.forEach(function (p) { if (p.getBoundingClientRect().top <= line) cur = p; });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) cur = panels[panels.length - 1];
      if (cur) show(cur.id);
    }
    links.forEach(function (a) {
      a.addEventListener("click", function () {
        lockUntil = Date.now() + 900;
        show(a.getAttribute("href").slice(1));
      });
    });
    window.addEventListener("scroll", mark, { passive: true });
    window.addEventListener("resize", mark);
    mark();
  })();

  LC.initTheme();
  loadModels();
  stg.timer = setTimeout(pollPrepare, 1200);
})();

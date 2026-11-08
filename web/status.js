(function () {
  "use strict";
  // The Status page: the server's /status and /admin/history as tiles, three charts and a few small tables.
  var $ = LC.$, el = LC.el, dl = LC.dl;
  var POLL_MS = 2500;
  var srv = null, rows = [], lastRequests = -1, timer = 0, offline = false;

  var speedChart = LC.lineChart($("#chartSpeed"), {
    title: "Decode speed per reply", color: "var(--viz-1)", average: true,
    fmt: LC.fmtTps, empty: "No replies yet. Send a message in the chat and the speed of each reply appears here.",
    tip: function (r) {
      return [replyHead(r), LC.fmtTps(r.decode_tok_s) + " decode", LC.fmtSecs(r.ttft_s) + " to the first token",
              LC.fmtTokens(r.completion_tokens) + " tokens written"];
    }
  });
  var firstChart = LC.lineChart($("#chartFirst"), {
    title: "Time to first token per reply", color: "var(--viz-2)", average: true,
    fmt: LC.fmtSecs, empty: "No replies yet. The wait for the first token of each reply appears here.",
    tip: function (r) {
      return [replyHead(r), LC.fmtSecs(r.ttft_s) + " to the first token",
              LC.fmtTokens(r.prompt_tokens) + " prompt tokens (" + LC.fmtTokens(r.cached_tokens) + " cached)"];
    }
  });
  var memOptions = {
    title: "Memory in use at each reply", color: "var(--viz-1)", height: 150, ref: null,
    fmt: LC.fmtGB, empty: "The memory in use at each reply appears here.",
    tip: function (r) { return [replyHead(r), LC.fmtGB(r.memory_gb) + " in use"]; }
  };
  var memChart = LC.lineChart($("#chartMemory"), memOptions);

  function clock(t, withDate) {
    var d = new Date(t * 1000);
    var hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return withDate ? d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + hm : hm;
  }
  function replyHead(r) {
    return "Reply " + r.n + " at " + new Date(r.t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function tiles() {
    var last = rows.length ? rows[rows.length - 1] : null;
    var speeds = rows.map(function (r) { return r.decode_tok_s; }).filter(function (v) { return v != null; });
    $("#tSpeed").textContent = last && last.decode_tok_s != null ? LC.fmtTps(last.decode_tok_s) : srv && srv.decode_tok_s != null ? LC.fmtTps(srv.decode_tok_s) : "-";
    $("#tSpeedSub").textContent = srv && srv.decode_tok_s != null
      ? "average " + LC.fmtTps(srv.decode_tok_s) + (speeds.length > 1 ? " · best " + LC.fmtTps(Math.max.apply(null, speeds)) : "")
      : "no replies yet";
    $("#tFirst").textContent = srv && srv.first_token_last_s != null ? LC.fmtSecs(srv.first_token_last_s) : "-";
    $("#tFirstSub").textContent = srv && srv.first_token_mean_s != null ? "average " + LC.fmtSecs(srv.first_token_mean_s) : "no replies yet";
    $("#tUp").textContent = srv ? LC.fmtUptime(srv.uptime_s) : "-";
    $("#tUpSub").textContent = srv && srv.started_at ? "since " + clock(srv.started_at, Date.now() / 1000 - srv.started_at > 43200) : "";
  }

  // Memory: in use against the budget, or against the computer's memory when no limit is set.
  function memory() {
    var used = srv && srv.memory_gb, peak = srv && srv.memory_peak_gb;
    var budget = srv && srv.memory_budget_gb, ram = srv && srv.machine && srv.machine.ram_gb;
    var scale = budget || ram || Math.max(used || 0, peak || 0) * 1.25 || 1;
    $("#tMem").textContent = used == null ? "-" : LC.fmtGB(used);
    var against = budget ? "of " + LC.fmtGB(budget) + " budget"
      : ram ? "no fixed budget · " + Math.round(ram) + " GB computer" : "no budget set";
    $("#tMemSub").textContent = used == null ? "" : against + (budget ? " · " + LC.fmtPct(used / budget * 100) : "");
    var fill = $("#gaugeFill"), gauge = $("#gauge"), peakEl = $("#gaugePeak");
    var pct = used == null ? 0 : Math.min(100, used / scale * 100);
    fill.style.width = pct.toFixed(1) + "%";
    gauge.setAttribute("aria-valuenow", String(Math.round(pct)));
    gauge.setAttribute("aria-valuetext", used == null ? "unknown" : LC.fmtGB(used) + " " + against);
    gauge.classList.toggle("warn", !!budget && used != null && used > budget * 0.9);
    peakEl.hidden = peak == null || peak <= (used || 0) + 0.02;
    if (!peakEl.hidden) peakEl.style.left = Math.min(100, peak / scale * 100).toFixed(1) + "%";
    var arena = (srv && srv.arena) || {}, store = (srv && srv.store) || {}, exp = (srv && srv.experts) || {};
    var held = arena.slots != null ? arena.slots + " slots, " + LC.fmtGB(arena.gb)
      : store.slots ? store.slots + " slots, " + LC.fmtGB((store.mb || 0) / 1000) : "-";
    dl($("#memStats"), [
      ["In use", used == null ? "-" : LC.fmtGB(used)],
      ["Peak", peak == null ? "-" : LC.fmtGB(peak)],
      ["Budget", budget ? LC.fmtGB(budget) + (srv.memory_budget_max_gb && srv.memory_budget_max_gb > budget ? ", up to " + LC.fmtGB(srv.memory_budget_max_gb) : "") : "none set"],
      ["Expert store", held],
      ["Expert reads", exp.rows_per_token != null ? Math.round(exp.rows_per_token) + " rows a token from the SSD" : "-"]
    ]);
  }

  // Conditions: what else on this computer changes the speed right now (the server's own reading).
  function conditions() {
    var c = (srv && srv.conditions) || {};
    var power = c.power === "battery" ? "on battery" : c.power === "ac" ? "on the power adapter" : "-";
    var tops = (c.top || []).map(function (t) { return t.name + " " + t.cpu.toFixed(1); }).join(", ");
    dl($("#condStats"), [
      ["Power", power],
      ["Low Power Mode", c.low_power == null ? "-" : c.low_power ? "on" : "off"],
      ["Thermal limit", c.thermal || "none reported"],
      ["Memory free", c.mem_free_gb == null ? "-" : LC.fmtGB(c.mem_free_gb)],
      ["Other apps", c.cpu_busy == null ? "-" : c.cpu_busy.toFixed(1) + " CPU cores" + (tops ? " (" + tops + ")" : "")],
      ["GPU", c.gpu_busy == null ? "-" : Math.round(c.gpu_busy) + "% in use by others"]
    ]);
    var line = srv && srv.conditions_text;
    $("#condLine").textContent = line || "Nothing on this computer is slowing the model down.";
  }

  function tokens() {
    var p = srv ? srv.prompt_tokens : 0, c = srv ? srv.cached_tokens : 0, o = srv ? srv.completion_tokens : 0;
    var fresh = Math.max(0, p - c), total = fresh + c + o;
    var bar = $("#tokenBar"), legend = $("#tokenLegend");
    bar.textContent = "";
    legend.textContent = "";
    var parts = [["new prompt", fresh, "seg-1"], ["cached prompt", c, "seg-2"], ["reply", o, "seg-3"]];
    bar.setAttribute("aria-label", "Tokens since the server started: " + parts.map(function (x) {
      return LC.fmtTokens(x[1]) + " " + x[0];
    }).join(", "));
    parts.forEach(function (x) {
      if (total && x[1]) {
        var seg = el("span", "seg " + x[2]);
        seg.style.flexGrow = String(x[1]);
        bar.appendChild(seg);
      }
      var li = el("li");
      li.appendChild(el("span", "swatch " + x[2]));
      li.appendChild(el("span", "lg-name", x[0]));
      li.appendChild(el("span", "lg-val", LC.fmtTokens(x[1])));
      legend.appendChild(li);
    });
    if (!total) bar.appendChild(el("span", "seg seg-empty"));
    var n = srv ? srv.requests : 0;
    dl($("#tokenStats"), [
      ["Prompt", LC.fmtTokens(p) + " tokens"],
      ["Served from cache", p ? LC.fmtTokens(c) + " (" + LC.fmtPct(c / p * 100) + ")" : "-"],
      ["Reply", LC.fmtTokens(o) + " tokens"],
      ["Per reply", n ? LC.fmtTokens(p / n) + " in, " + LC.fmtTokens(o / n) + " out" : "-"]
    ]);
  }

  function about() {
    var m = (srv && srv.machine) || {};
    var ctx = srv && srv.context_length;
    dl($("#modelStats"), [
      ["Model", srv ? LC.basename(srv.model) || "-" : "-"],
      ["Context length", ctx ? LC.fmtTokens(ctx) + " tokens" : "not in the config"],
      ["Images", srv ? (srv.vision ? "supported" : "not supported") : "-"],
      ["Replies served", srv ? LC.fmtNum(srv.requests) : "-"]
    ]);
    var backend = { metal: "Metal", cuda: "CUDA", cpu: "CPU" }[m.backend] || m.backend;
    var machine = [["Computer", (srv && srv.host) || "-"], ["Chip", m.chip || "-"],
                   ["Memory", m.ram_gb ? Math.round(m.ram_gb) + " GB" : "-"]];
    if (m.gpu) machine.push(["GPU", m.gpu]);
    machine.push(["System", m.os || "-"], ["Engine", (backend || "-") + (m.mlx ? ", MLX " + m.mlx : "")]);
    dl($("#machineStats"), machine);
  }

  function thisChat() {
    var conv = LC.activeConv(LC.loadStore());
    var done = conv.messages.filter(function (m) { return m.role === "assistant" && m.stats; });
    var sum = function (f) { return done.reduce(function (a, m) { return a + f(m.stats); }, 0); };
    var ctx = srv && srv.context_length;
    var counted = conv.messages.filter(function (m) { return m.role !== "marker"; }).length;
    var used = counted ? LC.contextUsed(conv.messages) : null;
    dl($("#chatStats"), [
      ["Chat", conv.title],
      ["Messages", LC.fmtNum(counted)],
      ["Prompt tokens", LC.fmtTokens(sum(function (s) { return s.prompt; }))],
      ["Reply tokens", LC.fmtTokens(sum(function (s) { return s.tokens; }))],
      ["Context now", used == null ? "-" : (used.exact ? "" : "~") + LC.fmtTokens(used.tokens) +
        (ctx ? " of " + LC.fmtTokens(ctx) + " (" + LC.fmtPct(used.tokens / ctx * 100) + ")" : " tokens")]
    ]);
  }

  function table() {
    var body = $("#replyTable tbody");
    body.textContent = "";
    rows.slice(-20).reverse().forEach(function (r) {
      var tr = el("tr");
      [r.n, clock(r.t), LC.fmtTokens(r.prompt_tokens), LC.fmtTokens(r.cached_tokens), LC.fmtTokens(r.completion_tokens),
       LC.fmtSecs(r.ttft_s), r.decode_tok_s == null ? "-" : LC.fmtTps(r.decode_tok_s), LC.fmtGB(r.memory_gb)].forEach(function (v) {
        tr.appendChild(el("td", null, String(v)));
      });
      body.appendChild(tr);
    });
  }

  function charts() {
    speedChart.set(rows.map(function (r) { return Object.assign({}, r, { y: r.decode_tok_s }); }));
    firstChart.set(rows.map(function (r) { return Object.assign({}, r, { y: r.ttft_s }); }));
    var budget = srv && srv.memory_budget_gb;
    memOptions.ref = budget ? { value: budget, label: "budget " + LC.fmtGB(budget) } : null;
    memChart.set(rows.map(function (r) { return Object.assign({}, r, { y: r.memory_gb }); }));
  }

  function setLive(ok) {
    offline = !ok;
    var badge = $("#liveBadge");
    badge.classList.toggle("off", !ok);
    $("#liveText").textContent = ok ? "Live · updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      : "Server not reachable";
  }

  function history() {
    return fetch("/admin/history", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (h) {
      var list = h.replies || [], first = Math.max(1, ((srv && srv.requests) || list.length) - list.length + 1);
      rows = list.map(function (r, i) { return Object.assign({ n: first + i }, r); });
      charts();
      table();
      tokens();
      tiles();
    });
  }

  function tick() {
    clearTimeout(timer);
    LC.refreshStatus().then(function (s) {
      srv = s;
      setLive(!!s);
      if (s) {
        memory();
        conditions();
        about();
        thisChat();
        tokens();
        tiles();
        if (s.requests !== lastRequests) {
          lastRequests = s.requests;
          return history().catch(function () {});
        }
      }
    }).then(function () {
      if (!document.hidden) timer = setTimeout(tick, POLL_MS);
    });
  }
  document.addEventListener("visibilitychange", function () { if (!document.hidden) tick(); });

  LC.initTheme();
  tick();
})();

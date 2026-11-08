(function () {
  "use strict";
  // What the chat, Settings and Status pages share: small DOM helpers, the theme menu and the /status reading.
  var LC = window.LC = {};

  LC.$ = function (s, root) { return (root || document).querySelector(s); };

  LC.el = function (tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  LC.dl = function (el, entries) {
    var key = JSON.stringify(entries);
    if (el._dlKey === key) return;
    el._dlKey = key;
    el.textContent = "";
    entries.forEach(function (r) {
      var t = document.createElement("dt"), v = document.createElement("dd");
      t.textContent = r[0];
      v.textContent = r[1];
      el.appendChild(t);
      el.appendChild(v);
    });
  };

  LC.basename = function (path) {
    return String(path || "").split(/[\\/]/).filter(Boolean).pop() || "";
  };

  LC.errorText = function (res) {
    return res.text().then(function (t) {
      try {
        var j = JSON.parse(t);
        return (j.error && j.error.message) || j.error || t;
      } catch (e) {
        return t || res.status + " " + res.statusText;
      }
    });
  };

  // ---- numbers with units ----------------------------------------------------
  LC.fmtNum = function (n) { return n == null ? "-" : Number(n).toLocaleString("en-US"); };
  // 842, 1.2k, 12.3k, 262k, 1.05M
  LC.fmtTokens = function (n) {
    if (n == null || !isFinite(n)) return "-";
    n = Math.max(0, n);
    if (n < 1000) return String(Math.round(n));
    if (n < 100000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    if (n < 1e6) return Math.round(n / 1000) + "k";
    return (n / 1e6).toFixed(2).replace(/\.?0+$/, "") + "M";
  };
  LC.fmtGB = function (gb) {
    if (gb == null || !isFinite(gb)) return "-";
    return (gb < 10 ? gb.toFixed(2) : gb.toFixed(1)) + " GB";
  };
  LC.fmtTps = function (t) { return t == null || !isFinite(t) ? "-" : t.toFixed(1) + " tok/s"; };
  LC.fmtPct = function (x) { return x == null || !isFinite(x) ? "-" : (x < 10 ? +x.toFixed(1) : Math.round(x)) + "%"; };
  // 0.5 s, 4.2 s, 18 s, 2 min 5 s
  LC.fmtSecs = function (s) {
    if (s == null || !isFinite(s)) return "-";
    if (s < 10) return s.toFixed(1) + " s";
    s = Math.round(s);
    return s < 60 ? s + " s" : Math.floor(s / 60) + " min " + (s % 60) + " s";
  };
  // Whole seconds, for a clock that ticks: 7 s, 2 min 5 s.
  LC.fmtElapsed = function (s) {
    if (s == null || !isFinite(s)) return "-";
    s = Math.floor(Math.max(0, s));
    return s < 60 ? s + " s" : Math.floor(s / 60) + " min " + (s % 60) + " s";
  };
  // 42 s, 5 min, 2 h 5 min, 3 d 4 h
  LC.fmtUptime = function (s) {
    if (s == null || !isFinite(s)) return "-";
    s = Math.round(s);
    if (s < 60) return s + " s";
    var m = Math.floor(s / 60);
    if (m < 60) return m + " min";
    var h = Math.floor(m / 60);
    if (h < 24) return h + " h " + (m % 60) + " min";
    return Math.floor(h / 24) + " d " + (h % 24) + " h";
  };

  // ---- storage ---------------------------------------------------------------
  LC.readJSON = function (key) {
    try { var raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : null; } catch (e) { return null; }
  };
  LC.stored = function (key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) {}
    return null;
  };

  // ---- api key ---------------------------------------------------------------
  // A shared device's server is local to the launcher but expects the key the menu put in localStorage.
  LC.apiKey = function () { return LC.stored("api.key") || ""; };
  LC.askApiKey = function () {
    var key = window.prompt("This shared model server asks for its api key.");
    if (key) LC.stored("api.key", key.trim());
    return LC.apiKey();
  };
  (function () {
    var realFetch = window.fetch;
    if (typeof realFetch !== "function") return;
    window.fetch = function (input, init) {
      init = init || {};
      var headers = new Headers(init.headers || {});
      var key = LC.apiKey();
      if (key && !headers.has("Authorization")) headers.set("Authorization", "Bearer " + key);
      function run() {
        return realFetch(input, Object.assign({}, init, { headers: headers })).then(function (res) {
          if (res.status === 401 && res.headers.get("X-Moe-Api-Key") === "required") {
            var again = LC.askApiKey();
            if (again && again !== key) {
              key = again;
              headers.set("Authorization", "Bearer " + key);
              return run();
            }
          }
          return res;
        });
      }
      return run();
    };
  })();

  // ---- saved chats -----------------------------------------------------------
  // They live in this browser's localStorage under chat.v2, so they belong to the address (and port) the page was opened at.
  LC.KEY = "chat.v2";
  var OLD_KEY = "chat.v1";

  LC.newId = function () {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  };
  LC.blankConversation = function () {
    return { id: LC.newId(), title: "New chat", created: Date.now(), updated: Date.now(), messages: [] };
  };
  // A message as it is saved: attached pictures become their count, and a compaction marker keeps its summary.
  LC.cleanMessage = function (m) {
    if (!m) return null;
    if (m.role === "marker") {
      if (typeof m.summary !== "string") return null;
      return { role: "marker", kind: "compact", summary: m.summary, folded: Number(m.folded) || 0,
               kept: Number(m.kept) || 0, at: Number(m.at) || 0 };
    }
    if (m.role !== "user" && m.role !== "assistant") return null;
    var out = { role: m.role, content: typeof m.content === "string" ? m.content : "" };
    if (typeof m.reasoning === "string" && m.reasoning) out.reasoning = m.reasoning;
    if (typeof m.meta === "string" && m.meta) out.meta = m.meta;
    if (typeof m.thinkSec === "number") out.thinkSec = m.thinkSec;
    if (m.stats && typeof m.stats === "object") out.stats = m.stats;
    var pictures = (m.images && m.images.length) || m.imageCount;
    if (pictures) out.imageCount = pictures;
    return out;
  };
  LC.titleFor = function (list) {
    for (var i = 0; i < list.length; i++) {
      if (list[i].role === "user") {
        var t = String(list[i].content || "").replace(/\s+/g, " ").trim();
        if (t) return t.slice(0, 60);
      }
    }
    return "";
  };
  // A compaction marker sits in the message list: everything before the last one is folded into its summary.
  LC.lastMarker = function (list) {
    for (var i = list.length - 1; i >= 0; i--) if (list[i].role === "marker") return i;
    return -1;
  };
  LC.estimateTokens = function (text) { return Math.ceil(String(text || "").length / 3.7); };
  // Tokens the next request would carry: the last reply's own count of its prompt plus what came after it, or an estimate.
  LC.contextUsed = function (list) {
    var mi = LC.lastMarker(list);
    var estimate = function (m) { return LC.estimateTokens(m.content) + (m.imageCount ? 256 * m.imageCount : 0) + 6; };
    var statsFrom = mi >= 0 ? mi + 1 + (list[mi].kept || 0) : 0, k = -1, j;
    for (j = list.length - 1; j >= statsFrom; j--) {
      if (list[j].role === "assistant" && list[j].stats && list[j].stats.prompt) { k = j; break; }
    }
    var tokens, exact;
    if (k >= 0) {
      var st = list[k].stats, c = (list[k].content || "").length, r = (list[k].reasoning || "").length;
      // The reasoning is not sent back, so only the answer's share of the reply counts.
      tokens = st.prompt + Math.round((st.tokens || 0) * (c + r ? c / (c + r) : 1));
      for (j = k + 1; j < list.length; j++) if (list[j].role !== "marker") tokens += estimate(list[j]);
      exact = k === list.length - 1 && !r;
    } else {
      tokens = mi >= 0 ? LC.estimateTokens(list[mi].summary) + 12 : 0;
      for (j = mi + 1; j < list.length; j++) tokens += estimate(list[j]);
      exact = false;
    }
    return { tokens: tokens, exact: exact };
  };

  // One saved conversation from anything the store or an imported file holds, or null.
  LC.cleanConv = function (c) {
    if (!c || typeof c !== "object") return null;
    var out = {
      id: String(c.id || LC.newId()),
      title: String(c.title || "New chat").slice(0, 120),
      created: Number(c.created) || Date.now(),
      updated: Number(c.updated) || Date.now(),
      messages: Array.isArray(c.messages) ? c.messages.map(LC.cleanMessage).filter(Boolean) : []
    };
    if (typeof c.model === "string" && c.model) out.model = c.model.slice(0, 120);
    if (typeof c.host === "string" && c.host) out.host = c.host.slice(0, 120);
    return out;
  };
  LC.loadStore = function () {
    var data = LC.readJSON(LC.KEY);
    if (!data || !Array.isArray(data.conversations)) {
      // One old single-thread page becomes the first saved conversation.
      var old = LC.readJSON(OLD_KEY), first = LC.blankConversation();
      if (Array.isArray(old)) first.messages = old.map(LC.cleanMessage).filter(Boolean);
      first.title = LC.titleFor(first.messages) || "New chat";
      return { conversations: [first], activeId: first.id };
    }
    var list = data.conversations.map(LC.cleanConv).filter(Boolean);
    if (!list.length) list = [LC.blankConversation()];
    var activeId = data.activeId;
    if (!list.some(function (c) { return c.id === activeId; })) activeId = list[0].id;
    return { conversations: list, activeId: activeId };
  };
  LC.saveStore = function (store) {
    var data = {
      activeId: store.activeId,
      conversations: store.conversations.map(function (c) {
        var out = { id: c.id, title: c.title, created: c.created, updated: c.updated };
        if (c.model) out.model = c.model;
        if (c.host) out.host = c.host;
        out.messages = c.messages.map(LC.cleanMessage).filter(Boolean);
        return out;
      })
    };
    LC.stored(LC.KEY, JSON.stringify(data));
  };
  // Add imported conversations to a store: new ones as they are, a known one only when the copy is newer.
  LC.mergeChats = function (store, incoming) {
    var added = 0, updated = 0, kept = 0;
    incoming.forEach(function (c) {
      var mine = store.conversations.find(function (x) { return x.id === c.id; });
      if (!mine) { store.conversations.push(c); added++; }
      else if (c.updated > mine.updated) { store.conversations[store.conversations.indexOf(mine)] = c; updated++; }
      else kept++;
    });
    return { added: added, updated: updated, kept: kept };
  };
  // The conversations in an exported file (an object with conversations, or the bare list), cleaned; throws with a plain reason.
  LC.readExport = function (text) {
    var data;
    try { data = JSON.parse(text); } catch (e) { throw new Error("That file is not JSON."); }
    var list = Array.isArray(data) ? data : data && Array.isArray(data.conversations) ? data.conversations : null;
    if (!list) throw new Error("That file has no chats in it.");
    var out = list.map(LC.cleanConv).filter(function (c) { return c && c.messages.length; });
    if (!out.length) throw new Error("That file has no chats with messages in it.");
    return out;
  };
  LC.activeConv = function (store) {
    return store.conversations.find(function (c) { return c.id === store.activeId; }) || store.conversations[0];
  };

  // ---- theme -----------------------------------------------------------------
  function resolveTheme(setting) {
    if (setting === "dark") return true;
    if (setting === "light") return false;
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  LC.initTheme = function () {
    var btn = LC.$("#themeBtn"), menu = LC.$("#themeMenu");
    if (!btn || !menu) return;
    var setting = document.documentElement.dataset.themeSetting || LC.stored("chat.theme") || "system";
    if (["system", "light", "dark"].indexOf(setting) < 0) setting = "system";
    function paint() {
      var dark = resolveTheme(setting);
      document.documentElement.dataset.theme = dark ? "dark" : "light";
      document.documentElement.dataset.themeSetting = setting;
      var color = LC.$('meta[name="theme-color"]');
      if (color) color.setAttribute("content", dark ? "#141416" : "#ffffff");
      Array.prototype.forEach.call(menu.querySelectorAll("[data-theme-choice]"), function (b) {
        b.setAttribute("aria-checked", b.getAttribute("data-theme-choice") === setting ? "true" : "false");
      });
      btn.title = "Theme: " + setting.charAt(0).toUpperCase() + setting.slice(1);
      btn.setAttribute("aria-label", btn.title);
    }
    function close() {
      menu.hidden = true;
      btn.setAttribute("aria-expanded", "false");
    }
    btn.addEventListener("click", function () {
      if (menu.hidden) {
        menu.hidden = false;
        btn.setAttribute("aria-expanded", "true");
      } else {
        close();
      }
    });
    Array.prototype.forEach.call(menu.querySelectorAll("[data-theme-choice]"), function (b) {
      b.addEventListener("click", function () {
        var choice = b.getAttribute("data-theme-choice");
        setting = ["system", "light", "dark"].indexOf(choice) < 0 ? "system" : choice;
        paint();
        LC.stored("chat.theme", setting);
        close();
        btn.focus();
      });
    });
    document.addEventListener("click", function (e) {
      if (!menu.hidden && !e.target.closest(".theme-wrap")) close();
    });
    if (window.matchMedia) {
      var query = window.matchMedia("(prefers-color-scheme: dark)");
      var onScheme = function () { if (setting === "system") paint(); };
      if (query.addEventListener) query.addEventListener("change", onScheme);
      else if (query.addListener) query.addListener(onScheme);
    }
    paint();
    LC.closeThemeMenu = function () {
      if (menu.hidden) return false;
      close();
      btn.focus();
      return true;
    };
  };

  // ---- the server's /status --------------------------------------------------
  LC.srv = null;
  LC.refreshStatus = function () {
    return fetch("/status", { cache: "no-store" }).then(function (r) { return r.json(); })
      .then(function (s) { LC.srv = s; return s; }, function () { LC.srv = null; return null; });
  };
})();

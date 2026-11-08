(function () {
  "use strict";
  var MAX_TOKENS = 2048;              // the reply limit sent with every request
  var MAX_THINK_TOKENS = 8192;        // with thinking on, the thinking counts too
  var MAX_SUMMARY_TOKENS = 1200;      // the limit for the summary a compaction asks for
  var KEEP_LAST = 2;                  // messages a compaction keeps word for word
  var MAX_IMAGE = 10 * 1024 * 1024;   // per image
  var MAX_IMAGES = 4;                 // per message
  var IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
  var WARN_AT = 0.8, FULL_AT = 0.95;  // context fill that turns the bar amber and red
  var COMPACT_PROMPT = "Write a summary of our conversation so far, so that it can replace the earlier messages " +
    "and the chat can go on from it. Keep what matters: the goal, facts and numbers, decisions, names, code and " +
    "file names, and anything still open. Write short notes in plain text, with no greeting and no closing remark.";
  var SUMMARY_LEAD = "Summary of the earlier part of this conversation, written by you:\n\n";
  var EXAMPLES = [
    { title: "Explain", text: "Explain how a hash map works, with a small code example." },
    { title: "Write code", text: "Write a Python function that merges overlapping intervals, and test it with three cases." },
    { title: "Compare", text: "Compare SQLite and Postgres for a small web app: when is each the better choice?" },
    { title: "Plan", text: "Help me plan three days in Lisbon with a relaxed pace and good food." }
  ];
  var $ = LC.$, el = LC.el, basename = LC.basename, errorText = LC.errorText;

  var log = $("#log"), msgs = $("#msgs"), box = $("#box"), sendBtn = $("#send");
  var modelEl = $("#model"), modelDot = $("#modelDot"), attachBtn = $("#attach");
  var thinkEl = $("#think");
  var convList = $("#convList");
  var scrim = $("#scrim"), menuBtn = $("#menuBtn"), confirmDialog = $("#confirmDialog");
  var compactBtn = $("#compactBtn"), clearBtn = $("#clearBtn");

  var messages = [];        // the open conversation's messages
  var ctl = null;           // AbortController while a reply is coming
  var epoch = 0;            // bumped when a reply's conversation is left behind
  var vision = false;       // whether the loaded model can look at images
  var pending = [];         // images attached to the message being written: {url}
  var loading = 0;          // images still being read
  var think = false;        // the Thinking switch
  var srv = null;           // the last /status
  var lastHost = null;      // the computer name the sidebar was last drawn for
  var confirmAction = null; // what the confirm dialog does on OK
  var working = null;       // the reply or compaction being streamed: {d, m, rate()}
  var warned = {};          // conversations whose context warning was already shown

  // ---- storage -------------------------------------------------------------
  var stored = LC.stored, titleFor = LC.titleFor, blankConversation = LC.blankConversation;
  var store = LC.loadStore();
  function activeConv() { return LC.activeConv(store); }
  function ensureTitle(conv) {
    if (conv.title && conv.title !== "New chat") return;
    var t = titleFor(conv.messages);
    if (!t) {
      t = conv.messages.some(function (m) { return m.imageCount; }) ? "Image chat" : "New chat";
    }
    conv.title = t;
  }
  function save() { LC.saveStore(store); }
  // A chat remembers the model and the computer that answered it.
  function tag(conv) {
    if (srv && srv.model) conv.model = basename(srv.model);
    if (srv && srv.host) conv.host = srv.host;
  }
  messages = activeConv().messages;

  // ---- sidebar -------------------------------------------------------------
  function closeDrawer() {
    document.body.classList.remove("drawer-open");
    scrim.hidden = true;
    menuBtn.setAttribute("aria-expanded", "false");
  }
  function openDrawer() {
    document.body.classList.add("drawer-open");
    scrim.hidden = false;
    menuBtn.setAttribute("aria-expanded", "true");
  }
  menuBtn.addEventListener("click", function () {
    if (document.body.classList.contains("drawer-open")) closeDrawer(); else openDrawer();
  });
  $("#closeSidebar").addEventListener("click", closeDrawer);
  scrim.addEventListener("click", closeDrawer);

  function icon(path) {
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    var p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", path);
    svg.appendChild(p);
    return svg;
  }
  var ICONS = {
    rename: "M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z",
    trash: "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v5M14 11v5",
    copy: "M9 9h10a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2Zm-4 6H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1",
    redo: "M3 12a9 9 0 0 1 15.5-6.2L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16M3 21v-5h5",
    edit: "M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"
  };

  function renderConvs() {
    convList.textContent = "";
    var sorted = store.conversations.slice().sort(function (a, b) { return b.updated - a.updated; });
    sorted.forEach(function (c) {
      var li = el("li", "conv-item" + (c.id === store.activeId ? " active" : ""));
      li.setAttribute("data-id", c.id);
      var open = el("button", "conv-open", c.title || "New chat");
      open.type = "button";
      open.title = c.title || "New chat";
      open.addEventListener("click", function () { selectConv(c.id); });
      var actions = el("div", "conv-actions");
      var rename = el("button", "icon-btn");
      rename.type = "button";
      rename.setAttribute("aria-label", "Rename conversation");
      rename.title = "Rename";
      rename.appendChild(icon(ICONS.rename));
      rename.addEventListener("click", function (e) { e.stopPropagation(); startRename(li, c); });
      var del = el("button", "icon-btn");
      del.type = "button";
      del.setAttribute("aria-label", "Delete conversation");
      del.title = "Delete";
      del.appendChild(icon(ICONS.trash));
      del.addEventListener("click", function (e) { e.stopPropagation(); askDelete(c); });
      actions.appendChild(rename);
      actions.appendChild(del);
      li.appendChild(open);
      li.appendChild(actions);
      // The model that answered it, and its computer when that is not this one.
      var who = [c.model, c.host && srv && srv.host && c.host !== srv.host ? c.host : ""].filter(Boolean).join(" · ");
      if (who) {
        var sub = el("span", "conv-sub", who);
        sub.title = "Answered by " + (c.model || "an unknown model") + (c.host ? " on " + c.host : "");
        li.appendChild(sub);
      }
      convList.appendChild(li);
    });
  }

  function startRename(li, conv) {
    var open = li.querySelector(".conv-open");
    if (!open) return;
    var input = document.createElement("input");
    input.className = "conv-title-input";
    input.type = "text";
    input.maxLength = 80;
    input.value = conv.title === "New chat" ? "" : conv.title;
    input.setAttribute("aria-label", "Conversation title");
    open.replaceWith(input);
    input.focus();
    input.select();
    var done = false;
    function finish(commit) {
      if (done) return;
      done = true;
      if (commit) {
        var value = input.value.replace(/\s+/g, " ").trim();
        conv.title = value || titleFor(conv.messages) || "New chat";
        save();
      }
      renderConvs();
    }
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    input.addEventListener("blur", function () { finish(true); });
  }

  function stopReply() {
    if (ctl) { epoch++; ctl.abort(); ctl = null; working = null; busy(false); }
  }

  function showConv() {
    messages = activeConv().messages;
    renderConvs();
    renderThread();
    paintContext();
  }

  function selectConv(id) {
    if (id === store.activeId) { closeDrawer(); return; }
    stopReply();
    store.activeId = id;
    save();
    showConv();
    closeDrawer();
    box.focus();
  }

  // One dialog for every question: its title, text and OK button are set each time.
  function askConfirm(title, text, ok, action) {
    $("#confirmTitle").textContent = title;
    $("#confirmText").textContent = text;
    $("#confirmOk").textContent = ok;
    confirmAction = action;
    if (typeof confirmDialog.showModal === "function") confirmDialog.showModal();
    else if (window.confirm(title)) { confirmAction = null; action(); }
  }
  confirmDialog.addEventListener("close", function () {
    var action = confirmAction;
    confirmAction = null;
    if (confirmDialog.returnValue === "ok" && action) action();
    confirmDialog.returnValue = "";
  });

  function askDelete(conv) {
    askConfirm("Delete chat?", '"' + (conv.title || "New chat") + '" will be removed from this browser.', "Delete",
      function () { removeConv(conv.id); });
  }

  function removeConv(id) {
    stopReply();
    store.conversations = store.conversations.filter(function (c) { return c.id !== id; });
    if (!store.conversations.length) store.conversations.push(blankConversation());
    if (store.activeId === id) store.activeId = store.conversations[0].id;
    save();
    showConv();
  }

  function newChat() {
    stopReply();
    var current = activeConv();
    if (current && !current.messages.length) { closeDrawer(); box.focus(); return false; }
    var conv = blankConversation();
    store.conversations.unshift(conv);
    store.activeId = conv.id;
    save();
    showConv();
    closeDrawer();
    box.focus();
    return true;
  }
  $("#new").addEventListener("click", newChat);

  // ---- markdown ------------------------------------------------------------
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  var purifierHooked = false;
  function sanitize(html) {
    if (!window.DOMPurify) return html;
    if (!purifierHooked) {
      window.DOMPurify.addHook("afterSanitizeAttributes", function (node) {
        if (node.tagName !== "A") return;
        var href = node.getAttribute("href") || "";
        var ok = false;
        try { ok = /^https?:$/.test(new URL(href, location.href).protocol); } catch (e) {}
        if (!ok) { node.removeAttribute("href"); return; }
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      });
      purifierHooked = true;
    }
    return window.DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true },
      FORBID_TAGS: ["style", "iframe", "object", "embed", "form", "input", "button", "textarea", "select"],
      FORBID_ATTR: ["style"]
    });
  }
  function markdown(src) {
    var text = String(src || "");
    if (!window.marked) return esc(text).replace(/\n/g, "<br>");
    try {
      return sanitize(window.marked.parse(text, { gfm: true, breaks: true }));
    } catch (e) {
      return esc(text).replace(/\n/g, "<br>");
    }
  }
  function decorateCode(root) {
    Array.prototype.forEach.call(root.querySelectorAll("pre > code"), function (code) {
      var pre = code.parentNode;
      if (!pre || !pre.parentNode) return;
      var lang = "";
      var m = String(code.className || "").match(/language-([\w+#.-]+)/);
      if (m) lang = m[1];
      var raw = code.textContent;
      var wrap = document.createElement("div");
      wrap.className = "code";
      var bar = document.createElement("div");
      bar.className = "bar";
      var label = document.createElement("span");
      label.className = "lang";
      label.textContent = lang || "code";
      var copy = document.createElement("button");
      copy.type = "button";
      copy.className = "copy";
      copy.textContent = "Copy";
      copy.setAttribute("aria-label", "Copy code");
      bar.appendChild(label);
      bar.appendChild(copy);
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(bar);
      wrap.appendChild(pre);
      if (window.hljs && lang && window.hljs.getLanguage(lang)) {
        try {
          code.innerHTML = window.hljs.highlight(raw, { language: lang, ignoreIllegals: true }).value;
          code.classList.add("hljs");
        } catch (e) {}
      }
    });
  }
  function copyText(text) {
    function legacy() {
      return new Promise(function (ok, fail) {
        var t = document.createElement("textarea");
        t.value = text;
        t.style.position = "fixed";
        t.style.opacity = "0";
        document.body.appendChild(t);
        t.select();
        var done = false;
        try { done = document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(t);
        done ? ok() : fail();
      });
    }
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text).catch(legacy);
    return legacy();
  }
  msgs.addEventListener("click", function (e) {
    var b = e.target.closest(".copy");
    if (!b) return;
    var block = b.closest(".code");
    var code = block && block.querySelector("code");
    if (!code) return;
    copyText(code.textContent).then(function () {
      b.textContent = "Copied";
      setTimeout(function () { b.textContent = "Copy"; }, 1500);
    }, function () { b.textContent = "Copy failed"; });
  });

  // ---- conversation view ----------------------------------------------------
  function nearBottom() { return log.scrollHeight - log.scrollTop - log.clientHeight < 90; }
  function toBottom() { log.scrollTop = log.scrollHeight; }
  function since(t0) { return t0 ? (performance.now() - t0) / 1000 : 0; }

  // The welcome screen: what this is and four prompts to start from.
  var welcomeFor = null;
  function whoIsThis() {
    return srv && srv.model ? basename(srv.model) + (srv.host ? " on " + srv.host : "") : "";
  }
  function emptyView() {
    var wrap = el("div", "empty");
    var mark = el("div", "mark");
    mark.setAttribute("aria-hidden", "true");
    mark.appendChild(icon("M21 12a8 8 0 0 1-8 8H6l-3 2 1-4.6A8 8 0 0 1 13 4a8 8 0 0 1 8 8ZM8 11h.01M12 11h.01M16 11h.01"));
    wrap.appendChild(mark);
    wrap.appendChild(el("h2", null, "How can I help?"));
    var who = whoIsThis();
    welcomeFor = who;
    wrap.appendChild(el("p", "empty-sub", (who ? "You are talking to " + who + ". " : "") +
      "Replies are written on this computer, and this chat stays in your browser."));
    var grid = el("div", "examples");
    EXAMPLES.forEach(function (ex) {
      var b = el("button", "example");
      b.type = "button";
      b.appendChild(el("span", "ex-title", ex.title));
      b.appendChild(el("span", "ex-text", ex.text));
      b.addEventListener("click", function () { box.value = ex.text; submit(); });
      grid.appendChild(b);
    });
    wrap.appendChild(grid);
    var hints = el("p", "hints");
    [["Enter", " to send"], ["Shift+Enter", " for a new line"], ["Esc", " to stop"], ["Ctrl/Cmd+K", " for a new chat"]]
      .forEach(function (h, i) {
        if (i) hints.appendChild(document.createTextNode(" · "));
        hints.appendChild(el("kbd", null, h[0]));
        hints.appendChild(document.createTextNode(h[1]));
      });
    wrap.appendChild(hints);
    return wrap;
  }

  // A blinking caret at the end of the text so far, inside the last text block or code line.
  function addCaret(body) {
    var host = body, deep = { P: 1, UL: 1, OL: 1, LI: 1, BLOCKQUOTE: 1, H1: 1, H2: 1, H3: 1, H4: 1, H5: 1, H6: 1,
                              STRONG: 1, EM: 1, DEL: 1, A: 1 };
    var last = host.lastElementChild;
    while (last && deep[last.tagName]) { host = last; last = host.lastElementChild; }
    if (last && last.classList.contains("code")) host = last.querySelector("code") || host;
    var c = document.createElement("span");
    c.className = "caret";
    c.setAttribute("aria-hidden", "true");
    host.appendChild(c);
  }

  function bubble(m) {
    var d = document.createElement("div");
    d._m = m;
    d.className = "msg " + m.role;
    var foot = '<div class="msg-foot"><div class="meta"></div><div class="actions"></div></div>';
    if (m.role === "assistant") {
      d.innerHTML = '<details class="think" hidden><summary><span class="t-dot" aria-hidden="true"></span>' +
        '<span class="t-label">Thinking</span><span class="t-time"></span></summary><div class="tbody"></div></details>' +
        '<div class="wait" hidden><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>' +
        '<span class="wait-text">Waiting for the first token</span><span class="wait-time"></span></div>' +
        '<div class="body"></div><div class="err" hidden><span class="err-text"></span></div>' + foot;
    } else {
      d.innerHTML = '<div class="body"></div>' + foot;
    }
    fill(d, m);
    return d;
  }

  function fill(d, m) {
    var body = d.querySelector(".body");
    if (m.role === "user") {
      body.textContent = "";
      if (m.images && m.images.length) {
        var row = document.createElement("div");
        row.className = "imgs";
        m.images.forEach(function (im, i) {
          var img = document.createElement("img");
          img.src = im.url;
          img.alt = "Attached image " + (i + 1);
          row.appendChild(img);
        });
        body.appendChild(row);
      } else if (m.imageCount) {
        var kept = document.createElement("div");
        kept.className = "kept";
        kept.textContent = m.imageCount + (m.imageCount === 1 ? " image was" : " images were") +
          " attached (not kept after a reload)";
        body.appendChild(kept);
      }
      if (m.content) {
        var text = document.createElement("div");
        text.className = "user-text";
        text.textContent = m.content;
        body.appendChild(text);
      }
    } else {
      var live = d.classList.contains("working");
      var thinkBox = d.querySelector(".think");
      if (thinkBox) {
        var thinking = live && !!m.reasoning && !m.content;
        thinkBox.hidden = !m.reasoning;
        if (m.reasoning) {
          var tbody = thinkBox.querySelector(".tbody");
          var follow = tbody.scrollHeight - tbody.scrollTop - tbody.clientHeight < 40;
          tbody.textContent = m.reasoning;
          if (thinking && follow) tbody.scrollTop = tbody.scrollHeight;
        }
        thinkBox.classList.toggle("is-thinking", thinking);
        thinkBox.querySelector(".t-label").textContent =
          thinking || m.thinkSec == null ? "Thinking" : "Thought for " + LC.fmtSecs(m.thinkSec);
        thinkBox.querySelector(".t-time").textContent = thinking ? " · " + LC.fmtElapsed(since(m.tThink0)) : "";
        d.classList.toggle("has-think", !!m.reasoning);
      }
      var wait = d.querySelector(".wait");
      if (wait) {
        wait.hidden = !(live && !m.content && !m.reasoning);
        d.classList.toggle("is-waiting", !wait.hidden);
      }
      body.innerHTML = markdown(m.content);
      decorateCode(body);
      if (live && m.content) addCaret(body);
    }
    var err = d.querySelector(".err");
    if (err) {
      err.querySelector(".err-text").textContent = m.error || "";
      err.hidden = !m.error;
      var retry = err.querySelector(".retry");
      if (m.error && !live && !retry) {
        retry = el("button", "mini-btn retry", "Try again");
        retry.type = "button";
        retry.addEventListener("click", function () {
          if (ctl) return;
          if (messages[messages.length - 1] === m) { regenerate(); return; }
          d.remove();
          ask();
        });
        err.appendChild(retry);
      }
    }
    d.querySelector(".meta").textContent = m.meta || "";
  }

  function actionButton(label, path, onClick) {
    var b = el("button", "icon-btn act");
    b.type = "button";
    b.setAttribute("aria-label", label);
    b.title = label;
    b.appendChild(icon(path));
    b.addEventListener("click", onClick);
    return b;
  }

  // Copy on every message, Edit on a message you wrote, Regenerate on the last reply (or Answer again on a last message with no reply).
  function paintActions() {
    var last = messages[messages.length - 1], mi = LC.lastMarker(messages);
    Array.prototype.forEach.call(msgs.querySelectorAll(".msg"), function (d) {
      var m = d._m, bar = d.querySelector(".actions");
      if (!m || !bar || d.classList.contains("working") || d.classList.contains("editing")) return;
      bar.textContent = "";
      var at = messages.indexOf(m);
      if (at < 0) return;
      bar.appendChild(actionButton("Copy message", ICONS.copy, function () {
        copyText(m.content || "").then(function () { note("Copied.", 1500); }, function () { note("Could not copy."); });
      }));
      if (m.role === "user" && at > mi) {
        bar.appendChild(actionButton("Edit and resend", ICONS.edit, function () { startEdit(d, m); }));
      }
      if (m === last) {
        bar.appendChild(actionButton(m.role === "user" ? "Answer again" : "Regenerate the reply", ICONS.redo, regenerate));
      }
    });
  }

  function markerView(m) {
    var wrap = el("div", "compaction");
    var line = el("div", "marker");
    line.setAttribute("role", "separator");
    line.appendChild(el("span", "rule"));
    line.appendChild(el("span", "label", "Conversation compacted · " + m.folded +
      (m.folded === 1 ? " message" : " messages") + " summarised"));
    line.appendChild(el("span", "rule"));
    wrap.appendChild(line);
    var det = el("details", "summary-view");
    det.appendChild(el("summary", null, "What the model keeps from them"));
    var body = el("div", "body");
    body.innerHTML = markdown(m.summary);
    det.appendChild(body);
    wrap.appendChild(det);
    return wrap;
  }

  function renderThread() {
    msgs.textContent = "";
    if (!messages.length) {
      msgs.classList.add("is-empty");
      msgs.appendChild(emptyView());
      return;
    }
    welcomeFor = null;
    msgs.classList.remove("is-empty");
    var mi = LC.lastMarker(messages);
    if (mi > 0) {
      var arch = el("details", "archive");
      arch.appendChild(el("summary", null, mi + (mi === 1 ? " earlier message" : " earlier messages") + ", folded"));
      var inner = el("div", "archive-body");
      messages.slice(0, mi).forEach(function (m) { inner.appendChild(m.role === "marker" ? markerView(m) : bubble(m)); });
      arch.appendChild(inner);
      msgs.appendChild(arch);
    }
    messages.forEach(function (m, i) {
      if (i < mi) return;
      msgs.appendChild(m.role === "marker" ? markerView(m) : bubble(m));
    });
    paintActions();
    toBottom();
  }

  function busy(on) {
    sendBtn.textContent = on ? "Stop" : "Send";
    sendBtn.classList.toggle("stop", on);
    sendBtn.setAttribute("aria-label", on ? "Stop the reply" : "Send the message");
    log.setAttribute("aria-busy", on ? "true" : "false");
    document.body.classList.toggle("generating", on);
    paintContext();
  }

  function grow() {
    box.style.height = "auto";
    box.style.height = Math.min(box.scrollHeight + 2, 200) + "px";
  }

  var noteTimer = 0;
  function note(text, ms) {
    var noteEl = $("#note");
    noteEl.textContent = text;
    clearTimeout(noteTimer);
    if (text) noteTimer = setTimeout(function () { noteEl.textContent = ""; }, ms || 8000);
  }

  // ---- how full the context is -----------------------------------------------
  function compactPlan() {
    var mi = LC.lastMarker(messages), start = mi + 1;
    if (messages.length - start < 4) return null;
    var cut = messages.length - KEEP_LAST;
    while (cut > start && messages[cut].role !== "user") cut--;
    if (cut - start < 2) return null;
    return { cut: cut, folded: cut - start };
  }

  function paintContext() {
    var limit = srv && srv.context_length;
    var u = LC.contextUsed(messages), used = u.tokens;
    if (working && working.m && !working.compacting) used += LC.estimateTokens(working.m.content);
    var ctx = $("#ctx"), fillEl = $("#ctxFill"), bar = $("#ctxBar");
    var ratio = limit ? Math.min(1, used / limit) : 0;
    var text = (used <= 0 || (u.exact && !working) ? "" : "~") + LC.fmtTokens(used) + (limit ? " / " + LC.fmtTokens(limit) : " tokens");
    if (limit) text += " · " + LC.fmtPct(ratio * 100);
    $("#ctxText").textContent = text;
    bar.hidden = !limit;
    fillEl.style.width = (ratio * 100).toFixed(1) + "%";
    bar.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
    bar.setAttribute("aria-valuetext", LC.fmtTokens(used) + " of " + (limit ? LC.fmtTokens(limit) : "an unknown number of") + " tokens");
    var level = ratio >= FULL_AT ? "full" : ratio >= WARN_AT ? "warn" : "";
    ctx.classList.toggle("warn", level === "warn");
    ctx.classList.toggle("full", level === "full");
    ctx.title = limit
      ? "About " + Math.round(used).toLocaleString("en-US") + " of " + limit.toLocaleString("en-US") + " tokens of context are used"
      : "About " + Math.round(used).toLocaleString("en-US") + " tokens of context are used; this model's limit is not known";
    var plan = compactPlan();
    compactBtn.disabled = !!ctl || !plan;
    compactBtn.classList.toggle("attention", !!level && !!plan);
    compactBtn.title = plan
      ? "Ask the model to summarise the conversation so far, then go on from the summary and the last messages"
      : "Nothing to compact yet: it needs at least two exchanges";
    clearBtn.disabled = !messages.length && !ctl;
    if (level && !warned[store.activeId] && !working) {
      warned[store.activeId] = true;
      note("The context is " + LC.fmtPct(ratio * 100) + " full. Compact keeps the gist of the chat and frees room.", 12000);
    }
  }

  // While a reply is coming: the seconds in the waiting line and the Thinking summary, a live speed, the context.
  setInterval(function () {
    if (!working) return;
    var d = working.d, m = working.m;
    var wt = d.querySelector(".wait-time");
    if (wt && !d.querySelector(".wait").hidden) {
      var w = since(m.t0);
      wt.textContent = w >= 2 ? " · " + LC.fmtElapsed(w) : "";
    }
    var tt = d.querySelector(".think.is-thinking .t-time");
    if (tt) tt.textContent = " · " + LC.fmtElapsed(since(m.tThink0));
    paintLive();
    paintContext();
  }, 250);

  // ---- one streamed request -------------------------------------------------------
  // Fills `run` (timing, usage, finish) and calls on(kind, text) for each piece; the promise ends when the stream does.
  function streamChat(body, run, on) {
    ctl = new AbortController();
    busy(true);
    function onEvent(raw) {
      var lines = raw.split(/\r?\n/), line = "";
      for (var i = 0; i < lines.length; i++) {
        if (lines[i].indexOf("data:") === 0) { line = lines[i]; break; }
      }
      if (!line) return;
      var data = line.slice(5).trim();
      if (data === "[DONE]") return;
      var j;
      try { j = JSON.parse(data); } catch (e) { return; }
      if (j.error) { run.error = j.error.message || String(j.error); return; }
      if (j.usage) run.usage = j.usage;
      var c = j.choices && j.choices[0];
      if (!c) return;
      if (c.finish_reason) run.finish = c.finish_reason;
      var thought = c.delta && c.delta.reasoning_content;
      var piece = c.delta && c.delta.content;
      if (thought || piece) {
        run.tLast = performance.now();
        if (!run.tFirst) run.tFirst = run.tLast;
        run.pieces++;
        if (thought) on("reasoning", thought);
        if (piece) on("content", piece);
      }
    }
    return fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: ctl.signal,
      body: JSON.stringify(body)
    }).then(function (res) {
      if (!res.ok) return errorText(res).then(function (t) { throw new Error(t); });
      var reader = res.body.getReader(), dec = new TextDecoder(), buf = "";
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) {
            if (buf.trim()) onEvent(buf);
            return;
          }
          buf += dec.decode(r.value, { stream: true });
          var parts = buf.split(/\r?\n\r?\n/);
          buf = parts.pop();
          parts.forEach(onEvent);
          return pump();
        });
      }
      return pump();
    }).catch(function (e) {
      if (e.name === "AbortError") run.finish = "aborted";
      else run.error = e instanceof TypeError ? "Cannot reach the server. Is it still running?" : e.message;
    });
  }

  function newRun() {
    return { t0: performance.now(), tFirst: 0, tLast: 0, pieces: 0, usage: null, finish: "", error: "" };
  }
  function rateOf(run) {
    return run.pieces > 2 && run.tLast > run.tFirst ? (run.pieces - 1) / ((run.tLast - run.tFirst) / 1000) : 0;
  }

  // A message as the API takes it: images first, then the text, as OpenAI's image_url parts.
  function wire(x) {
    if (x.images && x.images.length) {
      var parts = x.images.map(function (im) {
        return { type: "image_url", image_url: { url: im.url } };
      });
      if (x.content) parts.push({ type: "text", text: x.content });
      return { role: x.role, content: parts };
    }
    return { role: x.role, content: x.content || (x.imageCount ? "[image]" : "") };
  }
  // What the model is sent: after a compaction its summary stands in for everything before the marker.
  function wireMessages(list) {
    var mi = LC.lastMarker(list), out = [];
    if (mi >= 0) out.push({ role: "system", content: SUMMARY_LEAD + list[mi].summary });
    list.slice(mi + 1).forEach(function (x) { if (x.role !== "marker") out.push(wire(x)); });
    return out;
  }

  function clearFailed() {
    Array.prototype.forEach.call(msgs.querySelectorAll(".msg.failed"), function (n) { n.remove(); });
  }

  // ---- one reply ------------------------------------------------------------------
  function ask() {
    clearFailed();
    var m = { role: "assistant", content: "", meta: "" };
    var d = bubble(m), mine = epoch, asked = think, frame = 0, run = newRun();
    d.classList.add("working");
    msgs.appendChild(d);
    m.t0 = run.t0;
    working = { d: d, m: m, rate: function () { return rateOf(run); } };
    fill(d, m);
    toBottom();

    function paint() {
      frame = 0;
      var stick = nearBottom();
      fill(d, m);
      if (stick) toBottom();
    }
    return streamChat({
      model: "local", stream: true, stream_options: { include_usage: true },
      max_tokens: asked ? MAX_THINK_TOKENS : MAX_TOKENS,
      chat_template_kwargs: { enable_thinking: asked },
      messages: wireMessages(messages)
    }, run, function (kind, text) {
      if (kind === "reasoning") {
        if (!m.tThink0) m.tThink0 = run.tLast;
        m.reasoning = (m.reasoning || "") + text;
      } else {
        if (m.tThink0 && m.thinkSec == null) m.thinkSec = (run.tLast - m.tThink0) / 1000;
        m.content += text;
      }
      if (!frame) frame = requestAnimationFrame(paint);
    }).then(function () {
      if (frame) cancelAnimationFrame(frame);
      working = null;
      if (mine !== epoch) return;
      if (run.error) m.error = run.error;
      if (m.tThink0 && m.thinkSec == null) m.thinkSec = ((run.tLast || performance.now()) - m.tThink0) / 1000;
      d.classList.remove("working");
      d.classList.remove("is-waiting");
      var tokens = run.usage ? run.usage.completion_tokens : run.pieces;
      m.stats = {
        prompt: run.usage ? run.usage.prompt_tokens : 0,
        tokens: tokens,
        first: run.tFirst ? (run.tFirst - run.t0) / 1000 : 0,
        dtokens: 0,
        dsec: 0
      };
      var parts = [];
      if (run.tFirst) parts.push("first token " + ((run.tFirst - run.t0) / 1000).toFixed(1) + " s");
      if (run.tFirst && tokens > 1 && run.tLast > run.tFirst) {
        parts.push(((tokens - 1) / ((run.tLast - run.tFirst) / 1000)).toFixed(1) + " tok/s");
        m.stats.dtokens = tokens - 1;
        m.stats.dsec = (run.tLast - run.tFirst) / 1000;
      }
      if (run.tFirst) parts.push(tokens + " tokens");
      if (run.finish === "length") parts.push("stopped at the length limit");
      if (run.finish === "aborted") parts.push("stopped");
      m.meta = parts.join(" · ");
      var conv = activeConv();
      if (m.content) {
        messages.push(m);
        conv.updated = Date.now();
        tag(conv);
        ensureTitle(conv);
        save();
        renderConvs();
      } else if (m.error) {
        d.classList.add("failed");
      } else {
        // Stopped before anything came: nothing to keep, nothing to show.
        d.remove();
        note(run.finish === "aborted" ? "Stopped before the first token." : "The model sent nothing.", 5000);
      }
      if (d.isConnected) fill(d, m);
      ctl = null;
      busy(false);
      paintActions();
      box.focus();
      refresh();
    });
  }

  // Ask the model for a summary of the older messages, then keep it in their place, folded.
  function compact() {
    if (ctl) return;
    var plan = compactPlan();
    if (!plan) { note("There is not enough conversation to compact yet."); return; }
    clearFailed();
    var conv = activeConv(), mine = epoch, frame = 0, run = newRun();
    var m = { role: "assistant", content: "", meta: "" };
    var d = bubble(m);
    d.classList.add("working", "compacting");
    d.insertBefore(el("div", "compact-head", "Summarising " + plan.folded + " earlier messages"), d.firstChild);
    d.querySelector(".wait-text").textContent = "Waiting for the summary";
    msgs.appendChild(d);
    m.t0 = run.t0;
    working = { d: d, m: m, compacting: true, rate: function () { return rateOf(run); } };
    fill(d, m);
    toBottom();
    note("");

    function paint() {
      frame = 0;
      var stick = nearBottom();
      fill(d, m);
      if (stick) toBottom();
    }
    return streamChat({
      model: "local", stream: true, stream_options: { include_usage: true },
      max_tokens: MAX_SUMMARY_TOKENS, chat_template_kwargs: { enable_thinking: false },
      messages: wireMessages(messages.slice(0, plan.cut)).concat([{ role: "user", content: COMPACT_PROMPT }])
    }, run, function (kind, text) {
      if (kind === "content") m.content += text;
      if (!frame) frame = requestAnimationFrame(paint);
    }).then(function () {
      if (frame) cancelAnimationFrame(frame);
      working = null;
      if (mine !== epoch) return;
      ctl = null;
      busy(false);
      var summary = m.content.trim();
      if (run.finish === "aborted" || run.error || !summary) {
        d.remove();
        note(run.finish === "aborted" ? "Compaction stopped; the chat is as it was."
          : "Could not compact: " + (run.error || "the model wrote no summary") + ". The chat is as it was.");
        paintContext();
        return;
      }
      messages.splice(plan.cut, 0, {
        role: "marker", kind: "compact", summary: summary, folded: plan.folded,
        kept: messages.length - plan.cut, at: Date.now()
      });
      conv.updated = Date.now();
      save();
      renderConvs();
      renderThread();
      paintContext();
      note("Compacted " + plan.folded + " messages into a summary of about " +
        LC.fmtTokens(LC.estimateTokens(summary)) + " tokens.", 10000);
    });
  }

  // The last reply again, or an answer to a last message that never got one.
  function regenerate() {
    if (ctl) { note("Press Stop first."); return; }
    var last = messages[messages.length - 1];
    if (!last || last.role === "marker") return;
    if (last.role === "assistant") messages.pop();
    save();
    renderThread();
    ask();
  }

  // Edit a message you wrote and send it again: everything after it is dropped.
  function startEdit(d, m) {
    if (ctl) { note("Press Stop first, then edit."); return; }
    var body = d.querySelector(".body");
    d.classList.add("editing");
    d.querySelector(".actions").textContent = "";
    var area = el("textarea", "edit-box");
    area.value = m.content || "";
    area.rows = 3;
    area.setAttribute("aria-label", "Edit your message");
    var row = el("div", "edit-actions");
    var ok = el("button", "mini-btn primary", "Save and resend");
    var no = el("button", "mini-btn", "Cancel");
    ok.type = no.type = "button";
    row.appendChild(ok);
    row.appendChild(no);
    row.appendChild(el("span", "edit-note", "The replies after this message are removed."));
    body.textContent = "";
    body.appendChild(area);
    body.appendChild(row);
    area.focus();
    area.setSelectionRange(area.value.length, area.value.length);
    function fit() { area.style.height = "auto"; area.style.height = Math.min(area.scrollHeight + 2, 260) + "px"; }
    fit();
    area.addEventListener("input", fit);
    function cancel() { d.classList.remove("editing"); fill(d, m); paintActions(); }
    function resend() {
      var text = area.value.trim();
      if (!text && !(m.images && m.images.length)) { area.focus(); return; }
      resendFrom(m, text);
    }
    ok.addEventListener("click", resend);
    no.addEventListener("click", cancel);
    area.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancel(); }
      else if (e.key === "Enter" && (e.metaKey || e.ctrlKey || (!e.shiftKey && !matchMedia("(pointer: coarse)").matches))) {
        e.preventDefault();
        resend();
      }
    });
  }

  function resendFrom(m, text) {
    var i = messages.indexOf(m);
    if (i < 0 || ctl) return;
    var u = { role: "user", content: text };
    if (m.images && m.images.length) u.images = m.images;
    else if (m.imageCount) u.imageCount = m.imageCount;
    var conv = activeConv();
    var first = messages.findIndex(function (x) { return x.role === "user"; }) === i;
    messages.splice(i);
    messages.push(u);
    if (first) conv.title = "New chat";
    conv.updated = Date.now();
    tag(conv);
    ensureTitle(conv);
    save();
    renderConvs();
    renderThread();
    ask();
  }

  function submit() {
    if (ctl) { ctl.abort(); return; }
    var text = box.value.trim();
    if (!text && !pending.length) return;
    if (loading) { note("Still reading the images."); return; }
    clearFailed();
    var u = { role: "user", content: text };
    if (pending.length) u.images = pending;
    pending = [];
    showThumbs();
    note("");
    var conv = activeConv();
    conv.messages.push(u);
    conv.updated = Date.now();
    tag(conv);
    ensureTitle(conv);
    messages = conv.messages;
    save();
    renderConvs();
    if (messages.length === 1) {
      msgs.textContent = "";
      msgs.classList.remove("is-empty");
    }
    msgs.appendChild(bubble(u));
    box.value = "";
    grow();
    toBottom();
    ask();
  }

  // ---- images: the button, drag and drop, paste ----------------------------
  function showThumbs() {
    var wrap = $("#thumbs");
    wrap.textContent = "";
    pending.forEach(function (im, i) {
      var t = document.createElement("div");
      t.className = "thumb";
      var img = document.createElement("img");
      img.src = im.url;
      img.alt = "Attached image " + (i + 1);
      var x = document.createElement("button");
      x.type = "button";
      x.textContent = "×";
      x.setAttribute("aria-label", "Remove image " + (i + 1));
      x.addEventListener("click", function () { pending.splice(i, 1); showThumbs(); });
      t.appendChild(img);
      t.appendChild(x);
      wrap.appendChild(t);
    });
    wrap.hidden = !pending.length;
  }

  function addFiles(files) {
    if (!vision) { note("This model has no vision part, so images are not supported."); return; }
    Array.prototype.slice.call(files).forEach(function (f) {
      if (IMAGE_TYPES.indexOf(f.type) < 0) { note("Only PNG, JPEG, WebP and GIF images can be attached."); return; }
      if (f.size > MAX_IMAGE) { note((f.name || "The image") + " is larger than 10 MB."); return; }
      if (pending.length + loading >= MAX_IMAGES) { note("At most " + MAX_IMAGES + " images in one message."); return; }
      var r = new FileReader();
      loading++;
      r.onload = function () { loading--; pending.push({ url: r.result }); showThumbs(); };
      r.onerror = function () { loading--; note("Could not read " + (f.name || "the image") + "."); };
      r.readAsDataURL(f);
    });
  }

  function imagesOf(list) {
    return Array.prototype.filter.call(list || [], function (f) { return /^image\//.test(f.type); });
  }

  attachBtn.addEventListener("click", function () { $("#file").click(); });
  $("#file").addEventListener("change", function (e) { addFiles(e.target.files); e.target.value = ""; });
  document.addEventListener("paste", function (e) {
    var f = imagesOf(e.clipboardData && e.clipboardData.files);
    if (!f.length) return;
    e.preventDefault();
    addFiles(f);
  });
  var dragDepth = 0;
  function hasFiles(e) {
    return e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], "Files") >= 0;
  }
  document.addEventListener("dragenter", function (e) {
    if (hasFiles(e)) { dragDepth++; $("#drop").hidden = false; }
  });
  document.addEventListener("dragover", function (e) { if (hasFiles(e)) e.preventDefault(); });
  document.addEventListener("dragleave", function (e) {
    if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; $("#drop").hidden = true; }
  });
  document.addEventListener("drop", function (e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    $("#drop").hidden = true;
    addFiles(imagesOf(e.dataTransfer.files));
  });

  // ---- the live indicator in the header -------------------------------------
  function paintLive() {
    var pill = $("#liveBtn"), main = $("#liveMain"), sub = $("#liveSub");
    pill.classList.toggle("off", !srv);
    pill.classList.toggle("busy", !!working);
    var title;
    if (!srv) {
      main.textContent = "offline";
      sub.textContent = "";
      title = "The server is not reachable. Open the Status page.";
    } else if (working) {
      var rate = working.rate();
      main.textContent = rate ? LC.fmtTps(rate) : "writing";
      sub.textContent = rate ? "writing" : "";
      title = "A reply is being written. Open the Status page.";
    } else if (srv.decode_tok_s != null) {
      main.textContent = LC.fmtTps(srv.decode_tok_s);
      sub.textContent = srv.decode_replies ? srv.decode_replies + (srv.decode_replies === 1 ? " reply" : " replies")
        : srv.memory_gb != null ? LC.fmtGB(srv.memory_gb) : "";
      title = "Average decode speed " + LC.fmtTps(srv.decode_tok_s) +
        (srv.decode_replies ? " over " + srv.decode_replies + " replies" : "") +
        (srv.memory_gb != null ? ", memory in use " + LC.fmtGB(srv.memory_gb) : "") + ". Open the Status page.";
    } else {
      main.textContent = "ready";
      sub.textContent = srv.memory_gb != null ? LC.fmtGB(srv.memory_gb) : "";
      title = "No reply yet. Open the Status page.";
    }
    pill.title = title;
    pill.setAttribute("aria-label", "Status: " + main.textContent + (sub.textContent ? ", " + sub.textContent : "") + ". Open the Status page.");
  }

  // ---- the server's status and startup -------------------------------------------
  function refresh() {
    return LC.refreshStatus().then(function (s) {
      srv = s;
      if (s) {
        vision = !!s.vision;
        modelEl.textContent = basename(s.model) || "No model";
        $("#hostName").textContent = s.host || "";
        modelDot.classList.remove("offline");
        attachBtn.disabled = !vision;
        attachBtn.title = vision ? "Attach images" : "This model has no vision part, so images are not supported";
        attachBtn.setAttribute("aria-label", attachBtn.title);
        if (!vision && pending.length) {
          pending = [];
          showThumbs();
          note("This model has no vision part, so images are not supported.");
        }
        var conv = activeConv();
        if (conv.messages.length && !conv.model) { tag(conv); save(); }
      } else {
        modelEl.textContent = "Server offline";
        modelDot.classList.add("offline");
        attachBtn.disabled = true;
        attachBtn.title = "The server is not reachable, so images cannot be attached";
        attachBtn.setAttribute("aria-label", attachBtn.title);
      }
      paintLive();
      paintContext();
      // The sidebar names a chat's computer when it is not this one, so it is drawn again once this one is known.
      var host = s && s.host;
      if (host !== lastHost && !convList.querySelector(".conv-title-input")) { lastHost = host; renderConvs(); }
      // The welcome screen names the model once it is known.
      if (!messages.length && msgs.classList.contains("is-empty") && whoIsThis() !== welcomeFor) renderThread();
    });
  }
  setInterval(function () { if (!document.hidden && !ctl) refresh(); }, 5000);

  // Leaving the page would stop a reply being written: ask first.
  Array.prototype.forEach.call(document.querySelectorAll("#settingsBtn, #liveBtn, #footLink"), function (a) {
    a.addEventListener("click", function (e) {
      if (!ctl || e.metaKey || e.ctrlKey || e.shiftKey) return;
      e.preventDefault();
      var href = a.getAttribute("href");
      askConfirm("Leave while a reply is being written?", "Leaving stops the reply. What was written so far is not kept.",
        "Leave", function () { location.href = href; });
    });
  });

  $("#form").addEventListener("submit", function (e) { e.preventDefault(); submit(); });
  box.addEventListener("input", grow);
  box.addEventListener("keydown", function (e) {
    // Enter sends on a computer; on a touch screen it stays a new line.
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !matchMedia("(pointer: coarse)").matches) {
      e.preventDefault();
      if (!ctl) submit();
    }
  });
  thinkEl.addEventListener("change", function (e) {
    think = e.target.checked;
    stored("chat.think", think ? "1" : "0");
  });
  compactBtn.addEventListener("click", compact);
  clearBtn.addEventListener("click", function () {
    if (newChat()) note("Started a new chat with the same settings. The old one is in the list.", 6000);
  });
  document.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (!confirmDialog.open) newChat();
      return;
    }
    if (e.key !== "Escape" || confirmDialog.open) return;
    if (LC.closeThemeMenu()) return;
    if (document.body.classList.contains("drawer-open")) { closeDrawer(); menuBtn.focus(); return; }
    if (ctl) ctl.abort();
  });
  // Another tab of this page saved a chat: take it up when nothing is being written here.
  window.addEventListener("storage", function (e) {
    if (e.key !== LC.KEY || ctl) return;
    store = LC.loadStore();
    showConv();
  });

  think = stored("chat.think") === "1";
  thinkEl.checked = think;
  LC.initTheme();
  renderConvs();
  renderThread();
  paintContext();
  refresh();
  grow();
})();

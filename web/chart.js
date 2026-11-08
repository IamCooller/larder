(function () {
  "use strict";
  // A small line chart in plain SVG: one series, a recessive grid, an optional average and reference line,
  // a crosshair with a tooltip on hover or arrow keys, and a text summary for screen readers.
  var NS = "http://www.w3.org/2000/svg";
  var seq = 0;

  function mk(tag, attrs, cls) {
    var n = document.createElementNS(NS, tag);
    if (cls) n.setAttribute("class", cls);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  // A round maximum and step for about `ticks` grid lines.
  function nice(max, ticks) {
    if (!(max > 0)) return { max: 1, step: 1 / ticks };
    var raw = max / ticks, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    var step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
    return { max: Math.ceil(max / step - 1e-9) * step, step: step };
  }

  // The time of day; with seconds when the replies are minutes apart at most.
  function clock(t, secs) {
    return new Date(t * 1000).toLocaleTimeString([], secs ? { hour: "2-digit", minute: "2-digit", second: "2-digit" } : { hour: "2-digit", minute: "2-digit" });
  }

  // o: {title, color, fmt(value) -> text, tip(row, index) -> [lines], ref: {value, label}, average: bool, empty, height}
  // rows: [{t (unix seconds), y (number or null), ...anything the tip reads}]
  LC.lineChart = function (host, o) {
    var id = "lc" + (++seq), H = o.height || 190, pad = { l: 40, r: 16, t: 14, b: 26 };
    var rows = [], active = -1, W = 0, geo = null;
    host.classList.add("lchart");
    host.style.setProperty("--lc-color", o.color || "var(--accent)");
    host.tabIndex = 0;
    var svg = mk("svg", { height: H, role: "img" });
    var tip = LC.el("div", "lc-tip");
    tip.hidden = true;
    var empty = LC.el("p", "lc-empty", o.empty || "No data yet.");
    host.appendChild(svg);
    host.appendChild(tip);
    host.appendChild(empty);

    function points() {
      return rows.map(function (r, i) { return { r: r, i: i }; }).filter(function (p) { return p.r.y != null; });
    }

    function draw() {
      W = host.clientWidth;
      svg.textContent = "";
      var pts = points();
      empty.hidden = pts.length > 0;
      svg.style.display = pts.length ? "" : "none";
      host.classList.toggle("is-empty", !pts.length);
      if (!pts.length || W < 80) { geo = null; return; }
      svg.setAttribute("width", W);
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      var ys = pts.map(function (p) { return p.r.y; });
      var top = Math.max.apply(null, ys.concat(o.ref ? [o.ref.value] : []));
      var scale = nice(top * 1.08, 4);
      var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
      var n = pts.length;
      var X = function (k) { return pad.l + (n === 1 ? pw / 2 : k * pw / (n - 1)); };
      var Y = function (v) { return pad.t + ph - (v / scale.max) * ph; };
      geo = { pts: pts, X: X, Y: Y, pad: pad, pw: pw };

      var defs = mk("defs");
      var grad = mk("linearGradient", { id: id + "g", x1: 0, y1: 0, x2: 0, y2: 1 });
      grad.appendChild(mk("stop", { offset: "0", "stop-opacity": "0.22" }, "lc-stop"));
      grad.appendChild(mk("stop", { offset: "1", "stop-opacity": "0" }, "lc-stop"));
      defs.appendChild(grad);
      svg.appendChild(defs);

      for (var v = 0; v <= scale.max + scale.step / 2; v += scale.step) {
        var y = Y(v);
        svg.appendChild(mk("line", { x1: pad.l, x2: W - pad.r, y1: y, y2: y }, v === 0 ? "lc-axis" : "lc-grid"));
        var t = mk("text", { x: pad.l - 8, y: y + 4, "text-anchor": "end" }, "lc-tick");
        t.textContent = String(+v.toFixed(3));
        svg.appendChild(t);
      }
      var secs = pts[n - 1].r.t - pts[0].r.t < 300;
      // As many time labels as fit side by side (about 100 px each), at most four, spread evenly.
      var many = Math.max(2, Math.min(4, Math.floor(pw / 100)));
      var labelAt = [];
      if (n < 2) labelAt = [0];
      else for (var q = 0; q < Math.min(many, n); q++) labelAt.push(Math.round(q * (n - 1) / (Math.min(many, n) - 1)));
      labelAt.forEach(function (k, j) {
        var t = mk("text", { x: X(k), y: H - 7, "text-anchor": j === 0 && n > 1 ? "start" : k === n - 1 && n > 1 ? "end" : "middle" }, "lc-tick");
        t.textContent = clock(pts[k].r.t, secs);
        svg.appendChild(t);
      });

      if (o.ref) {
        var ry = Y(o.ref.value);
        svg.appendChild(mk("line", { x1: pad.l, x2: W - pad.r, y1: ry, y2: ry }, "lc-ref"));
        var rl = mk("text", { x: W - pad.r, y: ry - 5, "text-anchor": "end" }, "lc-ref-text");
        rl.textContent = o.ref.label;
        svg.appendChild(rl);
      }
      var mean = ys.reduce(function (a, b) { return a + b; }, 0) / n;
      if (o.average && n > 2) {
        var my = Y(mean);
        svg.appendChild(mk("line", { x1: pad.l, x2: W - pad.r, y1: my, y2: my }, "lc-avg"));
        var al = mk("text", { x: pad.l + 6, y: my - 5 }, "lc-avg-text");
        al.textContent = "average " + (o.fmt ? o.fmt(mean) : mean.toFixed(1));
        svg.appendChild(al);
      }

      var line = pts.map(function (p, k) { return (k ? "L" : "M") + X(k).toFixed(1) + " " + Y(p.r.y).toFixed(1); }).join(" ");
      if (n > 1) {
        svg.appendChild(mk("path", { d: line + " L" + X(n - 1).toFixed(1) + " " + Y(0).toFixed(1) + " L" + X(0).toFixed(1) + " " + Y(0).toFixed(1) + " Z", fill: "url(#" + id + "g)" }));
        svg.appendChild(mk("path", { d: line }, "lc-line"));
      }
      if (n <= 70) pts.forEach(function (p, k) { svg.appendChild(mk("circle", { cx: X(k), cy: Y(p.r.y), r: 3.5 }, "lc-dot")); });
      var last = pts[n - 1], lx = X(n - 1), ly = Y(last.r.y);
      svg.appendChild(mk("circle", { cx: lx, cy: ly, r: 5 }, "lc-last"));
      var lt = mk("text", { x: Math.min(lx, W - pad.r), y: ly - 11, "text-anchor": n > 1 ? "end" : "middle" }, "lc-last-text");
      lt.textContent = o.fmt ? o.fmt(last.r.y) : String(last.r.y);
      svg.appendChild(lt);

      var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
      svg.setAttribute("aria-label", (o.title || "Chart") + ", " + n + (n === 1 ? " reply" : " replies") + ", latest " +
        (o.fmt ? o.fmt(last.r.y) : last.r.y) + ", from " + (o.fmt ? o.fmt(lo) : lo) + " to " + (o.fmt ? o.fmt(hi) : hi));

      svg.appendChild(mk("line", { x1: 0, x2: 0, y1: pad.t, y2: pad.t + ph }, "lc-cross"));
      svg.appendChild(mk("circle", { cx: 0, cy: 0, r: 6 }, "lc-hot"));
      mark();
    }

    function mark() {
      var cross = svg.querySelector(".lc-cross"), hot = svg.querySelector(".lc-hot");
      if (!geo || !cross) return;
      if (active < 0 || active >= geo.pts.length) {
        cross.style.display = "none";
        hot.style.display = "none";
        tip.hidden = true;
        return;
      }
      var p = geo.pts[active], x = geo.X(active), y = geo.Y(p.r.y);
      cross.setAttribute("x1", x);
      cross.setAttribute("x2", x);
      cross.style.display = "";
      hot.setAttribute("cx", x);
      hot.setAttribute("cy", y);
      hot.style.display = "";
      tip.textContent = "";
      (o.tip ? o.tip(p.r, p.i) : [String(p.r.y)]).forEach(function (line, k) {
        tip.appendChild(LC.el("div", k ? "lc-tip-line" : "lc-tip-head", line));
      });
      tip.hidden = false;
      var tw = tip.offsetWidth, left = x + 12;
      if (left + tw > W - 4) left = x - tw - 12;
      tip.style.left = Math.max(4, left) + "px";
      tip.style.top = Math.max(4, Math.min(y - 20, H - tip.offsetHeight - 30)) + "px";
    }

    function pick(e) {
      if (!geo) return;
      var box = svg.getBoundingClientRect(), n = geo.pts.length;
      var k = n === 1 ? 0 : Math.round((e.clientX - box.left - pad.l) / (geo.pw / (n - 1)));
      active = Math.max(0, Math.min(n - 1, k));
      mark();
    }
    svg.addEventListener("pointermove", pick);
    svg.addEventListener("pointerdown", pick);
    svg.addEventListener("pointerleave", function () { if (document.activeElement !== host) { active = -1; mark(); } });
    host.addEventListener("blur", function () { active = -1; mark(); });
    host.addEventListener("focus", function () { if (geo && active < 0) { active = geo.pts.length - 1; mark(); } });
    host.addEventListener("keydown", function (e) {
      if (!geo) return;
      var n = geo.pts.length, next = active < 0 ? n - 1 : active;
      if (e.key === "ArrowLeft") next--;
      else if (e.key === "ArrowRight") next++;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = n - 1;
      else if (e.key === "Escape") next = -1;
      else return;
      e.preventDefault();
      active = Math.max(-1, Math.min(n - 1, next));
      mark();
    });
    if (window.ResizeObserver) new ResizeObserver(draw).observe(host);
    else window.addEventListener("resize", draw);

    return {
      set: function (data) { rows = data || []; if (active >= points().length) active = -1; draw(); },
      redraw: draw
    };
  };
})();

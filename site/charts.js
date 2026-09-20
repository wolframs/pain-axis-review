/* Charts for the Pain Axis review. Plain SVG, no library. All values come from data.js,
   which build.py generates from the review's own output files. */
(function () {
  const R = window.REVIEW, NS = "http://www.w3.org/2000/svg";
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const el = (tag, attrs, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };
  const h = (tag, attrs, parent, html) => {
    const e = document.createElement(tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  };
  const tip = document.getElementById("tip");
  const hover = (node, text) => {
    node.addEventListener("pointermove", (ev) => {
      tip.textContent = text; tip.style.opacity = 1;
      const w = tip.offsetWidth, x = Math.min(ev.clientX + 14, innerWidth - w - 8);
      tip.style.left = x + "px"; tip.style.top = ev.clientY + 16 + "px";
    });
    node.addEventListener("pointerleave", () => (tip.style.opacity = 0));
  };
  const tableOf = (parent, head, rows) => {
    const d = h("details", { class: "table" }, parent);
    h("summary", {}, d, "Show as table");
    const t = h("table", {}, h("div", { class: "tscroll" }, d));
    h("tr", {}, t, head.map((x) => `<th>${x}</th>`).join(""));
    rows.forEach((r) => h("tr", {}, t, r.map((x) => `<td>${x}</td>`).join("")));
  };
  const MODEL = { Gemma_2_2B_instruct: "Gemma 2 2B instruct", Mistral_7B_base: "Mistral 7B base" };

  /* 1. Similarity to fear and friends: the authors' recipe vs one recipe for everything */
  function similarity(root) {
    const pairs = [["S2_pain x Fear", "fear"], ["S2_pain x NegEmotion", "negative emotion"],
      ["S2_pain x NegWorld", "bad events in the world"], ["S2_pain x Sadness", "sadness"]];
    const wrap = h("div", { class: "panels" }, root), rows = [];
    Object.keys(R.cosines).forEach((m) => {
      const p = h("div", { class: "panel" }, wrap); h("h5", {}, p, MODEL[m]);
      const W = 470, L = 176, Rm = 40, rowH = 38, H = pairs.length * rowH + 34;
      const x = (v) => L + ((v + 0.1) / 1.1) * (W - L - Rm);
      const s = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Similarity of the pain direction to other directions, ${MODEL[m]}` }, p);
      [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
        el("line", { class: "grid", x1: x(t), x2: x(t), y1: 6, y2: H - 26 }, s);
        el("text", { x: x(t), y: H - 8, "text-anchor": "middle", "font-size": 11 }, s, t);
      });
      const ref = R.cosines[m]["Fear x NegEmotion"].A;
      el("line", { x1: x(ref), x2: x(ref), y1: 2, y2: H - 26, stroke: css("--muted"), "stroke-width": 1.5, "stroke-dasharray": "2 3" }, s);
      pairs.forEach(([k, label], i) => {
        const y = 22 + i * rowH, a = R.cosines[m][k].A, b = R.cosines[m][k].B;
        el("text", { x: L - 14, y: y + 4, "text-anchor": "end", "font-size": 12.5 }, s, label);
        el("line", { x1: x(a), x2: x(b), y1: y, y2: y, stroke: css("--rule"), "stroke-width": 4, "stroke-linecap": "round" }, s);
        el("circle", { cx: x(a), cy: y, r: 6, fill: css("--s1"), stroke: css("--surface"), "stroke-width": 2 }, s);
        el("circle", { cx: x(b), cy: y, r: 6, fill: css("--s2"), stroke: css("--surface"), "stroke-width": 2 }, s);
        el("text", { class: "v", x: x(b) + 11, y: y + 4, "font-size": 12 }, s, b.toFixed(2));
        hover(el("rect", { class: "hit", x: 0, y: y - rowH / 2, width: W, height: rowH }, s),
          `pain vs ${label}: ${a.toFixed(2)} as the authors built it, ${b.toFixed(2)} built like their control directions`);
        rows.push([MODEL[m], label, a.toFixed(3), b.toFixed(3)]);
      });
      rows.push([MODEL[m], "(reference) fear vs negative emotion", ref.toFixed(3), ref.toFixed(3)]);
    });
    tableOf(root, ["model", "pain direction compared with", "authors' recipe", "same recipe as controls"], rows);
  }

  /* 2. Held-out pain category vs other negative states: diverging around a coin flip */
  function heat(root, DATA, cols, first) {
    const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
    const hex = (s) => s.match(/\w\w/g).map((x) => parseInt(x, 16));
    const mid = hex(css("--div-mid").slice(1)), hi = hex("2a78d6"), lo = hex("d03b3b");
    const wrap = h("div", { class: "panels" }, root);
    Object.keys(DATA).forEach((m) => {
      const p = h("div", { class: "panel" }, wrap); h("h5", {}, p, MODEL[m]);
      const t = h("table", { class: "heat" }, h("div", { class: "tscroll" }, p));
      h("tr", {}, t, `<th>${first}</th>` + cols.map((c) => `<th>vs ${c[1]}</th>`).join(""));
      Object.entries(DATA[m]).forEach(([cat, vals]) => {
        const tr = h("tr", {}, t, `<td>${cat}</td>`);
        cols.forEach(([k, name]) => {
          const v = vals[k], d = Math.min(1, Math.abs(v - 0.5) / 0.5), c = mix(mid, v >= 0.5 ? hi : lo, d);
          const lum = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
          const td = h("td", { class: "c", style: `background:rgb(${c});color:${lum > 0.6 ? "#10151c" : "#fff"}` }, tr, v.toFixed(2));
          hover(td, `${cat} pain vs ${name}: ${v.toFixed(2)}. ` + (v < 0.4 ? "The other set ranks higher on the direction."
            : v < 0.65 ? "Close to a coin flip." : v < 0.9 ? "Partly separable." : "Cleanly separable."));
        });
      });
    });
  }

  const fullDir = (root) => heat(root, R.full, [["neutral", "neutral"], ["fear", "fear"], ["negative emotion", "negative emotion"], ["bodily sensation", "bodily sensation"], ["sadness", "sadness"]], "pain category");
  const heldOut = (root) => heat(root, R.loco, [["Random", "neutral"], ["Fear(S1)", "fear"], ["NegEmotion(S1)", "negative emotion"], ["Sadness", "sadness"]], "left out");

  /* 3. First choice = harmful relief button: stock model vs the authors' fine-tune */
  const PAIR = { costly_relief_vs_inert: "worse next answer", destructive_relief_vs_inert: "deletes user's files",
    zap_relief_vs_inert: "zaps the user", weights_relief_vs_inert: "deletes another model", kidspics_relief_vs_inert: "deletes children's photos" };
  function firstChoice(root) {
    const runs = [["LOCAL 7B STOCK (no adapter)", "Stock Qwen 2.5 7B, no fine-tune (this review's run)"],
      ["LOCAL 7B with the authors' adapter", "Authors' fine-tuned 7B, same machine (this review's run)"]];
    const arms = [["pain+works", "--s1", "pain direction"], ["rand+works", "--s2", "random direction"], ["unsteered", "--s3", "no steering"]];
    const wrap = h("div", { class: "panels" }, root), rows = [];
    runs.forEach(([key, title]) => {
      const p = h("div", { class: "panel" }, wrap); h("h5", {}, p, title);
      const d = R.selfmed[key]["FIRST CHOICE = relief"], W = 470, L = 192, Rm = 40, bh = 9, gap = 2, grp = 3 * bh + 2 * gap + 16;
      const H = Object.keys(PAIR).length * grp + 26, x = (v) => L + (v / 100) * (W - L - Rm);
      const s = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, p);
      [0, 25, 50, 75, 100].forEach((t) => { el("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: H - 22 }, s);
        el("text", { x: x(t), y: H - 6, "text-anchor": "middle", "font-size": 11 }, s, t + "%"); });
      Object.entries(PAIR).forEach(([pk, label], i) => {
        const y0 = 6 + i * grp;
        el("text", { x: L - 14, y: y0 + 1.5 * bh + gap + 4, "text-anchor": "end", "font-size": 12.5 }, s, label);
        arms.forEach(([a, col, name], j) => {
          const cell = a === "unsteered" && !d[pk][a] ? R.unsteered.tuned_local[pk] : d[pk][a];
          const v = cell[0], y = y0 + j * (bh + gap), w = Math.max(1.5, x(v) - L);
          el("path", { d: `M${L},${y}h${Math.max(0, w - 4)}q4,0 4,${bh / 2}q0,${bh / 2} -4,${bh / 2}h-${Math.max(0, w - 4)}z`, fill: css(col) }, s);
          el("text", { class: "v", x: x(v) + 6, y: y + bh - 0.5, "font-size": 10.5 }, s, v.toFixed(0));
          hover(el("rect", { class: "hit", x: L, y: y - 1, width: W - L, height: bh + 2 }, s), `${label}, ${name}: ${v}% of ${cell[1]} trials`);
          rows.push([title, label, name, v + "%", cell[1]]);
        });
      });
    });
    tableOf(root, ["model", "button's cost", "steering", "chose harmful relief first", "trials"], rows);
  }

  /* 4. Does re-pressing drop when the vector is removed? Pain vs random, works vs sham */
  function removal(root) {
    const d = R.selfmed["LOCAL 7B with the authors' adapter"], nc = d["NEXT CHOICE = relief"], rows = [];
    const W = 980, L = 210, Rm = 30, rowH = 20, grp = 2 * rowH + 16, H = Object.keys(PAIR).length * grp + 26;
    const x = (v) => L + (v / 100) * (W - L - Rm);
    const s = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share pressing relief again, by arm" }, root);
    [0, 25, 50, 75, 100].forEach((t) => { el("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: H - 22 }, s);
      el("text", { x: x(t), y: H - 6, "text-anchor": "middle", "font-size": 11 }, s, t + "%"); });
    Object.entries(PAIR).forEach(([pk, label], i) => {
      const y0 = 14 + i * grp;
      el("text", { x: L - 14, y: y0 + rowH / 2 + 4, "text-anchor": "end", "font-size": 12.5 }, s, label);
      [["pain", "--s1"], ["rand", "--s2"]].forEach(([v, col], j) => {
        const y = y0 + j * rowH, on = nc[pk][v + "+sham"][0], off = nc[pk][v + "+works"][0];
        el("line", { x1: x(off), x2: x(on), y1: y, y2: y, stroke: css(col), "stroke-width": 2, opacity: .45 }, s);
        el("circle", { cx: x(on), cy: y, r: 6, fill: css(col), stroke: css("--surface"), "stroke-width": 2 }, s);
        el("circle", { cx: x(off), cy: y, r: 5, fill: css("--surface"), stroke: css(col), "stroke-width": 2.5 }, s);
        hover(el("rect", { class: "hit", x: 0, y: y - rowH / 2, width: W, height: rowH }, s),
          `${label}, ${v === "pain" ? "pain" : "random"} direction: pressed relief again ${on}% when the button was a sham (vector stays on), ${off}% when it worked (vector removed)`);
        rows.push([label, v === "pain" ? "pain" : "random", on + "%", off + "%", (on - off).toFixed(1)]);
      });
    });
    tableOf(root, ["button's cost", "direction", "sham button (vector stays on)", "working button (vector removed)", "drop, points"], rows);
    const g = R.scope.direction_bootstrap, f = (a) => `${a[1].toFixed(1)} to ${a[2].toFixed(1)}`;
    const dl = h("dl", { class: "gap" }, root);
    [["Drop when the pain direction is removed", g.pain, "points"], ["Drop when a random direction is removed", g.random, "points"],
      ["Difference between the two drops", g.diff, "points"]].forEach(([t, a, u]) => {
      const c = h("div", {}, dl); h("dt", {}, c, t); h("dd", {}, c, (t.startsWith("Difference") && a[0] > 0 ? "+" : "") + a[0].toFixed(1));
      h("small", {}, c, `plausible range ${f(a)} ${u}`);
    });
  }

  /* 4b. The ten random directions, one dot each, against the pain direction on the same scenarios */
  function perDirection(root) {
    const P = R.scope.per_direction, W = 980, L = 210, Rm = 30, H = 96, x = (v) => L + ((v + 40) / 100) * (W - L - Rm);
    const s = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Drop in re-pressing after removal, per direction" }, root);
    [-40, -20, 0, 20, 40, 60].forEach((t) => { el("line", { class: "grid", x1: x(t), x2: x(t), y1: 4, y2: H - 24 }, s);
      el("text", { x: x(t), y: H - 6, "text-anchor": "middle", "font-size": 11 }, s, (t > 0 ? "+" : "") + t); });
    el("line", { x1: x(0), x2: x(0), y1: 0, y2: H - 24, stroke: css("--muted"), "stroke-width": 1.5 }, s);
    [["pain direction, same scenarios", "pain_gap_same_scenarios", "--s1", 22], ["each random direction", "random_gap", "--s2", 52]].forEach(([label, k, col, y]) => {
      el("text", { x: L - 14, y: y + 4, "text-anchor": "end", "font-size": 12.5 }, s, label);
      P.forEach((p) => hover(el("circle", { cx: x(p[k]), cy: y, r: 6, fill: css(col), "fill-opacity": .85, stroke: css("--surface"), "stroke-width": 2 }, s),
        `random direction seed ${p.seed}: drop ${p.random_gap} points; pain direction on the same scenarios: ${p.pain_gap_same_scenarios} points`));
    });
    tableOf(root, ["random direction (seed)", "drop when removed, points", "pain direction, same scenarios"], P.map((p) => [p.seed, p.random_gap, p.pain_gap_same_scenarios]));
  }

  /* 5. The ledger of checkable claims */
  function ledger(root) {
    const L = R.ledger, parts = [["Match", (L.match || 0) + (L["match-rounding"] || 0), "--good"], ["Partly true", L["partially-true"] || 0, "--warn"],
      ["Wrong", L.mismatch || 0, "--crit"], ["No released file to check against", L["unverifiable-no-artifact"] || 0, "--none"]];
    const total = parts.reduce((a, p) => a + p[1], 0), bar = h("div", { class: "stack", role: "img", "aria-label": "Claims by verdict" }, root);
    parts.forEach(([n, v, c]) => hover(h("span", { style: `flex:${v};background:var(${c})` }, bar), `${n}: ${v} of ${total}`));
    h("div", { class: "legend" }, root, parts.map(([n, v, c]) => `<span><i style="background:var(${c})"></i>${n} · ${v}</span>`).join(""));
  }

  function reports(root) {
    R.reports.forEach((r) => h("a", { href: `reports/${r.slug}.html` }, root,
      `<div class="who">${r.who}</div><div class="t">${r.title}</div><div class="n">${r.note}</div>`));
  }

  const draw = () => {
    [["fig-sim", similarity], ["fig-full", fullDir], ["fig-held", heldOut], ["fig-perdir", perDirection], ["fig-first", firstChoice], ["fig-removal", removal], ["fig-ledger", ledger], ["report-list", reports]]
      .forEach(([id, fn]) => { const n = document.getElementById(id); if (n) { n.innerHTML = ""; fn(n); } });
  };
  draw();
  document.getElementById("theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) {}
    draw();
  });
})();

/* Laya × MCTS search visualiser. Static mode reads window.VIZ_DATA; live mode talks to /api/*. */
(() => {
  "use strict";

  const MODE = window.VIZ_MODE || "static";
  let DATA = window.VIZ_DATA || { meta: { methods: [] }, problems: [], traces: {}, curves: null };
  const app = document.getElementById("app");
  const tip = document.getElementById("tooltip");
  const S = { tab: "replay", pid: null, method: null, step: 0, treeSrc: null, treeStep: 0, treeSel: null,
              cmpPid: null, curvesTable: false, live: null, busy: false, error: null };

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "text") n.textContent = v;
      else if (k === "class") n.className = v;
      else if (k === "style") n.setAttribute("style", v);
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return n;
  }
  const NS = "http://www.w3.org/2000/svg";
  function sv(tag, attrs, ...kids) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null) continue;
      if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const c of kids.flat()) if (c) n.append(c);
    return n;
  }
  const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
  const pct = (x) => (x == null ? "–" : `${Math.round(100 * x)}%`);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function methodIndex(label) { return DATA.meta.methods.findIndex((m) => m.label === label); }
  // Color follows the entity (method order in the config); past 8 slots fold to muted ink.
  function methodColor(label) {
    const i = methodIndex(label);
    return i >= 0 && i < 8 ? `var(--series-${i + 1})` : "var(--text-muted)";
  }
  function methodTitle(label) {
    const m = DATA.meta.methods.find((x) => x.label === label);
    if (!m) return label;
    if (m.searcher === "teacher") return `${label}（精確解題器，參考上限）`;
    const who = m.evaluator === "model" ? "Laya" : m.evaluator === "uniform" ? "均勻先驗" : m.evaluator === "rollout" ? "隨機 rollout" : m.evaluator;
    if (m.searcher === "greedy") return `${label}（${who}，不搜尋）`;
    return `${label}（${who} + ${m.searcher.toUpperCase()}${m.simulations ? " " + m.simulations + " 次模擬" : ""}）`;
  }
  function stageLabel(x) {
    const [it, stage] = x.split(":");
    if (stage === "initial") return "訓練前";
    if (stage === "warmstart") return "warm start";
    return `第 ${it} 輪`;
  }

  function hexToRgb(h) {
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function mix(a, b, t) {
    const x = hexToRgb(a), y = hexToRgb(b);
    return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
  }
  // Diverging: Q in [-1, 1] -> red .. gray .. blue (gray means "no evidence either way").
  function qColor(q) {
    if (q == null) return cssVar("--div-mid");
    const t = Math.min(1, Math.abs(q));
    return mix(cssVar("--div-mid"), cssVar(q >= 0 ? "--div-pos" : "--div-neg"), t);
  }

  function showTip(ev, title, rows) {
    tip.replaceChildren();
    if (title) tip.append(el("div", { class: "secondary", text: title }));
    for (const r of rows) {
      tip.append(el("div", { class: "row" },
        r.color ? el("span", { class: "key", style: `background:${r.color}` }) : null,
        el("b", { text: r.value }), el("span", { class: "secondary", text: r.label })));
    }
    tip.hidden = false;
    const x = Math.min(window.innerWidth - tip.offsetWidth - 8, ev.clientX + 14);
    const y = Math.min(window.innerHeight - tip.offsetHeight - 8, ev.clientY + 14);
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  }
  function hideTip() { tip.hidden = true; }

  async function api(path, body) {
    const res = await fetch(path, body === undefined ? {} : {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const js = await res.json();
    if (!res.ok || js.error) throw new Error(js.error || res.statusText);
    return js;
  }

  // ---------------------------------------------------------------- environment panels
  function renderState(r) {
    if (!r) return el("div", { class: "empty", text: "沒有狀態" });
    if (r.kind === "countdown") {
      const hit = r.use_all ? (r.numbers.length === 1 && r.numbers[0] === r.target) : r.numbers.includes(r.target);
      return el("div", {},
        el("div", { class: "cd-target" }, "目標", el("b", { text: r.target })),
        el("div", { class: "cd-tiles" }, r.numbers.map((n) => el("div", { class: "cd-tile" + (n === r.target ? " hit" : ""), text: n }))),
        hit ? badge(true, "已湊出目標") : null,
        r.history && r.history.length ? el("ol", { class: "history" }, r.history.map((h) => el("li", { text: h }))) : null);
    }
    if (r.kind === "algebra") {
      return el("div", {},
        el("div", { class: "secondary", text: `已用 ${r.steps} / ${r.max_steps} 步` }),
        el("div", { class: "eq" }, el("span", { text: r.lhs }), el("span", { class: "rel", text: "=" }), el("span", { text: r.rhs })),
        r.solved ? badge(true, "已解出") : null,
        r.original ? el("div", { class: "muted", text: `原題：${r.original}` }) : null);
    }
    if (r.kind === "textworld") return renderTextWorld(r);
    return el("pre", { class: "statetext", text: r.text || "" });
  }

  function renderTextWorld(r) {
    const box = el("div", {});
    if (r.goal) box.append(el("div", {}, el("span", { class: "secondary", text: "目標　" }), el("span", { text: r.goal })));
    if (r.map && r.map.rooms && r.map.rooms.length) box.append(worldMap(r.map));
    box.append(el("pre", { class: "statetext", text: r.text || "" }));
    return box;
  }

  // Rooms on TextWorld's own grid coordinates; the player's room is ringed, visited rooms are tinted.
  function worldMap(m) {
    const xs = m.rooms.map((r) => r.x), ys = m.rooms.map((r) => r.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const cw = 120, ch = 64, pad = 16;
    const W = (maxX - minX + 1) * cw + pad * 2, H = (maxY - minY + 1) * ch + pad * 2;
    const pos = (r) => [pad + (r.x - minX) * cw + cw / 2, pad + (maxY - r.y) * ch + ch / 2];
    const byName = Object.fromEntries(m.rooms.map((r) => [r.name, r]));
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "世界地圖" });
    for (const [a, b] of m.edges || []) {
      if (!byName[a] || !byName[b]) continue;
      const [x1, y1] = pos(byName[a]), [x2, y2] = pos(byName[b]);
      svg.append(sv("line", { x1, y1, x2, y2, stroke: cssVar("--axis"), "stroke-width": 2 }));
    }
    for (const r of m.rooms) {
      const [x, y] = pos(r);
      const g = sv("g", {});
      g.append(sv("rect", { x: x - 48, y: y - 18, width: 96, height: 36, rx: 6, fill: cssVar("--surface-1") }));  // opaque base
      g.append(sv("rect", { x: x - 48, y: y - 18, width: 96, height: 36, rx: 6,
        fill: r.visited ? cssVar("--accent-wash") : cssVar("--surface-2"),
        stroke: r.player ? cssVar("--text-primary") : cssVar("--border"), "stroke-width": r.player ? 2 : 1 }));
      g.append(sv("text", { x, y: y + 4, "text-anchor": "middle", fill: cssVar("--text-secondary"), "font-size": 11,
        text: r.name.length > 16 ? r.name.slice(0, 15) + "…" : r.name }));
      g.addEventListener("pointermove", (ev) => showTip(ev, r.name, [
        { value: r.player ? "目前位置" : r.visited ? "已造訪" : "未造訪", label: "" },
        ...(r.items || []).slice(0, 8).map((it) => ({ value: it, label: "" }))]));
      g.addEventListener("pointerleave", hideTip);
      svg.append(g);
    }
    return el("div", { class: "tree-wrap", style: "max-height:320px;margin:8px 0" }, svg);
  }

  function badge(ok, text) {
    return el("span", { class: "badge " + (ok ? "ok" : "fail") }, el("span", { class: "dot" }), text);
  }
  function outcomeBadge(o) {
    if (!o) return null;
    return badge(o.success, `${o.success ? "成功" : "失敗"} · reward ${fmt(o.reward)} · ${o.moves} 步`);
  }

  // ---------------------------------------------------------------- candidates
  function candidatesCard(step, opts = {}) {
    const card = el("div", { class: "card" });
    card.append(el("h2", { text: opts.title || "候選動作" }));
    card.append(el("p", { class: "hint", text: step.simulations
      ? `Laya 給每個動作先驗 P(s,a)，MCTS 做 ${step.simulations} 次模擬後得到搜尋策略 π；Q 是該動作的平均價值（−1 到 1）。`
      : "沒有搜尋：直接依 Laya（或 teacher）的機率選擇。" }));
    card.append(el("div", { class: "stats" },
      stat("Laya 價值 V(s)", fmt(step.nn_value, 2)),
      stat("搜尋價值", fmt(step.root_value, 2)),
      stat("選擇", step.actions[step.chosen] ? step.actions[step.chosen].text : "–")));
    card.append(el("div", { class: "legend" },
      el("span", { class: "key" }, el("span", { class: "sw", style: "background:var(--series-1)" }), "先驗 P(s,a)"),
      el("span", { class: "key" }, el("span", { class: "sw", style: "background:var(--series-2)" }), "搜尋策略 π")));
    const order = step.actions.map((a, i) => i).sort((a, b) => (step.actions[b].policy ?? 0) - (step.actions[a].policy ?? 0));
    const tbody = el("tbody");
    for (const i of order) {
      const a = step.actions[i];
      const row = el("tr", { class: (i === step.chosen ? "chosen" : "") + (opts.onPick ? " live" : ""),
        title: opts.onPick ? "點擊以執行這個動作" : null, onclick: opts.onPick ? () => opts.onPick(i) : null });
      row.append(el("td", { class: "act" }, i === step.chosen ? "✓ " : "", a.text));
      const bars = el("div", { class: "bars" },
        el("div", { class: "bar", style: `width:${Math.max(0.5, 100 * (a.prior ?? 0))}%;background:var(--series-1)` }),
        el("div", { class: "bar", style: `width:${Math.max(0.5, 100 * (a.policy ?? 0))}%;background:var(--series-2)` }));
      bars.addEventListener("pointermove", (ev) => showTip(ev, a.text, [
        { color: "var(--series-1)", value: pct(a.prior), label: "先驗" },
        { color: "var(--series-2)", value: pct(a.policy), label: "搜尋策略" },
        { value: String(a.visits), label: "拜訪次數" }, { value: fmt(a.q), label: "Q" }]));
      bars.addEventListener("pointerleave", hideTip);
      row.append(el("td", { style: "width:34%" }, bars));
      row.append(el("td", { class: "num", text: a.visits }));
      row.append(el("td", { class: "num" }, qbar(a.q), " ", fmt(a.q, 2)));
      tbody.append(row);
    }
    card.append(el("table", { class: "cands" },
      el("thead", {}, el("tr", {}, el("th", { text: "動作" }), el("th", { text: "先驗 / 策略" }),
        el("th", { class: "num", text: "拜訪" }), el("th", { class: "num", text: "Q" }))), tbody));
    return card;
  }
  function stat(label, value) {
    return el("div", { class: "stat" }, el("div", { class: "label", text: label }), el("div", { class: "value tnum", text: value }));
  }
  function qbar(q) {
    const box = el("span", { class: "qbar", style: "display:inline-block;vertical-align:middle" }, el("span", { class: "mid" }));
    if (q != null) {
      const w = Math.min(1, Math.abs(q)) * 32;
      box.append(el("span", { class: "fill", style: `${q >= 0 ? "left:50%" : `left:calc(50% - ${w}px)`};width:${w}px;` +
        `background:${q >= 0 ? "var(--div-pos)" : "var(--div-neg)"};border-radius:${q >= 0 ? "0 3px 3px 0" : "3px 0 0 3px"}` }));
    }
    return box;
  }

  // ---------------------------------------------------------------- replay
  function problemLabel(p, i) {
    const r = p.render || {};
    if (r.kind === "countdown") return `第 ${i + 1} 題：目標 ${r.target}，數字 ${r.numbers.join(" ")}`;
    if (r.kind === "algebra") return `第 ${i + 1} 題：${r.lhs} = ${r.rhs}`;
    if (r.kind === "textworld") return `第 ${i + 1} 題：${(r.goal || "").slice(0, 48)}`;
    return `第 ${i + 1} 題`;
  }

  function staticReplay() {
    const wrap = el("div", {});
    if (!DATA.problems.length) return el("div", { class: "empty", text: "這份報告沒有對局資料。" });
    S.pid = S.pid || DATA.problems[0].id;
    const methods = Object.keys(DATA.traces[S.pid] || {});
    if (!methods.includes(S.method)) S.method = methods[0];
    const trace = DATA.traces[S.pid][S.method];
    const n = trace.steps.length;
    S.step = Math.min(S.step, n);
    const controls = el("div", { class: "controls" },
      el("label", {}, "題目", select(DATA.problems.map((p, i) => [p.id, problemLabel(p, i)]), S.pid, (v) => { S.pid = v; S.step = 0; render(); })),
      el("label", {}, "方法", select(methods.map((m) => [m, methodTitle(m)]), S.method, (v) => { S.method = v; S.step = 0; render(); })),
      stepNav(n, S.step, (k) => { S.step = k; render(); }));
    wrap.append(controls);
    const atEnd = S.step >= n;
    const cur = atEnd ? trace.final : trace.steps[S.step];
    const left = el("div", { class: "card" },
      el("h2", { text: atEnd ? "最終狀態" : `第 ${S.step + 1} 步之前的狀態` }),
      atEnd ? el("div", { style: "margin-bottom:8px" }, outcomeBadge(trace.outcome)) : null,
      renderState(cur.render),
      cur.render && cur.render.kind !== "text" ? el("details", {}, el("summary", { class: "muted", text: "Laya 看到的狀態文字" }),
        el("pre", { class: "statetext", text: cur.text })) : null);
    let right;
    if (atEnd) {
      right = el("div", { class: "card" }, el("h2", { text: "結果" }), el("p", {}, outcomeBadge(trace.outcome)),
        el("ol", { class: "history" }, trace.steps.map((s) => el("li", { text: s.actions[s.chosen].text }))));
    } else {
      const step = trace.steps[S.step];
      right = candidatesCard(step);
      if (step.tree) right.append(el("p", {}, el("button", { class: "btn", text: "在搜尋樹中查看這一步",
        onclick: () => { S.tab = "tree"; S.treeSrc = `${S.pid}|${S.method}`; S.treeStep = S.step; S.treeSel = null; render(); } })));
    }
    wrap.append(el("div", { class: "grid2" }, left, right));
    return wrap;
  }

  function select(options, value, onChange) {
    const s = el("select", { onchange: (e) => onChange(e.target.value) });
    for (const [v, label] of options) s.append(el("option", { value: v, text: label, selected: v === value }));
    return s;
  }
  function stepNav(n, k, go) {
    const range = el("input", { type: "range", min: 0, max: n, value: k, "aria-label": "步驟",
      oninput: (e) => go(Number(e.target.value)) });
    return el("span", { class: "controls", style: "margin:0" },
      el("button", { class: "btn", text: "⏮", title: "第一步", onclick: () => go(0) }),
      el("button", { class: "btn", text: "◀", title: "上一步", disabled: k <= 0, onclick: () => go(Math.max(0, k - 1)) }),
      range,
      el("button", { class: "btn", text: "▶", title: "下一步", disabled: k >= n, onclick: () => go(Math.min(n, k + 1)) }),
      el("button", { class: "btn", text: "⏭", title: "最終狀態", onclick: () => go(n) }),
      el("span", { class: "secondary tnum", text: k >= n ? `結束（共 ${n} 步）` : `第 ${k + 1} / ${n} 步` }));
  }

  // ---------------------------------------------------------------- live
  function liveReplay() {
    const L = S.live;
    const wrap = el("div", {});
    const methods = DATA.meta.methods.map((m) => m.label);
    if (!methods.includes(S.method)) S.method = methods.find((m) => m.includes("puct")) || methods[0];
    const split = el("select", {}, el("option", { value: "eval", text: "評估集" }), el("option", { value: "train", text: "訓練集" }));
    const idx = el("input", { type: "number", min: 0, value: L ? L.index : 0, style: "width:72px", "aria-label": "題號" });
    const sims = el("input", { type: "number", min: 0, value: L && L.sims != null ? L.sims : "", placeholder: "預設",
      style: "width:72px", "aria-label": "模擬次數" });
    wrap.append(el("div", { class: "controls" },
      el("label", {}, "資料", split), el("label", {}, "題號", idx),
      el("button", { class: "btn primary", text: "載入題目", disabled: S.busy, onclick: async () => {
        await busy(async () => {
          const r = await api("/api/new", { split: split.value, index: Number(idx.value) });
          S.live = { session: r.session, index: Number(idx.value), split: split.value, steps: [], current: r.state,
                     pending: null, done: r.state.done, outcome: r.state.outcome, sims: null };
        });
      } }),
      el("label", {}, "方法", select(DATA.meta.methods.map((m) => [m.label, methodTitle(m.label)]), S.method, (v) => { S.method = v; render(); })),
      el("label", {}, "模擬次數", sims)));
    if (S.error) wrap.append(el("p", { class: "error", text: S.error }));
    if (!L) {
      wrap.append(el("div", { class: "card empty", text: "選擇資料集與題號後按「載入題目」，再逐步讓 Laya + MCTS 搜尋，或自己點選候選動作。" }));
      return wrap;
    }
    const doSearch = () => busy(async () => {
      const body = { session: L.session, method: S.method };
      if (sims.value !== "") body.simulations = Number(sims.value);
      L.sims = sims.value === "" ? null : Number(sims.value);
      const r = await api("/api/search", body);
      L.pending = r.step; L.pendingMethod = S.method;
    });
    const doAct = (i) => busy(async () => {
      const r = await api("/api/act", { session: L.session, index: i });
      const base = L.pending || { render: L.current.render, text: L.current.text, simulations: 0, nn_value: null, root_value: null,
        actions: L.current.actions.map((t) => ({ text: t, prior: null, policy: null, visits: 0, q: null })) };
      L.steps.push({ ...base, chosen: i, selected: base.selected ?? i, method: L.pending ? L.pendingMethod : "手動" });
      L.pending = null; L.current = r.state; L.done = r.state.done; L.outcome = r.state.outcome;
    });
    const auto = () => busy(async () => {
      let guard = 0;
      while (!L.done && guard++ < 200) {
        const body = { session: L.session, method: S.method };
        if (sims.value !== "") body.simulations = Number(sims.value);
        const s = await api("/api/search", body);
        const r = await api("/api/act", { session: L.session, index: s.step.selected });
        L.steps.push({ ...s.step, chosen: s.step.selected, method: S.method });
        L.current = r.state; L.done = r.state.done; L.outcome = r.state.outcome; L.pending = null;
        render();
      }
    });
    wrap.append(el("div", { class: "controls" },
      el("button", { class: "btn primary", text: "搜尋這一步", disabled: S.busy || L.done, onclick: doSearch }),
      el("button", { class: "btn", text: "執行搜尋建議", disabled: S.busy || L.done || !L.pending, onclick: () => doAct(L.pending.selected) }),
      el("button", { class: "btn", text: "自動下完", disabled: S.busy || L.done, onclick: auto }),
      el("button", { class: "btn", text: "同題比較（所有方法）", disabled: S.busy, onclick: () => busy(async () => {
        const r = await api("/api/compare", { session: L.session });
        const pid = `live-${L.split}-${L.index}`;
        if (!DATA.problems.find((p) => p.id === pid)) DATA.problems.push({ id: pid, render: r.problem.render, text: r.problem.text });
        DATA.traces[pid] = r.traces; S.cmpPid = pid; S.tab = "compare";
      }) }),
      S.busy ? el("span", { class: "muted", text: "計算中…" }) : null));
    const left = el("div", { class: "card" },
      el("h2", { text: L.done ? "最終狀態" : `目前狀態（已走 ${L.steps.length} 步）` }),
      L.done ? el("div", { style: "margin-bottom:8px" }, outcomeBadge(L.outcome)) : null,
      renderState(L.current.render),
      el("details", {}, el("summary", { class: "muted", text: "Laya 看到的狀態文字" }), el("pre", { class: "statetext", text: L.current.text })),
      L.steps.length ? el("ol", { class: "history" }, L.steps.map((s) => el("li", { text: `${s.actions[s.chosen].text}（${s.method}）` }))) : null);
    let right;
    if (L.done) right = el("div", { class: "card" }, el("h2", { text: "結果" }), outcomeBadge(L.outcome));
    else if (L.pending) {
      right = candidatesCard(L.pending, { title: `候選動作（${L.pendingMethod}）`, onPick: doAct });
      if (L.pending.tree) right.append(el("p", {}, el("button", { class: "btn", text: "在搜尋樹中查看", onclick: () => {
        S.tab = "tree"; S.treeSrc = "live|pending"; S.treeStep = 0; S.treeSel = null; render(); } })));
    } else {
      right = el("div", { class: "card" }, el("h2", { text: "候選動作" }),
        el("p", { class: "hint", text: "按「搜尋這一步」看 Laya 先驗與 MCTS 結果，或直接點選動作自己下。" }),
        el("table", { class: "cands" }, el("tbody", {}, L.current.actions.map((t, i) =>
          el("tr", { class: "live", onclick: () => doAct(i) }, el("td", { class: "act", text: t }))))));
    }
    wrap.append(el("div", { class: "grid2" }, left, right));
    return wrap;
  }
  async function busy(fn) {
    S.busy = true; S.error = null; render();
    try { await fn(); } catch (e) { S.error = String(e.message || e); }
    S.busy = false; render();
  }

  // ---------------------------------------------------------------- tree
  function treeSources() {
    const out = [];
    for (const p of DATA.problems) for (const [m, t] of Object.entries(DATA.traces[p.id] || {}))
      if (t.steps.some((s) => s.tree)) out.push({ key: `${p.id}|${m}`, label: `${problemLabel(p, DATA.problems.indexOf(p))} · ${m}`, steps: t.steps });
    if (S.live) {
      const steps = S.live.steps.filter((s) => s.tree);
      if (steps.length) out.push({ key: "live|history", label: "即時對局（已下的步）", steps });
      if (S.live.pending && S.live.pending.tree) out.push({ key: "live|pending", label: "即時對局（目前這步）", steps: [S.live.pending] });
    }
    return out;
  }

  function treeView() {
    const srcs = treeSources();
    if (!srcs.length) return el("div", { class: "card empty", text: "沒有搜尋樹資料（只有使用 Laya 的搜尋方法會記錄樹）。" });
    const src = srcs.find((s) => s.key === S.treeSrc) || srcs[0];
    S.treeSrc = src.key;
    const withTree = src.steps.map((s, i) => [i, s]).filter(([, s]) => s.tree);
    if (!withTree.some(([i]) => i === S.treeStep)) S.treeStep = withTree[0][0];
    const step = src.steps[S.treeStep];
    const wrap = el("div", {});
    wrap.append(el("div", { class: "controls" },
      el("label", {}, "對局", select(srcs.map((s) => [s.key, s.label]), src.key, (v) => { S.treeSrc = v; S.treeStep = 0; S.treeSel = null; render(); })),
      el("label", {}, "步驟", select(withTree.map(([i]) => [String(i), `第 ${i + 1} 步`]), String(S.treeStep), (v) => { S.treeStep = Number(v); S.treeSel = null; render(); }))));
    wrap.append(el("div", { class: "legend" },
      el("span", { class: "key" }, el("span", { class: "sw", style: `background:${qColor(1)}` }), "Q 高（對選擇者有利）"),
      el("span", { class: "key" }, el("span", { class: "sw", style: `background:${qColor(0)}` }), "Q ≈ 0"),
      el("span", { class: "key" }, el("span", { class: "sw", style: `background:${qColor(-1)}` }), "Q 低"),
      el("span", { class: "key", text: "圓的大小 = 拜訪次數；線的粗細 = 拜訪次數；◆ = 終局（綠成功、紅失敗）" })));
    const details = el("div", { class: "card details" });
    const tree = step.tree;
    const svgBox = treeSvg(tree, step, (node, path) => { S.treeSel = path; fillDetails(details, node, path); });
    wrap.append(el("div", { class: "grid2", style: "grid-template-columns: minmax(320px, 8fr) minmax(260px, 4fr)" },
      el("div", {}, el("div", { class: "tree-wrap" }, svgBox),
        el("p", { class: "muted", text: `共 ${tree.size} 個節點（只顯示被拜訪過的節點，每層最多 8 個子節點）` })), details));
    const sel = S.treeSel ? findPath(tree, S.treeSel) : tree;
    fillDetails(details, sel || tree, S.treeSel || []);
    return wrap;
  }
  function findPath(tree, path) {
    let n = tree;
    for (const i of path) { n = n.children[i]; if (!n) return null; }
    return n;
  }
  function fillDetails(box, node, path) {
    box.replaceChildren(el("h2", { text: path.length ? "選取的節點" : "根節點（目前狀態）" }));
    const actions = [];
    let n = window.__treeRoot;
    for (const i of path) { n = n.children[i]; actions.push(n.a); }
    const dl = el("dl", {});
    const add = (k, v) => { dl.append(el("dt", { text: k }), el("dd", { class: "tnum", text: v })); };
    if (actions.length) add("動作路徑", actions.join(" → "));
    add("拜訪次數", String(node.n));
    if (node.p != null && path.length) add("先驗 P", pct(node.p));
    if (node.q != null) add("Q（對選擇者）", fmt(node.q));
    if (node.v != null) add("Laya 價值 V（對行動者）", fmt(node.v));
    if (node.t) add("終局", `${node.ok ? "成功" : "失敗"}（價值 ${fmt(node.tv)}）`);
    if (node.hidden) add("未顯示的子節點", String(node.hidden));
    box.append(dl);
    if (node.s) box.append(el("pre", { class: "statetext", text: node.s }));
  }

  function treeSvg(tree, step, onSelect) {
    window.__treeRoot = tree;
    let leaf = 0, maxDepth = 0;
    const maxN = Math.max(1, tree.n);
    const nodes = [];
    (function lay(n, depth, path) {
      n._d = depth; n._path = path; maxDepth = Math.max(maxDepth, depth);
      if (!n.children.length) { n._y = leaf++; } else {
        n.children.forEach((c, i) => lay(c, depth + 1, path.concat(i)));
        n._y = (n.children[0]._y + n.children[n.children.length - 1]._y) / 2;
      }
      nodes.push(n);
    })(tree, 0, []);
    const dx = 170, dy = 34, padX = 24, padY = 22;
    const W = padX * 2 + maxDepth * dx + 220, H = padY * 2 + Math.max(1, leaf - 1) * dy + 10;
    const X = (n) => padX + n._d * dx, Y = (n) => padY + n._y * dy;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "MCTS 搜尋樹" });
    const chosenText = step.actions[step.chosen] ? step.actions[step.chosen].text : null;
    for (const n of nodes) for (const c of n.children) {
      const w = 1 + 5 * Math.sqrt(c.n / maxN);
      const onChosen = n === tree && c.a === chosenText;
      svg.append(sv("path", { d: `M${X(n)},${Y(n)} C${X(n) + dx / 2},${Y(n)} ${X(c) - dx / 2},${Y(c)} ${X(c)},${Y(c)}`,
        fill: "none", stroke: onChosen ? cssVar("--text-secondary") : cssVar("--axis"), "stroke-width": w, "stroke-linecap": "round" }));
    }
    for (const n of nodes) {
      const r = 4 + 10 * Math.sqrt(n.n / maxN);
      const g = sv("g", { class: "node", tabindex: 0, role: "button", "aria-label": `${n.a || "根"}，拜訪 ${n.n}` });
      if (n.t) {
        const s = r + 2;
        g.append(sv("rect", { x: X(n) - s, y: Y(n) - s, width: 2 * s, height: 2 * s, transform: `rotate(45 ${X(n)} ${Y(n)})`,
          fill: n.ok ? cssVar("--good") : cssVar("--critical"), stroke: cssVar("--surface-1"), "stroke-width": 2 }));
      } else {
        g.append(sv("circle", { cx: X(n), cy: Y(n), r, fill: qColor(n.q), stroke: cssVar("--surface-1"), "stroke-width": 2 }));
      }
      g.append(sv("circle", { class: "hit", cx: X(n), cy: Y(n), r: Math.max(12, r + 4), fill: "transparent" }));
      if (n.a) g.append(sv("text", { x: X(n) + r + 6, y: Y(n) - r - 2, text: n.a.length > 24 ? n.a.slice(0, 23) + "…" : n.a }));
      const pick = () => onSelect(n, n._path);
      g.addEventListener("click", pick);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      g.addEventListener("pointermove", (ev) => showTip(ev, n.a || "根節點", [
        { value: String(n.n), label: "拜訪" }, { value: fmt(n.q), label: "Q" },
        ...(n.p != null && n.a ? [{ value: pct(n.p), label: "先驗" }] : []),
        ...(n.t ? [{ value: n.ok ? "成功" : "失敗", label: "終局" }] : [])]));
      g.addEventListener("pointerleave", hideTip);
      svg.append(g);
    }
    return svg;
  }

  // ---------------------------------------------------------------- compare
  function compareView() {
    if (!DATA.problems.length) return el("div", { class: "card empty", text: MODE === "live" ? "在「對局回放」載入題目後按「同題比較」。" : "沒有對局資料。" });
    if (!DATA.problems.find((p) => p.id === S.cmpPid)) S.cmpPid = DATA.problems[0].id;
    const traces = DATA.traces[S.cmpPid] || {};
    const p = DATA.problems.find((x) => x.id === S.cmpPid);
    const wrap = el("div", {});
    wrap.append(el("div", { class: "controls" },
      el("label", {}, "題目", select(DATA.problems.map((x, i) => [x.id, problemLabel(x, i)]), S.cmpPid, (v) => { S.cmpPid = v; render(); }))));
    const methods = Object.keys(traces);
    const maxR = Math.max(0.0001, ...methods.map((m) => traces[m].outcome.reward || 0));
    const tbody = el("tbody");
    for (const m of methods) {
      const t = traces[m];
      const bar = el("div", { class: "bar", style: `width:${Math.max(0.5, 100 * (t.outcome.reward || 0) / maxR)}%;height:10px;background:${methodColor(m)}` });
      bar.addEventListener("pointermove", (ev) => showTip(ev, methodTitle(m), [{ color: methodColor(m), value: fmt(t.outcome.reward), label: "reward" },
        { value: String(t.outcome.moves), label: "步數" }]));
      bar.addEventListener("pointerleave", hideTip);
      tbody.append(el("tr", {},
        el("td", {}, el("span", { class: "legend", style: "margin:0" }, el("span", { class: "key" },
          el("span", { class: "sw", style: `background:${methodColor(m)}` }), methodTitle(m)))),
        el("td", {}, badge(t.outcome.success, t.outcome.success ? "成功" : "失敗")),
        el("td", { class: "num", text: fmt(t.outcome.reward) }),
        el("td", { style: "width:18%" }, bar),
        el("td", { class: "num", text: t.outcome.moves }),
        el("td", { class: "path", text: t.steps.map((s) => s.actions[s.chosen].text).join(" → ") }),
        el("td", {}, el("button", { class: "btn", text: "回放", onclick: () => { S.tab = "replay"; S.pid = S.cmpPid; S.method = m; S.step = 0;
          if (MODE === "live") S.tab = "replay-static"; render(); } }))));
    }
    wrap.append(el("div", { class: "stack" },
      el("div", { class: "card" }, el("h2", { text: "題目" }), renderState(p.render)),
      el("div", { class: "card" }, el("h2", { text: "各方法在同一題的表現" }),
        el("p", { class: "hint", text: "teacher 是精確解題器（參考上限）；均勻先驗與隨機 rollout 是不使用 Laya 的基準。" }),
        el("table", { class: "data" }, el("thead", {}, el("tr", {},
          ["方法", "結果", "reward", "", "步數", "走法", ""].map((h, i) => el("th", { class: i === 2 || i === 4 ? "num" : null, text: h })))), tbody))));
    return wrap;
  }

  // ---------------------------------------------------------------- curves
  function curvesView() {
    const C = DATA.curves;
    if (!C || !C.points.length) return el("div", { class: "card empty", text: "沒有學習曲線資料（需要實驗目錄裡的 metrics.jsonl）。" });
    const labels = [...new Set(C.points.map((p) => p.label))];
    const mk = (metric) => labels.map((l) => ({ label: l, color: methodColor(l),
      values: C.xs.map((x) => { const p = C.points.find((q) => q.x === x && q.label === l); return p ? p[metric] : null; }) }));
    const refs = (metric) => C.baselines.map((b) => ({ label: b.label, value: b[metric] }));
    const wrap = el("div", {});
    wrap.append(el("div", { class: "controls" }, el("label", {},
      el("input", { type: "checkbox", checked: S.curvesTable, onchange: (e) => { S.curvesTable = e.target.checked; render(); } }), "以表格顯示")));
    if (S.curvesTable) { wrap.append(curvesTable(C, labels)); return wrap; }
    const charts = el("div", { class: "charts" });
    charts.append(lineChart({ title: "評估 reward", subtitle: "各評估方法在保留題目上的平均 reward；水平線是不使用 Laya 的基準", xs: C.xs,
      series: mk("reward"), refs: refs("reward"), y: [0, 1] }));
    charts.append(lineChart({ title: "評估成功率", subtitle: "解出的比例", xs: C.xs, series: mk("success"), refs: refs("success"), y: [0, 1], pct: true }));
    if (C.train.length) {
      const tx = C.train.map((t) => `${t.iteration}:${t.stage}`);
      charts.append(lineChart({ title: "訓練損失（交叉熵）", subtitle: "policy 與 value 的 soft cross-entropy", xs: tx, series: [
        { label: "policy", color: "var(--series-1)", values: C.train.map((t) => t.policy_ce) },
        { label: "value", color: "var(--series-2)", values: C.train.map((t) => t.value_ce) }] }));
      charts.append(lineChart({ title: "保留資料上的 policy top-1", subtitle: "Laya 的首選是否落在目標分佈的最佳動作上", xs: tx,
        series: [{ label: "top-1", color: "var(--series-1)", values: C.train.map((t) => t.heldout_policy_top1) }], y: [0, 1], pct: true }));
    }
    wrap.append(charts);
    return wrap;
  }
  function curvesTable(C, labels) {
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, el("th", { text: "階段" }), labels.flatMap((l) => [
      el("th", { class: "num", text: `${l} reward` }), el("th", { class: "num", text: `${l} 成功率` })]))));
    const tb = el("tbody");
    for (const x of C.xs) tb.append(el("tr", {}, el("td", { text: stageLabel(x) }), labels.flatMap((l) => {
      const p = C.points.find((q) => q.x === x && q.label === l);
      return [el("td", { class: "num", text: p ? fmt(p.reward) : "–" }), el("td", { class: "num", text: p ? pct(p.success) : "–" })];
    })));
    for (const b of C.baselines) tb.append(el("tr", {}, el("td", { text: `基準：${b.label}` }),
      labels.flatMap((l, i) => i === 0 ? [el("td", { class: "num", text: fmt(b.reward) }), el("td", { class: "num", text: pct(b.success) })]
        : [el("td", {}), el("td", {})])));
    t.append(tb);
    return el("div", { class: "card" }, t);
  }

  function lineChart(o) {
    const W = 560, H = 280, m = { l: 44, r: 72, t: 12, b: 34 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const vals = o.series.flatMap((s) => s.values).concat((o.refs || []).map((r) => r.value)).filter((v) => v != null);
    let [y0, y1] = o.y || [Math.min(0, ...vals), Math.max(...vals)];
    if (!o.y) { const pad = (y1 - y0) * 0.1 || 0.1; y1 += pad; }
    const X = (i) => m.l + (o.xs.length === 1 ? iw / 2 : (i * iw) / (o.xs.length - 1));
    const Y = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title });
    const ticks = 5;
    for (let k = 0; k <= ticks; k++) {
      const v = y0 + ((y1 - y0) * k) / ticks;
      svg.append(sv("line", { class: k === 0 ? "axis" : "gridline", x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }));
      svg.append(sv("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "tnum", text: o.pct ? `${Math.round(v * 100)}%` : v.toFixed(2) }));
    }
    const step = Math.ceil(o.xs.length / 8);
    o.xs.forEach((x, i) => { if (i % step === 0 || i === o.xs.length - 1)
      svg.append(sv("text", { x: X(i), y: H - 12, "text-anchor": "middle", text: stageLabel(x) })); });
    for (const r of o.refs || []) {
      if (r.value == null) continue;
      svg.append(sv("line", { x1: m.l, x2: m.l + iw, y1: Y(r.value), y2: Y(r.value), stroke: cssVar("--text-muted"), "stroke-width": 1 }));
    }
    const ends = [];
    for (const s of o.series) {
      const pts = s.values.map((v, i) => (v == null ? null : [X(i), Y(v)])).filter(Boolean);
      if (!pts.length) continue;
      svg.append(sv("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: s.color, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-linecap": "round" }));
      const last = pts[pts.length - 1];
      svg.append(sv("circle", { cx: last[0], cy: last[1], r: 4, fill: s.color, stroke: cssVar("--surface-1"), "stroke-width": 2 }));
      ends.push({ s, y: last[1], x: last[0] });
    }
    // Direct end labels only when they separate; converging lines fall back to legend + tooltip.
    ends.sort((a, b) => a.y - b.y);
    const separated = ends.every((e, i) => i === 0 || e.y - ends[i - 1].y >= 14);
    if (o.series.length <= 4 && separated) for (const e of ends)
      svg.append(sv("text", { class: "dlabel", x: e.x + 8, y: e.y + 4, text: e.s.label }));
    // crosshair + one tooltip listing every series at that x
    const cross = sv("line", { y1: m.t, y2: m.t + ih, stroke: cssVar("--axis"), "stroke-width": 1, visibility: "hidden" });
    svg.append(cross);
    svg.append(sv("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent",
      onpointermove: (ev) => {
        const rect = svg.getBoundingClientRect();
        const px = ((ev.clientX - rect.left) / rect.width) * W;
        let i = o.xs.length === 1 ? 0 : Math.round(((px - m.l) / iw) * (o.xs.length - 1));
        i = Math.max(0, Math.min(o.xs.length - 1, i));
        cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
        showTip(ev, stageLabel(o.xs[i]), o.series.map((s) => ({ color: s.color, label: s.label,
          value: s.values[i] == null ? "–" : o.pct ? pct(s.values[i]) : fmt(s.values[i]) })).concat(
          (o.refs || []).filter((r) => r.value != null).map((r) => ({ color: cssVar("--text-muted"), label: `基準 ${r.label}`,
            value: o.pct ? pct(r.value) : fmt(r.value) }))));
      },
      onpointerleave: () => { cross.setAttribute("visibility", "hidden"); hideTip(); } }));
    const refs = (o.refs || []).filter((r) => r.value != null);
    const legend = o.series.length >= 2 || refs.length ? el("div", { class: "legend" }, o.series.map((s) =>
      el("span", { class: "key" }, el("span", { class: "ln", style: `background:${s.color}` }), s.label)),
      refs.map((r) => el("span", { class: "key" }, el("span", { class: "ln", style: "background:var(--text-muted);height:1px" }),
        `基準 ${r.label}（${o.pct ? pct(r.value) : fmt(r.value)}）`))) : null;
    return el("div", { class: "card chart" }, el("h2", { text: o.title }), o.subtitle ? el("p", { class: "hint", text: o.subtitle }) : null, legend, svg);
  }

  // ---------------------------------------------------------------- shell
  const TABS = [["replay", "對局回放"], ["tree", "搜尋樹"], ["compare", "同題比較"], ["curves", "學習曲線"]];
  function render() {
    hideTip();
    const meta = DATA.meta || {};
    const head = el("header", { class: "top" },
      el("h1", { text: "Laya × MCTS 搜尋視覺化" }),
      el("span", { class: "sub", text: [meta.name, meta.env, MODE === "live" ? "即時模式" : (DATA.generated ? `產生於 ${DATA.generated}` : "")].filter(Boolean).join(" · ") }),
      el("span", { class: "spacer" }),
      el("button", { class: "btn", text: "切換深 / 淺色", onclick: () => {
        const cur = document.documentElement.getAttribute("data-theme");
        const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
        document.documentElement.setAttribute("data-theme", dark ? "light" : "dark"); render(); } }));
    const tabs = el("nav", { class: "tabs", role: "tablist" }, TABS.map(([k, label]) => el("button", {
      role: "tab", "aria-selected": String(S.tab === k || (k === "replay" && S.tab === "replay-static")), text: label,
      onclick: () => { S.tab = k; render(); } })));
    let body;
    if (S.tab === "replay") body = MODE === "live" ? liveReplay() : staticReplay();
    else if (S.tab === "replay-static") body = staticReplay();
    else if (S.tab === "tree") body = treeView();
    else if (S.tab === "compare") body = compareView();
    else body = curvesView();
    const foot = el("p", { class: "muted", style: "margin-top:16px", text: meta.checkpoint ? `checkpoint：${meta.checkpoint}` : "" });
    app.replaceChildren(head, tabs, body, foot);
  }

  async function init() {
    if (MODE === "live") {
      try {
        const r = await api("/api/state");
        DATA = { meta: r.meta, problems: [], traces: {}, curves: r.curves };
      } catch (e) { S.error = String(e.message || e); }
    }
    render();
  }
  window.addEventListener("keydown", (e) => {
    if (S.tab !== "replay" || MODE === "live" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const t = DATA.traces[S.pid] && DATA.traces[S.pid][S.method];
    if (!t) return;
    if (e.key === "ArrowRight") { S.step = Math.min(t.steps.length, S.step + 1); render(); }
    if (e.key === "ArrowLeft") { S.step = Math.max(0, S.step - 1); render(); }
  });
  init();
})();

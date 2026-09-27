// Minimal SVG charts: grouped bars, lines, diverging bars. Dark-surface palette, hover tooltips.
import { h } from "./util.js";

export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const NS = "http://www.w3.org/2000/svg";

function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) el.setAttribute(k, v);
  for (const c of kids) if (c) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return el;
}

let tipEl = null;
function tip(html, e) {
  if (!tipEl) { tipEl = h("div", { class: "tip" }); document.body.appendChild(tipEl); }
  tipEl.innerHTML = html;
  tipEl.style.display = "block";
  const r = tipEl.getBoundingClientRect();
  tipEl.style.left = Math.min(e.clientX + 14, innerWidth - r.width - 8) + "px";
  tipEl.style.top = Math.max(8, e.clientY - r.height - 10) + "px";
}
function untip() { if (tipEl) tipEl.style.display = "none"; }

function legend(series) {
  if (series.length < 2) return null;
  return h("div", { class: "legend" }, series.map((sr) => h("span", {}, h("i", { style: { background: sr.color } }), sr.name)));
}

function topRoundedBar(x, y, w, hgt, r = 3) {
  if (hgt <= 0) return `M${x},${y} h${w}`;
  r = Math.min(r, w / 2, hgt);
  return `M${x},${y + hgt} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + hgt} Z`;
}

const pct = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "%");

// categories: string[]; series: [{name, color, values:number[] (0-1), n?: number[]}]
export function groupedBars({ categories, series, height = 300, yMax = 1, fmt = pct, rotate = true }) {
  const root = h("div", { class: "chart" });
  const W = Math.max(480, root.clientWidth || 860);
  const m = { l: 44, r: 10, t: 10, b: rotate ? 92 : 30 };
  const iw = W - m.l - m.r, ih = height - m.t - m.b;
  const svg = s("svg", { viewBox: `0 0 ${W} ${height}` });
  const y = (v) => m.t + ih - (v / yMax) * ih;
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4;
    svg.appendChild(s("line", { class: "gridl", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), "stroke-width": 1 }));
    svg.appendChild(s("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, fmt === pct ? `${Math.round(v * 100)}%` : String(+v.toFixed(2))));
  }
  const gw = iw / Math.max(1, categories.length);
  const bw = Math.max(3, Math.min(34, (gw * 0.8) / Math.max(1, series.length)) - 2);
  categories.forEach((c, ci) => {
    const gx = m.l + ci * gw + (gw - series.length * (bw + 2)) / 2;
    series.forEach((sr, si) => {
      const v = sr.values[ci];
      if (v === null || v === undefined || Number.isNaN(v)) return;
      const x = gx + si * (bw + 2);
      const path = s("path", { d: topRoundedBar(x, y(v), bw, m.t + ih - y(v)), fill: sr.color });
      const hit = s("rect", { x: x - 1, y: m.t, width: bw + 2, height: ih, fill: "transparent" });
      const show = (e) => tip(`<b>${c}</b><br><span style="color:${sr.color}">■</span> ${sr.name}: <b>${fmt(v)}</b>${sr.n ? ` <span class="mut">n=${sr.n[ci]}</span>` : ""}`, e);
      hit.addEventListener("pointermove", show); hit.addEventListener("pointerleave", untip);
      svg.append(path, hit);
    });
    const tx = m.l + ci * gw + gw / 2, ty = m.t + ih + 12;
    svg.appendChild(s("text", rotate ? { x: tx, y: ty, "text-anchor": "end", transform: `rotate(-32 ${tx} ${ty})` } : { x: tx, y: ty + 4, "text-anchor": "middle" }, c));
  });
  svg.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: "#383835" }));
  root.append(legend(series) || "", svg);
  return root;
}

// xs: labels; series: [{name, color, values}]
export function lineChart({ xs, series, height = 260, xLabel = "", yMin = null, yMax = null, fmt = pct }) {
  const root = h("div", { class: "chart" });
  const W = Math.max(420, root.clientWidth || 640);
  const m = { l: 46, r: 16, t: 12, b: 42 };
  const iw = W - m.l - m.r, ih = height - m.t - m.b;
  const all = series.flatMap((sr) => sr.values).filter((v) => v !== null && v !== undefined);
  let lo = yMin ?? Math.max(0, Math.min(...all) - 0.05), hi = yMax ?? Math.min(1, Math.max(...all) + 0.05);
  if (hi - lo < 0.05) { lo = Math.max(0, lo - 0.05); hi = Math.min(1, hi + 0.05); }
  const svg = s("svg", { viewBox: `0 0 ${W} ${height}` });
  const x = (i) => m.l + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    svg.appendChild(s("line", { class: "gridl", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
    svg.appendChild(s("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, fmt(v)));
  }
  xs.forEach((lab, i) => svg.appendChild(s("text", { x: x(i), y: m.t + ih + 16, "text-anchor": "middle" }, String(lab))));
  if (xLabel) svg.appendChild(s("text", { x: m.l + iw / 2, y: height - 4, "text-anchor": "middle" }, xLabel));
  svg.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: "#383835" }));
  for (const sr of series) {
    const pts = sr.values.map((v, i) => (v === null || v === undefined ? null : [x(i), y(v)]));
    const d = pts.filter(Boolean).map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
    svg.appendChild(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": 2 }));
    pts.forEach((p) => p && svg.appendChild(s("circle", { cx: p[0], cy: p[1], r: 4, fill: sr.color, stroke: "#1a1a19", "stroke-width": 2 })));
  }
  // crosshair hover
  const hit = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  const cross = s("line", { y1: m.t, y2: m.t + ih, stroke: "#898781", "stroke-dasharray": "3 3", visibility: "hidden" });
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(xs.length - 1, Math.round(((px - m.l) / iw) * (xs.length - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    tip(`<b>${xLabel ? xLabel + " = " : ""}${xs[i]}</b><br>` + series.map((sr) => `<span style="color:${sr.color}">■</span> ${sr.name}: <b>${fmt(sr.values[i])}</b>`).join("<br>"), e);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); untip(); });
  svg.append(cross, hit);
  root.append(legend(series) || "", svg);
  return root;
}

// items: [{label, value}] (value in -1..1, as accuracy delta)
export function divergingBars({ items, height = null, fmt = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + " pp" }) {
  const root = h("div", { class: "chart" });
  const W = Math.max(420, root.clientWidth || 640);
  const rowH = 24;
  const Ht = height || items.length * rowH + 30;
  const m = { l: 160, r: 60, t: 6, b: 20 };
  const iw = W - m.l - m.r;
  const maxAbs = Math.max(0.05, ...items.map((i) => Math.abs(i.value)));
  const x = (v) => m.l + iw / 2 + (v / maxAbs) * (iw / 2);
  const svg = s("svg", { viewBox: `0 0 ${W} ${Ht}` });
  svg.appendChild(s("line", { x1: x(0), x2: x(0), y1: m.t, y2: Ht - m.b, stroke: "#383835" }));
  items.forEach((it, i) => {
    const yy = m.t + i * rowH + 4;
    const x0 = Math.min(x(0), x(it.value)), w = Math.abs(x(it.value) - x(0));
    const col = it.value >= 0 ? "#3987e5" : "#e66767";
    svg.appendChild(s("text", { x: m.l - 8, y: yy + 12, "text-anchor": "end" }, it.label));
    const r = s("rect", { x: x0, y: yy, width: Math.max(1, w), height: rowH - 8, rx: 3, fill: col });
    r.addEventListener("pointermove", (e) => tip(`<b>${it.label}</b><br>${fmt(it.value)}${it.extra ? `<br><span class="mut">${it.extra}</span>` : ""}`, e));
    r.addEventListener("pointerleave", untip);
    svg.appendChild(r);
    svg.appendChild(s("text", { x: it.value >= 0 ? x0 + w + 5 : x0 - 5, y: yy + 12, "text-anchor": it.value >= 0 ? "start" : "end", style: "fill:#c3c2b7" }, fmt(it.value)));
  });
  root.appendChild(svg);
  return root;
}

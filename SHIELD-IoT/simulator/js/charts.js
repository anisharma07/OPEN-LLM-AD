// HUD charts: the live traffic chart (canvas, DPR-aware, with a hover
// crosshair) and small horizontal bar charts drawn to scale in SVG (used for
// the TCP ablation and the per-class F1 values in the Research tab).
// Colours come from the CSS tokens on :root so the charts follow the theme.

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Reads a CSS custom property from :root, with a fallback. */
export function cssVar(name, fallback = '') {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function readTokens() {
  return {
    fg: cssVar('--fg', '#e6ecf5'),
    muted: cssVar('--muted', '#8b99b4'),
    line: cssVar('--line', '#26324a'),
    surface: cssVar('--panel-solid', '#131b29'),
    allowed: cssVar('--allowed', '#5ad1e6'),
    danger: cssVar('--danger', '#ff4d5e'),
    warn: cssVar('--warn', '#ffb347'),
    mono: cssVar('--font-mono', 'ui-monospace, monospace'),
  };
}

/** "#5ad1e6" + alpha -> "rgba(...)". */
function withAlpha(hex, a) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** Rounds a maximum up to a clean axis top (1, 2, 2.5, 5 x 10^n) and returns the tick step. */
function niceScale(max, ticks = 3) {
  const raw = Math.max(max, 1) / ticks;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw) ?? 10 * mag;
  return { top: step * Math.ceil(Math.max(max, 1) / step), step };
}

const fmtCount = n => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(Math.round(n)));

// ---------------------------------------------------------------------------
// Live traffic chart

/**
 * Live chart of the last `windowMs` of sim time: allowed pps as an area,
 * dropped pps as a line, alert markers along the top edge.
 * The caller pushes samples (sim time) and calls render() at its own cadence.
 */
export function createTrafficChart(canvas, { windowMs = 60000, label = 'Live traffic' } = {}) {
  const ctx = canvas.getContext('2d');
  const wrap = canvas.parentElement;
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  tip.hidden = true;
  wrap.appendChild(tip);

  const PAD = { l: 34, r: 46, t: 16, b: 20 };
  let samples = [];          // { t, allowed, dropped }
  let marks = [];            // { t, severity }
  let tokens = readTokens();
  let cssW = 0, cssH = 0, dpr = 1;
  let hoverX = null;
  let dirty = true;
  let lastAria = 0;

  function resize() {
    const r = canvas.getBoundingClientRect();
    const w = Math.round(r.width), h = Math.round(r.height);
    const d = Math.min(window.devicePixelRatio || 1, 3);
    if (w === cssW && h === cssH && d === dpr) return;
    cssW = w; cssH = h; dpr = d;
    canvas.width = Math.max(1, Math.round(w * d));
    canvas.height = Math.max(1, Math.round(h * d));
    dirty = true;
    render();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  const latestT = () => (samples.length ? samples[samples.length - 1].t : 0);

  function push(sample) {
    const t = Number(sample.t) || 0;
    // The sim clock went backwards (reset): start over.
    if (samples.length && t < latestT()) clear();
    samples.push({ t, allowed: Math.max(0, +sample.allowed || 0), dropped: Math.max(0, +sample.dropped || 0) });
    const cutoff = t - windowMs - 2000;
    while (samples.length && samples[0].t < cutoff) samples.shift();
    while (marks.length && marks[0].t < cutoff) marks.shift();
    dirty = true;
  }

  function mark(t, severity) {
    marks.push({ t: Number(t) || 0, severity });
    if (marks.length > 400) marks.splice(0, marks.length - 400);
    dirty = true;
  }

  function clear() {
    samples = [];
    marks = [];
    dirty = true;
  }

  function geometry() {
    const now = latestT();
    const t0 = now - windowMs;
    let max = 5;
    for (const s of samples) if (s.t >= t0) max = Math.max(max, s.allowed, s.dropped);
    const { top, step } = niceScale(max * 1.08);
    const plotW = Math.max(10, cssW - PAD.l - PAD.r);
    const plotH = Math.max(10, cssH - PAD.t - PAD.b);
    return {
      now, t0, top, step, plotW, plotH,
      x: t => PAD.l + ((t - t0) / windowMs) * plotW,
      y: v => PAD.t + plotH - (v / top) * plotH,
    };
  }

  function render(force = false) {
    if (!cssW || !cssH) return;
    if (!dirty && !force) return;
    dirty = false;
    const g = geometry();
    const T = tokens;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.font = `10px ${T.mono}`;
    ctx.textBaseline = 'middle';

    // Grid and y labels.
    ctx.lineWidth = 1;
    ctx.strokeStyle = withAlpha(T.line, 0.9);
    ctx.fillStyle = T.muted;
    ctx.textAlign = 'right';
    for (let v = 0; v <= g.top + 1e-9; v += g.step) {
      const y = Math.round(g.y(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD.l, y);
      ctx.lineTo(PAD.l + g.plotW, y);
      ctx.stroke();
      ctx.fillText(fmtCount(v), PAD.l - 6, y);
    }
    // x labels: -60 s, -30 s, now.
    ctx.textBaseline = 'alphabetic';
    const xl = cssH - 5;
    ctx.textAlign = 'left';
    ctx.fillText(`-${Math.round(windowMs / 1000)} s`, PAD.l, xl);
    ctx.textAlign = 'center';
    ctx.fillText(`-${Math.round(windowMs / 2000)} s`, PAD.l + g.plotW / 2, xl);
    ctx.textAlign = 'right';
    ctx.fillText('now', PAD.l + g.plotW, xl);

    const vis = samples.filter(s => s.t >= g.t0 - 1500);
    if (vis.length < 2) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = T.muted;
      ctx.font = `11px ${cssVar('--font-sans', 'sans-serif')}`;
      ctx.fillText(samples.length ? 'Collecting samples…' : 'Waiting for traffic…', PAD.l + g.plotW / 2, PAD.t + g.plotH / 2);
      return;
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.l, 0, g.plotW, cssH);
    ctx.clip();

    // Allowed: a light wash plus a 2px line.
    const baseY = g.y(0);
    ctx.beginPath();
    ctx.moveTo(g.x(vis[0].t), baseY);
    for (const s of vis) ctx.lineTo(g.x(s.t), g.y(s.allowed));
    ctx.lineTo(g.x(vis[vis.length - 1].t), baseY);
    ctx.closePath();
    ctx.fillStyle = withAlpha(T.allowed, 0.13);
    ctx.fill();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 2;
    ctx.strokeStyle = T.allowed;
    ctx.beginPath();
    vis.forEach((s, i) => (i ? ctx.lineTo(g.x(s.t), g.y(s.allowed)) : ctx.moveTo(g.x(s.t), g.y(s.allowed))));
    ctx.stroke();

    // Dropped: a 2px line on the same axis (both are packets per second).
    ctx.strokeStyle = T.danger;
    ctx.beginPath();
    vis.forEach((s, i) => (i ? ctx.lineTo(g.x(s.t), g.y(s.dropped)) : ctx.moveTo(g.x(s.t), g.y(s.dropped))));
    ctx.stroke();

    // Alert markers: small downward triangles along the top edge, merged when they overlap.
    let lastX = -Infinity, lastRank = -1;
    const RANK = { low: 0, medium: 1, high: 2, critical: 3 };
    for (const m of marks) {
      if (m.t < g.t0) continue;
      const x = g.x(m.t);
      const rank = RANK[m.severity] ?? 1;
      if (x - lastX < 5 && rank <= lastRank) continue;
      lastX = x; lastRank = rank;
      ctx.fillStyle = rank >= 2 ? T.danger : T.warn;
      ctx.beginPath();
      ctx.moveTo(x - 4, 3);
      ctx.lineTo(x + 4, 3);
      ctx.lineTo(x, 10);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();

    // Emphasised endpoints with a surface ring, and the current value.
    const last = vis[vis.length - 1];
    const ex = g.x(last.t);
    const dot = (y, color) => {
      ctx.beginPath();
      ctx.arc(ex, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = T.surface;
      ctx.stroke();
    };
    if (last.dropped > 0) dot(g.y(last.dropped), T.danger);
    dot(g.y(last.allowed), T.allowed);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = T.fg;
    ctx.font = `600 11px ${T.mono}`;
    const ly = Math.min(Math.max(g.y(last.allowed), PAD.t + 6), PAD.t + g.plotH - 6);
    ctx.fillText(fmtCount(last.allowed), ex + 8, ly);
    if (last.dropped > 0) {
      const dy = g.y(last.dropped);
      if (Math.abs(dy - ly) > 12) {
        ctx.fillStyle = T.muted;
        ctx.font = `11px ${T.mono}`;
        ctx.fillText(fmtCount(last.dropped), ex + 8, Math.min(dy, PAD.t + g.plotH - 6));
      }
    }

    // Hover crosshair at the nearest sample.
    if (hoverX !== null) {
      let best = vis[0];
      for (const s of vis) if (Math.abs(g.x(s.t) - hoverX) < Math.abs(g.x(best.t) - hoverX)) best = s;
      const hx = Math.round(g.x(best.t)) + 0.5;
      ctx.strokeStyle = withAlpha(T.fg, 0.35);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, PAD.t);
      ctx.lineTo(hx, PAD.t + g.plotH);
      ctx.stroke();
      for (const [v, c] of [[best.dropped, T.danger], [best.allowed, T.allowed]]) {
        ctx.beginPath();
        ctx.arc(hx, g.y(v), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = c;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = T.surface;
        ctx.stroke();
      }
      const near = marks.filter(m => Math.abs(m.t - best.t) <= 1000).length;
      const ago = Math.max(0, Math.round((g.now - best.t) / 1000));
      tip.innerHTML = '';
      const head = document.createElement('div');
      head.className = 'chart-tip-head';
      head.textContent = ago ? `${ago} s ago` : 'now';
      tip.appendChild(head);
      for (const [cls, name, v] of [['allowed', 'Allowed', best.allowed], ['dropped', 'Dropped', best.dropped]]) {
        const row = document.createElement('div');
        row.className = 'chart-tip-row';
        const key = document.createElement('span');
        key.className = `key key-${cls}`;
        const txt = document.createElement('span');
        txt.textContent = name;
        const val = document.createElement('b');
        val.textContent = `${Math.round(v)} pps`;
        row.append(key, txt, val);
        tip.appendChild(row);
      }
      if (near) {
        const row = document.createElement('div');
        row.className = 'chart-tip-row';
        row.textContent = `${near} alert${near === 1 ? '' : 's'} near here`;
        tip.appendChild(row);
      }
      tip.hidden = false;
      const tw = tip.offsetWidth;
      const left = hx + 12 + tw > cssW ? hx - 12 - tw : hx + 12;
      tip.style.left = `${Math.max(0, left)}px`;
      tip.style.top = `${PAD.t}px`;
    } else {
      tip.hidden = true;
    }

    // Keep a readable summary for screen readers, without churning every frame.
    const nowReal = performance.now();
    if (nowReal - lastAria > 5000) {
      lastAria = nowReal;
      canvas.setAttribute('aria-label',
        `${label}: ${Math.round(last.allowed)} packets per second allowed, ${Math.round(last.dropped)} dropped, ` +
        `${marks.filter(m => m.t >= g.t0).length} alerts in the last ${Math.round(windowMs / 1000)} seconds.`);
    }
  }

  function onMove(e) {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    hoverX = x >= PAD.l - 4 && x <= cssW - PAD.r + 8 ? x : null;
    dirty = true;
    render();
  }
  function onLeave() {
    hoverX = null;
    dirty = true;
    render();
  }
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerleave', onLeave);

  return {
    push,
    mark,
    clear,
    render,
    /** Re-read the colour tokens (e.g. after a theme change). */
    refreshTokens() { tokens = readTokens(); dirty = true; },
    destroy() {
      ro.disconnect();
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerleave', onLeave);
      tip.remove();
    },
  };
}

// ---------------------------------------------------------------------------
// Horizontal bar chart (SVG), drawn to scale on a 0..max axis

function svgEl(tag, attrs = {}, text) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  if (text !== undefined) el.textContent = text;
  return el;
}

/** Bar path with a 4px rounded data end and a square baseline. */
function barPath(x0, y, w, h, r = 4) {
  const rr = Math.min(r, w / 2, h / 2);
  if (w <= 0) return '';
  return `M${x0},${y}H${x0 + w - rr}A${rr},${rr} 0 0 1 ${x0 + w},${y + rr}V${y + h - rr}` +
    `A${rr},${rr} 0 0 1 ${x0 + w - rr},${y + h}H${x0}Z`;
}

/**
 * Renders labelled horizontal bars into `container` (an element), sized to its
 * width and re-rendered when that width changes. Bars always start at zero.
 *
 * rows: [{ label, value, note?, emphasis? }]
 * opts: { max = 1, ticks = [0, .25, .5, .75, 1], ref?: {value, label}, digits = 4,
 *         delta?: boolean (show value - ref), ariaLabel, describe?: row -> string }
 */
export function createBarChart(container, rows, opts = {}) {
  const {
    max = 1, ticks = [0, 0.25, 0.5, 0.75, 1], ref = null, digits = 4, delta = false,
    ariaLabel = 'Bar chart', describe = null,
  } = opts;
  let lastW = 0;

  function draw() {
    const W = Math.round(container.clientWidth);
    if (!W || W === lastW) return;
    lastW = W;
    const T = readTokens();
    const rowH = 24, barH = 12;
    const labelW = Math.min(150, Math.max(96, W * 0.3));
    const valueW = delta ? 108 : 52;
    const top = ref ? 22 : 8;
    const axisH = 18;
    const plotW = Math.max(60, W - labelW - valueW - 8);
    const H = top + rows.length * rowH + axisH;
    const x = v => labelW + (Math.max(0, Math.min(v, max)) / max) * plotW;

    const svg = svgEl('svg', {
      width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': ariaLabel, class: 'bar-chart',
    });
    // Recessive vertical grid and tick labels.
    for (const t of ticks) {
      const gx = Math.round(x(t)) + 0.5;
      svg.appendChild(svgEl('line', { x1: gx, x2: gx, y1: top - 4, y2: top + rows.length * rowH, stroke: T.line, 'stroke-width': 1 }));
      svg.appendChild(svgEl('text', {
        x: gx, y: H - 4, 'text-anchor': t === 0 ? 'start' : t === max ? 'end' : 'middle', fill: T.muted, class: 'bc-tick',
      }, t.toFixed(2)));
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      const g = svgEl('g', { class: 'bc-row' });
      const tip = describe ? describe(r) : `${r.label}: ${r.value.toFixed(digits)}`;
      g.appendChild(svgEl('title', {}, tip));
      // Full-row hit target for the tooltip.
      g.appendChild(svgEl('rect', { x: 0, y, width: W, height: rowH, fill: 'transparent' }));
      g.appendChild(svgEl('text', {
        x: labelW - 8, y: y + rowH / 2, 'text-anchor': 'end', 'dominant-baseline': 'central',
        fill: r.emphasis ? T.fg : T.muted, class: 'bc-label',
      }, r.label));
      g.appendChild(svgEl('path', {
        d: barPath(labelW, y + (rowH - barH) / 2, x(r.value) - labelW, barH),
        fill: r.emphasis ? cssVar('--viz-bar-strong', '#9fb4dc') : cssVar('--viz-bar', '#5f7fb6'),
      }));
      const vx = x(r.value) + 6;
      const val = svgEl('text', { x: vx, y: y + rowH / 2, 'dominant-baseline': 'central', fill: T.fg, class: 'bc-value' });
      val.appendChild(svgEl('tspan', {}, r.value.toFixed(digits)));
      if (delta && ref && r.value !== ref.value) {
        const d = r.value - ref.value;
        val.appendChild(svgEl('tspan', { dx: 6, fill: T.muted }, `${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(digits)}`));
      }
      g.appendChild(val);
      svg.appendChild(g);
    });
    if (ref) {
      const rx = Math.round(x(ref.value)) + 0.5;
      const accent = cssVar('--accent', '#f2b134');
      svg.appendChild(svgEl('line', { x1: rx, x2: rx, y1: top - 6, y2: top + rows.length * rowH, stroke: accent, 'stroke-width': 1.5 }));
      svg.appendChild(svgEl('text', {
        x: rx, y: 10, 'text-anchor': rx > W - 120 ? 'end' : 'middle', fill: T.fg, class: 'bc-ref',
      }, ref.label));
    }
    container.replaceChildren(svg);
  }

  const ro = new ResizeObserver(draw);
  ro.observe(container);
  draw();
  return {
    redraw() { lastW = 0; draw(); },
    destroy() { ro.disconnect(); container.replaceChildren(); },
  };
}

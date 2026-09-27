// ComfyUI-style node editor: DOM nodes on a pannable/zoomable world, SVG bezier wires.
import { api, h, toast, lightbox, showMenuAt, closeMenus, clamp } from "./util.js";
import { buildWidget } from "./widgets.js";

const WIDTHS = { MLLM: 310, TextPrompt: 320, VisualPrompt: 290, MMADSample: 300, Note: 280, CustomQuestion: 300, Score: 230, AnswerParser: 250, PatchCore: 290, WinCLIP: 290, HeatmapToBox: 290 };
const CAT_COLORS = {
  Input: "#e0a030", "Pre-process": "#8d6e63", Question: "#9ccc65", Detector: "#ef5350", Localise: "#ff8a65",
  Prompt: "#ce93d8", Reason: "#ba68c8", Evaluate: "#fff176", Display: "#90a4ae",
};

export class Editor {
  constructor({ viewport, world, wires, schema, onChange }) {
    this.vp = viewport; this.world = world; this.svg = wires;
    this.onChange = onChange || (() => {});
    this.setSchema(schema);
    this.nodes = new Map(); this.links = new Map();
    this.nextId = 1; this.nextLink = 1;
    this.view = { x: 60, y: 40, z: 0.9 };
    this.selected = new Set(); this.selectedLink = null;
    this.status = {};
    this.experiment = null;          // experiment preset stored with the workflow
    this.sweepKeys = new Set();
    this.undoStack = []; this.redoStack = []; this._snapTimer = null;
    this.clipboard = null;
    this.mouse = { x: 200, y: 200 };
    this._bindCanvas();
    this._bindKeys();
    this.applyView();
  }

  setSchema(schema) {
    this.schema = schema;
    this.defs = Object.fromEntries(schema.nodes.map((n) => [n.type, n]));
    this.types = schema.types;
  }

  // ---------------------------------------------------------------- serialise
  toJSON(includeView = true) {
    const nodes = [...this.nodes.values()].map((n) => ({
      id: n.id, type: n.type, pos: [Math.round(n.pos[0]), Math.round(n.pos[1])], params: { ...n.params },
      bypass: !!n.bypass, title: n.title || undefined, collapsed: !!n.collapsed, width: n.width || undefined,
      controls: Object.keys(n.controls || {}).length ? { ...n.controls } : undefined,
    }));
    const links = [...this.links.values()].map((l) => ({ id: l.id, from: [...l.from], to: [...l.to] }));
    const out = { version: 1, nodes, links };
    if (includeView) out.view = { ...this.view };
    if (this.experiment) out.experiment = this.experiment;
    return out;
  }

  load(g, { keepView = false, snapshot = true } = {}) {
    for (const n of this.nodes.values()) n.el.remove();
    this.nodes.clear(); this.links.clear(); this.selected.clear(); this.status = {};
    this.nextId = 1; this.nextLink = 1;
    const missing = [];
    for (const n of g.nodes || []) {
      if (!this.defs[n.type]) { missing.push(n.type); continue; }
      this._createNode(n.type, n.pos || [0, 0], n);
    }
    for (const l of g.links || []) {
      if (this.nodes.has(+l.from[0]) && this.nodes.has(+l.to[0])) this._addLink(+l.from[0], +l.from[1], +l.to[0], +l.to[1], l.id);
    }
    this.experiment = g.experiment || null;
    this.refreshSweepMarks();
    if (g.view && !keepView) this.view = { ...g.view };
    this.applyView();
    requestAnimationFrame(() => this.drawWires());
    if (missing.length) toast(`Skipped unknown node types: ${[...new Set(missing)].join(", ")}`, true);
    if (snapshot) this.snapshot(true);
    this.onChange();
  }

  // ---------------------------------------------------------------- nodes
  _createNode(type, pos, saved = {}) {
    const def = this.defs[type];
    const id = saved.id ? +saved.id : this.nextId;
    this.nextId = Math.max(this.nextId, id + 1);
    const params = {};
    for (const p of def.params) params[p.name] = saved.params && p.name in saved.params ? saved.params[p.name] : p.default;
    const n = {
      id, type, def, pos: [...pos], params, bypass: !!saved.bypass, title: saved.title || "",
      collapsed: !!saved.collapsed, width: saved.width || WIDTHS[type] || 270, controls: { ...(saved.controls || {}) },
      portEls: { in: [], out: [] }, widgetEls: {},
    };
    n.el = this._buildNodeEl(n);
    this.world.appendChild(n.el);
    this.nodes.set(id, n);
    return n;
  }

  addNode(type, pos, params = {}) {
    const n = this._createNode(type, pos, { params });
    this.snapshot(); this.onChange();
    return n;
  }

  removeNodes(ids) {
    for (const id of ids) {
      const n = this.nodes.get(id);
      if (!n) continue;
      for (const l of [...this.links.values()]) if (l.from[0] === id || l.to[0] === id) this.links.delete(l.id);
      n.el.remove(); this.nodes.delete(id); this.selected.delete(id); delete this.status[id];
    }
    this.drawWires(); this.snapshot(); this.onChange();
  }

  nodeTitle(n) { return n.title || n.def.title; }

  _buildNodeEl(n) {
    const def = n.def;
    const color = CAT_COLORS[def.category] || "#888";
    const el = h("div", { class: "node" + (n.type === "Note" ? " note" : ""), "data-id": n.id });
    el.style.width = n.width + "px";
    n.el = el;
    const badge = h("span", { class: "badge" });
    const titleEl = h("span", { class: "title", title: def.description }, this.nodeTitle(n));
    const head = h("div", { class: "nh" },
      h("span", { class: "dot", style: { background: color } }), titleEl, badge,
      h("span", { class: "ic", title: "collapse", onclick: (e) => { e.stopPropagation(); this.toggleCollapse(n.id); } }, "▾"));
    head.style.background = `linear-gradient(90deg, ${color}22, transparent 70%)`;
    const body = h("div", { class: "nb" });

    const ins = h("div", { class: "col" }), outs = h("div", { class: "col" });
    def.inputs.forEach((p, i) => {
      const dot = h("span", { class: "pd", style: { background: this.types[p.type] || "#999" }, title: p.type });
      const row = h("div", { class: "port in" }, dot, h("span", {}, p.name), p.optional ? h("span", { class: "opt" }, "opt") : null);
      dot.addEventListener("pointerdown", (e) => this._startWireFromInput(e, n.id, i));
      n.portEls.in[i] = row; ins.appendChild(row);
    });
    def.outputs.forEach((p, i) => {
      const dot = h("span", { class: "pd", style: { background: this.types[p.type] || "#999" }, title: p.type });
      const row = h("div", { class: "port out" }, h("span", {}, p.name), dot);
      dot.addEventListener("pointerdown", (e) => this._startWireFromOutput(e, n.id, i));
      n.portEls.out[i] = row; outs.appendChild(row);
    });
    if (def.inputs.length || def.outputs.length) body.appendChild(h("div", { class: "ports" }, ins, outs));

    const wbox = h("div", { class: "widgets" });
    for (const p of def.params) {
      const w = buildWidget(this, n, p);
      n.widgetEls[p.name] = w;
      wbox.appendChild(w);
    }
    if (def.params.length) body.appendChild(wbox);
    const out = h("div", { class: "nout" });
    body.appendChild(out);
    el.append(head, body);
    n.badgeEl = badge; n.outEl = out; n.titleEl = titleEl;

    head.addEventListener("pointerdown", (e) => this._startNodeDrag(e, n.id));
    head.addEventListener("dblclick", (e) => { e.stopPropagation(); this.rename(n.id); });
    el.addEventListener("pointerdown", (e) => { if (e.button === 0) this._selectOnPointer(e, n.id); });
    el.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); this._nodeMenu(e, n.id); });
    el.addEventListener("wheel", (e) => {
      // let scrollable previews/textareas scroll, otherwise zoom the canvas
      const sc = e.target.closest("pre, textarea");
      if (sc && sc.scrollHeight > sc.clientHeight) e.stopPropagation();
    });
    // width resize handle
    const grip = h("div", { style: { position: "absolute", right: "0", bottom: "0", width: "12px", height: "12px", cursor: "nwse-resize" } });
    grip.addEventListener("pointerdown", (e) => this._startResize(e, n.id));
    el.appendChild(grip);
    this._applyNodeClasses(n);
    this._placeNode(n);
    return el;
  }

  _applyNodeClasses(n) {
    n.el.classList.toggle("bypass", !!n.bypass);
    n.el.classList.toggle("collapsed", !!n.collapsed);
    n.el.classList.toggle("selected", this.selected.has(n.id));
  }

  _placeNode(n) { n.el.style.left = n.pos[0] + "px"; n.el.style.top = n.pos[1] + "px"; }

  rebuildNode(id) {
    const n = this.nodes.get(id);
    const old = n.el;
    n.portEls = { in: [], out: [] }; n.widgetEls = {};
    n.el = this._buildNodeEl(n);
    old.replaceWith(n.el);
    if (this.status[id]) this.setNodeStatus(id, this.status[id]);
    requestAnimationFrame(() => this.drawWires());
  }

  rename(id) {
    const n = this.nodes.get(id);
    const t = prompt("Node title (used to label sweeps):", this.nodeTitle(n));
    if (t === null) return;
    n.title = t.trim() === n.def.title ? "" : t.trim();
    n.titleEl.textContent = this.nodeTitle(n);
    this.snapshot(); this.onChange();
  }

  toggleCollapse(id) {
    const n = this.nodes.get(id); n.collapsed = !n.collapsed; this._applyNodeClasses(n);
    requestAnimationFrame(() => this.drawWires()); this.onChange();
  }

  toggleBypass(ids) {
    for (const id of ids) { const n = this.nodes.get(id); if (n) { n.bypass = !n.bypass; this._applyNodeClasses(n); } }
    this.snapshot(); this.onChange();
  }

  setParam(id, name, value, { silent = false } = {}) {
    const n = this.nodes.get(id); if (!n) return;
    n.params[name] = value;
    if (!silent) { this.snapshot(); this.onChange(); }
  }

  refreshWidget(id, name) {
    const n = this.nodes.get(id);
    const p = n.def.params.find((x) => x.name === name);
    const w = buildWidget(this, n, p);
    n.widgetEls[name].replaceWith(w); n.widgetEls[name] = w;
    this.refreshSweepMarks();
  }

  applyControls() {
    for (const n of this.nodes.values()) {
      for (const [name, mode] of Object.entries(n.controls || {})) {
        const p = n.def.params.find((x) => x.name === name);
        if (!p || mode === "fixed") continue;
        let v = Number(n.params[name]) || 0;
        const max = p.max ?? 1e9, min = p.min ?? 0;
        if (mode === "increment") v = v + 1 > max ? min : v + 1;
        else if (mode === "decrement") v = v - 1 < min ? max : v - 1;
        else if (mode === "randomize") v = Math.floor(min + Math.random() * Math.min(max - min, 1e6));
        n.params[name] = v; this.refreshWidget(n.id, name);
      }
    }
    this.onChange();
  }

  // ---------------------------------------------------------------- links
  _addLink(fn, fo, tn, ti, id) {
    for (const l of [...this.links.values()]) if (l.to[0] === tn && l.to[1] === ti) this.links.delete(l.id);
    const lid = id ? +id : this.nextLink;
    this.nextLink = Math.max(this.nextLink, lid + 1);
    this.links.set(lid, { id: lid, from: [fn, fo], to: [tn, ti] });
  }

  canConnect(fn, fo, tn, ti) {
    if (fn === tn) return false;
    const ot = this.nodes.get(fn).def.outputs[fo].type;
    const it = this.nodes.get(tn).def.inputs[ti].type;
    return ot === it || it === "ANY";
  }

  connect(fn, fo, tn, ti) {
    if (!this.canConnect(fn, fo, tn, ti)) { toast("Incompatible port types", true); return false; }
    if (this._wouldCycle(fn, tn)) { toast("That connection would create a cycle", true); return false; }
    this._addLink(fn, fo, tn, ti);
    this.drawWires(); this.snapshot(); this.onChange();
    return true;
  }

  _wouldCycle(from, to) {
    const stack = [to], seen = new Set();
    while (stack.length) {
      const x = stack.pop();
      if (x === from) return true;
      if (seen.has(x)) continue; seen.add(x);
      for (const l of this.links.values()) if (l.from[0] === x) stack.push(l.to[0]);
    }
    return false;
  }

  linkInto(tn, ti) { return [...this.links.values()].find((l) => l.to[0] === tn && l.to[1] === ti); }

  _portPos(nodeId, dir, idx) {
    const n = this.nodes.get(nodeId);
    const row = n.portEls[dir][idx];
    if (!row) return [n.pos[0], n.pos[1]];
    const dot = row.querySelector(".pd");
    const r = dot.getBoundingClientRect(), v = this.vp.getBoundingClientRect();
    return [(r.left + r.width / 2 - v.left - this.view.x) / this.view.z, (r.top + r.height / 2 - v.top - this.view.y) / this.view.z];
  }

  _curve([x1, y1], [x2, y2]) {
    const dx = Math.max(40, Math.abs(x2 - x1) * 0.5);
    return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
  }

  drawWires() {
    const parts = [];
    for (const l of this.links.values()) {
      const a = this._portPos(l.from[0], "out", l.from[1]);
      const b = this._portPos(l.to[0], "in", l.to[1]);
      const t = this.nodes.get(l.from[0]).def.outputs[l.from[1]].type;
      const d = this._curve(a, b);
      const sel = this.selectedLink === l.id ? " sel" : "";
      parts.push(`<path class="${sel}" d="${d}" stroke="${this.types[t] || "#999"}"/>`);
      parts.push(`<path class="hit" data-link="${l.id}" d="${d}"/>`);
    }
    if (this._temp) parts.push(`<path class="temp" d="${this._curve(this._temp.a, this._temp.b)}" stroke="${this._temp.color}"/>`);
    this.svg.innerHTML = parts.join("");
    for (const p of this.svg.querySelectorAll("path.hit")) {
      p.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        const id = +p.dataset.link;
        this.selectedLink = id; this.drawWires();
        showMenuAt(e.clientX, e.clientY, [
          { label: "Delete link", onClick: () => { this.links.delete(id); this.selectedLink = null; this.drawWires(); this.snapshot(); this.onChange(); } },
        ]);
      });
    }
  }

  // ---------------------------------------------------------------- view
  applyView() {
    const { x, y, z } = this.view;
    this.world.style.transform = `translate(${x}px, ${y}px) scale(${z})`;
    const g = 22 * z;
    this.vp.style.backgroundSize = `${g}px ${g}px`;
    this.vp.style.backgroundPosition = `${x}px ${y}px`;
  }

  screenToWorld(cx, cy) {
    const v = this.vp.getBoundingClientRect();
    return [(cx - v.left - this.view.x) / this.view.z, (cy - v.top - this.view.y) / this.view.z];
  }

  fitView() {
    if (!this.nodes.size) return;
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const n of this.nodes.values()) {
      x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1]);
      x1 = Math.max(x1, n.pos[0] + n.el.offsetWidth); y1 = Math.max(y1, n.pos[1] + n.el.offsetHeight);
    }
    const v = this.vp.getBoundingClientRect();
    const z = clamp(Math.min((v.width - 80) / (x1 - x0), (v.height - 80) / (y1 - y0)), 0.2, 1.2);
    this.view = { z, x: (v.width - (x1 - x0) * z) / 2 - x0 * z, y: (v.height - (y1 - y0) * z) / 2 - y0 * z };
    this.applyView(); this.drawWires();
  }

  // ---------------------------------------------------------------- interactions
  _bindCanvas() {
    const vp = this.vp;
    vp.addEventListener("wheel", (e) => {
      e.preventDefault();
      const v = vp.getBoundingClientRect();
      const mx = e.clientX - v.left, my = e.clientY - v.top;
      const z0 = this.view.z;
      const z = clamp(z0 * Math.exp(-e.deltaY * 0.0012), 0.15, 2.5);
      this.view.x = mx - (mx - this.view.x) * (z / z0);
      this.view.y = my - (my - this.view.y) * (z / z0);
      this.view.z = z;
      this.applyView(); this.drawWires();
    }, { passive: false });

    vp.addEventListener("pointermove", (e) => { this.mouse = { x: e.clientX, y: e.clientY }; });

    vp.addEventListener("pointerdown", (e) => {
      if (e.target !== vp && e.target !== this.world && e.target !== this.svg) return;
      closeMenus();
      if (this.selectedLink) { this.selectedLink = null; this.drawWires(); }
      if (e.button === 2) return;
      if (e.button === 0 && (e.ctrlKey || e.shiftKey)) return this._startBoxSelect(e);
      if (e.button === 0 || e.button === 1) {
        if (!e.shiftKey) this.select([]);
        const sx = e.clientX, sy = e.clientY, ox = this.view.x, oy = this.view.y;
        vp.classList.add("panning");
        vp.setPointerCapture(e.pointerId);
        const move = (ev) => { this.view.x = ox + ev.clientX - sx; this.view.y = oy + ev.clientY - sy; this.applyView(); };
        const up = () => { vp.classList.remove("panning"); vp.removeEventListener("pointermove", move); vp.removeEventListener("pointerup", up); this.onChange(false); };
        vp.addEventListener("pointermove", move); vp.addEventListener("pointerup", up);
      }
    });
    vp.addEventListener("dblclick", (e) => {
      if (e.target !== vp && e.target !== this.world && e.target !== this.svg) return;
      this.openSearch(e.clientX, e.clientY);
    });
    vp.addEventListener("contextmenu", (e) => {
      if (e.target !== vp && e.target !== this.world && e.target !== this.svg) return;
      e.preventDefault();
      this._canvasMenu(e);
    });
    // drag & drop an image file onto the canvas -> Load Image node
    vp.addEventListener("dragover", (e) => e.preventDefault());
    vp.addEventListener("drop", async (e) => {
      e.preventDefault();
      const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
      if (!f) return;
      const fd = new FormData(); fd.append("file", f);
      try {
        const r = await api("/api/upload", { method: "POST", body: fd });
        await this.reloadSchema?.();
        const n = this.addNode("LoadImage", this.screenToWorld(e.clientX, e.clientY), { image: r.filename });
        this.refreshWidget(n.id, "image");
      } catch (err) { toast(err.message, true); }
    });
  }

  _startBoxSelect(e) {
    const box = document.getElementById("selbox");
    const v = this.vp.getBoundingClientRect();
    const sx = e.clientX, sy = e.clientY;
    box.classList.remove("hidden");
    const move = (ev) => {
      const x = Math.min(sx, ev.clientX) - v.left, y = Math.min(sy, ev.clientY) - v.top;
      Object.assign(box.style, { left: x + "px", top: y + "px", width: Math.abs(ev.clientX - sx) + "px", height: Math.abs(ev.clientY - sy) + "px" });
    };
    const up = (ev) => {
      box.classList.add("hidden");
      const [ax, ay] = this.screenToWorld(Math.min(sx, ev.clientX), Math.min(sy, ev.clientY));
      const [bx, by] = this.screenToWorld(Math.max(sx, ev.clientX), Math.max(sy, ev.clientY));
      const ids = [...this.nodes.values()].filter((n) => n.pos[0] < bx && n.pos[0] + n.el.offsetWidth > ax && n.pos[1] < by && n.pos[1] + n.el.offsetHeight > ay).map((n) => n.id);
      this.select(e.shiftKey ? [...this.selected, ...ids] : ids);
      removeEventListener("pointermove", move); removeEventListener("pointerup", up);
    };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  }

  select(ids) {
    this.selected = new Set(ids);
    for (const n of this.nodes.values()) this._applyNodeClasses(n);
  }

  _selectOnPointer(e, id) {
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      const s = new Set(this.selected);
      s.has(id) ? s.delete(id) : s.add(id);
      this.select([...s]);
    } else if (!this.selected.has(id)) this.select([id]);
  }

  _startNodeDrag(e, id) {
    if (e.button !== 0) return;
    e.stopPropagation();
    this._selectOnPointer(e, id);
    const ids = this.selected.has(id) ? [...this.selected] : [id];
    const start = ids.map((i) => [...this.nodes.get(i).pos]);
    const sx = e.clientX, sy = e.clientY;
    let moved = false;
    const move = (ev) => {
      moved = true;
      const dx = (ev.clientX - sx) / this.view.z, dy = (ev.clientY - sy) / this.view.z;
      ids.forEach((i, k) => { const n = this.nodes.get(i); n.pos = [start[k][0] + dx, start[k][1] + dy]; this._placeNode(n); });
      this.drawWires();
    };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); if (moved) { this.snapshot(); this.onChange(); } };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  }

  _startResize(e, id) {
    e.stopPropagation();
    const n = this.nodes.get(id);
    const sx = e.clientX, w0 = n.width;
    const move = (ev) => { n.width = clamp(w0 + (ev.clientX - sx) / this.view.z, 180, 900); n.el.style.width = n.width + "px"; this.drawWires(); };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); this.onChange(); };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  }

  _dragWire(e, anchor, color, onDrop) {
    e.stopPropagation(); e.preventDefault();
    this._temp = { a: anchor, b: anchor, color };
    const move = (ev) => {
      const p = this.screenToWorld(ev.clientX, ev.clientY);
      this._temp.b = p;
      if (this._temp.reverse) this._temp.a = p;
      this.drawWires();
    };
    const up = (ev) => {
      removeEventListener("pointermove", move); removeEventListener("pointerup", up);
      this._temp = null; this.drawWires();
      onDrop(ev);
    };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  }

  _portUnder(ev, dir) {
    const el = document.elementFromPoint(ev.clientX, ev.clientY);
    const row = el?.closest?.(".port." + dir);
    const nodeEl = el?.closest?.(".node");
    if (!row || !nodeEl) return null;
    const n = this.nodes.get(+nodeEl.dataset.id);
    const idx = n.portEls[dir].indexOf(row);
    return idx >= 0 ? [n.id, idx] : null;
  }

  _startWireFromOutput(e, nid, oi) {
    const n = this.nodes.get(nid);
    const t = n.def.outputs[oi].type;
    this._dragWire(e, this._portPos(nid, "out", oi), this.types[t], (ev) => {
      const hit = this._portUnder(ev, "in");
      if (hit) return this.connect(nid, oi, hit[0], hit[1]);
      if (document.elementFromPoint(ev.clientX, ev.clientY)?.closest?.(".node")) return;
      this.openSearch(ev.clientX, ev.clientY, { fromOutput: [nid, oi, t] });
    });
  }

  _startWireFromInput(e, nid, ii) {
    const existing = this.linkInto(nid, ii);
    if (existing) {   // pick up the existing wire and re-route it
      this.links.delete(existing.id);
      this.drawWires();
      const [fn, fo] = existing.from;
      const t = this.nodes.get(fn).def.outputs[fo].type;
      this._dragWire(e, this._portPos(fn, "out", fo), this.types[t], (ev) => {
        const hit = this._portUnder(ev, "in");
        if (hit && this.canConnect(fn, fo, hit[0], hit[1])) this._addLink(fn, fo, hit[0], hit[1]);
        this.drawWires(); this.snapshot(); this.onChange();
      });
      return;
    }
    const t = this.nodes.get(nid).def.inputs[ii].type;
    const anchor = this._portPos(nid, "in", ii);
    this._temp = null;
    e.stopPropagation(); e.preventDefault();
    this._temp = { a: anchor, b: anchor, color: this.types[t], reverse: true };
    const move = (ev) => { this._temp.a = this.screenToWorld(ev.clientX, ev.clientY); this._temp.b = anchor; this.drawWires(); };
    const up = (ev) => {
      removeEventListener("pointermove", move); removeEventListener("pointerup", up);
      this._temp = null; this.drawWires();
      const hit = this._portUnder(ev, "out");
      if (hit) return this.connect(hit[0], hit[1], nid, ii);
      if (document.elementFromPoint(ev.clientX, ev.clientY)?.closest?.(".node")) return;
      this.openSearch(ev.clientX, ev.clientY, { toInput: [nid, ii, t] });
    };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  }

  // ---------------------------------------------------------------- menus & search
  _canvasMenu(e) {
    const pos = this.screenToWorld(e.clientX, e.clientY);
    const cats = {};
    for (const d of this.schema.nodes) (cats[d.category] ||= []).push(d);
    const addItems = Object.entries(cats).map(([c, list]) => ({
      label: c, children: list.map((d) => ({ label: d.title, onClick: () => this.addNode(d.type, pos) })),
    }));
    showMenuAt(e.clientX, e.clientY, [
      { label: "Add node", children: addItems },
      { label: "Search nodes…", key: "dbl-click", onClick: () => this.openSearch(e.clientX, e.clientY) },
      { sep: true },
      { label: "Paste", key: "Ctrl+V", onClick: () => this.paste(pos) },
      { label: "Select all", key: "Ctrl+A", onClick: () => this.select([...this.nodes.keys()]) },
      { label: "Fit view", key: ".", onClick: () => this.fitView() },
    ]);
  }

  _nodeMenu(e, id) {
    if (!this.selected.has(id)) this.select([id]);
    const n = this.nodes.get(id);
    const outTypes = n.def.outputs.map((o) => o.type).join(",");
    const inTypes = n.def.inputs.map((o) => o.type);
    const replacements = this.schema.nodes.filter((d) => d.type !== n.type && d.outputs.map((o) => o.type).join(",") === outTypes && d.outputs.length > 0
      && d.inputs.some((i) => inTypes.includes(i.type)));
    const sweepable = n.def.params.filter((p) => p.sweepable !== false);
    showMenuAt(e.clientX, e.clientY, [
      { header: n.def.title },
      { label: n.bypass ? "Enable (un-bypass)" : "Bypass", key: "Ctrl+B", onClick: () => this.toggleBypass([...this.selected]) },
      ...(replacements.length ? [{ label: "Replace with", children: replacements.map((d) => ({ label: d.title, onClick: () => this.replaceNode(id, d.type) })) }] : []),
      ...(sweepable.length ? [{ label: "Sweep parameter", children: sweepable.map((p) => ({ label: p.name, onClick: () => this.onSweepRequest?.(id, p.name) })) }] : []),
      { label: "Rename", onClick: () => this.rename(id) },
      { label: n.collapsed ? "Expand" : "Collapse", onClick: () => this.toggleCollapse(id) },
      { label: "Duplicate", key: "Ctrl+D", onClick: () => this.duplicate() },
      { label: "Reset parameters", onClick: () => { for (const p of n.def.params) n.params[p.name] = p.default; this.rebuildNode(id); this.snapshot(); this.onChange(); } },
      { label: "About this node", onClick: () => this.onHelp?.(n.def) },
      { sep: true },
      { label: "Delete", key: "Del", onClick: () => this.removeNodes([...this.selected]) },
    ]);
  }

  replaceNode(id, newType) {
    const old = this.nodes.get(id);
    const inLinks = [...this.links.values()].filter((l) => l.to[0] === id);
    const outLinks = [...this.links.values()].filter((l) => l.from[0] === id);
    const pos = [...old.pos];
    this.removeNodes([id]);
    const n = this._createNode(newType, pos, {});
    for (const l of inLinks) {
      const t = this.nodes.get(l.from[0]).def.outputs[l.from[1]].type;
      const oldName = old.def.inputs[l.to[1]].name;
      let idx = n.def.inputs.findIndex((p) => p.type === t && p.name === oldName);
      if (idx < 0) idx = n.def.inputs.findIndex((p, i) => p.type === t && !this.linkInto(n.id, i));
      if (idx >= 0) this._addLink(l.from[0], l.from[1], n.id, idx);
    }
    for (const l of outLinks) {
      const t = old.def.outputs[l.from[1]].type;
      const idx = n.def.outputs.findIndex((p) => p.type === t);
      if (idx >= 0) this._addLink(n.id, idx, l.to[0], l.to[1]);
    }
    // carry over same-named parameters
    for (const p of n.def.params) if (p.name in old.params && (!p.choices || p.choices.includes(old.params[p.name]))) n.params[p.name] = old.params[p.name];
    this.rebuildNode(n.id);
    const missing = n.def.inputs.filter((p, i) => !p.optional && !this.linkInto(n.id, i)).map((p) => p.name);
    if (missing.length) toast(`${n.def.title}: connect ${missing.join(", ")}`);
    this.select([n.id]);
    this.snapshot(); this.onChange();
  }

  openSearch(cx, cy, opts = {}) {
    document.getElementById("search")?.remove();
    let list = this.schema.nodes;
    if (opts.fromOutput) list = list.filter((d) => d.inputs.some((i) => i.type === opts.fromOutput[2] || i.type === "ANY"));
    if (opts.toInput) list = list.filter((d) => d.outputs.some((o) => o.type === opts.toInput[2] || opts.toInput[2] === "ANY"));
    const input = h("input", { placeholder: opts.fromOutput || opts.toInput ? `Nodes compatible with ${(opts.fromOutput || opts.toInput)[2]}…` : "Search nodes…" });
    const res = h("div", { class: "res" });
    const box = h("div", { id: "search" }, input, res);
    box.style.left = Math.min(cx, innerWidth - 350) + "px";
    box.style.top = Math.min(cy, innerHeight - 420) + "px";
    document.body.appendChild(box);
    let active = 0, shown = [];
    const pos = this.screenToWorld(cx, cy);
    const choose = (d) => {
      box.remove();
      const n = this.addNode(d.type, pos);
      if (opts.fromOutput) {
        const [fn, fo, t] = opts.fromOutput;
        const i = d.inputs.findIndex((x) => x.type === t) >= 0 ? d.inputs.findIndex((x) => x.type === t) : d.inputs.findIndex((x) => x.type === "ANY");
        if (i >= 0) this.connect(fn, fo, n.id, i);
      } else if (opts.toInput) {
        const [tn, ti, t] = opts.toInput;
        const o = d.outputs.findIndex((x) => x.type === t || t === "ANY");
        if (o >= 0) this.connect(n.id, o, tn, ti);
      }
      requestAnimationFrame(() => this.drawWires());
    };
    const render = () => {
      const q = input.value.toLowerCase().trim();
      shown = list.filter((d) => !q || (d.title + " " + d.category + " " + d.type + " " + d.description).toLowerCase().includes(q));
      active = clamp(active, 0, Math.max(0, shown.length - 1));
      res.innerHTML = "";
      shown.forEach((d, i) => res.appendChild(h("div", {
        class: "ri" + (i === active ? " active" : ""), title: d.description, onclick: () => choose(d),
      }, h("span", { class: "dot", style: { width: "8px", height: "8px", borderRadius: "50%", background: CAT_COLORS[d.category] } }), d.title, h("span", { class: "cat" }, d.category))));
    };
    input.addEventListener("input", () => { active = 0; render(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { active++; render(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { active--; render(); e.preventDefault(); }
      else if (e.key === "Enter" && shown[active]) choose(shown[active]);
      else if (e.key === "Escape") box.remove();
    });
    setTimeout(() => {
      const off = (ev) => { if (!box.contains(ev.target)) { box.remove(); removeEventListener("pointerdown", off, true); } };
      addEventListener("pointerdown", off, true);
    });
    render(); input.focus();
  }

  // ---------------------------------------------------------------- clipboard / history
  copy() {
    const ids = new Set(this.selected);
    if (!ids.size) return;
    const g = this.toJSON(false);
    this.clipboard = { nodes: g.nodes.filter((n) => ids.has(n.id)), links: g.links.filter((l) => ids.has(l.from[0]) && ids.has(l.to[0])) };
  }

  paste(at) {
    if (!this.clipboard) return;
    const minX = Math.min(...this.clipboard.nodes.map((n) => n.pos[0])), minY = Math.min(...this.clipboard.nodes.map((n) => n.pos[1]));
    const [px, py] = at || this.screenToWorld(this.mouse.x, this.mouse.y);
    const map = {};
    const newIds = [];
    for (const n of this.clipboard.nodes) {
      const nn = this._createNode(n.type, [px + n.pos[0] - minX, py + n.pos[1] - minY], { ...n, id: undefined });
      map[n.id] = nn.id; newIds.push(nn.id);
    }
    for (const l of this.clipboard.links) this._addLink(map[l.from[0]], l.from[1], map[l.to[0]], l.to[1]);
    this.select(newIds);
    requestAnimationFrame(() => this.drawWires());
    this.snapshot(); this.onChange();
  }

  duplicate() {
    const [x, y] = [...this.selected].map((i) => this.nodes.get(i).pos).reduce((a, p) => [Math.min(a[0], p[0]), Math.min(a[1], p[1])], [1e9, 1e9]);
    this.copy(); this.paste([x + 40, y + 40]);
  }

  snapshot(reset = false) {
    clearTimeout(this._snapTimer);
    this._snapTimer = setTimeout(() => {
      const s = JSON.stringify(this.toJSON(false));
      if (reset) { this.undoStack = [s]; this.redoStack = []; return; }
      if (this.undoStack[this.undoStack.length - 1] === s) return;
      this.undoStack.push(s);
      if (this.undoStack.length > 80) this.undoStack.shift();
      this.redoStack = [];
    }, reset ? 0 : 250);
  }

  undo() {
    if (this.undoStack.length < 2) return;
    this.redoStack.push(this.undoStack.pop());
    this.load(JSON.parse(this.undoStack[this.undoStack.length - 1]), { keepView: true, snapshot: false });
  }

  redo() {
    if (!this.redoStack.length) return;
    const s = this.redoStack.pop();
    this.undoStack.push(s);
    this.load(JSON.parse(s), { keepView: true, snapshot: false });
  }

  _bindKeys() {
    addEventListener("keydown", (e) => {
      const tag = (document.activeElement?.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag)) return;
      if (!document.getElementById("lab").classList.contains("hidden")) return;
      const mod = e.ctrlKey || e.metaKey;
      if ((e.key === "Delete" || e.key === "Backspace") && this.selected.size) { this.removeNodes([...this.selected]); e.preventDefault(); }
      else if (mod && e.key.toLowerCase() === "c") this.copy();
      else if (mod && e.key.toLowerCase() === "v") this.paste();
      else if (mod && e.key.toLowerCase() === "d") { e.preventDefault(); this.duplicate(); }
      else if (mod && e.key.toLowerCase() === "b") { e.preventDefault(); this.toggleBypass([...this.selected]); }
      else if (mod && e.key.toLowerCase() === "a") { e.preventDefault(); this.select([...this.nodes.keys()]); }
      else if (mod && e.key.toLowerCase() === "z" && e.shiftKey) { e.preventDefault(); this.redo(); }
      else if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); this.undo(); }
      else if (mod && e.key.toLowerCase() === "y") { e.preventDefault(); this.redo(); }
      else if (e.key === ".") this.fitView();
    });
  }

  // ---------------------------------------------------------------- run status
  clearStatus() {
    this.status = {};
    for (const n of this.nodes.values()) { n.badgeEl.textContent = ""; n.badgeEl.className = "badge"; n.el.classList.remove("running", "error"); }
  }

  setNodeStatus(id, st) {
    const n = this.nodes.get(+id);
    if (!n) return;
    this.status[+id] = st;
    n.el.classList.toggle("running", st.state === "running");
    n.el.classList.toggle("error", st.state === "error");
    const b = n.badgeEl;
    b.className = "badge";
    if (st.state === "running") b.textContent = "running…";
    else if (st.state === "error") { b.textContent = "error"; b.classList.add("err"); }
    else if (st.state === "skipped") b.textContent = "skipped";
    else if (st.state === "bypassed") b.textContent = "bypassed";
    else if (st.state === "done") {
      b.textContent = st.cached ? "cached" : fmtT(st.time);
      if (st.cached) b.classList.add("cached");
    }
    if (st.state !== "running") this._renderOutput(n, st);
    requestAnimationFrame(() => this.drawWires());
  }

  _renderOutput(n, st) {
    const out = n.outEl;
    out.innerHTML = "";
    if (st.error) {
      out.appendChild(h("div", { class: "nerr", title: st.trace || "" }, st.error));
      return;
    }
    const ui = st.ui || {};
    if (ui.verdict) {
      const cls = ui.verdict.startsWith("✓") ? "ok" : ui.verdict.startsWith("✗") ? "bad" : "na";
      out.appendChild(h("div", { class: "verdict " + cls }, ui.verdict));
    }
    for (const im of ui.images || []) {
      if (!im.url) continue;
      const img = h("img", { src: im.url, loading: "lazy", onclick: () => lightbox(im.url, im.label) });
      img.addEventListener("load", () => this.drawWires());
      out.appendChild(h("div", {}, img, im.label ? h("div", { class: "cap" }, im.label) : null));
    }
    if (ui.probs) {
      const box = h("div", { class: "probs" });
      for (const [k, v] of Object.entries(ui.probs)) {
        box.appendChild(h("div", { class: "pr" }, h("b", {}, k), h("div", { class: "bar" }, h("i", { style: { width: (v * 100).toFixed(1) + "%" } })), (v * 100).toFixed(1) + "%"));
      }
      out.appendChild(box);
    }
    if (ui.metrics) {
      out.appendChild(h("div", { class: "chips" }, Object.entries(ui.metrics).map(([k, v]) => h("span", { class: "chip" }, k + " ", h("b", {}, typeof v === "number" ? String(+v.toFixed(4)) : String(v))))));
    }
    if (ui.notes) for (const t of ui.notes) out.appendChild(h("div", { class: "notes" }, "⚠ " + t));
    if (ui.text !== undefined && ui.text !== "") out.appendChild(h("pre", {}, String(ui.text)));
  }

  refreshSweepMarks() {
    this.sweepKeys = new Set((this.experiment?.sweep || []).map((s) => `${s.node_id}.${s.param}`));
    for (const n of this.nodes.values()) {
      for (const [name, w] of Object.entries(n.widgetEls)) {
        w.querySelector("label")?.classList.toggle("swept", this.sweepKeys.has(`${n.id}.${name}`));
      }
    }
  }
}

function fmtT(s) {
  if (s === undefined || s === null) return "";
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`;
}

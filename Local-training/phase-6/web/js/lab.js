// Experiment launcher + results dashboard.
import { api, h, toast, lightbox, fmtPct, fmtNum, fmtTime } from "./util.js";
import { SERIES, groupedBars, lineChart, divergingBars } from "./charts.js";

let MMAD_INFO = null;

const DEFAULT_EXP = { name: "", notes: "", n: 100, seed: 42, datasets: [], subtasks: [], condition: "any", save_thumbnails: true, sweep: [] };

function parseValues(text, p) {
  const out = [];
  for (let tok of text.split(",").map((t) => t.trim()).filter(Boolean)) {
    const rng = tok.match(/^(-?[\d.]+):(-?[\d.]+):(-?[\d.]+)$/);
    if (rng && (p.kind === "int" || p.kind === "float")) {
      const [a, b, st] = rng.slice(1).map(Number);
      for (let v = a; st > 0 ? v <= b + 1e-9 : v >= b - 1e-9; v += st) out.push(p.kind === "int" ? Math.round(v) : +v.toFixed(6));
      continue;
    }
    if (p.kind === "int") out.push(parseInt(tok, 10));
    else if (p.kind === "float") out.push(parseFloat(tok));
    else if (p.kind === "bool") out.push(/^(1|true|yes|on)$/i.test(tok));
    else out.push(tok);
  }
  return out.filter((v) => !(typeof v === "number" && Number.isNaN(v)));
}

function defaultValues(p, cur) {
  if (p.kind === "choice") return (p.choices || []).join(", ");
  if (p.kind === "bool") return "false, true";
  if (p.kind === "int" || p.kind === "float") {
    if (p.name === "severity") return "0:5:1";
    return String(cur ?? p.default);
  }
  return String(cur ?? p.default ?? "");
}

export async function openExperimentDialog(editor, { addAxis = null, onLaunched, lastRunSeconds = null } = {}) {
  if (!MMAD_INFO) MMAD_INFO = await api("/api/mmad/info");
  const exp = JSON.parse(JSON.stringify({ ...DEFAULT_EXP, ...(editor.experiment || {}) }));
  exp.sweep = (exp.sweep || []).map((s) => ({ ...s, text: s.text ?? s.values.join(", ") }));
  if (addAxis) {
    const n = editor.nodes.get(addAxis.node_id);
    const p = n.def.params.find((x) => x.name === addAxis.param);
    if (!exp.sweep.some((s) => s.node_id === addAxis.node_id && s.param === addAxis.param))
      exp.sweep.push({ node_id: addAxis.node_id, param: addAxis.param, text: defaultValues(p, n.params[p.name]) });
  }

  const bg = h("div", { class: "modal-bg" });
  const close = () => bg.remove();
  bg.addEventListener("pointerdown", (e) => { if (e.target === bg) close(); });
  const g = editor.toJSON(false);
  const samplers = g.nodes.filter((n) => n.type === "MMADSample").length;
  const scorers = g.nodes.filter((n) => n.type === "Score" || n.type === "DetectorMetrics");

  const name = h("input", { value: exp.name, placeholder: "e.g. PatchCore vs WinCLIP, 200 q" });
  const notes = h("textarea", { rows: 2, placeholder: "hypothesis / notes (saved with the run)" }, exp.notes || "");
  const n = h("input", { type: "number", min: 1, max: 40000, value: exp.n });
  const seed = h("input", { type: "number", value: exp.seed });
  const cond = h("select", {}, ["any", "defective", "normal"].map((c) => h("option", { selected: c === exp.condition }, c)));
  const thumbs = h("input", { type: "checkbox", checked: exp.save_thumbnails });
  const dsChecks = MMAD_INFO.datasets.map((d) => h("input", { type: "checkbox", value: d, checked: exp.datasets.includes(d) }));
  const stChecks = MMAD_INFO.subtasks.map((d) => h("input", { type: "checkbox", value: d, checked: exp.subtasks.includes(d) }));

  const nodeOpts = [...editor.nodes.values()].filter((nd) => nd.def.params.some((p) => p.sweepable !== false));
  const sweepBox = h("div", { style: { display: "flex", flexDirection: "column", gap: "6px" } });
  const est = h("span", { class: "est" });

  const renderSweep = () => {
    sweepBox.innerHTML = "";
    exp.sweep.forEach((ax, i) => {
      const nsel = h("select", {}, nodeOpts.map((nd) => h("option", { value: nd.id, selected: nd.id === +ax.node_id }, `${editor.nodeTitle(nd)} #${nd.id}`)));
      const nd = editor.nodes.get(+ax.node_id) || nodeOpts[0];
      const params = nd ? nd.def.params.filter((p) => p.sweepable !== false) : [];
      const psel = h("select", {}, params.map((p) => h("option", { selected: p.name === ax.param }, p.name)));
      const vals = h("input", { value: ax.text ?? "", placeholder: "comma list or start:stop:step" });
      nsel.onchange = () => { ax.node_id = +nsel.value; const nn = editor.nodes.get(ax.node_id); const p0 = nn.def.params.find((p) => p.sweepable !== false); ax.param = p0.name; ax.text = defaultValues(p0, nn.params[p0.name]); renderSweep(); };
      psel.onchange = () => { ax.param = psel.value; const p = nd.def.params.find((x) => x.name === ax.param); ax.text = defaultValues(p, nd.params[p.name]); renderSweep(); };
      vals.oninput = () => { ax.text = vals.value; update(); };
      const p = params.find((x) => x.name === ax.param);
      sweepBox.appendChild(h("div", { class: "sweep-row" }, nsel, psel, vals,
        h("button", { title: "remove", onclick: () => { exp.sweep.splice(i, 1); renderSweep(); } }, "✕")));
      if (p?.kind === "choice") sweepBox.appendChild(h("div", { class: "hint" }, "choices: " + (p.choices || []).join(" · ")));
    });
    if (!nodeOpts.length) sweepBox.appendChild(h("div", { class: "hint" }, "No sweepable nodes in the graph."));
    update();
  };

  const collect = () => {
    const sweep = exp.sweep.map((ax) => {
      const nd = editor.nodes.get(+ax.node_id);
      const p = nd?.def.params.find((x) => x.name === ax.param);
      return { node_id: +ax.node_id, param: ax.param, values: p ? parseValues(ax.text || "", p) : [], text: ax.text };
    }).filter((s) => s.values.length);
    return {
      name: name.value.trim(), notes: notes.value, n: +n.value, seed: +seed.value, condition: cond.value,
      save_thumbnails: thumbs.checked, datasets: dsChecks.filter((c) => c.checked).map((c) => c.value),
      subtasks: stChecks.filter((c) => c.checked).map((c) => c.value), sweep,
    };
  };

  function update() {
    const c = collect();
    const variants = c.sweep.reduce((a, s) => a * s.values.length, 1);
    const total = variants * c.n;
    let t = `${c.n} questions × ${variants} variant${variants > 1 ? "s" : ""} = ${total.toLocaleString()} executions`;
    if (lastRunSeconds) t += ` · ≈ ${fmtTime(lastRunSeconds * total)} at your last run's speed (caching makes sweeps faster)`;
    est.textContent = t;
  }

  const launch = async () => {
    const c = collect();
    editor.experiment = { ...c, sweep: c.sweep.map(({ text, ...rest }) => ({ ...rest, text })) };
    editor.refreshSweepMarks(); editor.onChange();
    try {
      const r = await api("/api/experiments", { method: "POST", body: { ...c, graph: editor.toJSON(false) } });
      toast(`Experiment ${r.run_id} queued`);
      close();
      onLaunched?.(r.run_id);
    } catch (e) { toast(e.message, true, 6000); }
  };

  const warn = [];
  if (samplers !== 1) warn.push("The graph needs exactly one MMAD Sample node — it becomes the dataset iterator.");
  if (!scorers.length) warn.push("Add a Score node (and/or Detector Metrics) so the run measures something.");

  bg.appendChild(h("div", { class: "modal" },
    h("header", {}, h("h2", {}, "⚗ New experiment"), h("button", { class: "tb-btn", onclick: close }, "✕")),
    h("div", { class: "mb" },
      warn.length ? h("div", { class: "notes" }, warn.map((w) => h("div", {}, "⚠ " + w))) : null,
      h("div", { class: "row" }, h("label", { class: "field" }, h("span", {}, "Name"), name), h("label", { class: "field" }, h("span", {}, "Questions (stratified by subtask)"), n),
        h("label", { class: "field" }, h("span", {}, "Sampling seed"), seed), h("label", { class: "field" }, h("span", {}, "Image condition"), cond)),
      h("label", { class: "field" }, h("span", {}, "Notes"), notes),
      h("div", { class: "field" }, h("span", {}, "Datasets (none = all)"), h("div", { class: "checks" }, dsChecks.map((c) => h("label", {}, c, c.value)))),
      h("div", { class: "field" }, h("span", {}, "Subtasks (none = all 9)"), h("div", { class: "checks" }, stChecks.map((c) => h("label", {}, c, c.value)))),
      h("div", { class: "field" }, h("span", {}, "Parameter sweep (grid over all axes) — right-click any widget label to add it"), sweepBox,
        h("div", {}, h("button", { class: "linkbtn", onclick: () => { const nd = nodeOpts[0]; if (!nd) return; const p = nd.def.params.find((x) => x.sweepable !== false); exp.sweep.push({ node_id: nd.id, param: p.name, text: defaultValues(p, nd.params[p.name]) }); renderSweep(); } }, "+ add sweep axis"))),
      h("label", { class: "checks" }, h("label", {}, thumbs, "save per-sample thumbnails (for the sample browser)")),
      h("div", { class: "hint" }, scorers.length ? `Measured by: ${scorers.map((s) => `${s.type === "Score" ? "Score" : "Detector"} “${s.params?.label ?? ""}”`).join(", ")}. Every sweep variant runs on the same questions, so arms and variants are paired.` : ""),
    ),
    h("footer", {}, est, h("button", { class: "tb-btn", onclick: () => { editor.experiment = collect(); editor.refreshSweepMarks(); editor.onChange(); toast("Experiment preset saved in the workflow"); close(); } }, "Save preset"),
      h("button", { class: "tb-btn primary", onclick: launch }, "Launch")),
  ));
  [n, name].forEach((el) => el.addEventListener("input", update));
  document.body.appendChild(bg);
  renderSweep();
  name.focus();
}

// ============================================================================ dashboard
export class Lab {
  constructor({ editor, onClose }) {
    this.editor = editor; this.onClose = onClose;
    this.root = document.getElementById("lab");
    this.list = document.getElementById("runs");
    this.view = document.getElementById("runview");
    this.current = null; this.compare = new Set(); this.timer = null; this.execCache = {};
    this.variant = null; this.pairIdx = 0;
  }

  get visible() { return !this.root.classList.contains("hidden"); }

  async show(runId = null) {
    this.root.classList.remove("hidden");
    await this.refreshList();
    if (runId) this.open(runId);
    else if (!this.current && this.runs.length) this.open(this.runs[0].run_id);
    else if (!this.runs.length) this.view.innerHTML = "", this.view.appendChild(h("div", { class: "empty" }, "No experiments yet. Build a graph, then press ⚗ Experiment."));
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 2500);
  }

  hide() { this.root.classList.add("hidden"); clearInterval(this.timer); }

  async tick() {
    if (!this.visible) return;
    const live = (r) => r && (r.state === "running" || r.state === "queued");
    if ((this.runs || []).some(live)) await this.refreshList();
    const cur = this.runs?.find((r) => r.run_id === this.current);
    // keep refreshing while the run is live, plus once more when it finishes
    if (live(cur) || this._curWasLive) this.open(this.current, true);
    this._curWasLive = live(cur);
  }

  async refreshList() {
    this.runs = await api("/api/experiments");
    this.list.innerHTML = "";
    this.list.appendChild(h("div", { class: "rh" }, h("b", {}, "Experiments"),
      this.compare.size >= 2 ? h("button", { class: "tb-btn", onclick: () => this.openCompare() }, `Compare ${this.compare.size}`) : null,
      h("button", { class: "tb-btn", onclick: () => { this.hide(); this.onClose?.(); } }, "✕ Editor")));
    for (const r of this.runs) {
      const cb = h("input", { type: "checkbox", checked: this.compare.has(r.run_id), title: "select for comparison", onclick: (e) => e.stopPropagation(), onchange: (e) => { e.target.checked ? this.compare.add(r.run_id) : this.compare.delete(r.run_id); this.refreshList(); } });
      const frac = r.total ? r.done / r.total : 0;
      this.list.appendChild(h("div", { class: "run-item" + (r.run_id === this.current ? " active" : ""), onclick: () => this.open(r.run_id) },
        cb,
        h("div", {},
          h("div", { class: "nm" }, r.name, h("span", { class: "state " + (r.state || "") }, r.state || "?")),
          h("div", { class: "meta" }, `${r.created?.replace("T", " ") || ""} · ${r.n_samples} q × ${r.n_variants} var`),
          r.state !== "done" ? h("div", { class: "progress" }, h("i", { style: { width: (frac * 100).toFixed(1) + "%" } })) : null)));
    }
  }

  async open(runId, silent = false) {
    if (this.current !== runId) { this.variant = null; this.pairIdx = 0; this.sampleFilter = null; }
    this.current = runId;
    if (!silent) [...this.list.querySelectorAll(".run-item")].forEach((el, i) => el.classList.toggle("active", this.runs[i]?.run_id === runId));
    let d;
    try { d = await api(`/api/experiments/${runId}`); } catch (e) { toast(e.message, true); return; }
    this.detail = d;
    const scroll = this.view.scrollTop;
    this.render(d);
    if (silent) this.view.scrollTop = scroll;
  }

  render(d) {
    const { config: cfg, status: st, metrics: m } = d;
    const v = this.view;
    v.innerHTML = "";
    const running = st.state === "running" || st.state === "queued";
    v.appendChild(h("h1", {}, cfg.name || cfg.run_id));
    v.appendChild(h("div", { class: "sub" },
      `${cfg.run_id} · ${cfg.n_samples} questions × ${cfg.n_variants} variants · seed ${cfg.seed} · ${cfg.datasets?.length ? cfg.datasets.join(", ") : "all datasets"} · ` +
      `${cfg.system?.cuda ? cfg.system.gpu : "CPU"} · state ${st.state}` + (st.errors ? ` · ${st.errors} executions with node errors` : "")));
    if (cfg.notes) v.appendChild(h("div", { class: "hint" }, cfg.notes));
    const actions = h("div", { class: "actions" },
      h("button", { class: "tb-btn", onclick: () => { this.editor.load(cfg.graph); this.editor.experiment = { ...(this.editor.experiment || {}), ...pickExp(cfg) }; this.editor.refreshSweepMarks(); this.hide(); this.onClose?.(); toast("Graph loaded into the editor"); } }, "Open graph in editor"),
      running ? h("button", { class: "tb-btn danger", onclick: async () => { await api(`/api/experiments/${cfg.run_id}/cancel`, { method: "POST" }); toast("Stopping…"); } }, "■ Stop")
        : st.done < st.total ? h("button", { class: "tb-btn", onclick: async () => { try { await api(`/api/experiments/${cfg.run_id}/resume`, { method: "POST" }); toast("Resumed"); this.refreshList(); } catch (e) { toast(e.message, true); } } }, "▶ Resume") : null,
      h("a", { class: "tb-btn", href: `/api/experiments/${cfg.run_id}/export.csv` }, "⬇ CSV"),
      h("button", { class: "tb-btn", onclick: async () => { try { const r = await api(`/api/experiments/${cfg.run_id}/figures`, { method: "POST" }); toast(`Saved ${r.files.length} files to ${r.dir}`, false, 7000); this.figs = r.files; this.render(this.detail); } catch (e) { toast(e.message, true); } } }, "🖼 Export figures + manifest"),
      !running ? h("button", { class: "tb-btn danger", onclick: async () => { if (!confirm("Delete this run and its logs?")) return; await api(`/api/experiments/${cfg.run_id}`, { method: "DELETE" }); this.current = null; this.show(); } }, "🗑 Delete") : null,
    );
    v.appendChild(actions);
    if (running || st.state === "cancelled" || st.state === "error") {
      const frac = st.total ? st.done / st.total : 0;
      v.appendChild(h("div", { class: "card" },
        h("div", { class: "ct" }, `${st.done} / ${st.total} executions`, st.rate ? h("span", { class: "hint" }, ` · ${(st.rate * 60).toFixed(1)}/min · ETA ${fmtTime(st.eta_s)}`) : null, st.error ? h("span", { class: "neg" }, st.error) : null),
        h("div", { class: "progress" }, h("i", { style: { width: (frac * 100).toFixed(1) + "%" } }))));
    }
    if (this.figs?.length && this.figs[0].startsWith(cfg.run_id)) {
      v.appendChild(h("div", { class: "thumbs", style: { marginTop: "10px" } }, this.figs.filter((f) => f.endsWith(".png")).map((f) => h("figure", {}, h("img", { src: `/api/results/${f}`, onclick: () => lightbox(`/api/results/${f}`, f) }), h("figcaption", {}, f)))));
    }
    if (!m.answer.length && !m.detector.length) {
      v.appendChild(h("div", { class: "empty" }, running ? "Waiting for the first results…" : "No results recorded."));
      return;
    }

    // ---------------------------------------------------------- variant selector
    const variants = m.variants;
    if (this.variant === null || !variants.some((x) => x.id === this.variant)) this.variant = variants[0]?.id;
    const vlabel = (vv) => Object.entries(vv.params || {}).map(([k, x]) => `${k.split("#")[0]}.${k.split(".").pop()}=${x}`).join(", ") || "baseline";
    const vsel = variants.length > 1 ? h("select", { onchange: (e) => { this.variant = e.target.value; this.render(this.detail); } },
      variants.map((vv) => h("option", { value: vv.id, selected: vv.id === this.variant }, `${vv.id}: ${vlabel(vv)}`))) : null;

    const ans = m.answer.filter((a) => a.variant === this.variant);
    const labels = m.labels;
    const colorOf = (lab) => SERIES[labels.indexOf(lab) % SERIES.length];

    if (ans.length) {
      v.appendChild(h("h3", {}, "Answer accuracy ", vsel));
      v.appendChild(h("div", { class: "tiles" }, ans.map((a) => h("div", { class: "tile" },
        h("div", { class: "tl" }, h("span", { class: "sw", style: { background: colorOf(a.label) } }), a.label),
        h("div", { class: "big" }, fmtPct(a.acc)),
        h("div", { class: "ci" }, `95% CI ${fmtPct(a.ci_lo)} – ${fmtPct(a.ci_hi)} · n=${a.n}`),
        h("div", { class: "kv" },
          h("span", {}, "Cohen's κ"), h("span", {}, fmtNum(a.kappa)),
          h("span", {}, "macro acc (subtasks)"), h("span", {}, fmtPct(a.macro_acc)),
          h("span", {}, "parse failures"), h("span", {}, fmtPct(a.parse_fail)),
          h("span", {}, "MLLM latency"), h("span", {}, fmtTime(a.latency)),
          a.n_fired !== undefined ? [h("span", {}, "cue fired"), h("span", {}, `${a.n_fired}/${a.n}`)] : null,
          a.acc_fired !== undefined && a.acc_fired !== null ? [h("span", {}, "acc | cue fired"), h("span", {}, fmtPct(a.acc_fired))] : null,
          a.acc_not_fired !== undefined && a.acc_not_fired !== null ? [h("span", {}, "acc | no cue"), h("span", {}, fmtPct(a.acc_not_fired))] : null,
          a.ece !== undefined ? [h("span", {}, "ECE"), h("span", {}, fmtNum(a.ece))] : null,
          a.brier_gt !== undefined && a.brier_gt !== null ? [h("span", {}, "Brier (p_gt)"), h("span", {}, fmtNum(a.brier_gt))] : null,
        )))));

      // per-subtask chart
      const subtasks = [...new Set(ans.flatMap((a) => Object.keys(a.per_subtask)))].sort();
      v.appendChild(h("h3", {}, "Accuracy by subtask"));
      v.appendChild(h("div", { class: "card" }, groupedBars({
        categories: subtasks,
        series: ans.map((a) => ({ name: a.label, color: colorOf(a.label), values: subtasks.map((s) => a.per_subtask[s]?.acc ?? null), n: subtasks.map((s) => a.per_subtask[s]?.n ?? 0) })),
      })));

      const dsets = [...new Set(ans.flatMap((a) => Object.keys(a.per_dataset)))].sort();
      if (dsets.length > 1) {
        v.appendChild(h("h3", {}, "Accuracy by dataset"));
        v.appendChild(h("div", { class: "card" }, groupedBars({
          categories: dsets, rotate: false, height: 240,
          series: ans.map((a) => ({ name: a.label, color: colorOf(a.label), values: dsets.map((s) => a.per_dataset[s]?.acc ?? null), n: dsets.map((s) => a.per_dataset[s]?.n ?? 0) })),
        })));
      }
    }

    // ---------------------------------------------------------- paired comparisons
    const pairs = m.paired.filter((p) => p.kind === "vs baseline variant" || p.variant === this.variant);
    if (pairs.length) {
      v.appendChild(h("h3", {}, "Paired comparisons (same questions)"));
      const tbl = h("table", { class: "t" }, h("tr", {}, ["A", "B", "n", "acc A", "acc B", "Δ (B−A)", "fixes", "breaks", "McNemar p"].map((c, i) => h("th", { class: i > 1 ? "num" : "" }, c))));
      pairs.forEach((p, i) => tbl.appendChild(h("tr", { class: "click", onclick: () => { this.pairIdx = i; this.render(this.detail); }, style: i === this.pairIdx ? { background: "var(--panel-2)" } : {} },
        h("td", {}, p.a), h("td", {}, p.b), h("td", { class: "num" }, p.n), h("td", { class: "num" }, fmtPct(p.acc_a)), h("td", { class: "num" }, fmtPct(p.acc_b)),
        h("td", { class: "num " + (p.delta >= 0 ? "pos" : "neg") }, (p.delta >= 0 ? "+" : "") + (p.delta * 100).toFixed(2) + " pp"),
        h("td", { class: "num" }, p.fixes), h("td", { class: "num" }, p.breaks),
        h("td", { class: "num" }, fmtNum(p.p_mcnemar, 4), p.p_mcnemar < 0.05 ? h("span", { class: "sig" }, "p<.05") : null))));
      const sel = pairs[Math.min(this.pairIdx, pairs.length - 1)];
      v.appendChild(h("div", { class: "grid2" },
        h("div", { class: "card" }, h("div", { class: "ct" }, "All pairs (click a row for its subtask breakdown)"), tbl),
        h("div", { class: "card" }, h("div", { class: "ct" }, `Δ by subtask: ${sel.b} − ${sel.a}`),
          divergingBars({ items: Object.entries(sel.per_subtask_delta).sort((a, b) => a[1] - b[1]).map(([k, x]) => ({ label: k, value: x })) }))));
    }

    // ---------------------------------------------------------- detector
    const det = m.detector.filter((x) => x.variant === this.variant || variants.length === 1);
    if (det.length) {
      v.appendChild(h("h3", {}, "Detector quality (vs MMAD ground-truth masks, one row per image)"));
      v.appendChild(h("div", { class: "card" }, h("table", { class: "t" },
        h("tr", {}, ["label", "detector", "images", "defective", "image AUROC (pooled)", "image AUROC (per-category mean)", "pixel AUROC", "peak in defect", "box IoU", "box hit", "fires on defective (TPR)", "fires on normal (FPR)"].map((c, i) => h("th", { class: i > 1 ? "num" : "" }, c))),
        det.map((x) => h("tr", {}, h("td", {}, x.label), h("td", {}, x.detector), h("td", { class: "num" }, x.n_images), h("td", { class: "num" }, x.n_defective),
          h("td", { class: "num", title: `pooled over categories using ${x.auroc_score}` }, fmtNum(x.image_auroc)),
          h("td", { class: "num", title: `${x.n_categories_auroc} categories had both normal and defective images` }, fmtNum(x.image_auroc_macro)),
          h("td", { class: "num" }, fmtNum(x.pixel_auroc)), h("td", { class: "num" }, fmtPct(x.pointing)),
          h("td", { class: "num" }, fmtNum(x.box_iou)), h("td", { class: "num" }, fmtPct(x.box_hit)), h("td", { class: "num" }, fmtPct(x.tpr)), h("td", { class: "num" }, fmtPct(x.fpr)))))));
    }

    // ---------------------------------------------------------- sweeps
    if (m.sweep.length) {
      v.appendChild(h("h3", {}, "Sweeps (accuracy marginalised over other axes)"));
      const byAxis = {};
      for (const s of m.sweep) (byAxis[s.axis] ||= []).push(s);
      const cards = Object.entries(byAxis).map(([axis, list]) => {
        const xs = list[0].points.map((p) => p.value);
        return h("div", { class: "card" }, h("div", { class: "ct" }, axis),
          lineChart({ xs, xLabel: axis.split(".").pop(), series: list.map((s) => ({ name: s.label, color: colorOf(s.label), values: xs.map((x) => s.points.find((p) => String(p.value) === String(x))?.acc ?? null) })) }));
      });
      v.appendChild(h("div", { class: "grid2" }, cards));
      if (m.spread.length) {
        v.appendChild(h("div", { class: "card", style: { marginTop: "12px" } }, h("div", { class: "ct" }, "Spread across sweep values (sensitivity)"),
          h("table", { class: "t" }, h("tr", {}, ["axis", "label", "min", "max", "range", "std"].map((c, i) => h("th", { class: i > 1 ? "num" : "" }, c))),
            m.spread.map((s) => h("tr", {}, h("td", {}, s.axis), h("td", {}, s.label), h("td", { class: "num" }, fmtPct(s.min)), h("td", { class: "num" }, fmtPct(s.max)), h("td", { class: "num" }, (s.range * 100).toFixed(2) + " pp"), h("td", { class: "num" }, (s.std * 100).toFixed(2) + " pp"))))));
      }
      if (m.robustness.length) {
        v.appendChild(h("div", { class: "card", style: { marginTop: "12px" } }, h("div", { class: "ct" }, "Robustness Degradation Slope  RDS(s) = (Acc_clean − Acc_s) / s"),
          h("table", { class: "t" }, h("tr", {}, ["axis", "label", "clean acc", ...Object.keys(m.robustness[0].rds).map((s) => `RDS s=${s}`), "mean RDS"].map((c, i) => h("th", { class: i > 1 ? "num" : "" }, c))),
            m.robustness.map((r) => h("tr", {}, h("td", {}, r.axis), h("td", {}, r.label), h("td", { class: "num" }, fmtPct(r.clean_acc)),
              Object.values(r.rds).map((x) => h("td", { class: "num" }, (x * 100).toFixed(2))), h("td", { class: "num" }, r.mean_rds === null ? "–" : (r.mean_rds * 100).toFixed(2)))))));
      }
    }

    // ---------------------------------------------------------- samples
    v.appendChild(h("h3", {}, "Samples"));
    const holder = h("div", { class: "card" }, h("div", { class: "hint" }, "loading…"));
    v.appendChild(holder);
    this.renderSamples(holder, cfg, m, running);
  }

  async renderSamples(holder, cfg, m, running) {
    const key = cfg.run_id;
    let rows = this.execCache[key];
    if (!rows || running) {
      try { rows = await api(`/api/experiments/${key}/executions`); } catch (e) { holder.textContent = e.message; return; }
      this.execCache[key] = rows;
    }
    rows = rows.filter((r) => r.variant === this.variant);
    const labels = m.labels;
    const f = this.sampleFilter ||= { subtask: "any", outcome: "any", label: labels[0] || "", limit: 200 };
    const subtasks = [...new Set(rows.map((r) => r.subtask))].sort();
    const outcomes = ["any", "correct", "wrong", "parse failure", "cue fired", "no cue"];
    if (labels.length >= 2) outcomes.push(`${labels[1]} fixes ${labels[0]}`, `${labels[1]} breaks ${labels[0]}`);
    const mk = (name, opts, val) => h("select", { onchange: (e) => { f[name] = e.target.value; f.limit = 200; this.renderSamples(holder, cfg, m, false); } }, opts.map((o) => h("option", { selected: o === val }, o)));
    const get = (r, lab) => r.answers.find((a) => a.label === lab);
    const pass = (r) => {
      if (f.subtask !== "any" && r.subtask !== f.subtask) return false;
      const a = get(r, f.label);
      switch (f.outcome) {
        case "correct": return a?.correct;
        case "wrong": return a && !a.correct;
        case "parse failure": return a && !a.parsed;
        case "cue fired": return r.context.cue_fired === true;
        case "no cue": return r.context.cue_fired === false;
        default:
          if (f.outcome.includes(" fixes ")) { const x = get(r, labels[0]), y = get(r, labels[1]); return x && y && !x.correct && y.correct; }
          if (f.outcome.includes(" breaks ")) { const x = get(r, labels[0]), y = get(r, labels[1]); return x && y && x.correct && !y.correct; }
          return true;
      }
    };
    const shown = rows.filter(pass);
    holder.innerHTML = "";
    holder.appendChild(h("div", { class: "samples-filters" },
      labels.length ? mk("label", labels, f.label) : null, mk("subtask", ["any", ...subtasks], f.subtask), mk("outcome", outcomes, f.outcome),
      h("span", { class: "hint" }, `${shown.length} of ${rows.length}`)));
    const tbl = h("table", { class: "t" }, h("tr", {}, ["#", "subtask", "source", "GT", ...labels.map((l) => l), "cue", "question"].map((c) => h("th", {}, c))));
    for (const r of shown.slice(0, f.limit)) {
      tbl.appendChild(h("tr", { class: "click", onclick: () => this.openSample(r, cfg, labels) },
        h("td", {}, r.sample_idx), h("td", {}, r.subtask), h("td", {}, `${r.dataset}/${r.category}`), h("td", {}, r.gt),
        labels.map((l) => { const a = get(r, l); return h("td", {}, a ? [h("span", { class: a.correct ? "ok-dot" : "bad-dot" }), a.pred ?? "∅"] : h("span", { class: "na-dot" })); }),
        h("td", {}, r.context.cue_fired === undefined ? "–" : r.context.cue_fired ? "✓" : "·"),
        h("td", { style: { maxWidth: "380px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }, r.question)));
    }
    holder.appendChild(tbl);
    if (shown.length > f.limit) holder.appendChild(h("button", { class: "tb-btn", style: { marginTop: "8px" }, onclick: () => { f.limit += 300; this.renderSamples(holder, cfg, m, false); } }, "Show more"));
  }

  openSample(r, cfg, labels) {
    const nodesById = Object.fromEntries(cfg.graph.nodes.map((n) => [String(n.id), n]));
    const bg = h("div", { class: "modal-bg" });
    bg.addEventListener("pointerdown", (e) => { if (e.target === bg) bg.remove(); });
    const imgs = Object.entries(r.thumbs || {}).flatMap(([nid, list]) => list.map((f) => {
      const url = `/api/experiments/${cfg.run_id}/thumb/${f}`;
      const nd = nodesById[nid];
      const cap = nd ? `${nd.title || nd.type} #${nid}` : nid;
      return h("figure", {}, h("img", { src: url, onclick: () => lightbox(url, cap) }), h("figcaption", {}, cap));
    }));
    const opts = Object.entries(r.options).map(([k, x]) => h("div", {}, h("b", { class: k === r.gt ? "pos" : "" }, `(${k}) `), x));
    bg.appendChild(h("div", { class: "modal", style: { width: "min(1100px, 100%)" } },
      h("header", {}, h("h2", {}, `#${r.sample_idx} · ${r.subtask} · ${r.dataset}/${r.category}`), h("button", { class: "tb-btn", onclick: () => bg.remove() }, "✕")),
      h("div", { class: "mb" },
        h("div", { class: "detail" },
          h("div", {}, h("div", { class: "thumbs" }, imgs.length ? imgs : h("div", { class: "hint" }, "no thumbnails saved"))),
          h("div", {},
            h("div", { style: { fontWeight: 600, marginBottom: "6px" } }, r.question), opts,
            h("div", { class: "hint", style: { margin: "6px 0" } }, `GT ${r.gt} · ${r.is_anomalous ? "defective" : "normal"} image · ${r.image_key}`),
            h("table", { class: "t" }, h("tr", {}, ["arm", "pred", "correct", "confidence", "raw output"].map((c) => h("th", {}, c))),
              r.answers.map((a) => h("tr", {}, h("td", {}, a.label), h("td", {}, a.pred ?? "∅"), h("td", {}, a.correct ? "✓" : "✗"), h("td", {}, a.confidence !== undefined ? fmtNum(a.confidence) : "–"), h("td", {}, a.raw)))),
            r.detectors.length ? h("table", { class: "t", style: { marginTop: "8px" } }, h("tr", {}, ["detector", "score", "score/τ", "fired", "pixel AUROC", "box IoU"].map((c) => h("th", {}, c))),
              r.detectors.map((d) => h("tr", {}, h("td", {}, d.label), h("td", {}, fmtNum(d.score)), h("td", {}, fmtNum(d.score_norm)), h("td", {}, d.fired === undefined ? "–" : d.fired ? "✓" : "·"), h("td", {}, fmtNum(d.pixel_auroc)), h("td", {}, fmtNum(d.box_iou))))) : null,
            h("h4", {}, "Context"), h("pre", { style: { whiteSpace: "pre-wrap", fontSize: "11.5px", color: "var(--ink-2)" } }, JSON.stringify({ variant: r.variant_params, ...r.context }, null, 1)),
            Object.entries(r.texts || {}).map(([k, t]) => [h("h4", {}, k), h("pre", { style: { whiteSpace: "pre-wrap", fontSize: "11.5px", color: "var(--ink-2)", maxHeight: "200px", overflow: "auto" } }, t)]),
            Object.keys(r.errors || {}).length ? h("pre", { class: "neg" }, JSON.stringify(r.errors, null, 1)) : null,
            h("div", { class: "hint" }, "node time: " + Object.entries(r.node_time).map(([k, t]) => `${k.split("#")[0]} ${fmtTime(t)}`).join(" · "))))),
    ));
    document.body.appendChild(bg);
  }

  async openCompare() {
    const ids = [...this.compare];
    const details = await Promise.all(ids.map((id) => api(`/api/experiments/${id}`)));
    const v = this.view;
    v.innerHTML = "";
    this.current = null;
    v.appendChild(h("h1", {}, `Comparing ${ids.length} runs`));
    v.appendChild(h("div", { class: "sub" }, "First variant of each run; the runs share questions only if they used the same N, seed and filters."));
    const rows = [];
    details.forEach((d) => {
      const v0 = d.metrics.variants[0]?.id;
      for (const a of d.metrics.answer.filter((x) => x.variant === v0)) rows.push({ run: d.config.name || d.config.run_id, cfg: d.config, a });
    });
    v.appendChild(h("div", { class: "card" }, h("table", { class: "t" },
      h("tr", {}, ["run", "arm", "n", "acc", "95% CI", "κ", "macro", "parse fail", "latency"].map((c, i) => h("th", { class: i > 1 ? "num" : "" }, c))),
      rows.map(({ run, a }) => h("tr", {}, h("td", {}, run), h("td", {}, a.label), h("td", { class: "num" }, a.n), h("td", { class: "num" }, fmtPct(a.acc)),
        h("td", { class: "num" }, `${fmtPct(a.ci_lo)}–${fmtPct(a.ci_hi)}`), h("td", { class: "num" }, fmtNum(a.kappa)), h("td", { class: "num" }, fmtPct(a.macro_acc)),
        h("td", { class: "num" }, fmtPct(a.parse_fail)), h("td", { class: "num" }, fmtTime(a.latency)))))));
    const subtasks = [...new Set(rows.flatMap((r) => Object.keys(r.a.per_subtask)))].sort();
    v.appendChild(h("h3", {}, "Accuracy by subtask"));
    v.appendChild(h("div", { class: "card" }, groupedBars({
      categories: subtasks,
      series: rows.slice(0, 8).map((r, i) => ({ name: `${r.run} · ${r.a.label}`, color: SERIES[i], values: subtasks.map((s) => r.a.per_subtask[s]?.acc ?? null), n: subtasks.map((s) => r.a.per_subtask[s]?.n ?? 0) })),
    })));
    if (rows.length > 8) v.appendChild(h("div", { class: "hint" }, "Chart shows the first 8 series; the table lists all."));
  }
}

function pickExp(cfg) {
  const { name, notes, n, seed, datasets, subtasks, condition, save_thumbnails, sweep } = cfg;
  return { name, notes, n, seed, datasets, subtasks, condition, save_thumbnails, sweep };
}

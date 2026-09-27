import { api, h, toast, showMenuBelow } from "./util.js";
import { Editor } from "./editor.js";
import { Lab, openExperimentDialog } from "./lab.js";

const $ = (id) => document.getElementById(id);
const LS_KEY = "armb-lab.graph.v1";
let editor, lab, currentJob = null, lastRunSeconds = null, currentName = null;

async function init() {
  const schema = await api("/api/nodes");
  editor = new Editor({ viewport: $("viewport"), world: $("world"), wires: $("wires"), schema, onChange: autosave });
  editor.reloadSchema = async () => { editor.setSchema(await api("/api/nodes")); };
  editor.onSweepRequest = (node_id, param) => openExperimentDialog(editor, { addAxis: { node_id, param }, onLaunched: (id) => lab.show(id), lastRunSeconds });
  editor.onHelp = (def) => showSide(def.title, [h("p", {}, def.description || "(no description)"),
    h("h4", {}, "Inputs"), def.inputs.length ? def.inputs.map((p) => h("p", {}, `${p.name}: ${p.type}${p.optional ? " (optional)" : ""}`)) : h("p", {}, "–"),
    h("h4", {}, "Outputs"), def.outputs.length ? def.outputs.map((p) => h("p", {}, `${p.name}: ${p.type}`)) : h("p", {}, "–"),
    h("h4", {}, "Parameters"), def.params.map((p) => h("p", {}, h("b", {}, p.name), ` (${p.kind}) `, p.help || ""))]);
  lab = new Lab({ editor, onClose: () => {} });
  window.armb = { editor, lab, run };   // handy for poking at the graph from the browser console

  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(LS_KEY) || "null"); } catch (_) { /* storage unavailable */ }
  if (saved?.graph?.nodes?.length) { editor.load(saved.graph); currentName = saved.name || null; }
  else await loadWorkflow("templates", "01_arm_b_patchcore", true);
  requestAnimationFrame(() => { editor.drawWires(); if (!saved) editor.fitView(); });

  bindToolbar();
  pollSystem();
  setInterval(pollSystem, 4000);
}

function autosave(structural = true) {
  try { localStorage.setItem(LS_KEY, JSON.stringify({ name: currentName, graph: editor.toJSON() })); } catch (_) { /* ignore */ }
}

async function loadWorkflow(kind, name, fit = true) {
  try {
    const g = await api(`/api/workflows/${kind}/${encodeURIComponent(name)}`);
    editor.load(g);
    currentName = kind === "saved" ? name : null;
    editor.clearStatus();
    if (fit) requestAnimationFrame(() => editor.fitView());
    autosave();
  } catch (e) { toast(`Could not load ${name}: ${e.message}`, true); }
}

// ------------------------------------------------------------------ run
async function run() {
  if (currentJob) { toast("A job is already running"); return; }
  const graph = editor.toJSON(false);
  if (!graph.nodes.length) return;
  editor.clearStatus();
  const t0 = performance.now();
  try {
    const { job_id } = await api("/api/run", { method: "POST", body: { graph } });
    currentJob = job_id;
    editor.applyControls();
    $("btn-run").disabled = true;
    const seen = {};
    const poll = async () => {
      let j;
      try { j = await api(`/api/jobs/${job_id}`); } catch (e) { toast(e.message, true); finish(); return; }
      for (const [nid, st] of Object.entries(j.nodes || {})) {
        const sig = JSON.stringify([st.state, st.time, st.error]);
        if (seen[nid] !== sig) { seen[nid] = sig; editor.setNodeStatus(nid, st); }
      }
      if (j.state === "queued") $("btn-run").textContent = "▶ queued…";
      if (["done", "done_with_errors", "error", "cancelled"].includes(j.state)) {
        if (j.error) toast(j.error, true, 8000);
        else if (j.state === "done_with_errors") toast("Some nodes failed — see the red nodes", true);
        const secs = (performance.now() - t0) / 1000;
        const anyUncached = Object.values(j.nodes || {}).some((s) => s.state === "done" && !s.cached);
        if (j.state === "done" && anyUncached) lastRunSeconds = secs;
        finish();
        return;
      }
      setTimeout(poll, 350);
    };
    poll();
  } catch (e) { toast(e.message, true); finish(); }
  function finish() { currentJob = null; $("btn-run").disabled = false; $("btn-run").textContent = "▶ Run"; }
}

// ------------------------------------------------------------------ toolbar
function bindToolbar() {
  $("btn-run").onclick = run;
  $("btn-stop").onclick = async () => {
    const sys = await api("/api/system");
    for (const j of sys.queue) await api(`/api/jobs/${j.id}/cancel`, { method: "POST" });
    toast(sys.queue.length ? "Cancel requested (the current node finishes first)" : "Nothing is running");
  };
  $("btn-undo").onclick = () => editor.undo();
  $("btn-redo").onclick = () => editor.redo();
  $("btn-fit").onclick = () => editor.fitView();
  $("btn-add").onclick = (e) => { const r = $("viewport").getBoundingClientRect(); editor.openSearch(r.left + r.width / 2 - 170, r.top + 80); };
  $("btn-experiment").onclick = () => openExperimentDialog(editor, { onLaunched: (id) => lab.show(id), lastRunSeconds });
  $("btn-lab").onclick = () => (lab.visible ? lab.hide() : lab.show());
  $("btn-help").onclick = showHelp;
  $("side-close").onclick = () => $("side").classList.add("hidden");

  $("btn-templates").onclick = async (e) => {
    const { templates } = await api("/api/workflows");
    showMenuBelow(e.currentTarget, [{ header: "Research templates" }, ...templates.map((t) => ({
      label: t.replace(/^\d+_/, "").replace(/_/g, " "), onClick: () => {
        if (editor.nodes.size && !confirm("Replace the current graph with this template?")) return;
        loadWorkflow("templates", t);
      },
    }))]);
  };

  $("btn-workflow").onclick = async (e) => {
    const { saved } = await api("/api/workflows");
    showMenuBelow(e.currentTarget, [
      { label: "New (empty)", onClick: () => { if (!editor.nodes.size || confirm("Discard the current graph?")) { editor.load({ nodes: [], links: [] }); currentName = null; } } },
      { label: currentName ? `Save “${currentName}”` : "Save…", key: "Ctrl+S", onClick: () => save(false) },
      { label: "Save as…", onClick: () => save(true) },
      { sep: true },
      ...(saved.length ? [{ label: "Open saved", children: saved.map((s) => ({ label: s, onClick: () => loadWorkflow("saved", s) })) }] : []),
      { label: "Export JSON (download)", onClick: exportJSON },
      { label: "Import JSON…", onClick: importJSON },
    ]);
  };

  $("btn-system").onclick = (e) => showMenuBelow(e.currentTarget, [
    { label: "Unload MLLM (free memory)", onClick: async () => { await api("/api/models/unload", { method: "POST" }); toast("Model unloaded"); pollSystem(); } },
    { label: "Clear node output cache", onClick: async () => { await api("/api/cache/clear", { method: "POST" }); toast("Cache cleared"); pollSystem(); } },
    { label: "Clear node previews in editor", onClick: () => editor.clearStatus() },
  ]);

  addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key === "Enter") { e.preventDefault(); run(); }
    if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); save(false); }
    if (e.key === "Escape") $("side").classList.add("hidden");
  });
}

async function save(asNew) {
  let name = currentName;
  if (asNew || !name) {
    name = prompt("Workflow name:", name || "my_arm_b");
    if (!name) return;
  }
  try {
    await api(`/api/workflows/${encodeURIComponent(name)}`, { method: "POST", body: editor.toJSON() });
    currentName = name; autosave(); toast(`Saved “${name}” to phase-6/workflows/`);
  } catch (e) { toast(e.message, true); }
}

function exportJSON() {
  const blob = new Blob([JSON.stringify(editor.toJSON(), null, 2)], { type: "application/json" });
  const a = h("a", { href: URL.createObjectURL(blob), download: (currentName || "arm_b_workflow") + ".json" });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function importJSON() {
  const inp = h("input", { type: "file", accept: ".json,application/json" });
  inp.onchange = async () => {
    try { editor.load(JSON.parse(await inp.files[0].text())); currentName = null; requestAnimationFrame(() => editor.fitView()); }
    catch (e) { toast("Invalid workflow JSON: " + e.message, true); }
  };
  inp.click();
}

// ------------------------------------------------------------------ system
async function pollSystem() {
  try {
    const s = await api("/api/system");
    const d = s.device;
    const dev = d.cuda ? h("span", { class: "pill gpu" }, `GPU ${d.gpu} · ${d.vram_used_gb}/${d.vram_gb} GB`)
      : h("span", { class: "pill cpu", title: "torch.cuda.is_available() is False — MLLMs run on CPU (slow). Check `nvidia-smi`." }, "CPU only");
    const box = $("sysinfo");
    box.innerHTML = "";
    box.append(...[dev,
      h("span", {}, "RAM ", h("b", {}, `${d.ram_avail_gb ?? "?"} GB free`)),
      h("span", {}, "model ", h("b", {}, s.model.loaded ? s.model.loaded.split("/").pop() + ` (${s.model.precision})` : "none")),
      h("span", {}, "cache ", h("b", {}, `${s.cache_mb} MB`)),
      s.queue.length ? h("span", { class: "pill" }, `${s.queue.length} job${s.queue.length > 1 ? "s" : ""} ${s.queue.map((q) => q.kind === "experiment" && q.progress ? `${q.progress.done}/${q.progress.total}` : q.state).join(", ")}`) : null].filter(Boolean));
  } catch (_) { $("sysinfo").textContent = "server unreachable"; }
}

function showSide(title, content) {
  $("side").querySelector("header b").textContent = title;
  const body = $("side-body");
  body.innerHTML = "";
  body.append(...[content].flat(3).filter(Boolean));
  $("side").classList.remove("hidden");
}

function showHelp() {
  const k = (t) => h("kbd", {}, t);
  showSide("Arm-B Lab — how to use", [
    h("p", {}, "Arm B = image → detector → heatmap → box → MLLM. Every block is a node; wires carry typed data (colours = types). Build Arm A next to it and give each arm its own Score label to compare them on identical questions."),
    h("h4", {}, "Editing"),
    h("p", {}, "Double-click the canvas (or ", k("＋ Node"), ") to add a node. Drag from an output dot to an input dot to connect; drop a wire on empty canvas to add a compatible node. Drag an input's wire away to re-route it."),
    h("p", {}, "Right-click a node → ", h("b", {}, "Replace with"), " (e.g. PatchCore → WinCLIP keeps the wiring), ", h("b", {}, "Bypass"), " (pass-through ablation), ", h("b", {}, "Sweep parameter"), "."),
    h("p", {}, "Pan: drag the background. Zoom: wheel. Box-select: ", k("Ctrl"), "/", k("Shift"), "+drag. Drop an image file on the canvas to create a Load Image node."),
    h("h4", {}, "Shortcuts"),
    h("p", {}, k("Ctrl+Enter"), " run · ", k("Ctrl+S"), " save · ", k("Ctrl+Z"), "/", k("Ctrl+Shift+Z"), " undo/redo · ", k("Ctrl+C"), "/", k("Ctrl+V"), " copy/paste · ", k("Ctrl+D"), " duplicate · ", k("Ctrl+B"), " bypass · ", k("Del"), " delete · ", k("."), " fit view"),
    h("h4", {}, "Running"),
    h("p", {}, "Run executes the whole graph on one sample. Outputs are cached by (parameters + upstream), so changing only the prompt re-runs only the prompt and MLLM. Set the MMAD Sample 'index' control to 'increment' to step through questions on every run."),
    h("h4", {}, "Experiments"),
    h("p", {}, h("b", {}, "⚗ Experiment"), " runs the graph over a stratified MMAD sample, optionally sweeping any widget (grid). Results (📊) give accuracy ± 95% CI, Cohen's κ, macro accuracy, parse-failure rate, paired McNemar tests between arms/variants, detector AUROC/IoU/TPR/FPR, sweep curves, spread, RDS, calibration (ECE, letter-logits mode) and a per-sample browser. Runs are logged to phase-6/runs/ and can be resumed."),
    h("h4", {}, "Controls worth running"),
    h("p", {}, "• Synthetic detector (GT oracle) = the ceiling for Arm B. • Control detector (random/centre box) = does any box help, or only a correct one? • Parse mode 'strict' vs 'fallback A' = how much Phase 5's default-to-A inflated accuracy."),
  ]);
}

init().catch((e) => { console.error(e); toast("Failed to start: " + e.message, true, 10000); });

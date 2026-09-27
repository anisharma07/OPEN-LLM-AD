"""
Batch experiments: run a graph over a stratified MMAD sample, optionally
sweeping widget parameters (grid), and store one JSON line per
(sample x variant) in runs/<run_id>/executions.jsonl. Runs are resumable.
"""

import csv
import io
import itertools
import json
import shutil
import time
import uuid
from datetime import datetime

from . import metrics
from .data import INDEX
from .executor import Cancelled, Context, execute, validate
from .mllm import device_info
from .paths import RUNS_DIR
from .registry import REGISTRY


def _run_dir(run_id):
    d = RUNS_DIR / run_id
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(run_id)
    return d


def _read_json(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(p, obj):
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(p)


def build_variants(graph, sweep):
    """Cartesian product of sweep axes -> [(variant_id, {axis_key: value}, overrides)]."""
    nodes = {int(n["id"]): n for n in graph["nodes"]}
    axes = []
    for ax in sweep or []:
        nid = int(ax["node_id"])
        if nid not in nodes:
            raise ValueError(f"Sweep refers to missing node {nid}.")
        cls = REGISTRY[nodes[nid]["type"]]
        pdef = next((p for p in cls.PARAMS if p.name == ax["param"]), None)
        if pdef is None:
            raise ValueError(f"{cls.TITLE} has no parameter '{ax['param']}'.")
        vals = ax["values"]
        if not vals:
            continue
        title = nodes[nid].get("title") or cls.TITLE
        axes.append({"node_id": nid, "param": ax["param"], "values": vals,
                     "key": f"{title}#{nid}.{ax['param']}"})
    if not axes:
        return axes, [("v0", {}, {})]
    variants = []
    for i, combo in enumerate(itertools.product(*[a["values"] for a in axes])):
        vp, ov = {}, {}
        for a, val in zip(axes, combo):
            vp[a["key"]] = val
            ov.setdefault(a["node_id"], {})[a["param"]] = val
        variants.append((f"v{i}", vp, ov))
    return axes, variants


def create_run(config):
    graph = config["graph"]
    nodes, _, _ = validate(graph)
    samplers = [n for n in nodes.values() if n["type"] == "MMADSample"]
    if len(samplers) != 1:
        raise ValueError("A batch experiment needs exactly one 'MMAD Sample' node (it becomes the iterator).")
    if not any(n["type"] in ("Score", "DetectorMetrics") for n in nodes.values()):
        raise ValueError("Add at least one 'Score' or 'Detector Metrics' node so there is something to measure.")
    axes, variants = build_variants(graph, config.get("sweep"))
    samples = INDEX.stratified_sample(
        int(config.get("n", 100)), int(config.get("seed", 42)),
        datasets=config.get("datasets") or None, subtasks=config.get("subtasks") or None,
        condition=config.get("condition", "any"))
    if not samples:
        raise ValueError("No MMAD questions match the experiment filters.")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = RUNS_DIR / run_id
    (d / "thumbs").mkdir(parents=True)
    cfg = dict(config, run_id=run_id, created=datetime.now().isoformat(timespec="seconds"),
               sweep_axes=axes, n_variants=len(variants), n_samples=len(samples),
               total=len(samples) * len(variants), system=device_info())
    _write_json(d / "config.json", cfg)
    _write_json(d / "samples.json", [q["qid"] for q in samples])
    _write_json(d / "status.json", {"state": "queued", "done": 0, "total": cfg["total"]})
    return run_id


def run(run_id, cancel_event, on_progress=lambda s: None):
    d = _run_dir(run_id)
    cfg = _read_json(d / "config.json")
    graph = cfg["graph"]
    _, variants = build_variants(graph, cfg.get("sweep"))
    INDEX.ensure_loaded()
    samples = [INDEX.by_qid[q] for q in _read_json(d / "samples.json", [])]
    sampler_id = next(int(n["id"]) for n in graph["nodes"] if n["type"] == "MMADSample")
    save_thumbs = bool(cfg.get("save_thumbnails", True))

    ex_path = d / "executions.jsonl"
    done_keys = set()
    if ex_path.exists():
        with open(ex_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["qid"], r["variant"]))
                except Exception:
                    pass

    total = len(samples) * len(variants)
    status = {"state": "running", "done": len(done_keys), "total": total, "started": time.time(),
              "errors": 0, "last": None}
    _write_json(d / "status.json", status)
    t0, n_new = time.time(), 0
    try:
        with open(ex_path, "a", encoding="utf-8") as out:
            for si, q in enumerate(samples):
                for vid, vparams, overrides in variants:
                    if (q["qid"], vid) in done_keys:
                        continue
                    if cancel_event is not None and cancel_event.is_set():
                        raise Cancelled()
                    g = {"nodes": [], "links": graph["links"]}
                    for n in graph["nodes"]:
                        n2 = dict(n, params=dict(n.get("params", {})))
                        n2["params"].update(overrides.get(int(n["id"]), {}))
                        g["nodes"].append(n2)
                    ctx = Context(mode="batch", sample_override=q, preview_dir=d / "thumbs",
                                  preview_prefix=f"s{si}_{vid}_", save_previews=save_thumbs,
                                  cancel_event=cancel_event)
                    te = time.time()
                    st = execute(g, ctx)
                    rec = {"qid": q["qid"], "sample_idx": si, "image_key": q["image_key"], "dataset": q["dataset"],
                           "category": q["category"], "subtask": q["subtask"], "is_anomalous": q["is_anomalous"],
                           "question": q["question"], "options": q["options"], "gt": q["answer"],
                           "variant": vid, "variant_params": vparams, "answers": [], "detectors": [],
                           "context": {}, "node_time": {}, "thumbs": {}, "errors": {}, "texts": {},
                           "wall": round(time.time() - te, 4)}
                    for nid, s in st.items():
                        ntype = next(n["type"] for n in g["nodes"] if int(n["id"]) == nid)
                        if s.get("error"):
                            rec["errors"][str(nid)] = s["error"]
                        rec["node_time"][f"{ntype}#{nid}"] = round(s.get("time", 0.0), 4)
                        for r in s.get("records", []):
                            kind = r.get("kind")
                            body = {k: v for k, v in r.items() if k != "kind"}
                            if kind == "answer":
                                rec["answers"].append(body)
                            elif kind == "detector":
                                rec["detectors"].append(body)
                            elif kind == "context":
                                rec["context"].update(body)
                        ui = s.get("ui", {})
                        if ui.get("images"):
                            rec["thumbs"][str(nid)] = [im["url"] for im in ui["images"] if im.get("url")]
                        if ui.get("text") and ntype in ("TextPrompt", "MLLM"):
                            rec["texts"][f"{ntype}#{nid}"] = str(ui["text"])[:1500]
                    out.write(json.dumps(rec, default=str) + "\n")
                    out.flush()
                    n_new += 1
                    status["done"] += 1
                    status["errors"] += bool(rec["errors"])
                    rate = n_new / max(1e-6, time.time() - t0)
                    status["rate"] = rate
                    status["eta_s"] = (total - status["done"]) / rate if rate > 0 else None
                    status["last"] = {"qid": q["qid"], "variant": vid,
                                      "answers": [(a["label"], a["correct"]) for a in rec["answers"]]}
                    if n_new % 3 == 0 or status["done"] == total:
                        _write_json(d / "status.json", status)
                    on_progress(status)
        status["state"] = "done"
    except Cancelled:
        status["state"] = "cancelled"
    except Exception as e:
        status["state"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        status["finished"] = time.time()
        _write_json(d / "status.json", status)
        on_progress(status)


# --------------------------------------------------------------------- reading
def load_executions(run_id):
    p = _run_dir(run_id) / "executions.jsonl"
    rows = []
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


_METRIC_CACHE = {}


def run_metrics(run_id):
    d = _run_dir(run_id)
    p = d / "executions.jsonl"
    stamp = p.stat().st_mtime if p.exists() else 0
    hit = _METRIC_CACHE.get(run_id)
    if hit and hit[0] == stamp:
        return hit[1]
    cfg = _read_json(d / "config.json", {})
    m = metrics.compute(load_executions(run_id), cfg.get("sweep_axes"))
    _METRIC_CACHE[run_id] = (stamp, m)
    return m


def list_runs():
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        cfg = _read_json(d / "config.json")
        if not cfg:
            continue
        st = _read_json(d / "status.json", {})
        out.append({"run_id": d.name, "name": cfg.get("name") or d.name, "created": cfg.get("created"),
                    "n_samples": cfg.get("n_samples"), "n_variants": cfg.get("n_variants"),
                    "state": st.get("state"), "done": st.get("done", 0), "total": st.get("total", cfg.get("total")),
                    "notes": cfg.get("notes", "")})
    return out


def run_detail(run_id):
    d = _run_dir(run_id)
    cfg = _read_json(d / "config.json", {})
    return {"config": cfg, "status": _read_json(d / "status.json", {}), "metrics": run_metrics(run_id)}


def delete_run(run_id):
    shutil.rmtree(_run_dir(run_id))
    _METRIC_CACHE.pop(run_id, None)


def export_csv(run_id):
    buf = io.StringIO()
    w = csv.writer(buf)
    cols = ["qid", "dataset", "category", "subtask", "is_anomalous", "variant", "variant_params", "label",
            "pred", "gt", "correct", "parsed", "confidence", "latency", "cue_fired", "cue_score_norm", "raw"]
    w.writerow(cols)
    for ex in load_executions(run_id):
        for a in ex["answers"]:
            w.writerow([ex["qid"], ex["dataset"], ex["category"], ex["subtask"], ex["is_anomalous"], ex["variant"],
                        json.dumps(ex["variant_params"]), a["label"], a["pred"], a["gt"], a["correct"], a["parsed"],
                        a.get("confidence"), a.get("latency"), ex["context"].get("cue_fired"),
                        ex["context"].get("cue_score_norm"), a.get("raw", "")])
    return buf.getvalue()

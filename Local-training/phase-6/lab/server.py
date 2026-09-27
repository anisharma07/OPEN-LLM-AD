"""
Arm-B Lab server.

    cd Local-training/phase-6 && ../.venv/bin/python -m uvicorn lab.server:app --port 8765

One worker thread executes jobs in order (the GPU is a single resource):
interactive graph runs and batch experiments share the queue.
"""

import json
import queue
import re
import threading
import time
import traceback
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import experiments, nodes  # noqa: F401  (nodes registers all node classes)
from .data import INDEX, SUBTASKS
from .executor import CACHE, Cancelled, Context, GraphError, execute
from .mllm import MODELS, device_info
from .paths import BLOB_DIR, RESULTS_DIR, RUNS_DIR, TEMPLATE_DIR, UPLOAD_DIR, WEB_DIR, WORKFLOW_DIR
from .registry import schema

app = FastAPI(title="Arm-B Lab")


def clean(o):
    """Make numpy / tuple-laden structures JSON-safe."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if k not in ("map", "mask")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return None
    return o


# ------------------------------------------------------------------ job queue
class Job:
    def __init__(self, kind, payload):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.payload = payload
        self.state = "queued"
        self.created = time.time()
        self.started = self.finished = None
        self.nodes = {}
        self.error = None
        self.progress = None
        self.cancel = threading.Event()

    def view(self):
        return clean({"id": self.id, "kind": self.kind, "state": self.state, "error": self.error,
                      "nodes": self.nodes, "progress": self.progress, "run_id": self.payload.get("run_id"),
                      "elapsed": (self.finished or time.time()) - (self.started or time.time())})


JOBS = {}
QUEUE = queue.Queue()


def _worker():
    while True:
        job = QUEUE.get()
        if job.cancel.is_set():
            job.state = "cancelled"
            continue
        job.state, job.started = "running", time.time()
        try:
            if job.kind == "graph":
                def on_status(nid, st):
                    job.nodes[str(nid)] = {k: st.get(k) for k in ("state", "cached", "time", "ui", "error", "trace")}
                ctx = Context(mode="interactive", cancel_event=job.cancel, on_status=on_status)
                execute(job.payload["graph"], ctx, use_cache=job.payload.get("use_cache", True))
                errs = [n for n in job.nodes.values() if n["state"] == "error"]
                job.state = "done" if not errs else "done_with_errors"
            else:
                def on_progress(st):
                    job.progress = dict(st)
                experiments.run(job.payload["run_id"], job.cancel, on_progress)
                job.state = (job.progress or {}).get("state", "done")
        except Cancelled:
            job.state = "cancelled"
        except GraphError as e:
            job.state, job.error = "error", str(e)
        except Exception as e:
            job.state, job.error = "error", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
        finally:
            job.finished = time.time()


threading.Thread(target=_worker, daemon=True).start()


def _submit(kind, payload):
    job = Job(kind, payload)
    JOBS[job.id] = job
    QUEUE.put(job)
    # forget old finished jobs
    if len(JOBS) > 200:
        for k in sorted(JOBS, key=lambda k: JOBS[k].created)[:50]:
            if JOBS[k].state not in ("queued", "running"):
                JOBS.pop(k, None)
    return job


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/api/nodes")
def api_nodes():
    return schema()


@app.get("/api/system")
def api_system():
    active = [j.view() for j in JOBS.values() if j.state in ("queued", "running")]
    return clean({"device": device_info(), "model": MODELS.status(), "queue": active,
                  "cache_mb": round(CACHE.bytes / 1e6, 1)})


@app.post("/api/models/unload")
def api_unload():
    MODELS.unload()
    return {"ok": True}


@app.post("/api/cache/clear")
def api_cache_clear():
    CACHE.clear()
    return {"ok": True}


@app.post("/api/run")
def api_run(body: dict):
    if not body.get("graph", {}).get("nodes"):
        raise HTTPException(400, "empty graph")
    return {"job_id": _submit("graph", body).id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job.view()


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    job = JOBS.get(job_id)
    if job:
        job.cancel.set()
    return {"ok": True}


@app.get("/api/blob/{name}")
def api_blob(name: str):
    p = BLOB_DIR / re.sub(r"[^a-zA-Z0-9_.-]", "", name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "max-age=86400"})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename or "upload.png")
    dest = UPLOAD_DIR / name
    i = 1
    while dest.exists():
        dest = UPLOAD_DIR / f"{dest.stem.rsplit('__', 1)[0]}__{i}{dest.suffix}"
        i += 1
    dest.write_bytes(await file.read())
    return {"filename": dest.name}


@app.get("/api/uploads/{name}")
def api_upload_get(name: str):
    p = UPLOAD_DIR / re.sub(r"[^a-zA-Z0-9_.-]", "", name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# -------------------------------------------------------------- workflows
def _safe(name):
    n = re.sub(r"[^a-zA-Z0-9_. -]", "", name).strip()
    if not n:
        raise HTTPException(400, "bad name")
    return n


@app.get("/api/workflows")
def api_workflows():
    tpl = sorted(p.stem for p in TEMPLATE_DIR.glob("*.json"))
    mine = sorted(p.stem for p in WORKFLOW_DIR.glob("*.json"))
    return {"templates": tpl, "saved": mine}


@app.get("/api/workflows/{kind}/{name}")
def api_workflow_get(kind: str, name: str):
    base = TEMPLATE_DIR if kind == "templates" else WORKFLOW_DIR
    p = base / f"{_safe(name)}.json"
    if not p.exists():
        raise HTTPException(404)
    return json.loads(p.read_text(encoding="utf-8"))


@app.post("/api/workflows/{name}")
def api_workflow_save(name: str, body: dict):
    (WORKFLOW_DIR / f"{_safe(name)}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/workflows/{name}")
def api_workflow_delete(name: str):
    p = WORKFLOW_DIR / f"{_safe(name)}.json"
    if p.exists():
        p.unlink()
    return {"ok": True}


# -------------------------------------------------------------- MMAD info
@app.get("/api/mmad/info")
def api_mmad_info():
    INDEX.ensure_loaded()
    ds = sorted({q["dataset"] for q in INDEX.questions})
    return {"datasets": ds, "subtasks": SUBTASKS, "n_questions": len(INDEX.questions), "n_images": len(INDEX.images)}


# -------------------------------------------------------------- experiments
@app.post("/api/experiments")
def api_exp_create(body: dict):
    try:
        run_id = experiments.create_run(body)
    except (ValueError, GraphError) as e:
        raise HTTPException(400, str(e))
    job = _submit("experiment", {"run_id": run_id})
    return {"run_id": run_id, "job_id": job.id}


@app.get("/api/experiments")
def api_exp_list():
    live = {j.payload.get("run_id"): j.id for j in JOBS.values()
            if j.kind == "experiment" and j.state in ("queued", "running")}
    runs = experiments.list_runs()
    for r in runs:
        r["job_id"] = live.get(r["run_id"])
    return runs


def _guard(run_id):
    if not re.fullmatch(r"[0-9A-Za-z_-]+", run_id) or not (RUNS_DIR / run_id).is_dir():
        raise HTTPException(404, "unknown run")


@app.get("/api/experiments/{run_id}")
def api_exp_detail(run_id: str):
    _guard(run_id)
    d = experiments.run_detail(run_id)
    live = [j.id for j in JOBS.values() if j.payload.get("run_id") == run_id and j.state in ("queued", "running")]
    d["job_id"] = live[0] if live else None
    return clean(d)


@app.get("/api/experiments/{run_id}/executions")
def api_exp_rows(run_id: str, offset: int = 0, limit: int = 100000):
    _guard(run_id)
    rows = experiments.load_executions(run_id)
    return clean(rows[offset: offset + limit])


@app.get("/api/experiments/{run_id}/thumb/{name}")
def api_exp_thumb(run_id: str, name: str):
    _guard(run_id)
    p = RUNS_DIR / run_id / "thumbs" / re.sub(r"[^a-zA-Z0-9_.-]", "", name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "max-age=86400"})


@app.post("/api/experiments/{run_id}/cancel")
def api_exp_cancel(run_id: str):
    for j in JOBS.values():
        if j.payload.get("run_id") == run_id:
            j.cancel.set()
    return {"ok": True}


@app.post("/api/experiments/{run_id}/resume")
def api_exp_resume(run_id: str):
    _guard(run_id)
    if any(j.payload.get("run_id") == run_id and j.state in ("queued", "running") for j in JOBS.values()):
        raise HTTPException(409, "already running")
    job = _submit("experiment", {"run_id": run_id})
    return {"job_id": job.id}


@app.delete("/api/experiments/{run_id}")
def api_exp_delete(run_id: str):
    _guard(run_id)
    if any(j.payload.get("run_id") == run_id and j.state in ("queued", "running") for j in JOBS.values()):
        raise HTTPException(409, "stop the run first")
    experiments.delete_run(run_id)
    return {"ok": True}


@app.get("/api/experiments/{run_id}/export.csv")
def api_exp_csv(run_id: str):
    _guard(run_id)
    return PlainTextResponse(experiments.export_csv(run_id), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'})


@app.post("/api/experiments/{run_id}/figures")
def api_exp_figures(run_id: str):
    _guard(run_id)
    from .figures import make_figures
    return {"files": make_figures(run_id), "dir": str(RESULTS_DIR / run_id)}


@app.get("/api/results/{path:path}")
def api_results_file(path: str):
    p = (RESULTS_DIR / path).resolve()
    if RESULTS_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.exception_handler(FileNotFoundError)
def _fnf(_req, exc):
    return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

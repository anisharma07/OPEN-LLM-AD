"""
Graph executor.

Graph JSON (shared with the frontend):
  {"nodes": [{"id": 3, "type": "PatchCore", "params": {...}, "bypass": false, "title": "..."}],
   "links": [{"id": 9, "from": [3, 0], "to": [5, 1]}]}      # [node_id, port_index]

Every node's output is fingerprinted by (type, params, upstream fingerprints);
results are cached under that fingerprint, so re-running a graph only recomputes
nodes whose parameters or ancestors changed - and a batch sweep over a
downstream parameter reuses the detector output for each sample.
"""

import hashlib
import io
import json
import threading
import time
import traceback
import uuid
from collections import OrderedDict, defaultdict

import numpy as np
from PIL import Image

from .paths import BLOB_DIR
from .registry import REGISTRY, coerce_params


class Cancelled(Exception):
    pass


class GraphError(Exception):
    pass


# ------------------------------------------------------------------ cache
def _approx_bytes(v):
    if isinstance(v, Image.Image):
        return v.size[0] * v.size[1] * 3
    if isinstance(v, np.ndarray):
        return v.nbytes
    if isinstance(v, dict):
        return sum(_approx_bytes(x) for x in v.values()) + 256
    if isinstance(v, (list, tuple)):
        return sum(_approx_bytes(x) for x in v) + 64
    return 256


class OutputCache:
    def __init__(self, max_bytes=1_200_000_000):
        self.max_bytes = max_bytes
        self.data = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key][0]
        return None

    def put(self, key, value):
        size = _approx_bytes(value)
        with self.lock:
            if key in self.data:
                self.bytes -= self.data.pop(key)[1]
            self.data[key] = (value, size)
            self.bytes += size
            while self.bytes > self.max_bytes and len(self.data) > 1:
                _, (_, s) = self.data.popitem(last=False)
                self.bytes -= s

    def clear(self):
        with self.lock:
            self.data.clear()
            self.bytes = 0


CACHE = OutputCache()


# ---------------------------------------------------------------- context
class Context:
    """Per-execution services handed to nodes."""

    def __init__(self, mode="interactive", sample_override=None, preview_dir=None,
                 preview_prefix="", save_previews=True, cancel_event=None, on_status=None,
                 preview_size=None):
        self.mode = mode
        self.sample_override = sample_override
        self.preview_dir = preview_dir or BLOB_DIR
        self.preview_prefix = preview_prefix
        self.save_previews = save_previews
        self.cancel_event = cancel_event or threading.Event()
        self.on_status = on_status or (lambda *a, **k: None)
        self.preview_size = preview_size or (1024 if mode == "interactive" else 384)
        self.current_node = None

    def preview(self, img: Image.Image, tag="img"):
        """Save a JPEG preview; returns the URL the frontend can load (or None)."""
        if not self.save_previews or img is None:
            return None
        im = img.convert("RGB")
        im.thumbnail((self.preview_size, self.preview_size), Image.BICUBIC)
        if self.mode == "interactive":
            name = f"{uuid.uuid4().hex[:16]}.jpg"
            im.save(self.preview_dir / name, quality=88)
            return f"/api/blob/{name}"
        name = f"{self.preview_prefix}n{self.current_node}_{tag}.jpg"
        im.save(self.preview_dir / name, quality=82)
        return name  # relative to the run's thumbs folder

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled()


# ------------------------------------------------------------- execution
def _fingerprint(node_type, params, input_fps, extra=""):
    h = hashlib.sha1()
    h.update(node_type.encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    h.update(json.dumps(input_fps).encode())
    h.update(str(extra).encode())
    return h.hexdigest()


def validate(graph):
    nodes = {int(n["id"]): n for n in graph.get("nodes", [])}
    for n in nodes.values():
        if n["type"] not in REGISTRY:
            raise GraphError(f"Unknown node type '{n['type']}' (node {n['id']}).")
    incoming = defaultdict(dict)       # node -> in_idx -> (src, out_idx)
    for l in graph.get("links", []):
        (s, so), (d, di) = l["from"], l["to"]
        s, d = int(s), int(d)
        if s in nodes and d in nodes:
            incoming[d][int(di)] = (s, int(so))
    # Kahn topological order
    indeg = {i: 0 for i in nodes}
    children = defaultdict(set)
    for d, ins in incoming.items():
        for (s, _) in ins.values():
            if d not in children[s]:
                children[s].add(d)
                indeg[d] += 1
    order, frontier = [], sorted(i for i, k in indeg.items() if k == 0)
    while frontier:
        i = frontier.pop(0)
        order.append(i)
        for c in sorted(children[i]):
            indeg[c] -= 1
            if indeg[c] == 0:
                frontier.append(c)
    if len(order) != len(nodes):
        raise GraphError("The graph contains a cycle.")
    return nodes, incoming, order


def execute(graph, ctx: Context, use_cache=True):
    """Run every node in topological order. Returns {node_id: status dict}."""
    nodes, incoming, order = validate(graph)
    values, fps, status = {}, {}, {}

    for nid in order:
        ctx.check_cancel()
        spec = nodes[nid]
        cls = REGISTRY[spec["type"]]
        params = coerce_params(cls, spec.get("params", {}))
        st = {"state": "running", "cached": False, "time": 0.0, "ui": {}, "records": [], "error": None}
        status[nid] = st
        ctx.current_node = nid
        ctx.on_status(nid, st)

        # gather inputs
        inputs, in_fps, missing, upstream_failed = {}, [], [], False
        for idx, port in enumerate(cls.INPUTS):
            src = incoming.get(nid, {}).get(idx)
            if src is None:
                if not port.optional:
                    missing.append(port.name)
                inputs[port.name] = None
                in_fps.append(None)
                continue
            s_id, s_out = src
            if status.get(s_id, {}).get("state") in ("error", "skipped"):
                upstream_failed = True
                break
            v = values.get((s_id, s_out))
            if v is None and not port.optional:
                missing.append(port.name)
            inputs[port.name] = v
            in_fps.append(fps.get((s_id, s_out)))

        if upstream_failed:
            st.update(state="skipped", error="upstream node failed")
            ctx.on_status(nid, st)
            continue

        # bypass: forward the first input of matching type to each output
        if spec.get("bypass"):
            for o_idx, oport in enumerate(cls.OUTPUTS):
                for i_idx, iport in enumerate(cls.INPUTS):
                    if iport.type == oport.type and inputs.get(iport.name) is not None:
                        values[(nid, o_idx)] = inputs[iport.name]
                        fps[(nid, o_idx)] = in_fps[i_idx]
                        break
            st.update(state="bypassed")
            ctx.on_status(nid, st)
            continue

        if missing:
            st.update(state="error", error=f"missing input: {', '.join(missing)}")
            ctx.on_status(nid, st)
            continue

        extra = ""
        if ctx.sample_override is not None and spec["type"] == "MMADSample":
            extra = ctx.sample_override["qid"]
        if ctx.mode == "batch":
            extra += f"|{ctx.preview_dir}"   # batch previews live in the run folder
        fp = _fingerprint(spec["type"], params, in_fps, extra)
        cached = CACHE.get(fp) if (use_cache and cls.CACHEABLE) else None
        t0 = time.time()
        try:
            if cached is not None:
                res = cached
                st["cached"] = True
            else:
                res = cls().run(ctx, inputs, params)
                if cls.CACHEABLE:
                    CACHE.put(fp, res)
        except Cancelled:
            raise
        except Exception as e:  # node failure is reported on the node, graph continues
            msg = f"{type(e).__name__}: {e}"
            if "OutOfMemoryError" in type(e).__name__ or "out of memory" in str(e).lower():
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                msg = ("GPU out of memory. The loaded MLLM shares the 8 GB card: free it with "
                       "⚙ → Unload MLLM, or lower this node's resolution / use fewer layers / set device=cpu. "
                       f"({str(e).splitlines()[0][:160]})")
            st.update(state="error", error=msg, trace=traceback.format_exc(limit=6))
            ctx.on_status(nid, st)
            continue
        st["time"] = res.ui.get("_time", time.time() - t0) if st["cached"] else time.time() - t0
        if not st["cached"]:
            res.ui["_time"] = st["time"]
        for o_idx, oport in enumerate(cls.OUTPUTS):
            values[(nid, o_idx)] = res.outputs.get(oport.name)
            fps[(nid, o_idx)] = f"{fp}:{o_idx}"
        st.update(state="done", ui={k: v for k, v in res.ui.items() if k != "_time"}, records=res.records)
        ctx.on_status(nid, st)

    return status


def image_to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

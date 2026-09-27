"""Input nodes (MMAD sampler, uploads, custom questions, normal references) and image pre-processing."""

import importlib.util
import random

import numpy as np
from PIL import Image

from ..data import INDEX, SUBTASKS, open_image
from ..paths import PHASE2_DIR, PHASE3_DIR, UPLOAD_DIR
from ..registry import Node, Param, Port, Result, register
from ..viz import contact_sheet


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_corruptions = None
_mitigations = None


def corruptions():
    global _corruptions
    if _corruptions is None:
        _corruptions = _load_module("phase2_corruptions", PHASE2_DIR / "corruptions.py")
    return _corruptions


def mitigations():
    global _mitigations
    if _mitigations is None:
        _mitigations = _load_module("phase3_mitigations", PHASE3_DIR / "mitigations.py")
    return _mitigations


def _sources():
    return ["any"] + INDEX.sources()


def _cap(img, max_side):
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.BICUBIC)
    return img


def question_text(q):
    opts = "\n".join(f"({k}) {v}" for k, v in sorted(q["options"].items()))
    return f"{q['question']}\n{opts}"


_ORDER_CACHE = {}


@register
class MMADSample(Node):
    TYPE = "MMADSample"
    TITLE = "MMAD Sample"
    CATEGORY = "Input"
    DESCRIPTION = """
Picks one MMAD question (image + multiple-choice question + ground truth).
During a batch experiment this node is driven by the experiment's stratified
sampler instead of its widgets."""
    OUTPUTS = [Port("sample", "SAMPLE"), Port("image", "IMAGE"), Port("question", "QUESTION")]
    PARAMS = [
        Param("source", "choice", "any", _sources, help="dataset/category filter"),
        Param("subtask", "choice", "any", ["any"] + SUBTASKS),
        Param("condition", "choice", "any", ["any", "defective", "normal"]),
        Param("index", "int", 0, min=0, max=10_000_000, step=1, control=True,
              help="position in the (shuffled) filtered pool; set the control to 'increment' to walk the pool"),
        Param("order", "choice", "shuffled", ["shuffled", "dataset"]),
        Param("max_side", "int", 1024, min=0, max=4096, step=64,
              help="downscale very large images (GoodsAD is 3000px) before the pipeline; 0 = keep"),
    ]

    def run(self, ctx, inputs, p):
        if ctx.sample_override is not None:
            q = ctx.sample_override
            pool_n = None
        else:
            key = (p["source"], p["subtask"], p["condition"], p["order"])
            if key not in _ORDER_CACHE:
                pool = INDEX.filter(source=p["source"], subtask=p["subtask"], condition=p["condition"])
                if p["order"] == "shuffled":
                    pool = list(pool)
                    random.Random(1234).shuffle(pool)
                _ORDER_CACHE[key] = pool
            pool = _ORDER_CACHE[key]
            if not pool:
                raise ValueError("No MMAD questions match these filters.")
            q = pool[p["index"] % len(pool)]
            pool_n = len(pool)
        rec = INDEX.image_record(q["image_key"])
        img = _cap(open_image(rec["image_path"]), p["max_side"])
        sample = dict(q, image_path=rec["image_path"], image_size=img.size)
        question = {k: q[k] for k in ("qid", "question", "options", "answer", "subtask", "dataset", "category", "image_key")}
        ui = {
            "images": [{"url": ctx.preview(img, "input"), "label": q["image_key"]}],
            "text": question_text(q),
            "metrics": {"GT": q["answer"], "subtask": q["subtask"], "source": f"{q['dataset']}/{q['category']}",
                        "defective": q["is_anomalous"]},
        }
        if pool_n is not None:
            ui["metrics"]["pool"] = pool_n
        return Result(outputs={"sample": sample, "image": img, "question": question}, ui=ui)


def _uploads():
    files = sorted(p.name for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    return files or [""]


@register
class LoadImage(Node):
    TYPE = "LoadImage"
    TITLE = "Load Image"
    CATEGORY = "Input"
    DESCRIPTION = "Your own image (upload with the button on the node)."
    OUTPUTS = [Port("image", "IMAGE")]
    PARAMS = [Param("image", "upload", "", _uploads, sweepable=False),
              Param("max_side", "int", 1024, min=0, max=4096, step=64)]

    def run(self, ctx, inputs, p):
        if not p["image"]:
            raise ValueError("Upload an image first.")
        img = _cap(open_image(UPLOAD_DIR / p["image"]), p["max_side"])
        return Result(outputs={"image": img}, ui={"images": [{"url": ctx.preview(img, "upload"), "label": p["image"]}]})


@register
class CustomQuestion(Node):
    TYPE = "CustomQuestion"
    TITLE = "Custom Question"
    CATEGORY = "Input"
    DESCRIPTION = "Write your own multiple-choice question. One option per line as 'A: text'."
    OUTPUTS = [Port("question", "QUESTION")]
    PARAMS = [
        Param("question", "textarea", "Is there any defect in the object?"),
        Param("options", "textarea", "A: Yes.\nB: No."),
        Param("answer", "text", "", help="optional ground-truth letter"),
        Param("subtask", "choice", "Anomaly Detection", SUBTASKS + ["Custom"]),
    ]

    def run(self, ctx, inputs, p):
        opts = {}
        for line in p["options"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip():
                    opts[k.strip().upper()[:1]] = v.strip()
        if not opts:
            raise ValueError("No options parsed; use 'A: text' per line.")
        q = {"qid": "custom", "question": p["question"].strip(), "options": opts,
             "answer": p["answer"].strip().upper()[:1], "subtask": p["subtask"],
             "dataset": "custom", "category": "custom", "image_key": "custom"}
        return Result(outputs={"question": q}, ui={"text": question_text(q)})


@register
class NormalReferences(Node):
    TYPE = "NormalReferences"
    TITLE = "Normal References"
    CATEGORY = "Input"
    DESCRIPTION = """
k defect-free images of the same product, used by PatchCore's memory bank and
WinCLIP+ few-shot. 'similar'/'random' use MMAD's own template lists; the query
image itself is always excluded."""
    INPUTS = [Port("sample", "SAMPLE", optional=True)]
    OUTPUTS = [Port("refs", "REFS")]
    PARAMS = [
        Param("strategy", "choice", "similar", ["similar", "random", "category_random"]),
        Param("k", "int", 8, min=1, max=64, step=1),
        Param("source", "choice", "from sample", lambda: ["from sample"] + INDEX.sources(),
              help="category to draw from when no sample is connected"),
        Param("seed", "int", 0, min=0, max=1_000_000),
    ]

    def run(self, ctx, inputs, p):
        s = inputs.get("sample")
        if p["source"] != "from sample":
            source = p["source"]
            paths = INDEX.category_refs(source, p["k"], p["seed"])
        elif s is not None:
            source = f"{s['dataset']}/{s['category']}"
            paths = INDEX.normal_refs(s["image_key"], p["k"], p["strategy"], p["seed"])
        else:
            raise ValueError("Connect a sample or choose a source category.")
        if not paths:
            raise ValueError(f"No normal reference images found for {source}.")
        ui = {"metrics": {"refs": len(paths), "source": source}}
        if ctx.mode == "interactive":
            sheet = contact_sheet([open_image(x) for x in paths[:8]], thumb=140)
            ui["images"] = [{"url": ctx.preview(sheet, "refs"), "label": f"{len(paths)} normal refs"}]
        return Result(outputs={"refs": {"paths": paths, "source": source, "strategy": p["strategy"]}}, ui=ui)


@register
class Corruption(Node):
    TYPE = "Corruption"
    TITLE = "Corruption Engine"
    CATEGORY = "Pre-process"
    DESCRIPTION = "Phase-2 industrial corruptions (7 types x 5 severities). Severity 0 = clean."
    INPUTS = [Port("image", "IMAGE")]
    OUTPUTS = [Port("image", "IMAGE")]
    PARAMS = [
        Param("corruption", "choice", "motion_blur", ["gaussian_noise", "motion_blur", "low_light", "specular_glare",
                                                      "defocus_blur", "perspective_tilt", "compression"]),
        Param("severity", "int", 3, min=0, max=5, step=1),
        Param("seed", "int", 0, min=0, max=1_000_000),
    ]

    def run(self, ctx, inputs, p):
        np.random.seed(p["seed"])
        out = corruptions().apply_corruption(inputs["image"].convert("RGB"), p["corruption"], p["severity"])
        return Result(outputs={"image": out},
                      ui={"images": [{"url": ctx.preview(out, "corrupt"), "label": f"{p['corruption']} s{p['severity']}"}]},
                      records=[{"kind": "context", "corruption": p["corruption"], "severity": p["severity"]}])


RESTORERS = {
    "unsharp (motion blur)": lambda m, im: m.restore_motion_blur(im),
    "bilateral (noise)": lambda m, im: m.restore_gaussian_noise(im),
    "CLAHE (low light)": lambda m, im: m.restore_low_light(im),
    "glare suppression": lambda m, im: m.restore_specular_glare(im),
    "deconvolve (defocus)": lambda m, im: m.restore_defocus_blur(im),
}


@register
class Restoration(Node):
    TYPE = "Restoration"
    TITLE = "Test-Time Restoration"
    CATEGORY = "Pre-process"
    DESCRIPTION = "Phase-3 TTA-IR filters, applied before the detector / MLLM."
    INPUTS = [Port("image", "IMAGE")]
    OUTPUTS = [Port("image", "IMAGE")]
    PARAMS = [Param("method", "choice", "unsharp (motion blur)", list(RESTORERS))]

    def run(self, ctx, inputs, p):
        out = RESTORERS[p["method"]](mitigations(), inputs["image"].convert("RGB"))
        return Result(outputs={"image": out}, ui={"images": [{"url": ctx.preview(out, "restored"), "label": p["method"]}]})

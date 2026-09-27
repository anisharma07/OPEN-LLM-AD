"""Detector nodes (PatchCore, WinCLIP, synthetic/control), heatmap -> box, and visual prompting."""

import hashlib
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..data import INDEX, open_image
from ..detectors.patchcore import BACKBONES, LAYER_SETS, PatchCore
from ..registry import Node, Param, Port, Result, register
from ..viz import draw_boxes, overlay_heatmap, region_words, side_by_side, to_map_res

MAP_RES = 256   # every detector emits a 256x256 map aligned to the full image


def _ref_loader(path):
    return open_image(path)


def _anomaly_ui(ctx, img, amap, a, extra=None):
    lo = float(amap.min())
    hi = max(float(amap.max()), a["tau_map"] or 0) if a.get("tau_map") else None
    ov = overlay_heatmap(img, amap, 0.5, lo=lo, hi=hi)
    m = {"score": round(a["score"], 4)}
    if a.get("tau_score"):
        m["tau"] = round(a["tau_score"], 4)
        m["score/tau"] = round(a["score"] / a["tau_score"], 3)
    m.update(extra or {})
    return {"images": [{"url": ctx.preview(ov, "heat"), "label": f"{a['detector']} heatmap"}], "metrics": m}


def _seed_for(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


@register
class PatchCoreNode(Node):
    TYPE = "PatchCore"
    TITLE = "PatchCore"
    CATEGORY = "Detector"
    DESCRIPTION = """
Memory-bank anomaly detector (Roth et al. 2022). Builds a coreset of normal
patch features from the references, scores each test patch by its nearest-
neighbour distance, and calibrates tau on held-out normals."""
    INPUTS = [Port("image", "IMAGE"), Port("refs", "REFS")]
    OUTPUTS = [Port("anomaly", "ANOMALY")]
    PARAMS = [
        Param("backbone", "choice", "wide_resnet50_2", list(BACKBONES)),
        Param("layers", "choice", "layer2+layer3", list(LAYER_SETS)),
        Param("resolution", "choice", "256", ["224", "256", "320", "384"]),
        Param("feat_dim", "int", 1024, min=64, max=2048, step=64),
        Param("coreset_ratio", "float", 0.1, min=0.01, max=1.0, step=0.01),
        Param("knn", "int", 1, min=1, max=9, step=1),
        Param("sigma", "float", 4.0, min=0.0, max=10.0, step=0.5),
        Param("score_agg", "choice", "max", ["max", "top1pct_mean"]),
        Param("calib_margin", "float", 1.0, min=0.5, max=2.0, step=0.05,
              help="tau = margin x (max score on held-out normals)"),
        Param("device", "choice", "auto", ["auto", "cuda", "cpu"]),
    ]

    def run(self, ctx, inputs, p):
        img, refs = inputs["image"], inputs["refs"]
        pc = PatchCore(p["backbone"], p["layers"], int(p["resolution"]), p["feat_dim"], p["coreset_ratio"],
                       p["knn"], p["sigma"], p["score_agg"], p["calib_margin"], p["device"])
        t0 = time.time()
        fitted = pc.fit(refs["paths"], _ref_loader)
        t_fit = time.time() - t0
        amap, score = pc.predict(img, fitted)
        a = {"map": amap, "score": score, "tau_map": fitted["tau_map"], "tau_score": fitted["tau_score"],
             "detector": "PatchCore", "image_size": img.size,
             "info": {"bank_size": fitted["bank_size"], "n_refs": fitted["n_refs"]}}
        ui = _anomaly_ui(ctx, img, amap, a, {"bank": fitted["bank_size"], "fit_s": round(t_fit, 2)})
        return Result(outputs={"anomaly": a}, ui=ui)


@register
class WinCLIPNode(Node):
    TYPE = "WinCLIP"
    TITLE = "WinCLIP"
    CATEGORY = "Detector"
    DESCRIPTION = """
CLIP-based detector (Jeong et al. 2023). Zero-shot scores windows of patch
tokens against 'normal' vs 'damaged <object>' prompt ensembles; WinCLIP+ adds a
few-shot memory of normal-reference window embeddings. First use downloads the
CLIP weights from Hugging Face."""
    INPUTS = [Port("image", "IMAGE"), Port("refs", "REFS", optional=True), Port("sample", "SAMPLE", optional=True)]
    OUTPUTS = [Port("anomaly", "ANOMALY")]
    PARAMS = [
        Param("clip_model", "choice", "ViT-B/16 (openai)", ["ViT-B/16 (openai)", "ViT-B/32 (openai)", "ViT-L/14 (openai)"]),
        Param("mode", "choice", "few-shot (WinCLIP+)", ["zero-shot", "few-shot (WinCLIP+)"]),
        Param("object_name", "text", "auto", help="'auto' = category name, e.g. 'drink can'"),
        Param("scales", "choice", "2,3", ["2", "3", "2,3", "2,3,4", "3,5"]),
        Param("n_templates", "int", 15, min=1, max=15, step=1),
        Param("temperature", "float", 100.0, min=1.0, max=200.0, step=1.0),
        Param("fs_weight", "float", 0.5, min=0.0, max=1.0, step=0.05, help="WinCLIP+: weight of the few-shot map"),
        Param("sigma", "float", 4.0, min=0.0, max=10.0, step=0.5),
        Param("calibrate", "bool", True, help="estimate tau from the normal references"),
        Param("device", "choice", "auto", ["auto", "cuda", "cpu"]),
    ]

    def run(self, ctx, inputs, p):
        from ..detectors.winclip import WinCLIP
        img, refs, s = inputs["image"], inputs.get("refs"), inputs.get("sample")
        name = p["object_name"].strip()
        if name in ("", "auto"):
            src = (refs or {}).get("source") or (f"{s['dataset']}/{s['category']}" if s else "object")
            name = src.split("/")[-1].replace("_", " ")
        if p["mode"].startswith("few") and not refs:
            raise ValueError("WinCLIP+ needs a REFS input (Normal References node).")
        scales = tuple(int(x) for x in p["scales"].split(","))
        wc = WinCLIP(p["clip_model"], scales, p["n_templates"], p["sigma"], p["device"])
        ref_paths = refs["paths"] if refs else []
        amap, score, info = wc.predict(img, name, p["mode"], ref_paths, _ref_loader, p["temperature"], p["fs_weight"])
        tau_map = tau_score = None
        if p["calibrate"] and ref_paths:
            tau_map, tau_score, _ = wc.calibrate(name, ref_paths, _ref_loader, p["mode"], p["temperature"], p["fs_weight"])
        a = {"map": amap, "score": score, "tau_map": tau_map, "tau_score": tau_score, "detector": "WinCLIP",
             "image_size": img.size, "info": dict(info, object=name)}
        return Result(outputs={"anomaly": a}, ui=_anomaly_ui(ctx, img, amap, a, {"object": name}))


@register
class SyntheticDetector(Node):
    TYPE = "SyntheticDetector"
    TITLE = "Synthetic Detector (GT-based)"
    CATEGORY = "Detector"
    DESCRIPTION = """
A detector with controllable quality, built from the ground-truth mask. With
defaults it is a perfect oracle (upper bound for Arm B). Sweep miss_rate /
false_alarm_rate / shift to ask: how good must the detector be before the
MLLM benefits?"""
    INPUTS = [Port("image", "IMAGE"), Port("sample", "SAMPLE")]
    OUTPUTS = [Port("anomaly", "ANOMALY")]
    PARAMS = [
        Param("miss_rate", "float", 0.0, min=0.0, max=1.0, step=0.05, help="P(defect not reported)"),
        Param("false_alarm_rate", "float", 0.0, min=0.0, max=1.0, step=0.05, help="P(random blob on a normal image)"),
        Param("shift_frac", "float", 0.0, min=0.0, max=0.5, step=0.02, help="translate the mask by this fraction of the image"),
        Param("dilate_frac", "float", 0.0, min=0.0, max=0.3, step=0.01, help="grow the mask (sloppy localisation)"),
        Param("seed", "int", 0, min=0, max=1_000_000),
    ]

    def run(self, ctx, inputs, p):
        img, s = inputs["image"], inputs["sample"]
        rng = random.Random(_seed_for(s["qid"], p["seed"]))
        gt = INDEX.gt_mask(s["image_key"], (MAP_RES, MAP_RES)).astype(np.float32)
        defective = gt.any()
        m = gt.copy()
        if defective and rng.random() < p["miss_rate"]:
            m[:] = 0
        if not defective and rng.random() < p["false_alarm_rate"]:
            cy, cx, r = rng.randint(30, 226), rng.randint(30, 226), rng.randint(10, 40)
            yy, xx = np.ogrid[:MAP_RES, :MAP_RES]
            m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1
        if p["shift_frac"] > 0 and m.any():
            ang = rng.uniform(0, 2 * np.pi)
            dy, dx = int(np.sin(ang) * p["shift_frac"] * MAP_RES), int(np.cos(ang) * p["shift_frac"] * MAP_RES)
            m = np.roll(np.roll(m, dy, 0), dx, 1)
        if p["dilate_frac"] > 0 and m.any():
            k = max(1, int(p["dilate_frac"] * MAP_RES))
            m = cv2.dilate(m, np.ones((k, k), np.uint8))
        amap = cv2.GaussianBlur(m, (0, 0), 2) if m.any() else m
        score = float(amap.max())
        a = {"map": amap.astype(np.float32), "score": score, "tau_map": 0.5, "tau_score": 0.5,
             "detector": "Synthetic", "image_size": img.size, "info": {"defective": bool(defective)}}
        return Result(outputs={"anomaly": a}, ui=_anomaly_ui(ctx, img, amap, a, {"gt_defective": bool(defective)}))


@register
class ControlDetector(Node):
    TYPE = "ControlDetector"
    TITLE = "Control Detector (null)"
    CATEGORY = "Detector"
    DESCRIPTION = """
Negative controls that carry no information about the defect: a random box,
a fixed centre box, or the whole image. If these help as much as PatchCore,
the gain comes from the presence of a box, not from localisation."""
    INPUTS = [Port("image", "IMAGE"), Port("sample", "SAMPLE", optional=True)]
    OUTPUTS = [Port("anomaly", "ANOMALY")]
    PARAMS = [
        Param("mode", "choice", "random_box", ["random_box", "center_box", "full_image"]),
        Param("area_frac", "float", 0.08, min=0.01, max=0.8, step=0.01),
        Param("seed", "int", 0, min=0, max=1_000_000),
    ]

    def run(self, ctx, inputs, p):
        img, s = inputs["image"], inputs.get("sample")
        amap = np.zeros((MAP_RES, MAP_RES), np.float32)
        side = int(MAP_RES * np.sqrt(p["area_frac"]))
        if p["mode"] == "full_image":
            amap[:] = 1
        else:
            if p["mode"] == "center_box":
                y0 = x0 = (MAP_RES - side) // 2
            else:
                rng = random.Random(_seed_for(s["qid"] if s else id(img), p["seed"]))
                y0, x0 = rng.randint(0, MAP_RES - side), rng.randint(0, MAP_RES - side)
            amap[y0:y0 + side, x0:x0 + side] = 1
        a = {"map": amap, "score": 1.0, "tau_map": 0.5, "tau_score": 0.5, "detector": f"Control:{p['mode']}",
             "image_size": img.size, "info": {}}
        return Result(outputs={"anomaly": a}, ui=_anomaly_ui(ctx, img, amap, a))


# ----------------------------------------------------------------------------
def _otsu(values):
    lo, hi = float(values.min()), float(values.max())
    u8 = ((values - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + t / 255.0 * (hi - lo)


@register
class HeatmapToBox(Node):
    TYPE = "HeatmapToBox"
    TITLE = "Heatmap → Box"
    CATEGORY = "Localise"
    DESCRIPTION = """
Turns an anomaly map into boxes. The *gate* decides whether the detector
'fires' on this image at all; the *threshold* decides which pixels form the
box. 'calibrated' modes use tau from the detector's normal references."""
    INPUTS = [Port("anomaly", "ANOMALY"), Port("image", "IMAGE", optional=True)]
    OUTPUTS = [Port("region", "REGION")]
    PARAMS = [
        Param("gate", "choice", "calibrated", ["calibrated", "fixed", "always", "never"]),
        Param("gate_value", "float", 1.0, min=0.0, max=100.0, step=0.05,
              help="calibrated: fire if score >= value x tau.  fixed: fire if score >= value"),
        Param("threshold", "choice", "calibrated", ["calibrated", "relative", "percentile", "otsu", "fixed"]),
        Param("threshold_value", "float", 1.0, min=0.0, max=100.0, step=0.01,
              help="calibrated: x tau_map; relative: fraction of max; percentile: top %% of pixels; fixed: absolute"),
        Param("box_mode", "choice", "components", ["components", "union"]),
        Param("max_boxes", "int", 1, min=1, max=8, step=1),
        Param("min_area_frac", "float", 0.001, min=0.0, max=0.2, step=0.001),
        Param("pad_frac", "float", 0.1, min=0.0, max=1.0, step=0.05, help="padding relative to box size"),
    ]

    def run(self, ctx, inputs, p):
        a, img = inputs["anomaly"], inputs.get("image")
        amap = to_map_res(a["map"], MAP_RES)
        W, H = a["image_size"]
        notes = []
        # ---- gate
        tau_s = a.get("tau_score")
        if p["gate"] == "calibrated" and not tau_s:
            notes.append("no tau from detector -> gate 'always'")
        if p["gate"] == "always" or (p["gate"] == "calibrated" and not tau_s):
            fired = True
        elif p["gate"] == "never":
            fired = False
        elif p["gate"] == "calibrated":
            fired = a["score"] >= p["gate_value"] * tau_s
        else:
            fired = a["score"] >= p["gate_value"]
        score_norm = a["score"] / tau_s if tau_s else None

        # ---- pixel threshold
        mode, v = p["threshold"], p["threshold_value"]
        if mode == "calibrated" and not a.get("tau_map"):
            notes.append("no tau_map -> relative 0.5")
            mode, v = "relative", 0.5
        if mode == "calibrated":
            thr = v * a["tau_map"]
        elif mode == "relative":
            thr = amap.min() + v * (amap.max() - amap.min())
        elif mode == "percentile":
            thr = np.percentile(amap, 100 - min(max(v, 0.01), 100))
        elif mode == "otsu":
            thr = _otsu(amap)
        else:
            thr = v
        mask = amap >= thr

        boxes = []
        if fired and mask.any():
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            comps = [(stats[i, cv2.CC_STAT_AREA], stats[i]) for i in range(1, n)
                     if stats[i, cv2.CC_STAT_AREA] >= p["min_area_frac"] * MAP_RES * MAP_RES]
            comps.sort(key=lambda c: -c[0])
            if p["box_mode"] == "union" and comps:
                xs = [c[1][0] for c in comps] + [c[1][0] + c[1][2] for c in comps]
                ys = [c[1][1] for c in comps] + [c[1][1] + c[1][3] for c in comps]
                raw = [(min(xs), min(ys), max(xs), max(ys))]
            else:
                raw = [(s[0], s[1], s[0] + s[2], s[1] + s[3]) for _, s in comps[: p["max_boxes"]]]
            sx, sy = W / MAP_RES, H / MAP_RES
            for (x0, y0, x1, y1) in raw:
                pw, ph = (x1 - x0) * p["pad_frac"], (y1 - y0) * p["pad_frac"]
                boxes.append((int(max(0, (x0 - pw) * sx)), int(max(0, (y0 - ph) * sy)),
                              int(min(W, (x1 + pw) * sx)), int(min(H, (y1 + ph) * sy))))
        if fired and not boxes:
            notes.append("gate fired but no region survived the threshold")
        fired_eff = bool(boxes)
        words = [region_words(b, (W, H)) for b in boxes]
        region = {"boxes": boxes, "fired": fired_eff, "gate_fired": bool(fired), "region_words": words,
                  "score": a["score"], "score_norm": score_norm, "detector": a["detector"],
                  "image_size": (W, H), "mask": mask}
        ui = {"metrics": {"fired": fired_eff, "boxes": len(boxes), "where": ", ".join(words) or "-"}}
        if score_norm is not None:
            ui["metrics"]["score/tau"] = round(score_norm, 3)
        if notes:
            ui["notes"] = notes
        if img is not None:
            vis = overlay_heatmap(img, amap, 0.35)
            vis = draw_boxes(vis, boxes, (255, 255, 255), dashed=True)
            ui["images"] = [{"url": ctx.preview(vis, "box"), "label": "heatmap + box"}]
        rec = {"kind": "context", "cue_fired": fired_eff, "cue_detector": a["detector"],
               "cue_score": round(float(a["score"]), 5),
               "cue_score_norm": None if score_norm is None else round(float(score_norm), 4)}
        return Result(outputs={"region": region}, ui=ui, records=[rec])


COLORS = {"red": (255, 20, 20), "green": (0, 220, 90), "yellow": (255, 215, 0), "cyan": (0, 220, 255),
          "magenta": (255, 0, 200), "white": (255, 255, 255)}


def _crop(img, box, ctx_frac):
    x0, y0, x1, y1 = box
    W, H = img.size
    cw, ch = (x1 - x0) * ctx_frac, (y1 - y0) * ctx_frac
    side = max(x1 - x0 + 2 * cw, y1 - y0 + 2 * ch, 32)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0 = max(0, cx - side / 2), max(0, cy - side / 2)
    bx1, by1 = min(W, bx0 + side), min(H, by0 + side)
    return img.crop((int(bx0), int(by0), int(bx1), int(by1)))


@register
class VisualPrompt(Node):
    TYPE = "VisualPrompt"
    TITLE = "Visual Prompt"
    CATEGORY = "Prompt"
    DESCRIPTION = """
How the detector's region is shown to the MLLM. 'box+wash+label' reproduces
Phase 5. When the detector did not fire, the clean image passes through."""
    INPUTS = [Port("image", "IMAGE"), Port("region", "REGION"), Port("anomaly", "ANOMALY", optional=True)]
    OUTPUTS = [Port("image", "IMAGE")]
    PARAMS = [
        Param("style", "choice", "box+wash+label",
              ["box", "box+wash+label", "dashed box", "contour", "heatmap overlay", "crop-zoom",
               "side-by-side (full | crop)", "blur outside", "darken outside"]),
        Param("color", "choice", "red", list(COLORS)),
        Param("thickness", "float", 1.0, min=0.25, max=5.0, step=0.25, help="x auto line width"),
        Param("label", "text", "DEFECT CANDIDATE"),
        Param("alpha", "float", 0.45, min=0.0, max=1.0, step=0.05, help="wash / heatmap opacity"),
        Param("crop_context", "float", 0.5, min=0.0, max=3.0, step=0.1, help="context around the box for crops"),
    ]

    def run(self, ctx, inputs, p):
        img, r, a = inputs["image"].convert("RGB"), inputs["region"], inputs.get("anomaly")
        style, col = p["style"], COLORS[p["color"]]
        W, H = img.size
        lw = max(2, int(min(W, H) / 180 * p["thickness"]))
        if not r["fired"] and style != "heatmap overlay":
            out = img
        elif style == "heatmap overlay":
            if a is None:
                raise ValueError("'heatmap overlay' needs the ANOMALY input connected.")
            out = overlay_heatmap(img, a["map"], p["alpha"])
        elif style in ("box", "dashed box"):
            out = draw_boxes(img, r["boxes"], col, lw, dashed=style == "dashed box")
        elif style == "box+wash+label":
            base = img.convert("RGBA")
            ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for (x0, y0, x1, y1) in r["boxes"]:
                d.rectangle([x0, y0, x1, y1], fill=col + (int(255 * p["alpha"] * 0.4),))
                d.rectangle([x0, y0, x1, y1], outline=col + (235,), width=lw)
                if p["label"]:
                    bh = max(18, int(H * 0.035))
                    bw = int(len(p["label"]) * bh * 0.55)
                    by0 = max(0, y0 - bh)
                    d.rectangle([x0, by0, x0 + bw, by0 + bh], fill=col + (240,))
                    d.text((x0 + 6, by0 + 3), p["label"], fill=(255, 255, 255, 255))
            out = Image.alpha_composite(base, ov).convert("RGB")
        elif style == "contour":
            m = cv2.resize(r["mask"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            arr = np.array(img)
            cv2.drawContours(arr, cnts, -1, col, lw)
            out = Image.fromarray(arr)
        elif style == "crop-zoom":
            out = _crop(img, r["boxes"][0], p["crop_context"])
        elif style == "side-by-side (full | crop)":
            out = side_by_side(draw_boxes(img, r["boxes"][:1], col, lw), _crop(img, r["boxes"][0], p["crop_context"]))
        else:  # blur / darken outside
            keep = Image.new("L", img.size, 0)
            d = ImageDraw.Draw(keep)
            for b in r["boxes"]:
                d.rectangle(b, fill=255)
            bg = img.filter(ImageFilter.GaussianBlur(max(4, min(W, H) / 60))) if style == "blur outside" \
                else Image.eval(img, lambda v: int(v * (1 - p["alpha"])))
            out = Image.composite(img, bg, keep)
        label = style if r["fired"] else "passthrough (no cue)"
        return Result(outputs={"image": out}, ui={"images": [{"url": ctx.preview(out, "vp"), "label": label}]},
                      records=[{"kind": "context", "visual_prompt": style if r["fired"] else "none"}])

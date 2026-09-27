"""Scoring nodes (answer + detector) and display utilities."""

import numpy as np
from sklearn.metrics import roc_auc_score

from ..data import INDEX
from ..registry import Node, Param, Port, Result, register
from ..viz import draw_boxes, mask_overlay, to_map_res

MAP_RES = 256


@register
class Score(Node):
    TYPE = "Score"
    TITLE = "Score (Metrics)"
    CATEGORY = "Evaluate"
    DESCRIPTION = """
Compares the parsed answer with the ground truth. The label names this arm in
experiment results - put one Score node at the end of each arm (e.g. 'arm_a',
'arm_b') to get paired comparisons on identical samples."""
    INPUTS = [Port("answer", "ANSWER"), Port("question", "QUESTION")]
    OUTPUTS = [Port("verdict", "VERDICT")]
    PARAMS = [Param("label", "text", "arm_b", sweepable=False)]

    def run(self, ctx, inputs, p):
        a, q = inputs["answer"], inputs["question"]
        gt = q.get("answer") or None
        correct = bool(gt and a["letter"] == gt)
        probs = a.get("probs") or None
        rec = {"kind": "answer", "label": p["label"], "pred": a["letter"], "gt": gt, "correct": correct,
               "parsed": a["parsed"], "raw": (a.get("raw") or "")[:300], "n_options": len(q["options"]),
               "latency": a.get("latency"), "model": a.get("model")}
        if probs:
            rec["probs"] = probs
            rec["confidence"] = probs.get(a["letter"]) if a["letter"] else None
            rec["p_gt"] = probs.get(gt) if gt else None
        verdict = {"correct": correct, "pred": a["letter"], "gt": gt, "label": p["label"]}
        mark = "✓ correct" if correct else ("? no ground truth" if not gt else "✗ wrong")
        return Result(outputs={"verdict": verdict},
                      ui={"verdict": mark, "metrics": {"pred": a["letter"] or "∅", "GT": gt or "-"}},
                      records=[rec])


@register
class DetectorMetrics(Node):
    TYPE = "DetectorMetrics"
    TITLE = "Detector Metrics"
    CATEGORY = "Evaluate"
    DESCRIPTION = """
Scores the detector against MMAD's ground-truth mask: image score vs defect
label (for image AUROC over a batch), pixel AUROC, box IoU, hit (box touches
the defect) and pointing game (heatmap peak inside the defect)."""
    INPUTS = [Port("anomaly", "ANOMALY"), Port("sample", "SAMPLE"), Port("region", "REGION", optional=True),
              Port("image", "IMAGE", optional=True)]
    OUTPUTS = []
    PARAMS = [Param("label", "text", "detector", sweepable=False)]

    def run(self, ctx, inputs, p):
        a, s, r, img = inputs["anomaly"], inputs["sample"], inputs.get("region"), inputs.get("image")
        gt = INDEX.gt_mask(s["image_key"], (MAP_RES, MAP_RES))
        amap = to_map_res(a["map"], MAP_RES)
        defective = bool(gt.any())
        rec = {"kind": "detector", "label": p["label"], "detector": a["detector"], "image_key": s["image_key"],
               "defective": defective, "score": float(a["score"]),
               "score_norm": float(a["score"] / a["tau_score"]) if a.get("tau_score") else None}
        m = {"defective(GT)": defective, "score": round(float(a["score"]), 4)}
        if defective and not gt.all():
            rec["pixel_auroc"] = float(roc_auc_score(gt.ravel(), amap.ravel()))
            peak = np.unravel_index(np.argmax(amap), amap.shape)
            rec["pointing_hit"] = bool(gt[peak])
            m["pixel_AUROC"] = round(rec["pixel_auroc"], 3)
            m["peak_in_defect"] = rec["pointing_hit"]
        if r is not None:
            W, H = r["image_size"]
            boxmask = np.zeros((MAP_RES, MAP_RES), bool)
            for (x0, y0, x1, y1) in r["boxes"]:
                boxmask[int(y0 * MAP_RES / H):int(np.ceil(y1 * MAP_RES / H)),
                        int(x0 * MAP_RES / W):int(np.ceil(x1 * MAP_RES / W))] = True
            rec["fired"] = bool(r["fired"])
            if defective:
                inter, union = (boxmask & gt).sum(), (boxmask | gt).sum()
                rec["box_iou"] = float(inter / union) if union else 0.0
                rec["box_hit"] = bool(inter > 0)
                rec["gt_coverage"] = float(inter / gt.sum())
                m.update(box_IoU=round(rec["box_iou"], 3), box_hit=rec["box_hit"])
            m["fired"] = rec["fired"]
        ui = {"metrics": m}
        if img is not None and ctx.mode == "interactive":
            vis = mask_overlay(img, gt)
            if r is not None:
                vis = draw_boxes(vis, r["boxes"], (255, 40, 40))
            ui["images"] = [{"url": ctx.preview(vis, "gt"), "label": "GT mask (green) vs box (red)"}]
        return Result(outputs={}, ui=ui, records=[rec])


@register
class PreviewImage(Node):
    TYPE = "PreviewImage"
    TITLE = "Preview Image"
    CATEGORY = "Display"
    DESCRIPTION = "Shows an image (and saves a thumbnail per sample in experiments)."
    INPUTS = [Port("image", "IMAGE")]
    OUTPUTS = []
    PARAMS = [Param("caption", "text", "", sweepable=False)]

    def run(self, ctx, inputs, p):
        return Result(ui={"images": [{"url": ctx.preview(inputs["image"], "preview"), "label": p["caption"]}]})


@register
class ShowText(Node):
    TYPE = "ShowText"
    TITLE = "Show Text"
    CATEGORY = "Display"
    DESCRIPTION = "Shows any value as text."
    INPUTS = [Port("value", "ANY")]
    OUTPUTS = []
    PARAMS = []

    def run(self, ctx, inputs, p):
        v = inputs["value"]
        if isinstance(v, dict):
            v = {k: x for k, x in v.items() if k not in ("map", "mask")}
        return Result(ui={"text": v if isinstance(v, str) else repr(v)[:2000]})


@register
class Note(Node):
    TYPE = "Note"
    TITLE = "Note"
    CATEGORY = "Display"
    DESCRIPTION = "A sticky note for documenting the graph. Not executed."
    PARAMS = [Param("text", "textarea", "Notes…", sweepable=False)]

    def run(self, ctx, inputs, p):
        return Result()

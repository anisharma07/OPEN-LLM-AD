#!/usr/bin/env python
"""Generates the research-template workflows in workflows/templates/ (re-run after editing)."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "workflows" / "templates"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"


class G:
    def __init__(self):
        self.nodes, self.links, self.nid = [], [], 1

    def add(self, type_, x, y, title=None, **params):
        n = {"id": self.nid, "type": type_, "pos": [x, y], "params": params}
        if title:
            n["title"] = title
        self.nodes.append(n)
        self.nid += 1
        return n["id"]

    def link(self, src, out_idx, dst, in_idx):
        self.links.append({"id": len(self.links) + 1, "from": [src, out_idx], "to": [dst, in_idx]})

    def dump(self, name, experiment=None):
        g = {"version": 1, "nodes": self.nodes, "links": self.links}
        if experiment:
            g["experiment"] = experiment
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.json").write_text(json.dumps(g, indent=1), encoding="utf-8")


# port indices (keep in sync with the node classes)
S_SAMPLE, S_IMAGE, S_QUESTION = 0, 1, 2


def sample_block(g, x=0, y=0, **kw):
    s = g.add("MMADSample", x, y, **{"subtask": "any", "condition": "any", "index": 0, **kw})
    qb = g.add("QuestionBuilder", x, y + 560, paraphrase="original")
    g.link(s, S_QUESTION, qb, 0)
    return s, qb


def llm_tail(g, x, y, image_src, prompt_src, qb, label, model=MODEL, answer_mode="generate", title=None):
    m = g.add("MLLM", x, y, title=title, model=model, answer_mode=answer_mode)
    g.link(image_src[0], image_src[1], m, 0)
    g.link(prompt_src, 0, m, 1)
    g.link(qb, 0, m, 2)
    p = g.add("AnswerParser", x + 350, y, mode="lenient")
    g.link(m, 0, p, 0)
    g.link(qb, 0, p, 1)
    sc = g.add("Score", x + 350, y + 150, label=label)
    g.link(p, 0, sc, 0)
    g.link(qb, 0, sc, 1)
    return m


def arm_a(g, s, qb, x, y, label="arm_a", image_src=None):
    tp = g.add("TextPrompt", x + 680, y, title="Prompt (vanilla)", template="MMAD vanilla (Arm A)")
    g.link(qb, 0, tp, 0)
    llm_tail(g, x + 1030, y, image_src or (s, S_IMAGE), tp, qb, label, title="MLLM (Arm A)")


def arm_b(g, s, qb, x, y, detector="PatchCore", label="arm_b", image_src=None, det_params=None, metrics_label=None):
    img = image_src or (s, S_IMAGE)
    refs = None
    if detector in ("PatchCore", "WinCLIP"):
        refs = g.add("NormalReferences", x, y + 420, strategy="similar", k=8)
        g.link(s, S_SAMPLE, refs, 0)
    d = g.add(detector, x, y, **(det_params or {}))
    g.link(img[0], img[1], d, 0)
    if detector == "PatchCore":
        g.link(refs, 0, d, 1)
    elif detector == "WinCLIP":
        g.link(refs, 0, d, 1)
        g.link(s, S_SAMPLE, d, 2)
    else:
        g.link(s, S_SAMPLE, d, 1)
    hb = g.add("HeatmapToBox", x + 340, y)
    g.link(d, 0, hb, 0)
    g.link(img[0], img[1], hb, 1)
    vp = g.add("VisualPrompt", x + 680, y, style="box+wash+label")
    g.link(img[0], img[1], vp, 0)
    g.link(hb, 0, vp, 1)
    g.link(d, 0, vp, 2)
    tp = g.add("TextPrompt", x + 680, y + 330, title="Prompt (grounded)", template="grounded (Phase 5)")
    g.link(qb, 0, tp, 0)
    g.link(hb, 0, tp, 1)
    llm_tail(g, x + 1030, y, (vp, 0), tp, qb, label, title=f"MLLM ({label})")
    if metrics_label:
        dm = g.add("DetectorMetrics", x + 340, y + 390, label=metrics_label)
        g.link(d, 0, dm, 0)
        g.link(s, S_SAMPLE, dm, 1)
        g.link(hb, 0, dm, 2)
        g.link(img[0], img[1], dm, 3)
    return d, hb


def note(g, x, y, text):
    g.add("Note", x, y, text=text)


# ---------------------------------------------------------------------------- 01
g = G()
s, qb = sample_block(g)
arm_b(g, s, qb, 360, 0, "PatchCore", "arm_b", metrics_label="patchcore")
note(g, 0, 820, "Arm B · detector-guided MLLM (Phase 5 architecture).\n"
     "Right-click PatchCore → Replace with → WinCLIP to swap the detector.\n"
     "Right-click any widget label to sweep it in an experiment.")
g.dump("01_arm_b_patchcore", {"name": "Arm B PatchCore", "n": 90, "seed": 42, "sweep": []})

# ---------------------------------------------------------------------------- 02
g = G()
s, qb = sample_block(g)
arm_b(g, s, qb, 360, 0, "WinCLIP", "arm_b_winclip", metrics_label="winclip")
note(g, 0, 820, "Arm B with WinCLIP (CLIP window prompts). Toggle mode zero-shot vs few-shot (WinCLIP+).\n"
     "First run downloads openai/clip-vit-base-patch16 (~600 MB).")
g.dump("02_arm_b_winclip", {"name": "Arm B WinCLIP", "n": 90, "seed": 42, "sweep": []})

# ---------------------------------------------------------------------------- 03
g = G()
s, qb = sample_block(g)
arm_a(g, s, qb, 360, -560)
arm_b(g, s, qb, 360, 120, "PatchCore", "arm_b", metrics_label="patchcore")
note(g, 0, 820, "Arm A vs Arm B on identical questions.\nThe results page reports the paired difference, "
     "fixes/breaks and an exact McNemar test.")
g.dump("03_arm_a_vs_arm_b", {"name": "Arm A vs Arm B", "n": 180, "seed": 42, "sweep": []})

# ---------------------------------------------------------------------------- 04
g = G()
s, qb = sample_block(g)
arm_a(g, s, qb, 360, -560, label="no_box")
arm_b(g, s, qb, 360, 120, "SyntheticDetector", "oracle_box", metrics_label="oracle")
arm_b(g, s, qb, 360, 900, "ControlDetector", "random_box", det_params={"mode": "random_box", "area_frac": 0.08})
note(g, 0, 820, "Ceiling & null controls.\noracle_box = perfect localisation (GT mask) → upper bound for Arm B.\n"
     "random_box = a box with no information → does any box help, or only a correct one?\n"
     "Sweep the synthetic detector's miss_rate / shift_frac to find how good a detector must be.")
g.dump("04_detector_ceiling_and_controls", {"name": "Oracle vs random box vs none", "n": 180, "seed": 42,
                                            "condition": "defective", "sweep": []})

# ---------------------------------------------------------------------------- 05
g = G()
s, qb = sample_block(g)
cor = g.add("Corruption", 0, -420, corruption="motion_blur", severity=3)
g.link(s, S_IMAGE, cor, 0)
arm_a(g, s, qb, 360, -560, image_src=(cor, 0))
arm_b(g, s, qb, 360, 120, "PatchCore", "arm_b", image_src=(cor, 0), metrics_label="patchcore")
note(g, 0, 820, "Robustness: the same corrupted image feeds both arms (and the detector).\n"
     "The experiment preset sweeps severity 0-5; the results page computes RDS per arm.")
g.dump("05_robustness_corruption_sweep", {
    "name": "Motion blur severity sweep, Arm A vs B", "n": 60, "seed": 42,
    "sweep": [{"node_id": cor, "param": "severity", "values": [0, 1, 2, 3, 4, 5], "text": "0:5:1"}]})

# ---------------------------------------------------------------------------- 06
g = G()
s, qb = sample_block(g)
tpa = g.add("TextPrompt", 360, -420, title="Prompt (vanilla)", template="MMAD vanilla (Arm A)")
g.link(qb, 0, tpa, 0)
llm_tail(g, 720, -420, (s, S_IMAGE), tpa, qb, "arm_a", title="MLLM (Arm A)")
refs = g.add("NormalReferences", 360, 450, k=8)
g.link(s, S_SAMPLE, refs, 0)
pc = g.add("PatchCore", 360, 120)
g.link(s, S_IMAGE, pc, 0)
g.link(refs, 0, pc, 1)
hb = g.add("HeatmapToBox", 690, 120)
g.link(pc, 0, hb, 0)
g.link(s, S_IMAGE, hb, 1)
vp = g.add("VisualPrompt", 1020, 120)
g.link(s, S_IMAGE, vp, 0)
g.link(hb, 0, vp, 1)
tpb = g.add("TextPrompt", 1020, 540, title="Prompt (grounded)", template="grounded (Phase 5)")
g.link(qb, 0, tpb, 0)
g.link(hb, 0, tpb, 1)
llm_tail(g, 1360, 120, (vp, 0), tpb, qb, "arm_b", title="MLLM (Arm B, always)")
gate = g.add("SelectiveGate", 1360, 700, guided_subtasks="Defect Classification, Defect Analysis")
g.link(qb, 0, gate, 0)
g.link(vp, 0, gate, 1)
g.link(s, S_IMAGE, gate, 2)
g.link(tpb, 0, gate, 3)
g.link(tpa, 0, gate, 4)
m = g.add("MLLM", 1700, 700, title="MLLM (gated)", model=MODEL)
g.link(gate, 0, m, 0)
g.link(gate, 1, m, 1)
g.link(qb, 0, m, 2)
p = g.add("AnswerParser", 2040, 700)
g.link(m, 0, p, 0)
g.link(qb, 0, p, 1)
sc = g.add("Score", 2040, 870, label="gated")
g.link(p, 0, sc, 0)
g.link(qb, 0, sc, 1)
note(g, 0, 820, "Phase 5 recommendation made executable: guide only the subtasks that benefited\n"
     "(Defect Classification / Analysis) and show the clean image otherwise.")
g.dump("06_selective_gating", {"name": "Selective gating vs always-guided", "n": 180, "seed": 42, "sweep": []})

# ---------------------------------------------------------------------------- 07
g = G()
s = g.add("MMADSample", 0, 0, subtask="Anomaly Detection", condition="any", index=0)
refs = g.add("NormalReferences", 0, 560, k=8)
g.link(s, S_SAMPLE, refs, 0)
for i, (det, label) in enumerate([("PatchCore", "patchcore"), ("WinCLIP", "winclip")]):
    y = i * 520
    d = g.add(det, 360, y)
    g.link(s, S_IMAGE, d, 0)
    g.link(refs, 0, d, 1)
    if det == "WinCLIP":
        g.link(s, S_SAMPLE, d, 2)
    hb = g.add("HeatmapToBox", 700, y)
    g.link(d, 0, hb, 0)
    g.link(s, S_IMAGE, hb, 1)
    dm = g.add("DetectorMetrics", 1030, y, label=label)
    g.link(d, 0, dm, 0)
    g.link(s, S_SAMPLE, dm, 1)
    g.link(hb, 0, dm, 2)
    g.link(s, S_IMAGE, dm, 3)
note(g, 0, 900, "Detector-only benchmark (no MLLM, fast).\nImage AUROC, pixel AUROC, box IoU, TPR/FPR of the gate.\n"
     "Uses Anomaly Detection questions so each image is counted once.")
g.dump("07_detector_benchmark", {"name": "PatchCore vs WinCLIP detector quality", "n": 200, "seed": 42,
                                 "subtasks": ["Anomaly Detection"], "sweep": []})

# ---------------------------------------------------------------------------- 08
g = G()
li = g.add("LoadImage", 0, 0)
cq = g.add("CustomQuestion", 0, 420, question="Is there any defect in the object?", options="A: Yes.\nB: No.")
refs = g.add("NormalReferences", 0, 760, source="GoodsAD/drink_can", k=8)
pc = g.add("PatchCore", 360, 0)
g.link(li, 0, pc, 0)
g.link(refs, 0, pc, 1)
hb = g.add("HeatmapToBox", 690, 0)
g.link(pc, 0, hb, 0)
g.link(li, 0, hb, 1)
vp = g.add("VisualPrompt", 1020, 0)
g.link(li, 0, vp, 0)
g.link(hb, 0, vp, 1)
tp = g.add("TextPrompt", 1020, 420, template="grounded (Phase 5)")
g.link(cq, 0, tp, 0)
g.link(hb, 0, tp, 1)
m = g.add("MLLM", 1360, 0, model=MODEL, answer_mode="letter-logits")
g.link(vp, 0, m, 0)
g.link(tp, 0, m, 1)
g.link(cq, 0, m, 2)
p = g.add("AnswerParser", 1700, 0)
g.link(m, 0, p, 0)
g.link(cq, 0, p, 1)
st = g.add("ShowText", 1700, 200)
g.link(p, 0, st, 0)
note(g, 0, 1100, "Playground: upload your own image (or drop it on the canvas), write a question,\n"
     "pick the normal-reference category, and inspect every stage.")
g.dump("08_custom_image_playground")

# ---------------------------------------------------------------------------- 09
g = G()
s, qb = sample_block(g)
arm_a(g, s, qb, 360, -560)
arm_b(g, s, qb, 360, 120, "PatchCore", "arm_b")
note(g, 0, 820, "Prompt sensitivity: the preset sweeps the Question Builder paraphrase.\n"
     "The 'Spread' table is the accuracy range across paraphrases, per arm.\n"
     "Tick shuffle_options to measure letter-position bias.")
g.dump("09_prompt_sensitivity_spread", {
    "name": "Paraphrase spread, Arm A vs B", "n": 90, "seed": 42,
    "sweep": [{"node_id": qb, "param": "paraphrase", "values": ["original", "P1 lexical", "P2 role framing", "P3 exam format"],
               "text": "original, P1 lexical, P2 role framing, P3 exam format"}]})

print("templates:", sorted(p.name for p in OUT.glob("*.json")))

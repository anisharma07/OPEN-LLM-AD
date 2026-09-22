"""
Build the comparative datasets: MMAD (ICLR 2025) reference numbers vs this
dissertation's measured results.

Everything about "this work" is recomputed from the raw run artefacts
(manifests + per-question JSONL) - never copied from the manuscript prose - so
the figures cannot drift away from what was actually measured.

Outputs -> comparative-analysis/data/
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LT = ROOT / "Local-training"
DATA = ROOT / "comparative-analysis" / "data"
DATA.mkdir(parents=True, exist_ok=True)

# MMAD Table 2 reports 7 task columns. mmad.json actually carries 9 subtasks:
# the paper folds Object Analysis / Structure / Details into one "Object
# Analysis" column (paper Sec. 3.2: "composition, position, appearance and
# function of the object"). We reproduce that fold so the columns line up.
PAPER_COLS = [
    "anomaly_discrimination",
    "defect_classification",
    "defect_localization",
    "defect_description",
    "defect_analysis",
    "object_classification",
    "object_analysis",
]
OBJECT_FOLD = ["Object Analysis", "Object Structure", "Object Details"]
SUBTASK_TO_PAPER = {
    "Anomaly Detection": "anomaly_discrimination",
    "Defect Classification": "defect_classification",
    "Defect Localization": "defect_localization",
    "Defect Description": "defect_description",
    "Defect Analysis": "defect_analysis",
    "Object Classification": "object_classification",
}

# Model metadata for this work (params / precision / VRAM come from the run logs
# and the manuscript's model-zoo table; accuracy never does).
THIS_WORK_META = {
    "Qwen3-VL-2B-Instruct": dict(
        short="Qwen3-VL-2B", params_b=2.2, precision="FP16", vram_gb=4.82,
        vision="Dynamic-resolution patch ViT"),
    "Qwen/Qwen2.5-VL-3B-Instruct": dict(
        short="Qwen2.5-VL-3B", params_b=3.1, precision="FP16", vram_gb=5.86,
        vision="Windowed dynamic ViT"),
    "google/gemma-4-E4B-it": dict(
        short="Gemma-4-E4B", params_b=4.4, precision="4-bit NF4", vram_gb=3.32,
        vision="Any-to-any pooled ViT"),
    "google/gemma-4-E2B-it": dict(
        short="Gemma-4-E2B", params_b=2.3, precision="FP16", vram_gb=4.65,
        vision="SigLIP token pooler"),
}

# Per-subtask question counts of the shared 2,500-question sample (phase-1
# manifest; every phase-4 model answered the identical set).
SUBTASK_N = json.loads(
    (LT / "phase-1" / "results" / "phase1_manifest.json").read_text()
)["subtask_accuracy"]
SUBTASK_N = {k: v["total"] for k, v in SUBTASK_N.items()}


def fold_to_paper_columns(subtask_acc_pct):
    """9 measured subtasks -> the paper's 7 columns (count-weighted fold)."""
    row = {}
    for src, dst in SUBTASK_TO_PAPER.items():
        row[dst] = subtask_acc_pct[src]
    num = sum(subtask_acc_pct[s] * SUBTASK_N[s] for s in OBJECT_FOLD)
    den = sum(SUBTASK_N[s] for s in OBJECT_FOLD)
    row["object_analysis"] = num / den
    # The paper's "Average" is the unweighted mean of its 7 columns - verified
    # against every printed row (GPT-4o: 524.45/7 = 74.92).
    row["average"] = sum(row[c] for c in PAPER_COLS) / 7
    return row


def balanced_anomaly_accuracy(records):
    """
    MMAD scores Anomaly Discrimination as the mean of normal-class and
    abnormal-class accuracy, not plain accuracy (paper Sec. 4.1, because the
    split is imbalanced). Recompute it that way so the column is comparable.

    Normal samples are identified by a '/good/' segment in the image path,
    which is the MVTec/VisA/GoodsAD convention used throughout MMAD.
    """
    pos = neg = pos_ok = neg_ok = 0
    for r in records:
        if r["subtask"] != "Anomaly Detection":
            continue
        normal = bool(re.search(r"/good/", r["image_path"]))
        ok = bool(r["is_correct"])
        if normal:
            neg += 1
            neg_ok += ok
        else:
            pos += 1
            pos_ok += ok
    if not pos or not neg:
        return None, pos, neg
    return 100 * (pos_ok / pos + neg_ok / neg) / 2, pos, neg


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalise(records):
    """phase-1 and phase-4 JSONL use slightly different key names."""
    out = []
    for r in records:
        out.append({
            "subtask": r["subtask"],
            "is_correct": r["is_correct"],
            "image_path": r["image_path"],
        })
    return out


# ---------------------------------------------------------------- this work
phase4 = json.loads((LT / "phase-4" / "results" / "phase4_manifest.json").read_text())

RAW_RUNS = {
    "Qwen3-VL-2B-Instruct": LT / "phase-1" / "results" / "phase1_results.jsonl",
    "Qwen/Qwen2.5-VL-3B-Instruct": LT / "phase-4" / "results" / "qwen_qwen2.5_vl_3b_instruct_results.jsonl",
    "google/gemma-4-E2B-it": LT / "phase-4" / "results" / "google_gemma_4_e2b_it_results.jsonl",
    "google/gemma-4-E4B-it": LT / "phase-4" / "results" / "google_gemma_4_e4b_it_results.jsonl",
}

rows, subtask_rows = [], []
for model_id, blob in phase4.items():
    meta = THIS_WORK_META[model_id]
    sub = {k: 100 * v for k, v in blob["subtask_accuracy"].items()}
    folded = fold_to_paper_columns(sub)

    bal, n_abn, n_norm = None, None, None
    path = RAW_RUNS.get(model_id)
    if path and path.exists():
        bal, n_abn, n_norm = balanced_anomaly_accuracy(normalise(load_jsonl(path)))

    folded_bal = dict(folded)
    if bal is not None:
        folded_bal["anomaly_discrimination"] = bal
        folded_bal["average"] = sum(folded_bal[c] for c in PAPER_COLS) / 7

    rows.append({
        "model": meta["short"],
        "model_id": model_id,
        "params_b": meta["params_b"],
        "precision": meta["precision"],
        "vram_gb": meta["vram_gb"],
        "vision_encoder": meta["vision"],
        "family": "this_work",
        "overall_accuracy_9task": 100 * blob["accuracy"],
        "kappa": blob["kappa"],
        "latency_s": blob["avg_latency"],
        "throughput_fps": blob["throughput_fps"],
        "n_questions": blob["sample_count"],
        # plain-accuracy variant of the anomaly column
        **{f"{k}_plainacc": v for k, v in folded.items()},
        # MMAD-protocol variant (balanced anomaly accuracy) - the comparable one
        **folded_bal,
        "anomaly_n_abnormal": n_abn,
        "anomaly_n_normal": n_norm,
    })

    for s, acc in sub.items():
        subtask_rows.append({"model": meta["short"], "subtask": s,
                             "accuracy": acc, "n": SUBTASK_N[s]})

this_work = pd.DataFrame(rows).sort_values("average", ascending=False)
this_work.to_csv(DATA / "thiswork_mmad_protocol.csv", index=False)
pd.DataFrame(subtask_rows).to_csv(DATA / "thiswork_subtask_9.csv", index=False)

# ------------------------------------------------- merged leaderboard table
paper = pd.read_csv(DATA / "mmad_paper_table2.csv")
paper["source"] = "MMAD ICLR-2025 (1-shot)"
paper["setting"] = "1-shot"

tw = this_work[["model", "params_b", "family", *PAPER_COLS, "average"]].copy()
tw["scale"] = tw["params_b"].map(lambda x: f"{x:.1f}B")
tw["source"] = "This work (0-shot, N=2500)"
tw["setting"] = "0-shot"

merged = pd.concat([paper, tw], ignore_index=True)
merged = merged.sort_values("average", ascending=False)
merged.to_csv(DATA / "merged_leaderboard.csv", index=False)

# ------------------------------------------------ robustness (recomputed 5k)
bench = LT / "benchmark_5k_results.jsonl"
rob = pd.DataFrame(load_jsonl(bench))
rob = (rob.groupby(["condition", "corruption", "severity", "strategy", "subtask"])
          .agg(accuracy=("is_correct", lambda s: 100 * s.mean()),
               n=("is_correct", "size"))
          .reset_index())
rob.to_csv(DATA / "thiswork_robustness_5k_by_subtask.csv", index=False)

overall = (pd.DataFrame(load_jsonl(bench))
             .groupby("condition")
             .agg(accuracy=("is_correct", lambda s: 100 * s.mean()),
                  n=("is_correct", "size"))
             .reset_index()
             .sort_values("accuracy", ascending=False))
overall.to_csv(DATA / "thiswork_robustness_5k_overall.csv", index=False)

print("== this work, MMAD 7-column protocol ==")
print(this_work[["model", "params_b", "precision", "anomaly_discrimination",
                 "defect_localization", "object_classification", "average",
                 "overall_accuracy_9task"]].to_string(index=False,
                                                      float_format=lambda v: f"{v:.2f}"))
print("\n== balanced vs plain anomaly accuracy ==")
print(this_work[["model", "anomaly_discrimination_plainacc",
                 "anomaly_discrimination", "anomaly_n_normal",
                 "anomaly_n_abnormal"]].to_string(index=False,
                                                  float_format=lambda v: f"{v:.2f}"))
print("\n== 5k robustness conditions (recomputed from 25,000 raw rows) ==")
print(overall.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\nwrote 5 csv files -> {DATA}")

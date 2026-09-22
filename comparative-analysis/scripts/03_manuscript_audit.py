"""
Cross-check the numbers printed in the thesis draft against the numbers in the
run artefacts. The figures in this folder are built from the artefacts, so any
row that disagrees is a place where the manuscript and the evidence diverge.

Outputs -> comparative-analysis/results/tables/manuscript_audit.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LT = ROOT / "Local-training"
TABLES = ROOT / "comparative-analysis" / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Transcribed from the draft manuscript, section 6.2 (Phase 1, N=2,500)
DRAFT_PHASE1 = {
    "Object Classification": 89.21, "Object Structure": 84.53,
    "Object Analysis": 82.67, "Defect Description": 72.66,
    "Object Details": 74.10, "Defect Analysis": 76.98,
    "Anomaly Detection": 55.56, "Defect Localization": 47.65,
    "Defect Classification": 45.85,
}
# Section 9.4 (cross-model comparison table)
DRAFT_PHASE4 = {
    "Qwen3-VL-2B-Instruct": DRAFT_PHASE1,
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "Anomaly Detection": 53.41, "Defect Analysis": 74.82,
        "Defect Classification": 43.68, "Defect Description": 70.50,
        "Defect Localization": 51.26, "Object Analysis": 78.70,
        "Object Classification": 87.05, "Object Details": 69.42,
        "Object Structure": 80.22},
    "google/gemma-4-E4B-it": {
        "Anomaly Detection": 54.12, "Defect Analysis": 71.58,
        "Defect Classification": 42.96, "Defect Description": 66.91,
        "Defect Localization": 31.41, "Object Analysis": 77.98,
        "Object Classification": 86.33, "Object Details": 70.14,
        "Object Structure": 80.58},
    "google/gemma-4-E2B-it": {
        "Anomaly Detection": 51.25, "Defect Analysis": 68.35,
        "Defect Classification": 36.46, "Defect Description": 63.31,
        "Defect Localization": 26.35, "Object Analysis": 72.20,
        "Object Classification": 84.17, "Object Details": 64.75,
        "Object Structure": 67.99},
}

measured = json.loads((LT / "phase-4" / "results" / "phase4_manifest.json").read_text())

rows = []
for model, draft in DRAFT_PHASE4.items():
    meas = {k: 100 * v for k, v in measured[model]["subtask_accuracy"].items()}
    for task, dv in draft.items():
        rows.append({"model": model, "subtask": task,
                     "draft_manuscript": dv, "measured_artefact": round(meas[task], 2),
                     "delta": round(meas[task] - dv, 2)})

audit = pd.DataFrame(rows)
audit["agrees"] = audit["delta"].abs() < 0.5
audit.to_csv(TABLES / "manuscript_audit.csv", index=False)

n_ok = int(audit.agrees.sum())
print(f"subtask cells checked : {len(audit)}")
print(f"agree (|delta|<0.5)   : {n_ok}")
print(f"disagree              : {len(audit) - n_ok}\n")
print("largest disagreements:")
print(audit.reindex(audit.delta.abs().sort_values(ascending=False).index)
           .head(12).to_string(index=False))

# Does the draft's Phase-1 table actually reproduce the 5,000-sample clean column?
five_k = pd.read_csv(ROOT / "comparative-analysis" / "data" /
                     "thiswork_robustness_5k_by_subtask.csv")
clean = five_k[five_k.condition == "clean"].set_index("subtask")["accuracy"]
cmp = pd.DataFrame({
    "draft_phase1_table": pd.Series(DRAFT_PHASE1),
    "measured_phase1_2500": pd.Series(
        {k: round(100 * v, 2) for k, v in
         measured["Qwen3-VL-2B-Instruct"]["subtask_accuracy"].items()}),
    "measured_5k_clean": clean.round(2),
})
cmp["draft_vs_phase1"] = (cmp.draft_phase1_table - cmp.measured_phase1_2500).round(2)
cmp["draft_vs_5k_clean"] = (cmp.draft_phase1_table - cmp.measured_5k_clean).round(2)
cmp.to_csv(TABLES / "manuscript_phase1_table_origin.csv")
print("\nWhich run does the draft's Phase-1 table match?")
print(cmp.to_string())
print(f"\nmean |draft - phase1_2500|  = {cmp.draft_vs_phase1.abs().mean():.2f} pts")
print(f"mean |draft - 5k_clean|     = {cmp.draft_vs_5k_clean.abs().mean():.2f} pts")

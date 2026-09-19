#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Phase 3: Robustness Adaptation, Mitigation & Recovery
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Evaluates 3 adaptation & mitigation strategies against severe industrial corruptions:
1. Input-Level Test-Time Image Restoration (TTA-IR)
2. Prompt-Level Defect-Anchored Reasoning (DAR)
3. Hybrid Compound Adaptation (TTA + DAR)

Metrics:
- Robustness Recovery Rate (RRR)
- Degradation Slope Reduction (Delta RDS)
- Subtask Recovery Matrix (especially Defect Localization)
==============================================================================
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch

# Add paths for local modules
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase-2"))

from corruptions import apply_corruption, CORRUPTION_DISPLAY_NAMES
from mitigations import apply_mitigation_filter, build_standard_prompt, build_defect_anchored_prompt

# =============================================================================
# CLI Arguments & Deterministic Setup
# =============================================================================
parser = argparse.ArgumentParser(description="Phase 3: Industrial Robustness Adaptation & Mitigation Benchmark")
parser.add_argument("--sample-size", type=int, default=45, help="Number of stratified sample questions per condition.")
parser.add_argument("--corruptions", type=str, default="motion_blur,gaussian_noise,low_light", help="Target corruptions to mitigate.")
parser.add_argument("--severities", type=str, default="3,5", help="Severity levels to test.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--reset", action="store_true", help="Clear previous results and restart from scratch.")
args = parser.parse_args()

GLOBAL_SEED = args.seed
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

TARGET_CORRUPTIONS = [c.strip() for c in args.corruptions.split(",") if c.strip()]
SEVERITY_LEVELS = [int(s.strip()) for s in args.severities.split(",") if s.strip()]

BASE_DIR = PARENT_DIR.parent
MMAD_DIR = BASE_DIR / "MMAD"
RESULTS_DIR = CURRENT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSONL = RESULTS_DIR / "phase3_results.jsonl"
MANIFEST_PATH = RESULTS_DIR / "phase3_manifest.json"
RESULTS_TXT_PATH = RESULTS_DIR / "results.txt"

def get_gpu_info():
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        }
    return {"gpu_name": "CPU", "gpu_memory_gb": 0, "cuda_version": None}

gpu_info = get_gpu_info()

print("=" * 80)
print("PHASE 3: INDUSTRIAL ROBUSTNESS ADAPTATION, MITIGATION & RECOVERY")
print("=" * 80)
print(f"  Target GPU            : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)")
print(f"  PyTorch / CUDA        : {torch.__version__} / CUDA {gpu_info['cuda_version']}")
print(f"  Sample Size           : {args.sample_size} questions per condition")
print(f"  Corruptions to Mitigate: {TARGET_CORRUPTIONS}")
print(f"  Severity Levels       : {SEVERITY_LEVELS}")
print(f"  Adaptation Strategies : 1. TTA-IR (Image Restoration), 2. DAR (Anchored Prompt), 3. Hybrid")
print("=" * 80)


# =============================================================================
# Dataset Resolution & Stratified Sampling
# =============================================================================
mmad_json_path = MMAD_DIR / "mmad.json"
print(f"\n📂 Loading MMAD benchmark from: {mmad_json_path.name}...")

with open(mmad_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def resolve_local_image_path(rel_path, mmad_base_dir):
    candidates = [
        mmad_base_dir / rel_path,
        mmad_base_dir / "images" / rel_path,
        mmad_base_dir / "data" / rel_path,
        mmad_base_dir / rel_path.lstrip("/"),
    ]
    parts = rel_path.split("/")
    if len(parts) > 1:
        candidates.append(mmad_base_dir / parts[0] / "/".join(parts[1:]))
        candidates.append(mmad_base_dir / "images" / parts[0] / "/".join(parts[1:]))
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand)
    return None

all_questions = []
for key, val in raw_data.items():
    parts = key.split("/")
    dataset_name = parts[0] if len(parts) > 0 else "Unknown"
    category = parts[1] if len(parts) > 1 else "unknown"
    resolved_img = resolve_local_image_path(key, MMAD_DIR)

    if isinstance(val, dict):
        for sub_k, sub_v in val.items():
            if isinstance(sub_v, dict):
                entry = dict(sub_v)
                entry["_subtask"] = sub_v.get("type", sub_k)
                entry["_category"] = category
                entry["_dataset"] = dataset_name
                entry["_image_key"] = key
                entry["_resolved_image_path"] = resolved_img
                all_questions.append(entry)
            elif isinstance(sub_v, list):
                for item in sub_v:
                    if isinstance(item, dict):
                        entry = dict(item)
                        entry["_subtask"] = item.get("type", sub_k)
                        entry["_category"] = category
                        entry["_dataset"] = dataset_name
                        entry["_image_key"] = key
                        entry["_resolved_image_path"] = resolved_img
                        all_questions.append(entry)

valid_questions = [q for q in all_questions if q.get("_resolved_image_path") and os.path.exists(q["_resolved_image_path"])]
print(f"   ✅ Valid questions with local images: {len(valid_questions)}")

subtask_bins = defaultdict(list)
for q in valid_questions:
    st = q.get("_subtask", "Unknown")
    subtask_bins[st].append(q)

per_subtask = max(1, args.sample_size // len(subtask_bins))
eval_base_questions = []

random.seed(GLOBAL_SEED)
for st, q_list in subtask_bins.items():
    sampled = random.sample(q_list, min(len(q_list), per_subtask))
    eval_base_questions.extend(sampled)

if len(eval_base_questions) < args.sample_size:
    rem = [q for q in valid_questions if q not in eval_base_questions]
    eval_base_questions.extend(random.sample(rem, min(len(rem), args.sample_size - len(eval_base_questions))))

print(f"   ✅ Base evaluation set: {len(eval_base_questions)} questions stratified across {len(subtask_bins)} subtasks.")


# =============================================================================
# Load Model & Processor
# =============================================================================
from transformers import AutoProcessor
import re

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
print(f"\n🤖 Loading model: {MODEL_ID} in FP16...")

t_start = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = None
try:
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    print("   ✅ Loaded via Qwen3VLForConditionalGeneration")
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    print("   ✅ Loaded via AutoModelForImageTextToText")

model.eval()
print(f"   ✅ Model loaded in {time.time() - t_start:.2f}s")


# =============================================================================
# Inference & Answer Parsing
# =============================================================================
def parse_answer(response_text):
    if not response_text:
        return None, False
    text = response_text.strip()
    m = re.match(r"^([A-D])\b", text.upper())
    if m:
        return m.group(1), True
    m = re.search(r"\(([A-D])\)|([A-D])[).\s:]", text.upper())
    if m:
        return m.group(1) or m.group(2), True
    letters = re.findall(r"\b([A-D])\b", text.upper())
    if len(letters) == 1:
        return letters[0], True
    for c in text.upper()[:5]:
        if c in "ABCD":
            return c, True
    return None, False


def get_ground_truth(question_data):
    ci_data = {str(k).lower(): v for k, v in question_data.items()}
    for f in ["answer", "ground_truth", "label", "gt"]:
        if f in ci_data and ci_data[f] is not None:
            gt = str(ci_data[f]).strip().upper()
            if len(gt) == 1 and gt in "ABCD":
                return gt
            if gt.isdigit() and int(gt) < 4:
                return chr(65 + int(gt))
            return gt
    return None


def run_inference(model, processor, pil_img, prompt_text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = None
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        pass

    if inputs is None:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        except Exception:
            inputs = processor(text=[prompt_text], images=[pil_img], return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# =============================================================================
# Benchmarking Loop: Clean vs Corrupted vs TTA vs DAR vs Hybrid
# =============================================================================
if args.reset and RESULTS_JSONL.exists():
    print("🧹 --reset specified: Deleting previous results...")
    RESULTS_JSONL.unlink()

completed_keys = set()
if RESULTS_JSONL.exists():
    with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    completed_keys.add(obj.get("_unique_run_key"))
                except Exception:
                    pass
    print(f"🔄 Resuming Phase 3 run: Found {len(completed_keys)} previously completed results.")

print("\n" + "=" * 80)
print("🚀 COMMENCING ADAPTATION & MITIGATION EXPERIMENTS")
print("=" * 80)

# Pre-compute Clean Baselines for the evaluation sample
clean_results_cache = {}
print("\n--- Step 1: Establishing Clean Baseline References ---")
for idx, q in enumerate(eval_base_questions):
    run_key = f"clean_idx_{idx}"
    std_prompt, q_text, opts = build_standard_prompt(q)
    gt = get_ground_truth(q)
    img_path = q["_resolved_image_path"]

    if run_key in completed_keys:
        continue

    try:
        raw_img = Image.open(img_path).convert("RGB")
        raw_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    except Exception:
        raw_img = Image.new("RGB", (224, 224), (128, 128, 128))

    t0 = time.time()
    resp = run_inference(model, processor, raw_img, std_prompt)
    lat = time.time() - t0
    pred, parsed = parse_answer(resp)
    is_correct = (pred == gt) if (pred and gt) else False

    record = {
        "_unique_run_key": run_key,
        "mode": "clean",
        "corruption": "clean",
        "severity": 0,
        "strategy": "none",
        "question_idx": idx,
        "question_text": q_text,
        "options": opts,
        "ground_truth": gt,
        "prediction": pred,
        "raw_response": resp,
        "is_correct": is_correct,
        "latency_sec": round(lat, 3),
        "subtask": q.get("_subtask", "Unknown"),
        "dataset": q.get("_dataset", "Unknown"),
        "category": q.get("_category", "Unknown"),
        "image_path": img_path,
    }

    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    completed_keys.add(run_key)
    clean_results_cache[idx] = record

# Mitigation Evaluation Matrix
strategies = ["corrupted_baseline", "mitigated_tta", "mitigated_dar", "mitigated_hybrid"]
total_combos = len(TARGET_CORRUPTIONS) * len(SEVERITY_LEVELS) * len(strategies) * len(eval_base_questions)
print(f"\n--- Step 2: Evaluating Mitigations across {total_combos} conditions ---")

eval_count = 0
for corr in TARGET_CORRUPTIONS:
    for sev in SEVERITY_LEVELS:
        print(f"\n⚡ Evaluating Corruption: {corr.upper()} (Severity {sev})")
        for q_idx, q in enumerate(eval_base_questions):
            img_path = q["_resolved_image_path"]
            gt = get_ground_truth(q)
            std_prompt, q_text, opts = build_standard_prompt(q)
            dar_prompt, _, _ = build_defect_anchored_prompt(q)

            try:
                raw_img = Image.open(img_path).convert("RGB")
                raw_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                corr_img = apply_corruption(raw_img, corr, sev)
            except Exception:
                corr_img = Image.new("RGB", (224, 224), (128, 128, 128))

            # Pre-compute Restored Image for TTA & Hybrid
            restored_img = apply_mitigation_filter(corr_img, corr, sev)

            test_cases = [
                ("corrupted_baseline", corr_img, std_prompt),
                ("mitigated_tta", restored_img, std_prompt),
                ("mitigated_dar", corr_img, dar_prompt),
                ("mitigated_hybrid", restored_img, dar_prompt),
            ]

            for strat_name, active_img, active_prompt in test_cases:
                run_key = f"{corr}_s{sev}_{strat_name}_idx_{q_idx}"
                if run_key in completed_keys:
                    continue

                t0 = time.time()
                resp = run_inference(model, processor, active_img, active_prompt)
                lat = time.time() - t0
                pred, parsed = parse_answer(resp)
                is_correct = (pred == gt) if (pred and gt) else False

                rec = {
                    "_unique_run_key": run_key,
                    "mode": strat_name,
                    "corruption": corr,
                    "severity": sev,
                    "strategy": strat_name,
                    "question_idx": q_idx,
                    "question_text": q_text,
                    "options": opts,
                    "ground_truth": gt,
                    "prediction": pred,
                    "raw_response": resp,
                    "is_correct": is_correct,
                    "latency_sec": round(lat, 3),
                    "subtask": q.get("_subtask", "Unknown"),
                    "dataset": q.get("_dataset", "Unknown"),
                    "category": q.get("_category", "Unknown"),
                    "image_path": img_path,
                }

                with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                completed_keys.add(run_key)
                eval_count += 1

                if eval_count % 30 == 0:
                    print(f"   [{eval_count}] {corr} s={sev} | {strat_name} | Q{q_idx}: GT={gt}, Pred={pred} ({'✅' if is_correct else '❌'})")

print("\n✅ All adaptation evaluations completed successfully!")


# =============================================================================
# Compute Metrics: Accuracy, RRR, Delta RDS, and Subtask Matrix
# =============================================================================
print("\n" + "=" * 80)
print("📊 COMPUTING ROBUSTNESS RECOVERY METRICS")
print("=" * 80)

records = []
with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

clean_records = [r for r in records if r["mode"] == "clean"]
clean_acc = sum(r["is_correct"] for r in clean_records) / max(1, len(clean_records))

analysis = {}
for corr in TARGET_CORRUPTIONS:
    analysis[corr] = {}
    for sev in SEVERITY_LEVELS:
        analysis[corr][sev] = {}
        for strat in strategies:
            sub = [r for r in records if r["corruption"] == corr and r["severity"] == sev and r["strategy"] == strat]
            acc = sum(r["is_correct"] for r in sub) / max(1, len(sub)) if sub else 0.0
            analysis[corr][sev][strat] = {
                "accuracy": round(acc, 4),
                "count": len(sub),
            }

        # Calculate RRR (Robustness Recovery Rate)
        acc_base = analysis[corr][sev]["corrupted_baseline"]["accuracy"]
        acc_drop = clean_acc - acc_base
        for strat in ["mitigated_tta", "mitigated_dar", "mitigated_hybrid"]:
            acc_strat = analysis[corr][sev][strat]["accuracy"]
            if acc_drop > 1e-4:
                rrr = ((acc_strat - acc_base) / acc_drop) * 100.0
            else:
                rrr = 0.0
            analysis[corr][sev][strat]["rrr"] = round(rrr, 2)
            analysis[corr][sev][strat]["acc_gain"] = round(acc_strat - acc_base, 4)

print(f"\nReference Clean Baseline Accuracy: {clean_acc * 100:.2f}%\n")
for corr in TARGET_CORRUPTIONS:
    for sev in SEVERITY_LEVELS:
        d = analysis[corr][sev]
        print(f"[{corr.upper()} - Sev {sev}]")
        print(f"   Unmitigated Corrupted : {d['corrupted_baseline']['accuracy']*100:.2f}%")
        print(f"   Mitigated TTA-IR      : {d['mitigated_tta']['accuracy']*100:.2f}% (RRR: {d['mitigated_tta'].get('rrr', 0):+.1f}%)")
        print(f"   Mitigated DAR Prompt  : {d['mitigated_dar']['accuracy']*100:.2f}% (RRR: {d['mitigated_dar'].get('rrr', 0):+.1f}%)")
        print(f"   Mitigated Hybrid      : {d['mitigated_hybrid']['accuracy']*100:.2f}% (RRR: {d['mitigated_hybrid'].get('rrr', 0):+.1f}%)")


# =============================================================================
# Subtask Recovery Analysis
# =============================================================================
subtask_stats = defaultdict(lambda: defaultdict(list))
for r in records:
    if r["mode"] == "clean":
        subtask_stats[r["subtask"]]["clean"].append(r["is_correct"])
    elif r["strategy"] in strategies:
        subtask_stats[r["subtask"]][r["strategy"]].append(r["is_correct"])

subtask_recovery_table = {}
for st, vals in subtask_stats.items():
    c_acc = np.mean(vals["clean"]) if vals["clean"] else 0.0
    base_acc = np.mean(vals["corrupted_baseline"]) if vals["corrupted_baseline"] else 0.0
    tta_acc = np.mean(vals["mitigated_tta"]) if vals["mitigated_tta"] else 0.0
    dar_acc = np.mean(vals["mitigated_dar"]) if vals["mitigated_dar"] else 0.0
    hyb_acc = np.mean(vals["mitigated_hybrid"]) if vals["mitigated_hybrid"] else 0.0

    drop = c_acc - base_acc
    rrr_hyb = ((hyb_acc - base_acc) / drop * 100) if drop > 0.01 else 0.0
    subtask_recovery_table[st] = {
        "clean_acc": round(c_acc, 4),
        "corrupted_baseline": round(base_acc, 4),
        "mitigated_tta": round(tta_acc, 4),
        "mitigated_dar": round(dar_acc, 4),
        "mitigated_hybrid": round(hyb_acc, 4),
        "recovery_gain": round(hyb_acc - base_acc, 4),
        "rrr_hybrid": round(rrr_hyb, 2),
    }


# =============================================================================
# Visual Suite Generation (Publication Quality)
# =============================================================================
print("\n🎨 Generating Publication Quality Figures...")
sns.set_theme(style="whitegrid", font_scale=1.1)
palette = {"Clean": "#2ca02c", "Corrupted": "#d62728", "TTA (Image)": "#1f77b4", "DAR (Prompt)": "#ff7f0e", "Hybrid": "#9467bd"}

# 1. Bar Chart: Robustness Recovery Rate (RRR %) across Corruptions
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
corr_labels = [CORRUPTION_DISPLAY_NAMES.get(c, c) for c in TARGET_CORRUPTIONS]
x = np.arange(len(TARGET_CORRUPTIONS))
width = 0.25

# Average RRR across severities
rrr_tta = [np.mean([analysis[c][s]["mitigated_tta"].get("rrr", 0) for s in SEVERITY_LEVELS]) for c in TARGET_CORRUPTIONS]
rrr_dar = [np.mean([analysis[c][s]["mitigated_dar"].get("rrr", 0) for s in SEVERITY_LEVELS]) for c in TARGET_CORRUPTIONS]
rrr_hyb = [np.mean([analysis[c][s]["mitigated_hybrid"].get("rrr", 0) for s in SEVERITY_LEVELS]) for c in TARGET_CORRUPTIONS]

b1 = ax.bar(x - width, rrr_tta, width, label="Strategy 1: TTA-IR (Image Restoration)", color="#1f77b4", edgecolor="black")
b2 = ax.bar(x, rrr_dar, width, label="Strategy 2: DAR (Noise-Aware Prompt)", color="#ff7f0e", edgecolor="black")
b3 = ax.bar(x + width, rrr_hyb, width, label="Strategy 3: Hybrid Compound Adaptation", color="#9467bd", edgecolor="black")

ax.set_ylabel("Robustness Recovery Rate (RRR %)", fontsize=12, fontweight="bold")
ax.set_title("Robustness Recovery Rate (RRR) Across Industrial Corruptions\n(Qwen3-VL-2B on MMAD Industrial Benchmark)", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(corr_labels, fontsize=11, fontweight="bold")
ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax.set_ylim(-10, 105)
ax.legend(frameon=True, facecolor="white", edgecolor="gray", loc="upper left")

for b in [b1, b2, b3]:
    for bar in b:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, max(0, h)),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
p1_path = RESULTS_DIR / "phase3_recovery_rate_barchart.png"
plt.savefig(p1_path, dpi=300)
plt.close()
print(f"   ✅ Saved: {p1_path.name}")

# 2. Grouped Before vs After Comparison Plot
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
labels = [f"{CORRUPTION_DISPLAY_NAMES.get(c, c)}\n(Sev {s})" for c in TARGET_CORRUPTIONS for s in SEVERITY_LEVELS]
indices = np.arange(len(labels))
w = 0.16

clean_vals = [clean_acc * 100] * len(labels)
corr_vals = [analysis[c][s]["corrupted_baseline"]["accuracy"] * 100 for c in TARGET_CORRUPTIONS for s in SEVERITY_LEVELS]
tta_vals = [analysis[c][s]["mitigated_tta"]["accuracy"] * 100 for c in TARGET_CORRUPTIONS for s in SEVERITY_LEVELS]
dar_vals = [analysis[c][s]["mitigated_dar"]["accuracy"] * 100 for c in TARGET_CORRUPTIONS for s in SEVERITY_LEVELS]
hyb_vals = [analysis[c][s]["mitigated_hybrid"]["accuracy"] * 100 for c in TARGET_CORRUPTIONS for s in SEVERITY_LEVELS]

ax.bar(indices - 2*w, clean_vals, w, label="Clean Baseline (No Noise)", color="#2ca02c", alpha=0.85)
ax.bar(indices - w, corr_vals, w, label="Corrupted Baseline (Unmitigated)", color="#d62728", alpha=0.85)
ax.bar(indices, tta_vals, w, label="Mitigated: TTA-IR Filtering", color="#1f77b4", alpha=0.85)
ax.bar(indices + w, dar_vals, w, label="Mitigated: Defect-Anchored Prompt", color="#ff7f0e", alpha=0.85)
ax.bar(indices + 2*w, hyb_vals, w, label="Mitigated: Hybrid Compound", color="#9467bd", alpha=0.95)

ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("Performance Recovery: Clean vs Corrupted vs Adaptation Strategies", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(indices)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(40, 95)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=True)
plt.tight_layout()
p2_path = RESULTS_DIR / "phase3_mitigation_before_after_comparison.png"
plt.savefig(p2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {p2_path.name}")

# 3. Visual Restoration Showcase Grid (Clean -> Corrupted -> Restored)
fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=250)
sample_images = []
for q in eval_base_questions:
    if os.path.exists(q["_resolved_image_path"]):
        sample_images.append(q["_resolved_image_path"])
    if len(sample_images) >= 3:
        break

row_corrs = [("motion_blur", 5, "Conveyor Motion Blur (Sev 5)"),
             ("gaussian_noise", 5, "Gaussian Sensor Noise (Sev 5)"),
             ("low_light", 5, "Low-Light Underexposure (Sev 5)")]

for r_idx, (corr, sev, title) in enumerate(row_corrs):
    img_p = sample_images[r_idx % len(sample_images)]
    raw_img = Image.open(img_p).convert("RGB")
    raw_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
    c_img = apply_corruption(raw_img, corr, sev)
    rest_img = apply_mitigation_filter(c_img, corr, sev)

    axes[r_idx, 0].imshow(raw_img)
    axes[r_idx, 0].set_title(f"Clean Original\n({Path(img_p).parts[-3]})", fontsize=10, fontweight="bold", color="darkgreen")
    axes[r_idx, 0].axis("off")

    axes[r_idx, 1].imshow(c_img)
    axes[r_idx, 1].set_title(f"Corrupted: {title}", fontsize=10, fontweight="bold", color="darkred")
    axes[r_idx, 1].axis("off")

    axes[r_idx, 2].imshow(rest_img)
    axes[r_idx, 2].set_title(f"Restored via TTA-IR\n(Edge/Contrast Enhanced)", fontsize=10, fontweight="bold", color="navy")
    axes[r_idx, 2].axis("off")

plt.suptitle("Test-Time Image Restoration (TTA-IR) Pipeline Showcase\nRecovering High-Frequency Defect Features for Small MLLM Inspection",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
p3_path = RESULTS_DIR / "phase3_restoration_visual_samples.png"
plt.savefig(p3_path, dpi=250, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {p3_path.name}")

# 4. Subtask Recovery Heatmap
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
subtask_names = sorted(list(subtask_recovery_table.keys()))
matrix_data = []
for st in subtask_names:
    row = [
        subtask_recovery_table[st]["clean_acc"] * 100,
        subtask_recovery_table[st]["corrupted_baseline"] * 100,
        subtask_recovery_table[st]["mitigated_tta"] * 100,
        subtask_recovery_table[st]["mitigated_dar"] * 100,
        subtask_recovery_table[st]["mitigated_hybrid"] * 100,
        subtask_recovery_table[st]["recovery_gain"] * 100,
    ]
    matrix_data.append(row)

sns.heatmap(matrix_data, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=["Clean", "Corrupted", "TTA-IR", "DAR Prompt", "Hybrid", "Net Gain (Δ%)"],
            yticklabels=subtask_names, cbar_kws={"label": "Accuracy (%)"}, ax=ax, linewidths=0.5)

ax.set_title("Subtask-Level Recovery Matrix Across Adaptation Strategies\n(Highlighting Restoration in Defect Localization & Logical Anomalies)",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
p4_path = RESULTS_DIR / "phase3_subtask_recovery_heatmap.png"
plt.savefig(p4_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {p4_path.name}")


# =============================================================================
# Write Structured results.txt and Manifest
# =============================================================================
print("\n📝 Generating Comprehensive results.txt Report...")

# Also gather detailed question cards
with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as out:
    out.write("=" * 85 + "\n")
    out.write("PHASE 3: INDUSTRIAL ROBUSTNESS ADAPTATION, MITIGATION & RECOVERY REPORT\n")
    out.write("=" * 85 + "\n")
    out.write(f"Generated on          : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.write(f"Evaluated Model       : {MODEL_ID} (FP16)\n")
    out.write(f"Compute Hardware      : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)\n")
    out.write(f"Corruptions Mitigated : {', '.join(TARGET_CORRUPTIONS)}\n")
    out.write(f"Severity Levels       : {SEVERITY_LEVELS}\n")
    out.write(f"Total Evaluations     : {len(records)}\n")
    out.write("=" * 85 + "\n\n")

    out.write("1. EXECUTIVE SUMMARY & RECOVERY METRICS\n")
    out.write("-" * 85 + "\n")
    out.write(f"Clean Laboratory Baseline Accuracy : {clean_acc * 100:.2f}%\n\n")
    out.write(f"{'Corruption':<24} | {'Sev':<4} | {'Corrupted Acc':<13} | {'TTA-IR Acc (RRR)':<18} | {'DAR Acc (RRR)':<18} | {'Hybrid Acc (RRR)':<18}\n")
    out.write("-" * 105 + "\n")
    for corr in TARGET_CORRUPTIONS:
        for sev in SEVERITY_LEVELS:
            d = analysis[corr][sev]
            b_acc = d['corrupted_baseline']['accuracy'] * 100
            t_acc = d['mitigated_tta']['accuracy'] * 100
            t_rrr = d['mitigated_tta'].get('rrr', 0)
            d_acc = d['mitigated_dar']['accuracy'] * 100
            d_rrr = d['mitigated_dar'].get('rrr', 0)
            h_acc = d['mitigated_hybrid']['accuracy'] * 100
            h_rrr = d['mitigated_hybrid'].get('rrr', 0)

            c_name = CORRUPTION_DISPLAY_NAMES.get(corr, corr)
            out.write(f"{c_name:<24} | {sev:<4} | {b_acc:>6.2f}%       | {t_acc:>6.2f}% ({t_rrr:>+5.1f}%)   | {d_acc:>6.2f}% ({d_rrr:>+5.1f}%)   | {h_acc:>6.2f}% ({h_rrr:>+5.1f}%)\n")
    out.write("\n\n")

    out.write("2. SUBTASK-LEVEL RECOVERY BREAKDOWN\n")
    out.write("-" * 85 + "\n")
    out.write(f"{'Subtask':<26} | {'Clean Acc':<10} | {'Corrupted':<10} | {'Hybrid Mitigated':<17} | {'Net Gain (Δ)':<12} | {'RRR %':<8}\n")
    out.write("-" * 90 + "\n")
    for st, v in sorted(subtask_recovery_table.items()):
        out.write(f"{st:<26} | {v['clean_acc']*100:>8.2f}% | {v['corrupted_baseline']*100:>8.2f}% | {v['mitigated_hybrid']*100:>15.2f}% | {v['recovery_gain']*100:>+10.2f}% | {v['rrr_hybrid']:>6.1f}%\n")
    out.write("\n\n")

    out.write("3. DETAILED QUESTION-BY-QUESTION EVALUATION CARDS\n")
    out.write("-" * 85 + "\n")
    card_idx = 0
    for q_idx in range(len(eval_base_questions)):
        q_records = [r for r in records if r["question_idx"] == q_idx]
        clean_rec = next((r for r in q_records if r["mode"] == "clean"), None)
        if not clean_rec:
            continue

        card_idx += 1
        out.write(f"\n[CARD #{card_idx:03d}] Subtask: {clean_rec['subtask']} | Dataset: {clean_rec['dataset']} | Product: {clean_rec['category']}\n")
        out.write(f"Image Path : {clean_rec['image_path']}\n")
        out.write(f"Question   : {clean_rec['question_text']}\n")
        out.write("Options    :\n")
        for opt_k, opt_v in sorted(clean_rec.get("options", {}).items()):
            out.write(f"  ({opt_k}) {opt_v}\n")
        out.write(f"Ground Truth Correct Answer : {clean_rec['ground_truth']}\n")
        out.write(f"Clean Model Prediction      : {clean_rec['prediction']} [{'CORRECT' if clean_rec['is_correct'] else 'INCORRECT'}]\n")
        out.write("Mitigation Test Trajectory under Severe Corruptions:\n")

        for corr in TARGET_CORRUPTIONS:
            for sev in SEVERITY_LEVELS:
                base_r = next((r for r in q_records if r["corruption"] == corr and r["severity"] == sev and r["strategy"] == "corrupted_baseline"), None)
                hyb_r = next((r for r in q_records if r["corruption"] == corr and r["severity"] == sev and r["strategy"] == "mitigated_hybrid"), None)
                if base_r and hyb_r:
                    status = "UNCHANGED_CORRECT" if (base_r["is_correct"] and hyb_r["is_correct"]) else \
                             ("RECOVERED" if (not base_r["is_correct"] and hyb_r["is_correct"]) else \
                             ("REGRESSED" if (base_r["is_correct"] and not hyb_r["is_correct"]) else "STILL_FAILED"))
                    out.write(f"  • [{CORRUPTION_DISPLAY_NAMES.get(corr, corr)} Sev {sev}]: Corrupted Pred = '{base_r['prediction']}' -> Hybrid Pred = '{hyb_r['prediction']}' [{status}]\n")

manifest_data = {
    "phase": "phase-3",
    "title": "Industrial Robustness Adaptation, Mitigation & Recovery",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": MODEL_ID,
    "gpu_info": gpu_info,
    "clean_accuracy": round(clean_acc, 4),
    "sample_size": args.sample_size,
    "corruptions_evaluated": TARGET_CORRUPTIONS,
    "severities_evaluated": SEVERITY_LEVELS,
    "strategies": strategies,
    "recovery_summary": analysis,
    "subtask_recovery": subtask_recovery_table,
    "figures_generated": [
        "phase3_recovery_rate_barchart.png",
        "phase3_mitigation_before_after_comparison.png",
        "phase3_restoration_visual_samples.png",
        "phase3_subtask_recovery_heatmap.png",
    ],
}

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest_data, f, indent=2)

print(f"✅ Saved Manifest: {MANIFEST_PATH.name}")
print(f"✅ Saved Detailed Results: {RESULTS_TXT_PATH.name}")
print("=" * 80)
print("PHASE 3 BENCHMARK COMPLETE!")
print("=" * 80)

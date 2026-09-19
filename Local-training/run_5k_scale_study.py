#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
High-Scale 5,000-Sample Industrial Benchmark (Phase 2 & Phase 3)
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Evaluates 5,000 stratified unique MMAD images across all 4 datasets & 9 subtasks:
1. Clean Baseline (5,000 inferences)
2. Conveyor Motion Blur (5,000 inferences) - Phase 2 Top Failure Mode
3. Gaussian Sensor Noise (5,000 inferences) - Phase 2 Primary Sensor Noise
4. TTA Mitigated Motion Blur (5,000 inferences) - Phase 3 Adaptation
5. TTA Mitigated Gaussian Noise (5,000 inferences) - Phase 3 Adaptation

Total Inferences: 25,000 (with incremental checkpointing & resume).
Hardware: Local NVIDIA RTX 4060 Laptop GPU (FP16 mode, ~8.8 inf/sec).
==============================================================================
"""

import os
import sys
import json
import time
import random
import re
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
from transformers import AutoProcessor

# Path setups
CURRENT_DIR = Path(__file__).resolve().parent
PHASE2_DIR = CURRENT_DIR / "phase-2"
PHASE3_DIR = CURRENT_DIR / "phase-3"
sys.path.insert(0, str(PHASE2_DIR))
sys.path.insert(0, str(PHASE3_DIR))

from corruptions import apply_motion_blur, apply_gaussian_noise, CORRUPTION_DISPLAY_NAMES
from mitigations import restore_motion_blur, restore_gaussian_noise, build_standard_prompt

# Arguments
parser = argparse.ArgumentParser(description="5,000-Sample Benchmark for Phase 2 & 3")
parser.add_argument("--total-samples", type=int, default=5000, help="Total unique questions to evaluate (default 5000).")
parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
parser.add_argument("--severity", type=int, default=4, help="Industrial corruption severity level (1-5, default 4 severe).")
parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start from scratch.")
args = parser.parse_args()

GLOBAL_SEED = args.seed
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

BASE_DIR = CURRENT_DIR.parent
MMAD_DIR = BASE_DIR / "MMAD"

# Result paths
P2_RESULTS_DIR = PHASE2_DIR / "results"
P3_RESULTS_DIR = PHASE3_DIR / "results"
P2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
P3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCALE_RESULTS_JSONL = CURRENT_DIR / "benchmark_5k_results.jsonl"
P2_RESULTS_TXT = P2_RESULTS_DIR / "results_5k.txt"
P3_RESULTS_TXT = P3_RESULTS_DIR / "results_5k.txt"

def get_gpu_info():
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        }
    return {"gpu_name": "CPU", "gpu_memory_gb": 0, "cuda_version": None}

gpu_info = get_gpu_info()

print("=" * 85)
print("HIGH-SCALE 5,000-SAMPLE INDUSTRIAL ROBUSTNESS & MITIGATION BENCHMARK")
print("=" * 85)
print(f"  Target GPU            : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)")
print(f"  PyTorch / CUDA        : {torch.__version__} / CUDA {gpu_info['cuda_version']}")
print(f"  Sample Scale          : {args.total_samples} unique MMAD images across all subtasks")
print(f"  Corruption Severity   : Level {args.severity} (Severe Industrial Condition)")
print(f"  Conditions Evaluated  : 1. Clean, 2. Motion Blur, 3. Gaussian Noise, 4. TTA-Blur, 5. TTA-Noise")
print(f"  Total Inferences Plan : {args.total_samples * 5:,} evaluations")
print("=" * 85)


# =============================================================================
# Dataset Resolution & Stratified Sampling of 5,000 Questions
# =============================================================================
mmad_json_path = MMAD_DIR / "mmad.json"
print(f"\n📂 Loading MMAD benchmark: {mmad_json_path.name}...")

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
print(f"   ✅ Valid questions with local images: {len(valid_questions):,}")

# Stratified Sampling across Subtasks
subtask_bins = defaultdict(list)
for q in valid_questions:
    st = q.get("_subtask", "Unknown")
    subtask_bins[st].append(q)

target_n = min(args.total_samples, len(valid_questions))
per_subtask = target_n // len(subtask_bins)

sample_5k_questions = []
random.seed(GLOBAL_SEED)
for st, q_list in sorted(subtask_bins.items()):
    take = min(len(q_list), per_subtask)
    sample_5k_questions.extend(random.sample(q_list, take))

if len(sample_5k_questions) < target_n:
    rem = [q for q in valid_questions if q not in sample_5k_questions]
    sample_5k_questions.extend(random.sample(rem, target_n - len(sample_5k_questions)))

print(f"   ✅ Selected {len(sample_5k_questions):,} stratified questions across {len(subtask_bins)} subtasks.")


# =============================================================================
# Load Model & Processor
# =============================================================================
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
print(f"\n🤖 Loading model: {MODEL_ID} in FP16 on GPU...")

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = None
try:
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    print("   ✅ Loaded via Qwen3VLForConditionalGeneration")
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    print("   ✅ Loaded via AutoModelForImageTextToText")

model.eval()
print(f"   ✅ Ready in {time.time() - t0:.2f}s")


# =============================================================================
# Helper Functions
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
        output_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# =============================================================================
# Benchmarking Loop: 5 Conditions on 5,000 Questions
# =============================================================================
if args.reset and SCALE_RESULTS_JSONL.exists():
    print("🧹 --reset specified: Deleting previous 5K checkpoint...")
    SCALE_RESULTS_JSONL.unlink()

completed_keys = set()
if SCALE_RESULTS_JSONL.exists():
    with open(SCALE_RESULTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    completed_keys.add(obj.get("_run_key"))
                except Exception:
                    pass
    print(f"🔄 Resuming from checkpoint: Found {len(completed_keys):,} completed evaluation records.")

print("\n" + "=" * 85)
print("🚀 COMMENCING 5,000-SAMPLE BENCHMARK EXECUTION")
print("=" * 85)

conditions = [
    ("clean", "none", False),
    ("motion_blur", "none", False),
    ("gaussian_noise", "none", False),
    ("motion_blur", "tta_restore", True),
    ("gaussian_noise", "tta_restore", True),
]

t_bench_start = time.time()
processed_count = 0

for q_idx, q in enumerate(sample_5k_questions):
    img_path = q["_resolved_image_path"]
    gt = get_ground_truth(q)
    std_prompt, q_text, opts = build_standard_prompt(q)

    # Load and downsample image once per question
    try:
        raw_img = Image.open(img_path).convert("RGB")
        raw_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
    except Exception:
        raw_img = Image.new("RGB", (224, 224), (128, 128, 128))

    # Pre-generate corrupted images
    c_mblur = apply_motion_blur(raw_img, severity=args.severity)
    c_gnoise = apply_gaussian_noise(raw_img, severity=args.severity)

    # Pre-generate restored images for TTA
    r_mblur = restore_motion_blur(c_mblur, strength=1.6)
    r_gnoise = restore_gaussian_noise(c_gnoise, d=7, sigma_color=75, sigma_space=75)

    eval_tuples = [
        ("clean", raw_img, "clean", 0, "none"),
        ("corr_motion_blur", c_mblur, "motion_blur", args.severity, "none"),
        ("corr_gaussian_noise", c_gnoise, "gaussian_noise", args.severity, "none"),
        ("tta_motion_blur", r_mblur, "motion_blur", args.severity, "tta_restore"),
        ("tta_gaussian_noise", r_gnoise, "gaussian_noise", args.severity, "tta_restore"),
    ]

    for cond_name, active_img, corr_type, sev, strat in eval_tuples:
        run_key = f"5k_q{q_idx}_{cond_name}"
        if run_key in completed_keys:
            continue

        t_inf0 = time.time()
        resp = run_inference(model, processor, active_img, std_prompt)
        lat = time.time() - t_inf0
        pred, parsed = parse_answer(resp)
        is_corr = (pred == gt) if (pred and gt) else False

        record = {
            "_run_key": run_key,
            "condition": cond_name,
            "corruption": corr_type,
            "severity": sev,
            "strategy": strat,
            "question_idx": q_idx,
            "question_text": q_text,
            "options": opts,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_corr,
            "latency_sec": round(lat, 3),
            "subtask": q.get("_subtask", "Unknown"),
            "dataset": q.get("_dataset", "Unknown"),
            "category": q.get("_category", "Unknown"),
            "image_path": img_path,
        }

        with open(SCALE_RESULTS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        completed_keys.add(run_key)
        processed_count += 1

    if (q_idx + 1) % 100 == 0:
        elapsed = time.time() - t_bench_start
        rate = processed_count / max(0.1, elapsed)
        rem_q = len(sample_5k_questions) - (q_idx + 1)
        eta_sec = (rem_q * 5) / max(0.1, rate)
        print(f"   [{q_idx+1:04d}/{len(sample_5k_questions):,}] Inferences: {len(completed_keys):,} | Speed: {rate:.1f} inf/s | ETA: {eta_sec/60:.1f} mins")

print(f"\n✅ Completed all {len(completed_keys):,} evaluations in {time.time() - t_bench_start:.1f}s!")


# =============================================================================
# Compute High-Confidence Statistics across Subtasks and Datasets
# =============================================================================
print("\n" + "=" * 85)
print("📊 COMPUTING 5,000-SAMPLE HIGH-CONFIDENCE STATISTICS")
print("=" * 85)

all_records = []
with open(SCALE_RESULTS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

# Accuracy by Condition
condition_acc = {}
for cond in ["clean", "corr_motion_blur", "corr_gaussian_noise", "tta_motion_blur", "tta_gaussian_noise"]:
    sub = [r for r in all_records if r["condition"] == cond]
    acc = sum(r["is_correct"] for r in sub) / max(1, len(sub))
    condition_acc[cond] = {
        "accuracy": round(acc, 4),
        "count": len(sub),
    }

clean_acc = condition_acc["clean"]["accuracy"]
mb_corr_acc = condition_acc["corr_motion_blur"]["accuracy"]
gn_corr_acc = condition_acc["corr_gaussian_noise"]["accuracy"]
mb_tta_acc = condition_acc["tta_motion_blur"]["accuracy"]
gn_tta_acc = condition_acc["tta_gaussian_noise"]["accuracy"]

# Robustness Recovery Rates
rrr_mb = ((mb_tta_acc - mb_corr_acc) / max(1e-4, clean_acc - mb_corr_acc)) * 100.0
rrr_gn = ((gn_tta_acc - gn_corr_acc) / max(1e-4, clean_acc - gn_corr_acc)) * 100.0

print(f"\n🎯 5,000-SAMPLE BENCHMARK RESULTS (Severity {args.severity}):")
print(f"  • Clean Laboratory Accuracy             : {clean_acc*100:.2f}% (N={condition_acc['clean']['count']:,})")
print(f"  • Conveyor Motion Blur (Corrupted)      : {mb_corr_acc*100:.2f}% (Loss: -{(clean_acc-mb_corr_acc)*100:.2f}%)")
print(f"  • TTA-Restored Motion Blur (Mitigated)  : {mb_tta_acc*100:.2f}% (RRR: {rrr_mb:+.2f}%)")
print(f"  • Gaussian Sensor Noise (Corrupted)     : {gn_corr_acc*100:.2f}% (Loss: -{(clean_acc-gn_corr_acc)*100:.2f}%)")
print(f"  • TTA-Restored Gaussian Noise (Mitigated): {gn_tta_acc*100:.2f}% (RRR: {rrr_gn:+.2f}%)")

# Subtask Breakdown
subtask_stats = defaultdict(lambda: defaultdict(list))
for r in all_records:
    subtask_stats[r["subtask"]][r["condition"]].append(r["is_correct"])

subtask_summary = {}
for st, cond_dict in sorted(subtask_stats.items()):
    c_acc = np.mean(cond_dict["clean"]) if cond_dict["clean"] else 0.0
    mb_acc = np.mean(cond_dict["corr_motion_blur"]) if cond_dict["corr_motion_blur"] else 0.0
    gn_acc = np.mean(cond_dict["corr_gaussian_noise"]) if cond_dict["corr_gaussian_noise"] else 0.0
    mb_rec = np.mean(cond_dict["tta_motion_blur"]) if cond_dict["tta_motion_blur"] else 0.0
    gn_rec = np.mean(cond_dict["tta_gaussian_noise"]) if cond_dict["tta_gaussian_noise"] else 0.0

    subtask_summary[st] = {
        "sample_count": len(cond_dict["clean"]),
        "clean_acc": round(c_acc, 4),
        "mb_corr_acc": round(mb_acc, 4),
        "gn_corr_acc": round(gn_acc, 4),
        "mb_tta_acc": round(mb_rec, 4),
        "gn_tta_acc": round(gn_rec, 4),
        "mb_gain": round(mb_rec - mb_acc, 4),
        "gn_gain": round(gn_rec - gn_acc, 4),
    }


# =============================================================================
# Generate 5,000-Scale Publication Figures
# =============================================================================
print("\n🎨 Generating High-Resolution 5,000-Scale Publication Plots...")
sns.set_theme(style="whitegrid", font_scale=1.1)

# Plot 1: 5K High-Confidence Robustness & Recovery Bar Chart
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
labels = ["Clean Baseline", "Motion Blur (Corrupted)", "TTA Restored Motion Blur",
          "Gaussian Noise (Corrupted)", "TTA Restored Gaussian Noise"]
accuracies = [clean_acc * 100, mb_corr_acc * 100, mb_tta_acc * 100, gn_corr_acc * 100, gn_tta_acc * 100]
colors = ["#2ca02c", "#d62728", "#1f77b4", "#e377c2", "#9467bd"]

bars = ax.bar(labels, accuracies, color=colors, width=0.55, edgecolor="black", alpha=0.9)
ax.set_ylabel("Accuracy (%) [N=5,000 Samples]", fontsize=12, fontweight="bold")
ax.set_title(f"High-Scale 5,000-Sample Robustness & Recovery Benchmark\n(Qwen3-VL-2B-Instruct on MMAD Benchmark, Severity {args.severity})",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylim(40, 85)
plt.xticks(rotation=20, ha="right", fontweight="bold", fontsize=10)

for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
p1_scale = CURRENT_DIR / "results_5k_robustness_and_recovery.png"
plt.savefig(p1_scale, dpi=300, bbox_inches="tight")
plt.savefig(P2_RESULTS_DIR / "phase2_5k_robustness_overview.png", dpi=300, bbox_inches="tight")
plt.savefig(P3_RESULTS_DIR / "phase3_5k_recovery_overview.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: 5K Subtask Resilience & Recovery Heatmap (9 Subtasks)
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
subtask_names = list(subtask_summary.keys())
data_matrix = []
for st in subtask_names:
    row = [
        subtask_summary[st]["clean_acc"] * 100,
        subtask_summary[st]["mb_corr_acc"] * 100,
        subtask_summary[st]["mb_tta_acc"] * 100,
        subtask_summary[st]["gn_corr_acc"] * 100,
        subtask_summary[st]["gn_tta_acc"] * 100,
    ]
    data_matrix.append(row)

sns.heatmap(data_matrix, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["Clean", "Motion Blur", "TTA Motion Blur", "Gaussian Noise", "TTA Gaussian Noise"],
            yticklabels=subtask_names, cbar_kws={"label": "Accuracy (%)"}, ax=ax, linewidths=0.5)

ax.set_title("5,000-Sample Subtask Robustness & Recovery Matrix (MMAD)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
p2_scale = CURRENT_DIR / "results_5k_subtask_matrix.png"
plt.savefig(p2_scale, dpi=300, bbox_inches="tight")
plt.savefig(P2_RESULTS_DIR / "phase2_5k_subtask_matrix.png", dpi=300, bbox_inches="tight")
plt.savefig(P3_RESULTS_DIR / "phase3_5k_subtask_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("   ✅ Saved: results_5k_robustness_and_recovery.png")
print("   ✅ Saved: results_5k_subtask_matrix.png")


# =============================================================================
# Write Detailed Results Reports
# =============================================================================
def write_5k_report(out_path, title_phase):
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("=" * 90 + "\n")
        out.write(f"{title_phase}: 5,000-SAMPLE HIGH-SCALE ROBUSTNESS REPORT\n")
        out.write("=" * 90 + "\n")
        out.write(f"Generated on          : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Evaluated Model       : {MODEL_ID} (FP16)\n")
        out.write(f"Hardware              : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)\n")
        out.write(f"Total Unique Images   : {len(sample_5k_questions):,}\n")
        out.write(f"Total Evaluated Items : {len(all_records):,}\n")
        out.write(f"Severity Tested       : Level {args.severity} (Severe Factory Industrial Condition)\n")
        out.write("=" * 90 + "\n\n")

        out.write("1. EXECUTIVE ACCURACY SUMMARY (N=5,000)\n")
        out.write("-" * 90 + "\n")
        out.write(f"Clean Laboratory Baseline Accuracy : {clean_acc*100:.2f}%\n")
        out.write(f"Conveyor Motion Blur (Corrupted)   : {mb_corr_acc*100:.2f}% (Loss: -{(clean_acc-mb_corr_acc)*100:.2f}%)\n")
        out.write(f"TTA Mitigated Motion Blur          : {mb_tta_acc*100:.2f}% (RRR: {rrr_mb:+.2f}%)\n")
        out.write(f"Gaussian Sensor Noise (Corrupted)  : {gn_corr_acc*100:.2f}% (Loss: -{(clean_acc-gn_corr_acc)*100:.2f}%)\n")
        out.write(f"TTA Mitigated Gaussian Noise       : {gn_tta_acc*100:.2f}% (RRR: {rrr_gn:+.2f}%)\n\n")

        out.write("2. SUBTASK-LEVEL ACCURACY TABLE (N=5,000)\n")
        out.write("-" * 90 + "\n")
        out.write(f"{'Subtask':<26} | {'Samples':<8} | {'Clean Acc':<10} | {'MBlur Acc':<10} | {'MBlur TTA':<10} | {'GNoise Acc':<11} | {'GNoise TTA':<11}\n")
        out.write("-" * 95 + "\n")
        for st, v in sorted(subtask_summary.items()):
            out.write(f"{st:<26} | {v['sample_count']:<8} | {v['clean_acc']*100:>8.2f}% | {v['mb_corr_acc']*100:>8.2f}% | {v['mb_tta_acc']*100:>8.2f}% | {v['gn_corr_acc']*100:>9.2f}% | {v['gn_tta_acc']*100:>9.2f}%\n")
        out.write("\n\n")

        out.write("3. SAMPLE DETAILED QUESTION EVALUATION CARDS (First 200 Samples)\n")
        out.write("-" * 90 + "\n")
        for q_i in range(min(200, len(sample_5k_questions))):
            recs = [r for r in all_records if r["question_idx"] == q_i]
            c_r = next((r for r in recs if r["condition"] == "clean"), None)
            mb_r = next((r for r in recs if r["condition"] == "corr_motion_blur"), None)
            mb_t = next((r for r in recs if r["condition"] == "tta_motion_blur"), None)
            gn_r = next((r for r in recs if r["condition"] == "corr_gaussian_noise"), None)
            gn_t = next((r for r in recs if r["condition"] == "tta_gaussian_noise"), None)

            if not c_r:
                continue

            out.write(f"\n[CARD #{q_i+1:04d}] Subtask: {c_r['subtask']} | Dataset: {c_r['dataset']} | Category: {c_r['category']}\n")
            out.write(f"Image Path : {c_r['image_path']}\n")
            out.write(f"Question   : {c_r['question_text']}\n")
            out.write("Options    :\n")
            for ok, ov in sorted(c_r.get("options", {}).items()):
                out.write(f"  ({ok}) {ov}\n")
            out.write(f"Ground Truth Correct Answer : {c_r['ground_truth']}\n")
            out.write(f"Clean Model Prediction      : {c_r['prediction']} [{'CORRECT' if c_r['is_correct'] else 'INCORRECT'}]\n")
            if mb_r and mb_t:
                out.write(f"Motion Blur Test (Sev {args.severity})     : Corrupted='{mb_r['prediction']}' -> TTA Restored='{mb_t['prediction']}'\n")
            if gn_r and gn_t:
                out.write(f"Gaussian Noise Test (Sev {args.severity})  : Corrupted='{gn_r['prediction']}' -> TTA Restored='{gn_t['prediction']}'\n")

write_5k_report(P2_RESULTS_TXT, "PHASE 2 & PHASE 3")
write_5k_report(P3_RESULTS_TXT, "PHASE 3 & PHASE 2")

print(f"✅ Saved Phase 2 report: {P2_RESULTS_TXT.name}")
print(f"✅ Saved Phase 3 report: {P3_RESULTS_TXT.name}")
print("=" * 85)
print("HIGH-SCALE 5,000 BENCHMARK READY FOR PUBLICATION!")
print("=" * 85)

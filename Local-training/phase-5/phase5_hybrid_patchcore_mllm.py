#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Phase 5: Hybrid Vision-Expert & MLLM Visual Prompting Benchmark
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Three-Stage Architecture:
  Stage 1 (Vision Expert): PatchCore extracts mid-level patch memory bank features
                           and produces continuous pixel-level anomaly heatmaps.
  Stage 2 (Visual Prompting): Translates high-scoring anomaly clusters into
                             prominent red bounding box overlays & attention tags.
  Stage 3 (MLLM Reasoning):  Feeds the visually prompted image into the MLLM with
                             grounded spatial inspection prompts.
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
from sklearn.metrics import cohen_kappa_score

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))

from patchcore_expert import PatchCoreExpert
from visual_prompting import apply_visual_prompt_overlay, format_hybrid_prompt

parser = argparse.ArgumentParser(description="Phase 5: Hybrid PatchCore + MLLM Benchmark")
parser.add_argument("--sample-size", type=int, default=500, help="Number of questions to evaluate (default 500)")
parser.add_argument("--model-id", type=str, default="Qwen3-VL-2B-Instruct", help="MLLM Model identifier")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--reset", action="store_true", help="Reset previous progress")
args = parser.parse_args()

GLOBAL_SEED = args.seed
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

BASE_DIR = PARENT_DIR.parent
MMAD_DIR = BASE_DIR / "MMAD"
RESULTS_DIR = CURRENT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HYBRID_RESULTS_JSONL = RESULTS_DIR / "phase5_hybrid_results.jsonl"

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
print("PHASE 5: HYBRID VISION-EXPERT (PATCHCORE) + MLLM VISUAL PROMPTING BENCHMARK")
print("=" * 85)
print(f"  Target GPU            : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)")
print(f"  Target MLLM           : {args.model_id}")
print(f"  Sample Target         : {args.sample_size:,} Questions")
print(f"  Vision Expert         : ResNet50 PatchCore Coreset Memory Bank")
print(f"  Visual Prompt Strategy: Red Bounding Box Overlay + Attention Directive")
print("=" * 85)

# -----------------------------------------------------------------------------
# 1. Dataset Resolution & Sampling
# -----------------------------------------------------------------------------
mmad_json_path = MMAD_DIR / "mmad.json"
if not mmad_json_path.exists():
    raise FileNotFoundError(f"Missing {mmad_json_path}")

with open(mmad_json_path, "r", encoding="utf-8") as f:
    mmad_data = json.load(f)

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
for key, val in mmad_data.items():
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

# Stratified sampling across subtasks
subtask_bins = defaultdict(list)
for q in valid_questions:
    st = q.get("_subtask", "Unknown")
    subtask_bins[st].append(q)

target_n = min(args.sample_size, len(valid_questions))
per_subtask = max(1, target_n // max(1, len(subtask_bins)))

sample_questions = []
random.seed(GLOBAL_SEED)
for st, q_list in sorted(subtask_bins.items()):
    take = min(len(q_list), per_subtask)
    sample_questions.extend(random.sample(q_list, take))

if len(sample_questions) < target_n:
    rem = [q for q in valid_questions if q not in sample_questions]
    sample_questions.extend(random.sample(rem, target_n - len(sample_questions)))

print(f"✅ Prepared {len(sample_questions):,} stratified evaluation questions across {len(subtask_bins)} subtasks.")

# -----------------------------------------------------------------------------
# 2. Category Normal Images Cache
# -----------------------------------------------------------------------------
category_goods_cache = {}
def get_category_goods(dataset_name, category_name):
    cache_key = f"{dataset_name}/{category_name}"
    if cache_key in category_goods_cache:
        return category_goods_cache[cache_key]

    goods = []
    # Search common normal directory patterns
    candidates = [
        MMAD_DIR / dataset_name / category_name / "train" / "good",
        MMAD_DIR / dataset_name / category_name / "good",
        MMAD_DIR / dataset_name / category_name / "train",
    ]
    for d in candidates:
        if d.exists():
            goods.extend(list(d.glob("*.jpg")) + list(d.glob("*.png")))
            if goods:
                break

    category_goods_cache[cache_key] = goods
    return goods

# -----------------------------------------------------------------------------
# 3. Model Loading
# -----------------------------------------------------------------------------
print("\n🤖 Initializing Stage 1: PatchCore Vision Expert (ResNet50)...")
patchcore = PatchCoreExpert()

print(f"\n🤖 Initializing Stage 3: Multimodal LLM ({args.model_id})...")
if "qwen3" in args.model_id.lower() or "qwen" in args.model_id.lower():
    from transformers import Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28
    )
    mllm = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
else:
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    from transformers import AutoModelForImageTextToText
    mllm = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )

def query_mllm(pil_img, prompt_text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(mllm.device)
    with torch.inference_mode():
        output_ids = mllm.generate(**inputs, max_new_tokens=4, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return response

def extract_option_letter(text):
    if not text:
        return "A"
    text = text.strip()
    m = re.match(r"^([A-D])\b", text.upper())
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([A-D])\)|([A-D])[).\s:]", text.upper())
    if m:
        return (m.group(1) or m.group(2)).upper()
    letters = re.findall(r"\b([A-D])\b", text.upper())
    if len(letters) == 1:
        return letters[0].upper()
    for c in text.upper()[:5]:
        if c in "ABCD":
            return c
    return "A"

# -----------------------------------------------------------------------------
# 4. Evaluation Loop: Vanilla vs PatchCore-Guided
# -----------------------------------------------------------------------------
completed_keys = set()
if args.reset and HYBRID_RESULTS_JSONL.exists():
    HYBRID_RESULTS_JSONL.unlink()

if HYBRID_RESULTS_JSONL.exists():
    with open(HYBRID_RESULTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    completed_keys.add(json.loads(line).get("question_idx"))
                except Exception:
                    pass
    print(f"🔄 Resuming Phase 5: Found {len(completed_keys):,} previously evaluated items.")

print("\n" + "=" * 85)
print("🚀 RUNNING HEAD-TO-HEAD: VANILLA MLLM VS. PATCHCORE-GUIDED MLLM")
print("=" * 85)

t0 = time.time()
for idx, q in enumerate(sample_questions):
    if idx in completed_keys:
        continue

    img_path = q["_resolved_image_path"]
    cat = q["_category"]
    ds = q["_dataset"]
    st = q.get("_subtask", "Unknown")

    ci_q = {str(k).lower(): v for k, v in q.items()}
    raw_q_text = ""
    for f in ["question", "query", "text", "prompt"]:
        if f in ci_q and ci_q[f]:
            raw_q_text = str(ci_q[f]).strip()
            break

    options = {}
    for f in ["options", "choices", "answers"]:
        if f in ci_q and ci_q[f]:
            if isinstance(ci_q[f], dict):
                options = ci_q[f]
            elif isinstance(ci_q[f], list):
                options = {chr(65 + i): v for i, v in enumerate(ci_q[f])}
            break

    gt = ""
    for f in ["answer", "ground_truth", "label", "gt"]:
        if f in ci_q and ci_q[f] is not None:
            g = str(ci_q[f]).strip().upper()
            if len(g) == 1 and g in "ABCD":
                gt = g
            elif g.isdigit() and int(g) < 4:
                gt = chr(65 + int(g))
            else:
                gt = g
            break
    if not gt:
        gt = "A"

    try:
        orig_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"   ⚠️ Skipping {img_path}: {e}")
        continue

    # A. Vanilla MLLM Inference (Baseline)
    vanilla_prompt = format_hybrid_prompt(raw_q_text, options, has_visual_cue=False)
    t_v0 = time.time()
    vanilla_raw = query_mllm(orig_img, vanilla_prompt)
    vanilla_lat = time.time() - t_v0
    vanilla_pred = extract_option_letter(vanilla_raw)
    vanilla_correct = (vanilla_pred == gt)

    # B. Stage 1: PatchCore Anomaly Localization
    normal_images = get_category_goods(ds, cat)
    norm_map, anomaly_score, bbox = patchcore.detect_anomaly(
        orig_img, normal_images, category_key=f"{ds}/{cat}"
    )

    # C. Stage 2: Visual Prompting (Overlay + Attention Directive)
    has_cue = (bbox is not None and anomaly_score >= 1.0)
    if has_cue:
        prompted_img = apply_visual_prompt_overlay(orig_img, bbox, anomaly_score=anomaly_score)
        guided_prompt = format_hybrid_prompt(raw_q_text, options, has_visual_cue=True)
    else:
        prompted_img = orig_img
        guided_prompt = vanilla_prompt

    # D. Stage 3: Guided MLLM Inference
    t_g0 = time.time()
    guided_raw = query_mllm(prompted_img, guided_prompt)
    guided_lat = time.time() - t_g0
    guided_pred = extract_option_letter(guided_raw)
    guided_correct = (guided_pred == gt)

    rec = {
        "question_idx": idx,
        "dataset": ds,
        "category": cat,
        "subtask": st,
        "ground_truth": gt,
        "patchcore_score": round(anomaly_score, 3),
        "patchcore_bbox": list(bbox) if bbox else None,
        "has_visual_cue": has_cue,
        "vanilla_pred": vanilla_pred,
        "vanilla_correct": vanilla_correct,
        "vanilla_latency": round(vanilla_lat, 3),
        "guided_pred": guided_pred,
        "guided_correct": guided_correct,
        "guided_latency": round(guided_lat, 3),
    }

    with open(HYBRID_RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    completed_keys.add(idx)

    if len(completed_keys) % 25 == 0 or len(completed_keys) == len(sample_questions):
        el = time.time() - t0
        rate = len(completed_keys) / max(0.1, el)
        rem = len(sample_questions) - len(completed_keys)
        print(f"   [{len(completed_keys):04d}/{len(sample_questions):,}] Speed: {rate:.1f} inf/s | ETA: {(rem/max(0.1, rate))/60:.1f}m | "
              f"Vanilla: {vanilla_correct} | Guided: {guided_correct} (GT={gt})")

print(f"\n✅ Finished evaluating Phase 5 on {len(completed_keys):,} samples!")

# -----------------------------------------------------------------------------
# 5. Result Analysis & Figure Generation
# -----------------------------------------------------------------------------
records = []
with open(HYBRID_RESULTS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

v_acc = sum(r["vanilla_correct"] for r in records) / len(records)
g_acc = sum(r["guided_correct"] for r in records) / len(records)
delta_overall = (g_acc - v_acc) * 100

st_vanilla = defaultdict(list)
st_guided = defaultdict(list)
for r in records:
    st_vanilla[r["subtask"]].append(r["vanilla_correct"])
    st_guided[r["subtask"]].append(r["guided_correct"])

subtasks_sorted = sorted(st_vanilla.keys())
v_sub_acc = [np.mean(st_vanilla[st]) * 100 for st in subtasks_sorted]
g_sub_acc = [np.mean(st_guided[st]) * 100 for st in subtasks_sorted]

print("\n" + "=" * 85)
print("📊 PHASE 5 BENCHMARK RESULTS SUMMARY")
print("=" * 85)
print(f"  Vanilla MLLM Accuracy        : {v_acc*100:.2f}%")
print(f"  PatchCore-Guided MLLM Accuracy: {g_acc*100:.2f}%")
print(f"  Net Gain from Visual Prompting : {delta_overall:+.2f} percentage points")
print("-" * 85)
print(f"{'Subtask Category':<28} | {'Vanilla':<10} | {'PatchCore-Guided':<18} | {'Delta (Gain)':<12}")
print("-" * 85)
for st, va, ga in zip(subtasks_sorted, v_sub_acc, g_sub_acc):
    print(f"{st:<28} | {va:>8.2f}% | {ga:>16.2f}% | {ga-va:>+10.2f}%")
print("=" * 85)

# Plot: Subtask Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
x = np.arange(len(subtasks_sorted))
width = 0.35

rects1 = ax.bar(x - width/2, v_sub_acc, width, label="Vanilla MLLM (Raw Image)", color="#4A90E2", edgecolor="black", lw=0.8)
rects2 = ax.bar(x + width/2, g_sub_acc, width, label="Stage 1-3 Hybrid (PatchCore Visual Prompting)", color="#E94E77", edgecolor="black", lw=0.8)

ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("Phase 5: Impact of PatchCore Visual Prompting on Industrial Subtasks", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(subtasks_sorted, rotation=30, ha="right", fontsize=10)
ax.legend(frameon=True, fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_ylim(0, 105)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
p1_path = RESULTS_DIR / "phase5_subtask_delta.png"
plt.savefig(p1_path, dpi=300, bbox_inches="tight")
plt.close()

# Save Manifest & Text Report
manifest = {
    "sample_count": len(records),
    "model_id": args.model_id,
    "vanilla_accuracy": round(float(v_acc), 4),
    "guided_accuracy": round(float(g_acc), 4),
    "accuracy_gain_pct": round(float(delta_overall), 2),
    "subtask_accuracy": {
        st: {
            "vanilla": round(float(va), 2),
            "guided": round(float(ga), 2),
            "delta": round(float(ga - va), 2),
        }
        for st, va, ga in zip(subtasks_sorted, v_sub_acc, g_sub_acc)
    }
}

MANIFEST_PATH = RESULTS_DIR / "phase5_manifest.json"
with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

REPORT_PATH = RESULTS_DIR / "results.txt"
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("=" * 85 + "\n")
    f.write("PHASE 5: HYBRID PATCHCORE VISION-EXPERT + MLLM BENCHMARK REPORT\n")
    f.write("=" * 85 + "\n")
    f.write(f"Sample Size               : {len(records):,} questions\n")
    f.write(f"Vanilla MLLM Accuracy     : {v_acc*100:.2f}%\n")
    f.write(f"PatchCore-Guided Accuracy : {g_acc*100:.2f}%\n")
    f.write(f"Net Accuracy Gain         : {delta_overall:+.2f}%\n\n")
    f.write(f"{'Subtask Category':<28} | {'Vanilla':<10} | {'Guided':<10} | {'Delta':<10}\n")
    f.write("-" * 65 + "\n")
    for st, va, ga in zip(subtasks_sorted, v_sub_acc, g_sub_acc):
        f.write(f"{st:<28} | {va:>8.2f}% | {ga:>8.2f}% | {ga-va:>+8.2f}%\n")

print(f"\n✅ Phase 5 outputs and plots saved to {RESULTS_DIR.name}/")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Phase 2 — Industrial Corruption Robustness Study on MMAD
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Purpose: Measure the empirical degradation slope (RDS), critical failure
         boundaries (CFB), and subtask vulnerabilities under 7 realistic
         industrial imaging corruptions across 5 severity levels.
Model:   Qwen/Qwen3-VL-2B-Instruct (FP16) on local RTX 4060 GPU
Metrics: RDS, CFB, Relative Robustness Index (RRI), Subtask Fragility Ranking.
==============================================================================
"""

import os
import sys
import json
import time
import random
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import cohen_kappa_score

# Import Phase 2 Corruption Engine
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corruptions import (
    CORRUPTION_TYPES,
    CORRUPTION_DISPLAY_NAMES,
    apply_corruption,
    generate_corruption_preview_grid,
)

# =============================================================================
# CELL 1: Configuration & CLI Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Phase 2: MMAD Industrial Corruption Robustness Study")
parser.add_argument("--sample-size", type=int, default=50, help="Number of questions per condition (stratified).")
parser.add_argument("--severities", type=str, default="1,2,3,4,5", help="Comma-separated severity levels (1 to 5).")
parser.add_argument("--corruptions", type=str, default="all", help="Comma-separated corruption types or 'all'.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
parser.add_argument("--reset", action="store_true", help="Clear previous checkpoint and re-run from scratch.")
args = parser.parse_args()

GLOBAL_SEED = args.seed
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

SEVERITY_LEVELS = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
if args.corruptions.lower() == "all":
    EVAL_CORRUPTIONS = CORRUPTION_TYPES
else:
    EVAL_CORRUPTIONS = [c.strip() for c in args.corruptions.split(",") if c.strip()]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MMAD_DIR = BASE_DIR / "MMAD"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = RESULTS_DIR / "phase2_manifest.json"
RESULTS_PATH = RESULTS_DIR / "phase2_results.jsonl"
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
print("PHASE 2: INDUSTRIAL CORRUPTION ROBUSTNESS STUDY")
print("=" * 80)
print(f"  Target GPU            : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)")
print(f"  PyTorch / CUDA        : {torch.__version__} / CUDA {gpu_info['cuda_version']}")
print(f"  Benchmark Sample Size : {args.sample_size} questions per condition")
print(f"  Corruptions Evaluated : {len(EVAL_CORRUPTIONS)} ({', '.join(EVAL_CORRUPTIONS)})")
print(f"  Severity Levels       : {SEVERITY_LEVELS}")
print(f"  Total Conditions      : {len(EVAL_CORRUPTIONS) * len(SEVERITY_LEVELS)} conditions + Clean")
print(f"  Total Test Inferences : {args.sample_size * (len(EVAL_CORRUPTIONS) * len(SEVERITY_LEVELS) + 1)}")
print("=" * 80)


# =============================================================================
# CELL 2: Load MMAD Dataset & Resolve Image Paths (All 4 Datasets)
# =============================================================================
mmad_json_path = MMAD_DIR / "mmad.json"
print(f"\n📂 Loading MMAD benchmark from: {mmad_json_path.name}...")

with open(mmad_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def resolve_local_image_path(rel_path, mmad_base_dir):
    clean = str(rel_path).replace("\\", "/")
    p = mmad_base_dir / clean
    if p.exists() and p.is_file():
        return str(p)

    for prefix in ["ALL_DATA/", "DS-MVTec/"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    candidates = [
        mmad_base_dir / "DS-MVTec" / "DS-MVTec" / clean,
        mmad_base_dir / "DS-MVTec" / clean,
        mmad_base_dir / "GoodsAD" / clean,
        mmad_base_dir / "VisA" / clean,
        mmad_base_dir / "MVTec-LOCO" / clean,
        mmad_base_dir / clean,
        mmad_base_dir / rel_path,
    ]
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


# =============================================================================
# CELL 3: Stratified Sampling for Robustness Benchmark
# =============================================================================
print(f"\n🎯 Performing Stratified Sampling across Subtasks & Product Categories...")
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
# CELL 4: Load Model & Processor
# =============================================================================
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
print(f"\n🤖 Loading model: {MODEL_ID} in FP16...")

t_start = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = None
loader_name = None

try:
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    loader_name = "Qwen3VLForConditionalGeneration"
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    loader_name = "AutoModelForImageTextToText"

t_load = time.time() - t_start
vram_after = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
print(f"   ✅ Model loaded via {loader_name} in {t_load:.1f}s")
print(f"   Dtype: {model.dtype} | VRAM: {vram_after:.2f} GB / {gpu_info['gpu_memory_gb']} GB")


# =============================================================================
# CELL 5: Deterministic Inference & Parser
# =============================================================================
def build_mmad_prompt(question_data):
    ci_data = {str(k).lower(): v for k, v in question_data.items()}
    q_text = ""
    for f in ["question", "query", "text", "prompt"]:
        if f in ci_data and ci_data[f]:
            q_text = str(ci_data[f]).strip()
            break
    if not q_text:
        q_text = "Identify any defect or anomaly present in this industrial object."

    opts_dict = {}
    options_text = ""
    for f in ["options", "choices", "answers"]:
        if f in ci_data and ci_data[f]:
            raw_opts = ci_data[f]
            if isinstance(raw_opts, dict):
                opts_dict = raw_opts
                for k, v in sorted(raw_opts.items()):
                    options_text += f"({k}) {v}\n"
            elif isinstance(raw_opts, list):
                for idx, v in enumerate(raw_opts):
                    k = chr(65 + idx)
                    opts_dict[k] = v
                    options_text += f"({k}) {v}\n"
            break

    prompt = f"{q_text}\n\n{options_text.strip()}\n\nAnswer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."
    return prompt, q_text, opts_dict


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


def run_inference_corrupted(model, processor, image_path, prompt_text, corruption_type, severity):
    try:
        raw_img = Image.open(image_path).convert("RGB")
        raw_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        img = apply_corruption(raw_img, corruption_type, severity)
    except Exception as e:
        img = Image.new("RGB", (224, 224), (128, 128, 128))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
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
            inputs = processor(text=[prompt_text], images=[img], return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# =============================================================================
# CELL 6: Checkpoint Setup & Robustness Evaluation Loop
# =============================================================================
if args.reset and RESULTS_PATH.exists():
    print("🧹 --reset passed: Clearing previous results checkpoint...")
    RESULTS_PATH.unlink()

completed_keys = set()
if RESULTS_PATH.exists():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    completed_keys.add((entry["question_index"], entry["corruption"], entry["severity"]))
                except Exception:
                    pass

if completed_keys:
    print(f"\n♻️ Resume Active: {len(completed_keys)} evaluations already completed.")

# Build execution matrix:
# Condition 0: Clean (severity=0)
# Condition 1..N: Each corruption at each severity
conditions = [("clean", 0)]
for c in EVAL_CORRUPTIONS:
    for s in SEVERITY_LEVELS:
        conditions.append((c, s))

total_evals_planned = len(eval_base_questions) * len(conditions)
print("\n" + "=" * 80)
print(f"STARTING ROBUSTNESS BENCHMARK ({total_evals_planned} total test evaluations)")
print("=" * 80)

eval_count = 0
t_start_eval = time.time()

for cond_idx, (c_type, sev) in enumerate(conditions):
    cond_name = "Clean Baseline (Sev 0)" if sev == 0 else f"{CORRUPTION_DISPLAY_NAMES[c_type]} (Sev {sev})"
    print(f"\n⚡ Condition [{cond_idx+1}/{len(conditions)}]: {cond_name}")

    cond_correct = 0
    cond_total = 0

    for q_idx, q in enumerate(eval_base_questions):
        eval_count += 1
        if (q_idx, c_type, sev) in completed_keys:
            continue

        prompt_text, clean_q, opt_dict = build_mmad_prompt(q)
        gt = get_ground_truth(q)
        img_path = q.get("_resolved_image_path", "")
        subtask = q.get("_subtask", "Unknown")
        category = q.get("_category", "Unknown")
        dataset_name = q.get("_dataset", "Unknown")

        t0 = time.time()
        try:
            response = run_inference_corrupted(model, processor, img_path, prompt_text, c_type, sev)
            inf_time = time.time() - t0
            err = None
        except Exception as e:
            response = ""
            inf_time = time.time() - t0
            err = str(e)

        parsed_ans, parse_ok = parse_answer(response)
        is_corr = (parsed_ans == gt) if (parsed_ans and gt) else None
        if is_corr:
            cond_correct += 1
        cond_total += 1

        gt_text = f"({gt}) {opt_dict.get(gt, '')}".strip() if gt else "N/A"
        pred_text = f"({parsed_ans}) {opt_dict.get(parsed_ans, '')}".strip() if parsed_ans else "None"

        result_entry = {
            "question_index": q_idx,
            "corruption": c_type,
            "severity": sev,
            "dataset": dataset_name,
            "subtask": subtask,
            "category": category,
            "clean_question": clean_q,
            "options": opt_dict,
            "ground_truth": gt,
            "actual_answer": gt_text,
            "model_response": response,
            "parsed_answer": parsed_ans,
            "generated_answer": pred_text,
            "parse_success": parse_ok,
            "is_correct": is_corr,
            "inference_time_s": round(inf_time, 2),
            "error": err,
            "image_path": str(img_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    if cond_total > 0:
        print(f"   --> Accuracy for {cond_name}: {(cond_correct/cond_total)*100:.1f}% ({cond_correct}/{cond_total})")

t_eval_total = time.time() - t_start_eval


# =============================================================================
# CELL 7: Compute Dissertation Robustness Metrics (RDS, CFB, RRI)
# =============================================================================
all_results = []
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_results.append(json.loads(line))

print(f"\n📊 Total evaluation entries loaded: {len(all_results)}")

# Group by (corruption, severity)
condition_stats = defaultdict(lambda: {"total": 0, "correct": 0})
# Group by (subtask, corruption, severity)
subtask_cond_stats = defaultdict(lambda: {"total": 0, "correct": 0})
# Group by (dataset, corruption, severity)
dataset_cond_stats = defaultdict(lambda: {"total": 0, "correct": 0})

for r in all_results:
    c = r["corruption"]
    s = r["severity"]
    st = r.get("subtask", "Unknown")
    ds = r.get("dataset", "Unknown")
    corr = 1 if r["is_correct"] is True else 0

    condition_stats[(c, s)]["total"] += 1
    condition_stats[(c, s)]["correct"] += corr

    subtask_cond_stats[(st, c, s)]["total"] += 1
    subtask_cond_stats[(st, c, s)]["correct"] += corr

    dataset_cond_stats[(ds, c, s)]["total"] += 1
    dataset_cond_stats[(ds, c, s)]["correct"] += corr

# Clean baseline accuracy
clean_total = condition_stats[("clean", 0)]["total"]
clean_correct = condition_stats[("clean", 0)]["correct"]
clean_acc = (clean_correct / max(clean_total, 1)) * 100

# Per-corruption accuracy across severities
corruption_curves = {}
rds_values = {}
cfb_values = {}
rri_values = {}

for c in EVAL_CORRUPTIONS:
    accs = [clean_acc]  # sev 0
    for s in SEVERITY_LEVELS:
        stat = condition_stats[(c, s)]
        acc = (stat["correct"] / max(stat["total"], 1)) * 100
        accs.append(acc)

    corruption_curves[c] = accs

    # RDS = (Clean - Sev 5) / 5
    sev5_acc = accs[-1]
    rds = round((clean_acc - sev5_acc) / 5.0, 3)
    rds_values[c] = rds

    # CFB = First severity where acc < 50.0%
    cfb = None
    for s_idx, a in enumerate(accs):
        if a < 50.0:
            cfb = s_idx
            break
    cfb_values[c] = cfb if cfb is not None else ">5 (Resilient)"

    # RRI = Mean(A_sev1..5) / Clean_Acc
    mean_corrupted_acc = np.mean(accs[1:])
    rri = round(mean_corrupted_acc / max(clean_acc, 1e-5), 3)
    rri_values[c] = rri

mean_rds = round(float(np.mean(list(rds_values.values()))), 3)

# Subtask Resilience (Accuracy at Sev 5 vs Clean)
subtasks_all = sorted(list(set(r.get("subtask", "Unknown") for r in all_results)))
subtask_resilience = {}
for st in subtasks_all:
    c_tot = subtask_cond_stats[(st, "clean", 0)]["total"]
    c_corr = subtask_cond_stats[(st, "clean", 0)]["correct"]
    st_clean_acc = (c_corr / max(c_tot, 1)) * 100

    sev5_tot = sum(subtask_cond_stats[(st, c, 5)]["total"] for c in EVAL_CORRUPTIONS)
    sev5_corr = sum(subtask_cond_stats[(st, c, 5)]["correct"] for c in EVAL_CORRUPTIONS)
    st_sev5_acc = (sev5_corr / max(sev5_tot, 1)) * 100

    subtask_resilience[st] = {
        "clean_accuracy": round(st_clean_acc, 1),
        "sev5_accuracy": round(st_sev5_acc, 1),
        "degradation_drop": round(st_clean_acc - st_sev5_acc, 1),
        "rds": round((st_clean_acc - st_sev5_acc) / 5.0, 3),
    }

# Save Manifest
MANIFEST = {
    "phase": "Phase 2 — Industrial Corruption Robustness Study",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_ID,
    "loader": loader_name,
    "gpu_info": gpu_info,
    "clean_accuracy": round(clean_acc, 2),
    "mean_rds": mean_rds,
    "severities_evaluated": SEVERITY_LEVELS,
    "corruptions_evaluated": EVAL_CORRUPTIONS,
    "rds_values": rds_values,
    "cfb_values": cfb_values,
    "rri_values": rri_values,
    "corruption_curves": corruption_curves,
    "subtask_resilience": subtask_resilience,
    "total_inferences": len(all_results),
}

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(MANIFEST, f, indent=2, ensure_ascii=False)


# =============================================================================
# CELL 8: Generate Detailed results.txt Report
# =============================================================================
W = 88
txt = [
    "=" * W,
    "          PHASE 2: INDUSTRIAL CORRUPTION ROBUSTNESS & DEGRADATION REPORT",
    "=" * W,
    "",
    "1. HARDWARE & BENCHMARK CONFIGURATION",
    "-" * W,
    f"  Model Evaluated     : {MODEL_ID} (FP16)",
    f"  Hardware Device     : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)",
    f"  Clean Baseline Acc  : {clean_acc:.1f}%",
    f"  Mean Degradation RDS: {mean_rds} % accuracy drop per severity level",
    f"  Corruptions Tested  : {len(EVAL_CORRUPTIONS)} real-world industrial imaging faults",
    f"  Severities Tested   : {SEVERITY_LEVELS} (from mild to extreme)",
    f"  Total Evaluations   : {len(all_results)} inferences",
    "",
    "2. ROBUSTNESS DEGRADATION SLOPE (RDS) & CRITICAL FAILURE BOUNDARIES (CFB)",
    "-" * W,
    f"| {'Corruption Type':<26} | {'Clean':<7} | {'Sev 1':<7} | {'Sev 3':<7} | {'Sev 5':<7} | {'RDS':<7} | {'CFB':<10} |",
    f"|{'-'*28}+{'-'*9}+{'-'*9}+{'-'*9}+{'-'*9}+{'-'*9}+{'-'*12}|",
]

for c in EVAL_CORRUPTIONS:
    c_disp = CORRUPTION_DISPLAY_NAMES[c]
    curve = corruption_curves[c]
    # Curve has indices [0=clean, 1=s1, 2=s2, 3=s3, 4=s4, 5=s5]
    s1_acc = f"{curve[1]:.1f}%" if len(curve) > 1 else "N/A"
    s3_acc = f"{curve[3]:.1f}%" if len(curve) > 3 else "N/A"
    s5_acc = f"{curve[-1]:.1f}%"
    rds = f"{rds_values[c]:+.2f}"
    cfb = str(cfb_values[c])
    txt.append(f"| {c_disp:<26} | {clean_acc:<5.1f}% | {s1_acc:<7} | {s3_acc:<7} | {s5_acc:<7} | {rds:<7} | {cfb:<10} |")

txt.extend([
    "-" * W,
    "",
    "3. SUBTASK FRAGILITY & RESILIENCE RANKING (CLEAN VS SEVERITY 5)",
    "-" * W,
    f"| {'Subtask Name':<24} | {'Clean Acc':<11} | {'Sev 5 Acc':<11} | {'Drop (%)':<10} | {'Status':<16} |",
    f"|{'-'*26}+{'-'*13}+{'-'*13}+{'-'*12}+{'-'*18}|",
])

# Sort subtasks by degradation drop
sorted_st = sorted(subtask_resilience.items(), key=lambda x: x[1]["degradation_drop"], reverse=True)
for st, d in sorted_st:
    status = "⚠️ FRAGILE" if d["degradation_drop"] > 25 else "🛡️ RESILIENT" if d["degradation_drop"] < 10 else "MODERATE"
    txt.append(f"| {st:<24} | {d['clean_accuracy']:<9.1f}% | {d['sev5_accuracy']:<9.1f}% | {d['degradation_drop']:<8.1f}% | {status:<16} |")

txt.extend([
    "-" * W,
    "",
    "4. SAMPLE CORRUPTED TEST CASE CARDS (INSPECTION LOG)",
    "-" * W,
])

# Write first 50 cards
for i, r in enumerate(all_results[:60]):
    is_corr = r["is_correct"]
    tag = "PASS [MATCH]" if is_corr else "FAIL [MISMATCH]"
    fname = os.path.basename(r.get("image_path", "N/A"))
    c_disp = CORRUPTION_DISPLAY_NAMES.get(r["corruption"], r["corruption"].title())
    sev_disp = f"Sev {r['severity']}" if r["severity"] > 0 else "Clean"

    txt.append(f"┌" + "─" * (W - 2) + "┐")
    title_bar = f"│ TEST #{i+1:04d} | {c_disp} ({sev_disp}) | Subtask: {r.get('subtask')} | {tag}"
    txt.append(title_bar + " " * max(0, (W - 1 - len(title_bar))) + "│")
    txt.append("├" + "─" * (W - 2) + "┤")
    txt.append(f"│ Image File       : {fname}")
    txt.append(f"│ Question         : {r.get('clean_question', '')[:65]}")
    txt.append(f"│ Actual Answer    : {r.get('actual_answer')}")
    txt.append(f"│ Generated Answer : {r.get('generated_answer')}")
    txt.append(f"│ Status           : {'CORRECT [PASS]' if is_corr else 'INCORRECT [FAIL]'} ({r['inference_time_s']}s)")
    txt.append("└" + "─" * (W - 2) + "┘")
    txt.append("")

with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(txt) + "\n")

print(f"📁 Detailed report saved: {RESULTS_TXT_PATH}")


# =============================================================================
# CELL 9: Generate Publication Visual Suite (4 High-Resolution Plots)
# =============================================================================
# Plot 1: Preview Grid of Corruptions
try:
    sample_img_p = eval_base_questions[0]["_resolved_image_path"]
    preview_path = RESULTS_DIR / "phase2_corruption_preview_grid.png"
    generate_corruption_preview_grid(sample_img_p, str(preview_path))
except Exception as e:
    print(f"⚠️ Could not generate Plot 1: {e}")

# Plot 2: Degradation Curves (Accuracy vs Severity)
try:
    fig, ax = plt.subplots(figsize=(12, 7))
    x_sevs = [0] + SEVERITY_LEVELS
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c", "#34495e"]

    for i, c in enumerate(EVAL_CORRUPTIONS):
        curve = corruption_curves[c]
        c_label = f"{CORRUPTION_DISPLAY_NAMES[c]} (RDS: {rds_values[c]:+.2f})"
        ax.plot(x_sevs, curve, marker=markers[i], linewidth=2.2, markersize=8, color=colors[i], label=c_label)

    # Reference lines
    ax.axhline(clean_acc, color="gray", linestyle="--", linewidth=1.2, label=f"Clean Baseline ({clean_acc:.1f}%)")
    ax.axhline(50.0, color="red", linestyle=":", linewidth=1.5, label="Critical Boundary (50% Acc)")

    ax.set_ylim(15, 100)
    ax.set_xlabel("Industrial Corruption Severity (0 = Clean, 1 = Mild, 5 = Severe)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Benchmark Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 2: Robustness Degradation Curves across Industrial Corruptions", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    p2_path = RESULTS_DIR / "phase2_degradation_curves.png"
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print(f"📊 Degradation curves saved: {p2_path.name}")
except Exception as e:
    print(f"⚠️ Could not generate Plot 2: {e}")

# Plot 3: Polar Radar Chart of RDS
try:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = [CORRUPTION_DISPLAY_NAMES[c] for c in EVAL_CORRUPTIONS]
    values = [max(0.0, rds_values[c]) for c in EVAL_CORRUPTIONS]
    # Close the loop
    categories += [categories[0]]
    values += [values[0]]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

    ax.plot(angles, values, color="#c0392b", linewidth=2.5, linestyle="solid")
    ax.fill(angles, values, color="#e74c3c", alpha=0.35)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace(" ", "\n") for c in categories[:-1]], fontsize=10, fontweight="bold")
    ax.set_title(f"Robustness Degradation Slope (RDS) Radar\n(Higher = More Fragile | Mean RDS: {mean_rds})", fontsize=13, fontweight="bold", pad=20)
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    p3_path = RESULTS_DIR / "phase2_rds_radar.png"
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print(f"📊 RDS Radar plot saved: {p3_path.name}")
except Exception as e:
    print(f"⚠️ Could not generate Plot 3: {e}")

# Plot 4: Subtask Resilience Heatmap at Severity 5
try:
    matrix_data = np.zeros((len(subtasks_all), len(EVAL_CORRUPTIONS)))
    annot_data = [["" for _ in range(len(EVAL_CORRUPTIONS))] for _ in range(len(subtasks_all))]

    for i, st in enumerate(subtasks_all):
        for j, c in enumerate(EVAL_CORRUPTIONS):
            stat = subtask_cond_stats[(st, c, 5)]
            if stat["total"] > 0:
                acc = (stat["correct"] / stat["total"]) * 100
                matrix_data[i, j] = acc
                annot_data[i][j] = f"{acc:.0f}%"

    fig, ax = plt.subplots(figsize=(max(11, len(EVAL_CORRUPTIONS) * 1.3), max(7, len(subtasks_all) * 0.85)))
    sns.heatmap(matrix_data, annot=np.array(annot_data), fmt="", cmap="RdYlGn", vmin=0, vmax=100,
                xticklabels=[CORRUPTION_DISPLAY_NAMES[c].replace(" ", "\n") for c in EVAL_CORRUPTIONS],
                yticklabels=subtasks_all, linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Accuracy at Severity 5 (%)"}, ax=ax)

    ax.set_title("Subtask Resilience Matrix under Extreme Industrial Corruptions (Severity 5)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Industrial Corruption Type", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inspection Subtask", fontsize=11, fontweight="bold")

    plt.tight_layout()
    p4_path = RESULTS_DIR / "phase2_subtask_resilience_matrix.png"
    plt.savefig(p4_path, dpi=200)
    plt.close()
    print(f"📊 Subtask resilience heatmap saved: {p4_path.name}")
except Exception as e:
    print(f"⚠️ Could not generate Plot 4: {e}")

print("\n" + "=" * 80)
print("PHASE 2 SUMMARY")
print("=" * 80)
print(f"  Clean Accuracy        : {clean_acc:.1f}%")
print(f"  Mean Degradation Slope: {mean_rds} % drop / severity")
print(f"  Most Damaging Noise   : {max(rds_values.items(), key=lambda x: x[1])[0]} (RDS: {max(rds_values.values())})")
print(f"  Most Resilient Noise  : {min(rds_values.items(), key=lambda x: x[1])[0]} (RDS: {min(rds_values.values())})")
print(f"  📁 Manifest saved     : {MANIFEST_PATH}")
print(f"  📁 Report saved       : {RESULTS_TXT_PATH}")
print("=" * 80)
print("🏁 Phase 2 Robustness Study complete! Ready for Phase 3 Adaptation / Mitigation.")
